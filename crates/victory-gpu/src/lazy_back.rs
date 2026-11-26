//! Demand-paged / lazy GPU-to-CPU readback (`VICTORY_LAZY_BACK=1`).
//!
//! Instead of eagerly DMAing GPU-written `card` and `rebar` 64 KiB pages back
//! to title RAM on the relay landing thread, eligible 64 KiB blocks are armed
//! with `PAGE_NOACCESS` across all mapped title views. If the GPU overwrites the
//! same block across multiple flights without any CPU access, zero DMA occurs.
//!
//! On the first CPU read/write fault (via `watch.rs` VEH) or internal runtime
//! read/remap (`read_into`, `read_into_fast`, `copy_mapped`, `write_mapped`,
//! `kernel_writing`, `title_memory_remapping`, `unregister`), the block is
//! fetched once on a dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue, merged
//! into title RAM via `Shadow::card_merge_relay` / `Shadow::relay_merge_rebar`,
//! and restored to `PAGE_READWRITE`.
//!
//! # Lock ordering
//! - `LAZY_STATE` is NEVER held while acquiring `FETCHER`, `watch::GUARDS`,
//!   `inplace::VIEWS`, `shadow::BUFFERS`/`LOOSE`/`STRETCHES`, or any `Shadow` lock.
//! - `FETCHER` is a strict leaf lock used only for the `COPY`-queue DMA wait.
//! - `watch::GUARDS` may acquire `LAZY_STATE` (`has_block`, `try_arm_lazy_under_guards`),
//!   which is acyclic because `LAZY_STATE` never acquires `watch::GUARDS`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, OnceLock};

use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_QUEUE_DESC, D3D12_FENCE_FLAG_NONE,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12Resource,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
};

use crate::shadow::Shadow;
use crate::watch::PAGE;

/// Default number of distinct frames with CPU/internal materializations before
/// a block promotes to hot.
const DEFAULT_PROMOTE_FRAMES: u32 = 2;
/// Distinct idle frames without a fault or active GPU write after which a hot
/// entry decays back to lazy.
const HOT_DECAY_FRAMES: u64 = 120;
/// Base interval (in frames) before an actively GPU-written hot block allows a
/// single reprobe deferral (doubles on each re-promotion up to 16x).
const HOT_REPROBE_FRAMES: u64 = 1800;
/// Bounded ring of recently materialized/discarded blocks to absorb multi-thread read-fault races.
const RECENT_CAP: usize = 2048;
/// Initial reusable readback buffer size for the dedicated lazy fetcher (4 MiB).
const FETCHER_INIT_BYTES: usize = 4 << 20;

/// Fast atomic count of currently deferred primary 64 KiB blocks.
static LAZY_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Set once any block has ever been deferred so `handle_fault` can skip `LAZY_STATE`
/// completely when lazy readback has never been used.
static EVER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Whether `VICTORY_LAZY_BACK=1` is enabled for production relay readback.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_BACK").is_some_and(|v| v == "1"))
}

/// Number of distinct frames a block must fault on before promoting to hot (`VICTORY_LAZY_PROMOTE_FRAMES`, default 2).
fn promote_frames() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_LAZY_PROMOTE_FRAMES")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|n| n.max(1))
            .unwrap_or(DEFAULT_PROMOTE_FRAMES)
    })
}

/// Whether internal runtime materializations contribute to hot block promotion
/// (`VICTORY_LAZY_INTERNAL_PROMOTE`, default `1`).
fn internal_promote_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_LAZY_INTERNAL_PROMOTE").is_none_or(|v| v != "0")
    })
}

/// Fast lock-free check for whether any lazy blocks are currently armed.
#[inline]
pub(crate) fn active() -> bool {
    LAZY_COUNT.load(Ordering::Acquire) > 0
}

struct LazyEntry {
    shadow: Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
    armed_vas: Vec<usize>,
    is_view: bool,
    cpu_clean_after: Option<u64>,
    materializing: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaterializeReason {
    CpuFault { is_write: bool },
    InternalRead,
    Retire,
}

#[derive(Clone, Copy)]
struct HotEntry {
    faults: u32,
    last_frame: u64,
    promoted_frame: u64,
    reprobes: u32,
    reprobing: bool,
}

struct LazyTable {
    pages: BTreeMap<usize, LazyEntry>,
    alias_to_primary: BTreeMap<usize, usize>,
    by_shadow: HashMap<usize, BTreeSet<usize>>,
    hot: HashMap<usize, HotEntry>,
    recent_order: VecDeque<usize>,
    recent_set: HashSet<usize>,
}

impl LazyTable {
    fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
            alias_to_primary: BTreeMap::new(),
            by_shadow: HashMap::new(),
            hot: HashMap::new(),
            recent_order: VecDeque::new(),
            recent_set: HashSet::new(),
        }
    }

    fn note_recent(&mut self, block: usize) {
        if self.recent_set.insert(block) {
            self.recent_order.push_back(block);
            if self.recent_order.len() > RECENT_CAP
                && let Some(evicted) = self.recent_order.pop_front()
            {
                self.recent_set.remove(&evicted);
            }
        }
    }

    fn clear_recent(&mut self, block: usize) {
        if self.recent_set.remove(&block) {
            self.recent_order.retain(|&b| b != block);
        }
    }
}

static LAZY_STATE: LazyLock<Mutex<LazyTable>> = LazyLock::new(|| Mutex::new(LazyTable::new()));
static LAZY_CV: Condvar = Condvar::new();

struct LazyStats {
    deferred_new: AtomicU64,
    deferred_reused: AtomicU64,
    skipped_bytes: AtomicU64,
    materialized_bytes: AtomicU64,
    read_faults: AtomicU64,
    write_faults: AtomicU64,
    internal_materializes: AtomicU64,
    retire_materializes: AtomicU64,
    dma_batches: AtomicU64,
    hot_promoted: AtomicU64,
}

static STATS: LazyStats = LazyStats {
    deferred_new: AtomicU64::new(0),
    deferred_reused: AtomicU64::new(0),
    skipped_bytes: AtomicU64::new(0),
    materialized_bytes: AtomicU64::new(0),
    read_faults: AtomicU64::new(0),
    write_faults: AtomicU64::new(0),
    internal_materializes: AtomicU64::new(0),
    retire_materializes: AtomicU64::new(0),
    dma_batches: AtomicU64::new(0),
    hot_promoted: AtomicU64::new(0),
};

std::thread_local! {
    /// Primary blocks currently being materialized on this thread so nested
    /// `read_into` / `write_mapped` inside `card_merge_relay` / `relay_merge_rebar`
    /// never wait on `LAZY_CV` for themselves.
    static IN_MATERIALIZE: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn current_thread_materializing(block: usize) -> bool {
    IN_MATERIALIZE.with(|v| v.borrow().contains(&block))
}

struct MaterializeBatchScope {
    blocks: Vec<usize>,
    finished: bool,
}

impl MaterializeBatchScope {
    fn enter(blocks: Vec<usize>) -> Self {
        IN_MATERIALIZE.with(|v| v.borrow_mut().extend_from_slice(&blocks));
        Self {
            blocks,
            finished: false,
        }
    }
}

impl Drop for MaterializeBatchScope {
    fn drop(&mut self) {
        IN_MATERIALIZE.with(|v| {
            let mut list = v.borrow_mut();
            for &block in self.blocks.iter().rev() {
                if let Some(pos) = list.iter().rposition(|&b| b == block) {
                    list.remove(pos);
                }
            }
        });
        if !self.finished {
            // Unwind safety: ensure armed pages do not stay stuck in `materializing` or `PAGE_NOACCESS`.
            let mut removed = Vec::with_capacity(self.blocks.len());
            {
                let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                for &block in &self.blocks {
                    if let Some(entry) = remove_entry_locked_no_notify(&mut table, block, true) {
                        removed.push(entry);
                    }
                }
                LAZY_COUNT.store(table.pages.len(), Ordering::Release);
                LAZY_CV.notify_all();
            }
            for entry in &removed {
                for &va in &entry.armed_vas {
                    let mut old = 0u32;
                    // SAFETY: restoring title view protection on unwind.
                    unsafe { VirtualProtect(va as *const _, PAGE, PAGE_READWRITE, &mut old) };
                }
            }
            drop(removed);
        }
    }
}

struct LazyFetcher {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
    readback: Option<(ID3D12Resource, *mut u8, usize)>,
}

// SAFETY: D3D12 COM pointers and Win32 event HANDLE are accessed exclusively under `FETCHER`.
unsafe impl Send for LazyFetcher {}

impl Drop for LazyFetcher {
    fn drop(&mut self) {
        // SAFETY: `self.event` was created by `CreateEventW` and is owned solely by `LazyFetcher`.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

impl LazyFetcher {
    fn new() -> windows::core::Result<Self> {
        let host = crate::host::host().ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
        // SAFETY: plain D3D12 COPY queue, allocator, list, fence, and Win32 auto-reset event.
        unsafe {
            let queue: ID3D12CommandQueue = host.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator =
                host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                host.device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let readback = host
                .shared_buffer(FETCHER_INIT_BYTES as u64)
                .or_else(|| host.shared_buffer(PAGE as u64))
                .map(|(res, ptr)| {
                    crate::host::name(&res, || "lazy copy-back readback".to_owned());
                    let cap = if ptr.is_null() { 0 } else { res.GetDesc().Width as usize };
                    (res, ptr, cap)
                });
            Ok(Self {
                queue,
                allocator,
                list,
                fence,
                event,
                sent: 0,
                readback,
            })
        }
    }

    #[cfg(test)]
    fn fetch(&mut self, shadow: &Shadow, offset: usize, len: usize) -> windows::core::Result<Vec<u8>> {
        let (buf, _, _) = self.fetch_batch(&[(shadow, offset, len)], true)?;
        Ok(buf)
    }

    fn fetch_batch(
        &mut self,
        items: &[(&Shadow, usize, usize)],
        wait_direct: bool,
    ) -> windows::core::Result<(Vec<u8>, Vec<(usize, usize)>, u64)> {
        if items.is_empty() {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        let host = crate::host::host().ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
        let mut total_bytes = 0usize;
        let mut slices = Vec::with_capacity(items.len());
        for &(_, _, len) in items {
            let dst_off = total_bytes;
            total_bytes = total_bytes
                .checked_add(len)
                .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
            slices.push((dst_off, len));
        }
        let needed = total_bytes.max(PAGE);
        if self.readback.as_ref().is_none_or(|(_, _, cap)| *cap < needed) {
            let alloc_len = needed.max(FETCHER_INIT_BYTES).next_multiple_of(PAGE);
            let (res, ptr) = host
                .shared_buffer(alloc_len as u64)
                .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
            crate::host::name(&res, || "lazy copy-back readback".to_owned());
            self.readback = Some((res, ptr, alloc_len));
        }
        let (rb_res, rb_ptr, _) = self
            .readback
            .as_ref()
            .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;

        // SAFETY: every `shadow.buffer` is live (held by `Arc<Shadow>`) in `COMMON` state
        // (implicit promotion to `COPY_SOURCE` and decay back to `COMMON` at
        // `ExecuteCommandLists`), `rb_res` is live with capacity >= `total_bytes`, and
        // `FETCHER` serializes this queue/list/fence.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            for (&(shadow, offset, _), &(dst_off, len)) in items.iter().zip(&slices) {
                if len > 0 {
                    self.list.CopyBufferRegion(
                        rb_res,
                        dst_off as u64,
                        &shadow.buffer,
                        offset as u64,
                        len as u64,
                    );
                }
            }
            self.list.Close()?;
            let fetched_after = host.sync_copy_queue_for(&self.queue, wait_direct)?;
            let lists = [Some(self.list.cast::<ID3D12CommandList>()?)];
            self.queue.ExecuteCommandLists(&lists);
            self.sent += 1;
            STATS.dma_batches.fetch_add(1, Ordering::Relaxed);
            self.queue.Signal(&self.fence, self.sent)?;
            self.fence.SetEventOnCompletion(self.sent, self.event)?;
            WaitForSingleObject(self.event, INFINITE);
            let mut out = vec![0u8; total_bytes];
            if total_bytes > 0 {
                std::ptr::copy_nonoverlapping(*rb_ptr, out.as_mut_ptr(), total_bytes);
            }
            Ok((out, slices, fetched_after))
        }
    }
}

/// Leaf lock protecting the dedicated `COPY`-queue fetcher. Never acquired inside `LAZY_STATE`,
/// `watch::GUARDS`, `shadow::BUFFERS`, or any `Shadow` lock.
static FETCHER: Mutex<Option<LazyFetcher>> = Mutex::new(None);

#[cfg(test)]
pub(crate) fn fetcher_sent_count() -> u64 {
    let guard = FETCHER.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map_or(0, |f| f.sent)
}

fn fetch_gpu_batch(
    items: &[(&Shadow, usize, usize)],
) -> windows::core::Result<(Vec<u8>, Vec<(usize, usize)>, u64)> {
    if items.is_empty() {
        return Ok((Vec::new(), Vec::new(), 0));
    }
    // Check in-flight DIRECT-queue writers outside `FETCHER` (lock order).
    // When none of the batch's shadows have pending shader writes or unlanded
    // card uploads, the hardware COPY queue runs immediately without stalling
    // behind unrelated in-flight 3D work on `Host::queue`.
    let wait_direct = items
        .iter()
        .any(|&(shadow, _, _)| shadow.has_inflight_gpu_write());
    let mut guard = FETCHER.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(LazyFetcher::new()?);
    }
    guard
        .as_mut()
        .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?
        .fetch_batch(items, wait_direct)
}

#[cfg(test)]
fn fetch_gpu_bytes(shadow: &Shadow, offset: usize, len: usize) -> windows::core::Result<Vec<u8>> {
    let mut guard = FETCHER.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(LazyFetcher::new()?);
    }
    guard
        .as_mut()
        .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?
        .fetch(shadow, offset, len)
}

fn remove_entry_locked_no_notify(
    table: &mut LazyTable,
    primary_block: usize,
    record_recent: bool,
) -> Option<LazyEntry> {
    let entry = table.pages.remove(&primary_block)?;
    if entry.is_view {
        crate::watch::clear_lazy_view_armed(&entry.armed_vas);
    }
    for &va in &entry.armed_vas {
        let b = va / PAGE;
        table.alias_to_primary.remove(&b);
        if record_recent {
            table.note_recent(b);
        }
    }
    let shadow_ptr = Arc::as_ptr(&entry.shadow) as usize;
    if let Some(blocks) = table.by_shadow.get_mut(&shadow_ptr) {
        blocks.remove(&primary_block);
        if blocks.is_empty() {
            table.by_shadow.remove(&shadow_ptr);
        }
    }
    Some(entry)
}

fn is_readable_now(at: usize) -> bool {
    // SAFETY: plain `VirtualQuery` into a zeroed `MEMORY_BASIC_INFORMATION`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && matches!(
            info.Protect,
            PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE
        )
}

/// Attempts to defer copy-back of `[offset, offset + len)` in `shadow` by arming
/// `PAGE_NOACCESS` across its title virtual address and any mapped section aliases.
/// Returns `true` if the block was deferred (new or reused), `false` if the caller
/// must copy it back eagerly.
pub(crate) fn try_defer_page(
    shadow: &Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
) -> bool {
    if !enabled() {
        return false;
    }
    try_defer_page_inner(shadow, offset, len, fetched_after)
}

#[cfg(test)]
std::thread_local! {
    static BEFORE_PROTECT_HOOK: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}

fn try_defer_page_inner(
    shadow: &Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
) -> bool {
    if len != PAGE
        || shadow.is_placed()
        || (!shadow.carded() && !shadow.is_rebar())
        || offset.saturating_add(len) > shadow.len() as usize
    {
        return false;
    }
    let primary_va = (shadow.start() as usize).wrapping_add(offset);
    if !primary_va.is_multiple_of(PAGE) || crate::host::host().is_none() {
        return false;
    }
    let primary_block = primary_va / PAGE;
    let frame = crate::frame::current();

    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hot) = table.hot.get_mut(&primary_block) {
            let threshold = promote_frames();
            if frame > hot.last_frame.saturating_add(HOT_DECAY_FRAMES) {
                table.hot.remove(&primary_block);
            } else if hot.faults >= threshold {
                let reprobe_after = HOT_REPROBE_FRAMES << hot.reprobes.min(4);
                if frame > hot.promoted_frame.saturating_add(reprobe_after) {
                    hot.faults = threshold.saturating_sub(1);
                    hot.last_frame = frame;
                    hot.promoted_frame = frame;
                    hot.reprobes = hot.reprobes.saturating_add(1);
                    hot.reprobing = true;
                } else {
                    hot.last_frame = frame;
                    return false;
                }
            }
        }
        if let Some(existing) = table.pages.get_mut(&primary_block) {
            if existing.materializing || !Arc::ptr_eq(&existing.shadow, shadow) {
                return false;
            }
            existing.fetched_after = existing.fetched_after.max(fetched_after);
            if existing.is_view {
                crate::watch::note_lazy_view_reused(&existing.armed_vas);
            }
            STATS.deferred_reused.fetch_add(1, Ordering::Relaxed);
            STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
            return true;
        }
        if table.alias_to_primary.contains_key(&primary_block) {
            return false;
        }
    }

    let is_view =
        crate::inplace::has_view_at(primary_va) || crate::watch::is_known_view_block(primary_block);
    let mut vas = crate::inplace::pages_of(primary_va);
    if !vas.contains(&primary_va) {
        vas.push(primary_va);
    }
    if let Some(pos) = vas.iter().position(|&va| va == primary_va)
        && pos != 0
    {
        // Arm `primary_va` first so a protection failure on it never leaves aliases modified.
        vas.swap(0, pos);
    }
    if vas.len() > crate::watch::MANY_VIEWS || vas.iter().any(|&va| !va.is_multiple_of(PAGE)) {
        return false;
    }
    let blocks: Vec<usize> = vas.iter().map(|&va| va / PAGE).collect();

    crate::watch::ensure_handler();
    crate::watch::try_arm_lazy_under_guards(&blocks, |cpu_clean_after| {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = table.pages.get_mut(&primary_block) {
            if existing.materializing || !Arc::ptr_eq(&existing.shadow, shadow) {
                return false;
            }
            existing.fetched_after = existing.fetched_after.max(fetched_after);
            if existing.is_view {
                crate::watch::note_lazy_view_reused(&existing.armed_vas);
            }
            STATS.deferred_reused.fetch_add(1, Ordering::Relaxed);
            STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
            return true;
        }
        if table.alias_to_primary.contains_key(&primary_block)
            || vas
                .iter()
                .any(|&va| table.alias_to_primary.contains_key(&(va / PAGE)))
        {
            return false;
        }

        // Publish `EVER_ACTIVE` BEFORE arming `PAGE_NOACCESS` so any thread faulting on
        // the very first deferred page observes `EVER_ACTIVE == true` in `handle_fault`.
        EVER_ACTIVE.store(true, Ordering::Release);
        #[cfg(test)]
        BEFORE_PROTECT_HOOK.with(|hook| {
            if let Some(cb) = hook.borrow_mut().as_mut() {
                cb();
            }
        });

        let mut protected: Vec<(usize, u32)> = Vec::with_capacity(vas.len());
        for &va in &vas {
            let mut old = 0u32;
            // SAFETY: `va` is a 64 KiB-aligned page in title memory; failures on `primary_va`
            // immediately roll back all previously modified protections.
            let ok = unsafe { VirtualProtect(va as *const _, PAGE, PAGE_NOACCESS, &mut old) != 0 };
            if ok {
                protected.push((va, old));
            } else if va == primary_va {
                for (prev_va, prev_prot) in protected {
                    let mut dummy = 0u32;
                    // SAFETY: rolling back protection changed moments ago.
                    unsafe { VirtualProtect(prev_va as *const _, PAGE, prev_prot, &mut dummy) };
                }
                return false;
            }
        }

        let armed_vas: Vec<usize> = protected.into_iter().map(|(va, _)| va).collect();
        if is_view {
            crate::watch::note_lazy_view_armed(&armed_vas);
        }
        for &va in &armed_vas {
            let b = va / PAGE;
            table.clear_recent(b);
            table.alias_to_primary.insert(b, primary_block);
        }
        table
            .by_shadow
            .entry(Arc::as_ptr(shadow) as usize)
            .or_default()
            .insert(primary_block);
        table.pages.insert(
            primary_block,
            LazyEntry {
                shadow: shadow.clone(),
                offset,
                len,
                fetched_after,
                armed_vas,
                is_view,
                cpu_clean_after: if is_view { cpu_clean_after } else { None },
                materializing: false,
            },
        );
        LAZY_COUNT.store(table.pages.len(), Ordering::Release);
        STATS.deferred_new.fetch_add(1, Ordering::Relaxed);
        STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
        true
    })
}

/// Returns `true` if `block` is currently registered as a lazy block (or alias of one).
pub(crate) fn has_block(block: usize) -> bool {
    if !active() {
        return false;
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table.alias_to_primary.contains_key(&block)
}

/// Returns `true` if `shadow_ptr` currently owns `page` as a deferred primary block.
pub(crate) fn owns_block(shadow_ptr: usize, page: usize) -> bool {
    if !active() {
        return false;
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table
        .pages
        .get(&page)
        .is_some_and(|e| Arc::as_ptr(&e.shadow) as usize == shadow_ptr)
}

/// Returns all primary 64 KiB block indices in `[first_page, last_page)` owned by `shadow_ptr`.
pub(crate) fn owned_blocks_in(shadow_ptr: usize, first_page: usize, last_page: usize) -> Vec<usize> {
    if !active() || first_page >= last_page {
        return Vec::new();
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(blocks) = table.by_shadow.get(&shadow_ptr) else {
        return Vec::new();
    };
    blocks.range(first_page..last_page).copied().collect()
}

/// Returns sorted `(block, pre_epoch)` pairs for non-`materializing` view
/// blocks in `[first_page, last_page)` owned by `shadow_ptr` that were
/// guard-clean before being armed.
pub(crate) fn owned_cpu_clean_blocks_in(
    shadow_ptr: usize,
    first_page: usize,
    last_page: usize,
) -> Vec<(usize, u64)> {
    if !active() || first_page >= last_page {
        return Vec::new();
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(blocks) = table.by_shadow.get(&shadow_ptr) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &b in blocks.range(first_page..last_page) {
        if let Some(entry) = table.pages.get(&b)
            && !entry.materializing
            && let Some(pre_epoch) = entry.cpu_clean_after
        {
            out.push((b, pre_epoch));
        }
    }
    out
}

fn note_fault_hot_locked(table: &mut LazyTable, primary_block: usize) {
    let frame = crate::frame::current();
    let threshold = promote_frames();
    let entry = table.hot.entry(primary_block).or_insert(HotEntry {
        faults: 0,
        last_frame: frame,
        promoted_frame: frame,
        reprobes: 0,
        reprobing: false,
    });
    if entry.faults == 0
        || (!entry.reprobing && frame > entry.last_frame.saturating_add(HOT_DECAY_FRAMES))
    {
        entry.faults = 1;
        entry.last_frame = frame;
        entry.reprobing = false;
        if threshold == 1 {
            entry.promoted_frame = frame;
            STATS.hot_promoted.fetch_add(1, Ordering::Relaxed);
        }
    } else if entry.reprobing || frame != entry.last_frame {
        let was_below = entry.faults < threshold;
        entry.faults = entry.faults.saturating_add(1);
        entry.last_frame = frame;
        entry.reprobing = false;
        if was_below && entry.faults >= threshold {
            entry.promoted_frame = frame;
            STATS.hot_promoted.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct ClaimedBlock {
    primary_block: usize,
    shadow: Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
    armed_vas: Vec<usize>,
}

fn materialize_claimed_batch(claimed: Vec<ClaimedBlock>, reason: MaterializeReason) {
    if claimed.is_empty() {
        return;
    }
    let scope_blocks: Vec<usize> = claimed.iter().map(|c| c.primary_block).collect();
    let mut scope = MaterializeBatchScope::enter(scope_blocks);
    let fetch_items: Vec<(&Shadow, usize, usize)> = claimed
        .iter()
        .map(|c| (&*c.shadow, c.offset, c.len))
        .collect();
    let fetched = fetch_gpu_batch(&fetch_items);

    let restore_prot = |vas: &[usize]| {
        for &va in vas {
            let mut old = 0u32;
            // SAFETY: `va` was armed by us with `PAGE_NOACCESS`; restoring `PAGE_READWRITE`.
            unsafe { VirtualProtect(va as *const _, PAGE, PAGE_READWRITE, &mut old) };
        }
    };

    match fetched {
        Ok((buf, slices, live_fetched_after)) => {
            for (item, (buf_off, buf_len)) in claimed.iter().zip(slices) {
                let gpu_bytes = &buf[buf_off..buf_off + buf_len];
                let merge_after = item.fetched_after.max(live_fetched_after);
                // Section-backed views merge GPU->title bytes via `inplace` own `PAGE_READWRITE`
                // views BEFORE restoring `PAGE_READWRITE` on `armed_vas`, so concurrent title
                // threads never observe stale pre-GPU-write RAM without faulting.
                let merged_via_own = if crate::inplace::has_view_at(item.primary_block * PAGE) {
                    if item.shadow.carded() {
                        let mut scratch = Vec::new();
                        item.shadow
                            .card_merge_relay_via_own(
                                item.offset,
                                gpu_bytes,
                                merge_after,
                                &mut scratch,
                            )
                            .is_some()
                    } else if item.shadow.is_rebar() {
                        if item
                            .shadow
                            .relay_merge_rebar_via_own(item.offset, gpu_bytes)
                            .is_some()
                        {
                            item.shadow.mark_rebar_dirty_range(item.offset, item.len);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                // Private non-section allocations (or section views whose own view could not be
                // acquired, e.g. during an active remap) have no `PAGE_READWRITE` own view, so
                // `armed_vas` must be restored to `PAGE_READWRITE` before `ReadProcessMemory` /
                // `NtWriteVirtualMemory` in `card_merge_relay` / `relay_merge_rebar` can run.
                restore_prot(&item.armed_vas);

                if !merged_via_own {
                    if item.shadow.carded() {
                        let mut scratch = Vec::new();
                        item.shadow.card_merge_relay(
                            item.offset,
                            gpu_bytes,
                            merge_after,
                            &mut scratch,
                        );
                    } else if item.shadow.is_rebar() {
                        item.shadow.relay_merge_rebar(
                            &[(item.offset, item.offset + item.len)],
                            &[gpu_bytes],
                        );
                        item.shadow.mark_rebar_dirty_range(item.offset, item.len);
                    }
                }
                if reason == (MaterializeReason::CpuFault { is_write: true }) {
                    for &va in &item.armed_vas {
                        crate::watch::wrote(va, PAGE);
                    }
                }
                STATS
                    .materialized_bytes
                    .fetch_add(item.len as u64, Ordering::Relaxed);
            }
        }
        Err(e) => {
            for item in &claimed {
                restore_prot(&item.armed_vas);
            }
            tracing::warn!(
                error = %e,
                count = claimed.len(),
                "lazy copy-back batched DMA fetch failed"
            );
        }
    }

    let mut removed = Vec::with_capacity(claimed.len());
    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        for item in &claimed {
            if let Some(entry) = remove_entry_locked_no_notify(&mut table, item.primary_block, true)
            {
                removed.push(entry);
            }
        }
        LAZY_COUNT.store(table.pages.len(), Ordering::Release);
        LAZY_CV.notify_all();
    }
    scope.finished = true;
    drop(scope);
    drop(removed);
    drop(claimed);
}

/// Materializes `primary_block` outside `LAZY_STATE`.
/// Returns `true` if `primary_block` was active (or waited on another thread materializing it).
fn materialize_primary(primary_block: usize, reason: MaterializeReason) -> bool {
    let claimed = {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let mut waited = false;
        loop {
            match table.pages.get_mut(&primary_block) {
                Some(entry) if entry.materializing => {
                    waited = true;
                    table = LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner());
                }
                Some(entry) => {
                    entry.materializing = true;
                    let claimed = ClaimedBlock {
                        primary_block,
                        shadow: entry.shadow.clone(),
                        offset: entry.offset,
                        len: entry.len,
                        fetched_after: entry.fetched_after,
                        armed_vas: entry.armed_vas.clone(),
                    };
                    match reason {
                        MaterializeReason::CpuFault { is_write } => {
                            note_fault_hot_locked(&mut table, primary_block);
                            if is_write {
                                STATS.write_faults.fetch_add(1, Ordering::Relaxed);
                            } else {
                                STATS.read_faults.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        MaterializeReason::InternalRead => {
                            if internal_promote_enabled() {
                                note_fault_hot_locked(&mut table, primary_block);
                            }
                            STATS.internal_materializes.fetch_add(1, Ordering::Relaxed);
                        }
                        MaterializeReason::Retire => {
                            STATS.retire_materializes.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    break claimed;
                }
                None => return waited,
            }
        }
    };

    materialize_claimed_batch(vec![claimed], reason);
    true
}

/// Handles a CPU `EXCEPTION_ACCESS_VIOLATION` (`is_write == true` for write, `false` for read)
/// at `at`. Called by `watch.rs`'s VEH handler before checking write guards.
pub(crate) fn handle_fault(at: usize, is_write: bool) -> bool {
    if !active() && !EVER_ACTIVE.load(Ordering::Acquire) {
        return false;
    }
    let va_block = at / PAGE;
    let primary = {
        let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&p) = table.alias_to_primary.get(&va_block) {
            Some(p)
        } else if !is_write && table.recent_set.contains(&va_block) && is_readable_now(at) {
            // Another thread just materialized this block to readable between our CPU fault and now.
            return true;
        } else {
            None
        }
    };
    let Some(primary_block) = primary else {
        return false;
    };
    if materialize_primary(primary_block, MaterializeReason::CpuFault { is_write }) {
        return true;
    }
    if !is_write {
        let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        if table.recent_set.contains(&va_block) && is_readable_now(at) {
            return true;
        }
    }
    false
}

/// Materializes any deferred lazy blocks overlapping any `[at, at + len)` in `ranges`
/// in a single batched `COPY`-queue DMA before internal runtime reads, kernel writes,
/// copies, or memory remaps.
pub(crate) fn materialize_ranges(ranges: &[(usize, usize)]) {
    if ranges.is_empty() || !active() {
        return;
    }
    loop {
        let (claimed, wait_only) = {
            let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            let mut to_claim = BTreeSet::new();
            let mut wait_only = false;
            for &(at, len) in ranges {
                if len == 0 {
                    continue;
                }
                let end = at.saturating_add(len);
                let first = at / PAGE;
                let last = end.div_ceil(PAGE);
                for (_, &primary_block) in table.alias_to_primary.range(first..last) {
                    if current_thread_materializing(primary_block) {
                        continue;
                    }
                    if table
                        .pages
                        .get(&primary_block)
                        .is_some_and(|e| e.materializing)
                    {
                        wait_only = true;
                    } else {
                        to_claim.insert(primary_block);
                    }
                }
            }
            if to_claim.is_empty() {
                if wait_only {
                    drop(LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner()));
                    continue;
                }
                return;
            }
            let do_promote = internal_promote_enabled();
            let mut claimed = Vec::with_capacity(to_claim.len());
            for primary_block in to_claim {
                if let Some(entry) = table.pages.get_mut(&primary_block)
                    && !entry.materializing
                {
                    entry.materializing = true;
                    claimed.push(ClaimedBlock {
                        primary_block,
                        shadow: entry.shadow.clone(),
                        offset: entry.offset,
                        len: entry.len,
                        fetched_after: entry.fetched_after,
                        armed_vas: entry.armed_vas.clone(),
                    });
                    if do_promote {
                        note_fault_hot_locked(&mut table, primary_block);
                    }
                    STATS.internal_materializes.fetch_add(1, Ordering::Relaxed);
                }
            }
            (claimed, wait_only)
        };
        if !claimed.is_empty() {
            materialize_claimed_batch(claimed, MaterializeReason::InternalRead);
        }
        if !wait_only {
            return;
        }
    }
}

/// Materializes any deferred lazy blocks overlapping `[at, at + len)` before an internal
/// runtime read, kernel write, copy, or memory remap.
pub(crate) fn materialize_range(at: usize, len: usize) {
    if len > 0 && active() {
        materialize_ranges(&[(at, len)]);
    }
}

/// Materializes all deferred lazy blocks owned by `shadow_ptr` before retirement or unregister
/// in a single batched `COPY`-queue DMA without hot promotion.
pub(crate) fn materialize_shadow(shadow_ptr: usize) {
    if !active() {
        return;
    }
    loop {
        let (claimed, wait_only) = {
            let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            let mut to_claim = Vec::new();
            let mut wait_only = false;
            if let Some(blocks) = table.by_shadow.get(&shadow_ptr) {
                for &b in blocks {
                    if current_thread_materializing(b) {
                        continue;
                    }
                    if table.pages.get(&b).is_some_and(|e| e.materializing) {
                        wait_only = true;
                    } else {
                        to_claim.push(b);
                    }
                }
            }
            if to_claim.is_empty() {
                if wait_only {
                    drop(LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner()));
                    continue;
                }
                return;
            }
            let mut claimed = Vec::with_capacity(to_claim.len());
            for primary_block in to_claim {
                if let Some(entry) = table.pages.get_mut(&primary_block)
                    && !entry.materializing
                {
                    entry.materializing = true;
                    STATS.retire_materializes.fetch_add(1, Ordering::Relaxed);
                    claimed.push(ClaimedBlock {
                        primary_block,
                        shadow: entry.shadow.clone(),
                        offset: entry.offset,
                        len: entry.len,
                        fetched_after: entry.fetched_after,
                        armed_vas: entry.armed_vas.clone(),
                    });
                }
            }
            (claimed, wait_only)
        };
        if !claimed.is_empty() {
            materialize_claimed_batch(claimed, MaterializeReason::Retire);
        }
        if !wait_only {
            return;
        }
    }
}

/// Discards deferred entries (or unmapped alias VAs) in `[at, at + len)` after
/// `title_memory_remapped`.
pub(crate) fn discard_range(at: usize, len: usize) {
    if len == 0 || !active() {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / PAGE;
    let last = end.div_ceil(PAGE);
    let mut dropped: Vec<LazyEntry> = Vec::new();
    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let waiting = table
                .alias_to_primary
                .range(first..last)
                .any(|(_, &p)| table.pages.get(&p).is_some_and(|e| e.materializing));
            if waiting {
                table = LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner());
            } else {
                break;
            }
        }
        let hits: Vec<(usize, usize)> = table
            .alias_to_primary
            .range(first..last)
            .map(|(&va_block, &primary)| (va_block, primary))
            .collect();
        for (va_block, primary) in hits {
            if va_block == primary {
                if let Some(entry) = remove_entry_locked_no_notify(&mut table, primary, true) {
                    dropped.push(entry);
                }
            } else {
                table.alias_to_primary.remove(&va_block);
                table.note_recent(va_block);
                if let Some(entry) = table.pages.get_mut(&primary) {
                    if entry.is_view {
                        crate::watch::clear_lazy_view_armed_block(va_block);
                    }
                    entry.armed_vas.retain(|&va| va / PAGE != va_block);
                }
            }
        }
        if !dropped.is_empty() {
            LAZY_COUNT.store(table.pages.len(), Ordering::Release);
            LAZY_CV.notify_all();
        }
        for b in first..last {
            if table.hot.is_empty() {
                break;
            }
            table.hot.remove(&b);
        }
    }
    drop(dropped);
}

/// Drops any lazy entries tracked for `shadow_ptr` on `Shadow::drop`.
pub(crate) fn discard_shadow(shadow_ptr: usize) {
    if !active() {
        return;
    }
    let mut dropped: Vec<LazyEntry> = Vec::new();
    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let blocks: Vec<usize> = table
            .by_shadow
            .get(&shadow_ptr)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        for b in blocks {
            if let Some(entry) = remove_entry_locked_no_notify(&mut table, b, true) {
                dropped.push(entry);
            }
        }
        if !dropped.is_empty() {
            LAZY_COUNT.store(table.pages.len(), Ordering::Release);
            LAZY_CV.notify_all();
        }
    }
    drop(dropped);
}

/// Logs the periodic 5s telemetry summary when `VICTORY_LAZY_BACK=1` is enabled.
pub(crate) fn log_stats(frames: f64) {
    if !enabled() || frames <= 0.0 {
        return;
    }
    let deferred_new = STATS.deferred_new.swap(0, Ordering::Relaxed) as f64 / frames;
    let deferred_reused = STATS.deferred_reused.swap(0, Ordering::Relaxed) as f64 / frames;
    let skipped_mb =
        STATS.skipped_bytes.swap(0, Ordering::Relaxed) as f64 / (1024.0 * 1024.0) / frames;
    let materialized_mb =
        STATS.materialized_bytes.swap(0, Ordering::Relaxed) as f64 / (1024.0 * 1024.0) / frames;
    let read_faults = STATS.read_faults.swap(0, Ordering::Relaxed) as f64 / frames;
    let write_faults = STATS.write_faults.swap(0, Ordering::Relaxed) as f64 / frames;
    let internal = STATS.internal_materializes.swap(0, Ordering::Relaxed) as f64 / frames;
    let retire = STATS.retire_materializes.swap(0, Ordering::Relaxed) as f64 / frames;
    let batches = STATS.dma_batches.swap(0, Ordering::Relaxed) as f64 / frames;
    let hot_promoted = STATS.hot_promoted.swap(0, Ordering::Relaxed) as f64;
    tracing::info!(
        "lazy copy-back (VICTORY_LAZY_BACK=1): deferred {deferred_new:.1} new / {deferred_reused:.1} reused ({skipped_mb:.1} MB/frame skipped); materialized {materialized_mb:.1} MB/frame (read faults {read_faults:.1}, write faults {write_faults:.1}, internal {internal:.1}, retire {retire:.1}, dma batches {batches:.1}); hot promoted {hot_promoted:.0}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MEM_COMMIT, MEM_RELEASE,
        MEM_RESERVE, MapViewOfFile, UnmapViewOfFile, VirtualAlloc, VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    fn alloc_watch_pages(pages: usize) -> *mut u8 {
        // SAFETY: fresh private write-watched title allocation for tests.
        let p = unsafe {
            VirtualAlloc(
                ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!p.is_null(), "VirtualAlloc failed");
        p
    }

    unsafe fn free_watch_pages(p: *mut u8) {
        // SAFETY: allocation from `alloc_watch_pages`.
        unsafe {
            VirtualFree(p.cast(), 0, MEM_RELEASE);
        }
    }

    fn make_card_shadow_at(title_start: u64, pages: usize, init_byte: u8) -> Option<(Arc<Shadow>, Vec<u8>)> {
        let len = pages * PAGE;
        let mut side = vec![init_byte; len];
        let shadow = Shadow::new_card_sized_for_test(side.as_mut_ptr(), title_start, len as u64)?;
        // Seed `shadow.buffer` (card side) with `init_byte` so non-written bytes match `side`.
        let h = crate::host::host()?;
        let (up, up_ptr) = h.upload_buffer(len as u64)?;
        // SAFETY: `up_ptr` is mapped for `len` bytes.
        unsafe { ptr::write_bytes(up_ptr, init_byte, len) };
        let list = h.new_list().ok()?;
        // SAFETY: live D3D12 buffers in COMMON / GENERIC_READ states.
        unsafe {
            list.list.CopyBufferRegion(&shadow.buffer, 0, &up, 0, len as u64);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();
        Some((shadow, side))
    }

    fn gpu_write_shadow(shadow: &Shadow, offset: usize, bytes: &[u8]) {
        let h = crate::host::host().expect("host GPU");
        let (up, up_ptr) = h.upload_buffer(bytes.len() as u64).expect("upload buffer");
        // SAFETY: `up_ptr` is mapped for `bytes.len()` bytes.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), up_ptr, bytes.len()) };
        let list = h.new_list().expect("host list");
        // SAFETY: `shadow.buffer` is live in COMMON state.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, offset as u64, &up, 0, bytes.len() as u64);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();
        let _ = D3D12_RESOURCE_STATES::default();
    }

    fn advance_frames(n: usize) {
        let mut token = 0u64;
        for _ in 0..n {
            let _ = crate::frame::wait(ptr::null(), 0, 0, ptr::null(), 2, &mut token);
        }
    }

    fn clear_hot_for_test(block: usize) {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        table.hot.remove(&block);
    }

    #[test]
    fn lazy_read_fault_materializes_gpu_bytes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(2);
        // SAFETY: `ptr` owns `2 * PAGE` committed bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, 2 * PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);
        clear_hot_for_test(va / PAGE + 1);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 2, 0x11) else {
            // SAFETY: freeing test allocation.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 64, &[0xAA; 64]);

        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(va / PAGE));
        assert!(!is_readable_now(va));

        // Real CPU read fault via VEH.
        // SAFETY: reading inside the deferred allocation triggers VEH materialization.
        let got = unsafe { std::slice::from_raw_parts(ptr.add(64), 64).to_vec() };
        assert_eq!(got, vec![0xAA; 64]);
        // Untouched byte in the same page remains 0x11.
        // SAFETY: inside `ptr`.
        assert_eq!(unsafe { ptr.read() }, 0x11);
        assert!(!has_block(va / PAGE));
        assert!(is_readable_now(va));
        // Page 1 was never deferred or touched.
        assert!(is_readable_now(va + PAGE));

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: test allocation cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_write_fault_preserves_both_gpu_and_cpu_bytes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);
        // Reset initial write-watch bits from the seed memset.
        let _ = crate::watch::writes(va, va + PAGE, false);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 32, &[0xBB; 8]);

        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        // Real CPU write fault via VEH at offset 128.
        // SAFETY: writing inside deferred page triggers VEH write materialization.
        unsafe { ptr.add(128).write(0x77) };

        assert!(!has_block(va / PAGE));
        // SAFETY: page is now readable/writable.
        let gpu_slice = unsafe { std::slice::from_raw_parts(ptr.add(32), 8) };
        assert_eq!(gpu_slice, &[0xBB; 8], "GPU bytes must land before CPU write");
        // SAFETY: byte 128 was written by the CPU instruction after VEH resumed.
        assert_eq!(unsafe { ptr.add(128).read() }, 0x77, "CPU write must survive");
        // Write watch must report the page dirty so next upload picks up `0x77`.
        let (_, written) = crate::watch::writes(va, va + PAGE, false).expect("write-watched");
        assert!(
            written.iter().any(|&(p, _)| p == va / PAGE),
            "write fault must leave page dirty in write-watch"
        );

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_multi_flight_reuse_skips_rearming() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x22; 16]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 1));

        let reused_before = STATS.deferred_reused.load(Ordering::Relaxed);
        gpu_write_shadow(&shadow, 16, &[0x33; 16]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 2));
        assert!(
            STATS.deferred_reused.load(Ordering::Relaxed) > reused_before,
            "second flight on same block must reuse armed entry"
        );

        // SAFETY: CPU read materializes both flights' GPU writes in one DMA.
        let first = unsafe { std::slice::from_raw_parts(ptr, 16).to_vec() };
        // SAFETY: reading bytes 16..32.
        let second = unsafe { std::slice::from_raw_parts(ptr.add(16), 16).to_vec() };
        assert_eq!(first, vec![0x22; 16]);
        assert_eq!(second, vec![0x33; 16]);

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_hot_promotion_after_three_frames_and_decay() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        let threshold = promote_frames();
        for f in 0..threshold {
            advance_frames(1);
            gpu_write_shadow(&shadow, 0, &[0x40 + f as u8; 8]);
            assert!(
                try_defer_page_inner(&shadow, 0, PAGE, (f + 1) as u64),
                "frame {f} before promotion should defer"
            );
            // SAFETY: CPU read fault increments hot frame counter.
            let b = unsafe { ptr.read() };
            assert_eq!(b, 0x40 + f as u8);
        }

        // Now promoted: next flight must NOT defer.
        advance_frames(1);
        assert!(
            !try_defer_page_inner(&shadow, 0, PAGE, 10),
            "hot block must skip lazy deferral"
        );

        // Decay after > 120 unused frames.
        advance_frames((HOT_DECAY_FRAMES + 2) as usize);
        assert!(
            try_defer_page_inner(&shadow, 0, PAGE, 11),
            "unused hot block must decay back to lazy after >120 frames"
        );
        materialize_range(va, PAGE);

        clear_hot_for_test(block);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_internal_materialization_on_read_into_kernel_writing_and_unregister() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // 1 MiB write-watched allocation so `shadow::register` + `find` gives a real card/ReBAR shadow.
        let pages = 16;
        let ptr = alloc_watch_pages(pages);
        // SAFETY: `ptr` owns `16 * PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, pages * PAGE) };
        let va0 = ptr as usize;
        let va1 = va0 + PAGE;
        let va2 = va0 + 2 * PAGE;
        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);
        clear_hot_for_test(va2 / PAGE);

        let Some((card_shadow, _side)) = make_card_shadow_at(va0 as u64, 3, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&card_shadow, 0, &[0xA1; 16]);
        gpu_write_shadow(&card_shadow, PAGE, &[0xB2; 16]);

        // 1. `host::read_into` materializes page 0.
        assert!(try_defer_page_inner(&card_shadow, 0, PAGE, 0));
        let mut dst = [0u8; 16];
        assert!(crate::host::read_into(va0 as *const u8, &mut dst));
        assert_eq!(dst, [0xA1; 16]);
        assert!(!has_block(va0 / PAGE));

        // 2. `watch::kernel_writing` materializes page 1.
        assert!(try_defer_page_inner(&card_shadow, PAGE, PAGE, 0));
        crate::watch::kernel_writing(va1, PAGE);
        assert!(!has_block(va1 / PAGE));
        // SAFETY: page 1 is now readable with GPU bytes.
        assert_eq!(unsafe { std::slice::from_raw_parts(va1 as *const u8, 16) }, &[0xB2; 16]);
        // Clear kernel pin via `title_memory_remapped` cleanup at end or unpin by remapped.
        crate::watch::title_memory_remapped(va1, PAGE);

        // 3. `shadow::unregister` materializes registered shadow on page 2.
        drop(card_shadow);
        let reg_len = (pages * PAGE) as u64;
        crate::shadow::register(va0, va0 as u64, reg_len);
        let reg_shadow = crate::shadow::find(va0 as u64, reg_len).expect("registered shadow");
        if reg_shadow.carded() || reg_shadow.is_rebar() {
            if reg_shadow.is_rebar() {
                let list = h.new_list().expect("host list");
                reg_shadow.refresh(&list, va0 as u64, reg_len);
                list.close();
                h.execute(&[&list]);
                list.reset();
                reg_shadow.writing(va2, PAGE, |_, size| Some(vec![0x11u8; size as usize]));
            }
            gpu_write_shadow(&reg_shadow, 2 * PAGE, &[0xC3; 16]);
            assert!(try_defer_page_inner(&reg_shadow, 2 * PAGE, PAGE, 0));
            assert!(has_block(va2 / PAGE));
            drop(reg_shadow);
            crate::shadow::unregister(va0, va0 as u64);
            assert!(!has_block(va2 / PAGE));
            // SAFETY: unregister materialized GPU bytes to title RAM.
            assert_eq!(unsafe { std::slice::from_raw_parts(va2 as *const u8, 16) }, &[0xC3; 16]);
        } else {
            drop(reg_shadow);
            crate::shadow::unregister(va0, va0 as u64);
        }

        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);
        clear_hot_for_test(va2 / PAGE);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_aliased_section_views_protect_and_fault_across_aliases() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: creating an anonymous page-file-backed section for two aliased views.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null(), "CreateFileMappingW failed");
        // SAFETY: mapping two distinct views of offset 0 of `section`.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: second aliased view.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let addr_a = view_a.Value as usize;
        let addr_b = view_b.Value as usize;
        assert_ne!(addr_a, addr_b);
        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);

        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        crate::inplace::title_memory_view(addr_a, PAGE, section as usize, 0);
        crate::inplace::title_memory_view(addr_b, PAGE, section as usize, 0);

        // SAFETY: `addr_a` is mapped for `PAGE` bytes.
        unsafe { ptr::write_bytes(addr_a as *mut u8, 0x11, PAGE) };
        let Some((shadow, _side)) = make_card_shadow_at(addr_a as u64, 1, 0x11) else {
            crate::watch::title_memory_remapped(addr_a, PAGE);
            crate::watch::title_memory_remapped(addr_b, PAGE);
            // SAFETY: cleanup views and section handle.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_a.Value });
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_b.Value });
                CloseHandle(section);
            }
            return;
        };
        gpu_write_shadow(&shadow, 64, &[0xDD; 16]);

        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(addr_a / PAGE));
        assert!(has_block(addr_b / PAGE));
        assert!(!is_readable_now(addr_a));
        assert!(!is_readable_now(addr_b));

        // Fault on the ALIAS view (`addr_b`), not `addr_a`.
        // SAFETY: reading from `addr_b` triggers VEH fault, resolving alias -> primary block.
        let got_b = unsafe { std::slice::from_raw_parts((addr_b + 64) as *const u8, 16).to_vec() };
        assert_eq!(got_b, vec![0xDD; 16]);
        // Both views are now restored and show the materialized GPU bytes.
        assert!(is_readable_now(addr_a));
        assert!(is_readable_now(addr_b));
        // SAFETY: reading from `addr_a`.
        let got_a = unsafe { std::slice::from_raw_parts((addr_a + 64) as *const u8, 16).to_vec() };
        assert_eq!(got_a, vec![0xDD; 16]);
        assert!(!has_block(addr_a / PAGE));
        assert!(!has_block(addr_b / PAGE));

        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);
        drop(shadow);
        crate::watch::title_memory_remapped(addr_a, PAGE);
        crate::watch::title_memory_remapped(addr_b, PAGE);
        // SAFETY: cleanup views and section handle.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_a.Value });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_b.Value });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_merge_under_writes_update_busy_and_copy_runs_never_deadlock() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let pages = 16;
        let ptr = alloc_watch_pages(pages);
        // SAFETY: `ptr` owns `pages * PAGE` committed write-watched bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, pages * PAGE) };
        let va0 = ptr as usize;
        clear_hot_for_test(va0 / PAGE);

        let Some((shadow_a, _side_a)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        let Some((shadow_b, side_b)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            drop(shadow_a);
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        let Some((shadow_c, side_c)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            drop(shadow_b);
            drop(shadow_a);
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Declare pending ranges on `shadow_a` and `shadow_b` so `merge_under_writes`
        // and `update` -> `update_busy` exercise the busy-page paths.
        shadow_a.writing(va0, PAGE, |_, size| Some(vec![0x11u8; size as usize]));
        shadow_b.writing(va0, PAGE, |_, size| Some(vec![0x11u8; size as usize]));

        // 1. Defer page 0 on `shadow_a`: `shadow_a.merge_under_writes` must skip its own
        //    deferred block without locking + materializing itself, keeping page 0 deferred.
        gpu_write_shadow(&shadow_a, 0, &[0xA1; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert!(has_block(va0 / PAGE));
        let mut runs_left = 1024usize;
        assert!(!shadow_a.merge_under_writes(va0, PAGE, &mut runs_left));
        assert!(
            has_block(va0 / PAGE),
            "merge_under_writes must skip own deferred block"
        );

        // 2. Overlapping `shadow_b.merge_under_writes` must materialize `shadow_a`'s
        //    deferred block outside `shadow_b.before.write()` and see `[0xA1; 32]`.
        let mut runs_left = 1024usize;
        assert!(!shadow_b.merge_under_writes(va0, PAGE, &mut runs_left));
        assert!(!has_block(va0 / PAGE));
        assert_eq!(&side_b[..32], &[0xA1; 32]);

        // 3. `update` -> `update_busy`: `shadow_a.update` skips own deferred block, while
        //    overlapping `shadow_b.update` materializes outside `before` lock and copies `[0xB2; 32]`.
        gpu_write_shadow(&shadow_a, 0, &[0xB2; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert_eq!(shadow_a.update(va0, PAGE), 0);
        assert!(has_block(va0 / PAGE));
        assert!(shadow_b.update(va0, PAGE) >= 32);
        assert!(!has_block(va0 / PAGE));
        assert_eq!(&side_b[..32], &[0xB2; 32]);

        // 4. `update` -> `copy_runs`: `shadow_c` (no pending ranges) refreshes across
        //    `shadow_a`'s deferred block without holding any `Shadow` lock during materialization.
        gpu_write_shadow(&shadow_a, 0, &[0xC3; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert!(has_block(va0 / PAGE));
        assert!(shadow_c.update(va0, PAGE) > 0);
        assert!(!has_block(va0 / PAGE));
        assert_eq!(&side_c[..32], &[0xC3; 32]);

        // 5. `card_merge_relay` directly against an otherwise-deferred block and
        //    `relay_merge_rebar` / `merge_under_writes` when registered shadow uses ReBAR.
        let Some((shadow_d, _side_d)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            drop(side_c);
            drop(shadow_c);
            drop(side_b);
            drop(shadow_b);
            drop(_side_a);
            drop(shadow_a);
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        let mut scratch = Vec::new();
        let mut gpu_card = vec![0x11u8; 64];
        gpu_card[32..64].copy_from_slice(&[0xD4; 32]);
        assert!(shadow_d.card_merge_relay(0, &gpu_card, 0, &mut scratch) > 0);
        assert!(!has_block(va0 / PAGE));
        // SAFETY: merged bytes 32..64 landed in title RAM.
        assert_eq!(unsafe { std::slice::from_raw_parts(ptr.add(32), 32) }, &[0xD4; 32]);

        drop(_side_d);
        drop(shadow_d);
        drop(side_c);
        drop(shadow_c);
        drop(side_b);
        drop(shadow_b);
        drop(_side_a);
        drop(shadow_a);

        // SAFETY: reset test memory before registering a fresh shadow over `pages * PAGE`.
        unsafe { ptr::write_bytes(ptr, 0x11, pages * PAGE) };
        let reg_len = (pages * PAGE) as u64;
        crate::shadow::register(va0, va0 as u64, reg_len);
        let reg_shadow = crate::shadow::find(va0 as u64, reg_len).expect("registered shadow");
        if reg_shadow.is_rebar() {
            let list = h.new_list().expect("host list");
            reg_shadow.refresh(&list, va0 as u64, reg_len);
            list.close();
            h.execute(&[&list]);
            list.reset();
            reg_shadow.writing(va0, PAGE, |_, size| Some(vec![0x11u8; size as usize]));
            gpu_write_shadow(&reg_shadow, 0, &[0xE5; 32]);
            assert!(try_defer_page_inner(&reg_shadow, 0, PAGE, 0));
            let mut runs_left = 1024usize;
            assert!(!reg_shadow.merge_under_writes(va0, PAGE, &mut runs_left));
            assert!(has_block(va0 / PAGE));
            let mut gpu_rebar = vec![0x11u8; 64];
            gpu_rebar[..32].copy_from_slice(&[0xE5; 32]);
            let _ = reg_shadow.relay_merge_rebar(&[(0, 64)], &[&gpu_rebar]);
            assert!(!has_block(va0 / PAGE));
            // SAFETY: materialized ReBAR GPU bytes landed in title RAM.
            assert_eq!(unsafe { std::slice::from_raw_parts(ptr, 32) }, &[0xE5; 32]);
        }
        drop(reg_shadow);
        crate::shadow::unregister(va0, va0 as u64);

        clear_hot_for_test(va0 / PAGE);
        // SAFETY: test allocation cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_first_page_sets_ever_active_before_virtual_protect() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` committed bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x5A; 16]);

        let saw_ever_active_before_protect = std::rc::Rc::new(std::cell::Cell::new(false));
        let was_active = active();
        let prev_ever = EVER_ACTIVE.load(Ordering::Acquire);
        if !was_active {
            EVER_ACTIVE.store(false, Ordering::Release);
            assert!(
                !handle_fault(va, false),
                "handle_fault skips LAZY_STATE when !active && !EVER_ACTIVE"
            );
        }
        let saw_clone = saw_ever_active_before_protect.clone();
        BEFORE_PROTECT_HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                saw_clone.set(EVER_ACTIVE.load(Ordering::Acquire));
            }));
        });
        let deferred = try_defer_page_inner(&shadow, 0, PAGE, 0);
        BEFORE_PROTECT_HOOK.with(|hook| {
            *hook.borrow_mut() = None;
        });
        assert!(deferred);
        assert!(
            saw_ever_active_before_protect.get(),
            "EVER_ACTIVE must be set BEFORE VirtualProtect(PAGE_NOACCESS)"
        );
        // SAFETY: real read fault materializes the page.
        assert_eq!(unsafe { std::slice::from_raw_parts(ptr, 16) }, &[0x5A; 16]);
        if !prev_ever && !active() {
            // Keep EVER_ACTIVE monotonic once a lazy page has been deferred.
            assert!(EVER_ACTIVE.load(Ordering::Acquire));
        }

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_try_arm_under_guards_skips_pinned_and_clears_cold_state() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x6B; 16]);

        // Pin via `kernel_writing`: `try_arm_lazy_under_guards` must reject deferral.
        crate::watch::kernel_writing(va, PAGE);
        assert!(
            !try_defer_page_inner(&shadow, 0, PAGE, 0),
            "pinned block must not be deferred"
        );
        assert!(!has_block(block));
        assert!(is_readable_now(va));
        crate::watch::title_memory_remapped(va, PAGE);

        // Set `ColdRo`-like `PAGE_READONLY` guard via section view + `writes(..., true)`.
        // SAFETY: anonymous section mapped to test guard clearing under `try_arm_lazy_under_guards`.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping view.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let sva = view.Value as usize;
        let sblock = sva / PAGE;
        clear_hot_for_test(sblock);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        crate::inplace::title_memory_view(sva, PAGE, section as usize, 0);
        // SAFETY: seed view.
        unsafe { ptr::write_bytes(sva as *mut u8, 0x11, PAGE) };
        assert!(crate::watch::writes(sva, sva + PAGE, true).is_some());
        assert!(crate::watch::clean(sblock), "view page must be guarded clean");

        let Some((view_shadow, _vside)) = make_card_shadow_at(sva as u64, 1, 0x11) else {
            crate::watch::title_memory_remapped(sva, PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
                free_watch_pages(ptr);
            }
            return;
        };
        gpu_write_shadow(&view_shadow, 0, &[0x7C; 16]);

        assert!(
            try_defer_page_inner(&view_shadow, 0, PAGE, 0),
            "non-pinned clean guarded block must defer atomically"
        );
        assert!(
            !crate::watch::clean(sblock),
            "try_arm_lazy_under_guards must clear GuardClean/ColdRo before arming PAGE_NOACCESS"
        );
        // SAFETY: CPU read materializes back to PAGE_READWRITE (still not marked clean).
        assert_eq!(unsafe { std::slice::from_raw_parts(sva as *const u8, 16) }, &[0x7C; 16]);
        assert!(!crate::watch::clean(sblock));

        clear_hot_for_test(sblock);
        clear_hot_for_test(block);
        drop(view_shadow);
        drop(shadow);
        crate::watch::title_memory_remapped(sva, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
            free_watch_pages(ptr);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_fast_copy_rip_routes_to_fast_fill_without_materializing() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        assert!(!crate::host::is_fast_copy_rip(0));

        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x8D; 16]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(va / PAGE));

        // `copy_mapped_raw_checked` with `VICTORY_FAST_COPY=1` runs `victory_copy_try`,
        // which faults on `PAGE_NOACCESS`; `watch::on_write` must recognize `is_fast_copy_rip`
        // and pass to `fast_fill::on_fault` without materializing `shadow`'s deferred block.
        let mut dst = vec![0xEEu8; PAGE];
        let ok = crate::host::copy_mapped_raw_checked(dst.as_mut_ptr(), ptr, PAGE, false);
        assert!(
            !ok,
            "copy_mapped_raw_checked must return false on deferred PAGE_NOACCESS"
        );
        assert!(
            has_block(va / PAGE),
            "victory_copy_try fault must route to fast_fill without materializing lazy block"
        );

        materialize_range(va, PAGE);
        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_fetcher_cross_queue_sync_reads_latest_direct_queue_writes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        // Write on `Host::queue` (DIRECT) and fetch via `LazyFetcher` (COPY) using `sync_copy_queue`.
        gpu_write_shadow(&shadow, 128, &[0x9E; 64]);
        let fetched = fetch_gpu_bytes(&shadow, 0, PAGE).expect("LazyFetcher::fetch");
        assert_eq!(&fetched[128..192], &[0x9E; 64]);
        assert_eq!(&fetched[..128], &[0x11; 128]);

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_section_view_merges_via_own_view_before_restoring_prot() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: anonymous section with two aliased title views.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping two views of the same section page.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: second aliased view.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let addr_a = view_a.Value as usize;
        let addr_b = view_b.Value as usize;
        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);

        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        crate::inplace::title_memory_view(addr_a, PAGE, section as usize, 0);
        crate::inplace::title_memory_view(addr_b, PAGE, section as usize, 0);

        // SAFETY: seed physical memory via `addr_a`.
        unsafe { ptr::write_bytes(addr_a as *mut u8, 0x11, PAGE) };
        let Some((shadow, side)) = make_card_shadow_at(addr_a as u64, 1, 0x11) else {
            crate::watch::title_memory_remapped(addr_a, PAGE);
            crate::watch::title_memory_remapped(addr_b, PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_a.Value });
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_b.Value });
                CloseHandle(section);
            }
            return;
        };
        gpu_write_shadow(&shadow, 64, &[0xE1; 32]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(!is_readable_now(addr_a));
        assert!(!is_readable_now(addr_b));

        // While both title views remain `PAGE_NOACCESS`, `read_own` + `card_merge_relay_via_own`
        // must merge GPU bytes into both `shadow` CPU side and physical section RAM.
        let mut own_buf = [0u8; 32];
        assert!(crate::inplace::read_own(addr_a + 64, &mut own_buf));
        assert_eq!(own_buf, [0x11; 32]);

        let gpu_bytes = fetch_gpu_bytes(&shadow, 0, PAGE).expect("fetch");
        let mut scratch = Vec::new();
        assert_eq!(
            shadow.card_merge_relay_via_own(0, &gpu_bytes, 0, &mut scratch),
            Some(32)
        );
        assert!(
            !is_readable_now(addr_a) && !is_readable_now(addr_b),
            "title views must stay PAGE_NOACCESS during own-view merge"
        );
        assert!(crate::inplace::read_own(addr_a + 64, &mut own_buf));
        assert_eq!(own_buf, [0xE1; 32]);
        assert_eq!(&side[64..96], &[0xE1; 32]);

        // Full materialize path also succeeds cleanly.
        materialize_range(addr_a, PAGE);
        assert!(is_readable_now(addr_a) && is_readable_now(addr_b));
        // SAFETY: reading restored views.
        assert_eq!(
            unsafe { std::slice::from_raw_parts((addr_b + 64) as *const u8, 32) },
            &[0xE1; 32]
        );

        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);
        drop(shadow);
        crate::watch::title_memory_remapped(addr_a, PAGE);
        crate::watch::title_memory_remapped(addr_b, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_a.Value });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_b.Value });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_view_owner_skips_self_refresh_while_other_shadow_materializes_and_promotes_hot() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let pages = 2;
        // SAFETY: anonymous section mapped for a 2-page view.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                (pages * PAGE) as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping 2-page view.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, pages * PAGE) };
        assert!(!view.Value.is_null());
        let va0 = view.Value as usize;
        let va1 = va0 + PAGE;
        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);

        crate::inplace::title_memory_section(section as usize, (pages * PAGE) as u64, true);
        crate::inplace::title_memory_view(va0, pages * PAGE, section as usize, 0);

        // SAFETY: `va0` is mapped for `2 * PAGE` bytes.
        unsafe { ptr::write_bytes(va0 as *mut u8, 0x11, pages * PAGE) };
        let Some((shadow_a, side_a)) = make_card_shadow_at(va0 as u64, pages, 0x11) else {
            crate::watch::title_memory_remapped(va0, pages * PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
            }
            return;
        };
        let Some((shadow_b, side_b)) = make_card_shadow_at(va0 as u64, pages, 0x11) else {
            drop(side_a);
            drop(shadow_a);
            crate::watch::title_memory_remapped(va0, pages * PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
            }
            return;
        };

        // 1. View owner skips self-refresh when page 0 was guard-clean before deferring.
        let _ = shadow_a.update(va0, pages * PAGE);
        shadow_a.clear_card_dirty_for_test();
        let _ = shadow_b.update(va0, pages * PAGE);
        assert!(crate::watch::clean(va0 / PAGE));
        gpu_write_shadow(&shadow_a, 0, &[0xA1; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert!(has_block(va0 / PAGE));
        assert!(
            !owned_cpu_clean_blocks_in(
                Arc::as_ptr(&shadow_a) as usize,
                va0 / PAGE,
                va0 / PAGE + 1
            )
            .is_empty()
        );
        assert_eq!(shadow_a.update(va0, PAGE), 0);
        assert!(has_block(va0 / PAGE), "view owner must skip self-refresh");
        assert!(!is_readable_now(va0));

        // 2. Pre-defer CPU write safety on page 1:
        // (a) Un-guarded CPU write before deferring leaves `cpu_clean_after == None`.
        // SAFETY: `va1` is committed RW title view memory.
        unsafe { ((va1 + 16) as *mut u8).write(0x22) };
        assert!(!crate::watch::clean(va1 / PAGE));
        gpu_write_shadow(&shadow_a, PAGE, &[0xB1; 16]);
        assert!(try_defer_page_inner(&shadow_a, PAGE, PAGE, 0));
        assert!(
            owned_cpu_clean_blocks_in(
                Arc::as_ptr(&shadow_a) as usize,
                va1 / PAGE,
                va1 / PAGE + 1
            )
            .is_empty(),
            "dirty view block before defer must not record cpu_clean_after"
        );
        materialize_range(va1, PAGE);
        assert!(shadow_a.update(va1, PAGE) > 0);
        assert_eq!(side_a[PAGE + 16], 0x22);

        // (b) CPU write after `shadow_a` refreshed, guarded by `shadow_b` before `shadow_a` defers:
        // `cpu_clean_after > shadow_a.copied[1]` so `shadow_a` must still refresh after materialize.
        // SAFETY: `va1` is committed RW title view memory.
        unsafe { ((va1 + 20) as *mut u8).write(0x23) };
        assert!(shadow_b.update(va1, PAGE) > 0);
        assert!(crate::watch::clean(va1 / PAGE));
        gpu_write_shadow(&shadow_a, PAGE, &[0xB2; 16]);
        assert!(try_defer_page_inner(&shadow_a, PAGE, PAGE, 0));
        assert_eq!(shadow_a.update(va1, PAGE), 0);
        materialize_range(va1, PAGE);
        assert!(shadow_a.update(va1, PAGE) > 0);
        assert_eq!(side_a[PAGE + 20], 0x23);

        // 3. Overlapping `shadow_b.update` materializes `shadow_a`'s deferred block via
        //    `InternalRead` and promotes it to hot across `promote_frames()` distinct frames.
        materialize_range(va0, PAGE);
        clear_hot_for_test(va0 / PAGE);
        let threshold = promote_frames();
        for f in 0..threshold {
            advance_frames(1);
            let _ = shadow_a.update(va0, PAGE);
            shadow_a.clear_card_dirty_for_test();
            gpu_write_shadow(&shadow_a, 0, &[0xD0 + f as u8; 16]);
            assert!(try_defer_page_inner(&shadow_a, 0, PAGE, (f + 1) as u64));
            assert!(
                shadow_b.update(va0, PAGE) >= 16,
                "overlapping shadow_b must materialize and observe GPU bytes on frame {f}"
            );
            assert!(!has_block(va0 / PAGE));
            assert_eq!(&side_b[..16], &[0xD0 + f as u8; 16]);
        }
        advance_frames(1);
        assert!(
            !try_defer_page_inner(&shadow_a, 0, PAGE, 99),
            "InternalRead must promote overlapping block to hot"
        );

        // 4. `materialize_shadow` (`Retire`) must NOT promote to hot.
        clear_hot_for_test(va1 / PAGE);
        for f in 0..threshold {
            advance_frames(1);
            gpu_write_shadow(&shadow_a, PAGE, &[0xE0 + f as u8; 16]);
            assert!(try_defer_page_inner(&shadow_a, PAGE, PAGE, (f + 10) as u64));
            materialize_shadow(Arc::as_ptr(&shadow_a) as usize);
            assert!(!has_block(va1 / PAGE));
        }
        advance_frames(1);
        assert!(
            try_defer_page_inner(&shadow_a, PAGE, PAGE, 99),
            "Retire (materialize_shadow) must not promote block to hot"
        );
        materialize_shadow(Arc::as_ptr(&shadow_a) as usize);

        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);
        drop(side_b);
        drop(shadow_b);
        drop(side_a);
        drop(shadow_a);
        crate::watch::title_memory_remapped(va0, pages * PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_materialize_ranges_batches_multi_block_and_multi_shadow_into_single_dma() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr1 = alloc_watch_pages(2);
        let ptr2 = alloc_watch_pages(2);
        let ptr3 = alloc_watch_pages(3);
        // SAFETY: fresh allocations own their respective page counts.
        unsafe {
            ptr::write_bytes(ptr1, 0x11, 2 * PAGE);
            ptr::write_bytes(ptr2, 0x11, 2 * PAGE);
            ptr::write_bytes(ptr3, 0x11, 3 * PAGE);
        }
        let va1 = ptr1 as usize;
        let va2 = ptr2 as usize;
        let va3 = ptr3 as usize;
        for b in [
            va1 / PAGE,
            va1 / PAGE + 1,
            va2 / PAGE,
            va2 / PAGE + 1,
            va3 / PAGE,
            va3 / PAGE + 1,
            va3 / PAGE + 2,
        ] {
            clear_hot_for_test(b);
        }

        let Some((shadow1, _side1)) = make_card_shadow_at(va1 as u64, 2, 0x11) else {
            // SAFETY: cleanup.
            unsafe {
                free_watch_pages(ptr1);
                free_watch_pages(ptr2);
                free_watch_pages(ptr3);
            }
            return;
        };
        let Some((shadow2, _side2)) = make_card_shadow_at(va2 as u64, 2, 0x11) else {
            drop(shadow1);
            // SAFETY: cleanup.
            unsafe {
                free_watch_pages(ptr1);
                free_watch_pages(ptr2);
                free_watch_pages(ptr3);
            }
            return;
        };
        let Some((shadow3, _side3)) = make_card_shadow_at(va3 as u64, 3, 0x11) else {
            drop(shadow2);
            drop(shadow1);
            // SAFETY: cleanup.
            unsafe {
                free_watch_pages(ptr1);
                free_watch_pages(ptr2);
                free_watch_pages(ptr3);
            }
            return;
        };

        gpu_write_shadow(&shadow1, 0, &[0xA1; 16]);
        gpu_write_shadow(&shadow1, PAGE, &[0xA2; 16]);
        gpu_write_shadow(&shadow2, 0, &[0xB1; 16]);
        gpu_write_shadow(&shadow2, PAGE, &[0xB2; 16]);

        assert!(try_defer_page_inner(&shadow1, 0, PAGE, 0));
        assert!(try_defer_page_inner(&shadow1, PAGE, PAGE, 0));
        assert!(try_defer_page_inner(&shadow2, 0, PAGE, 0));
        assert!(try_defer_page_inner(&shadow2, PAGE, PAGE, 0));

        let sent_before = fetcher_sent_count();
        materialize_ranges(&[(va1, 2 * PAGE), (va2, 2 * PAGE)]);
        assert_eq!(
            fetcher_sent_count(),
            sent_before + 1,
            "materialize_ranges must batch all 4 blocks across 2 shadows into 1 DMA"
        );
        assert!(!has_block(va1 / PAGE) && !has_block(va1 / PAGE + 1));
        assert!(!has_block(va2 / PAGE) && !has_block(va2 / PAGE + 1));
        // SAFETY: all 4 pages are materialized to readable RAM.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(ptr1, 16), &[0xA1; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr1.add(PAGE), 16), &[0xA2; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr2, 16), &[0xB1; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr2.add(PAGE), 16), &[0xB2; 16]);
        }

        // Batched `materialize_shadow` across 3 deferred blocks of `shadow3`.
        gpu_write_shadow(&shadow3, 0, &[0xC1; 16]);
        gpu_write_shadow(&shadow3, PAGE, &[0xC2; 16]);
        gpu_write_shadow(&shadow3, 2 * PAGE, &[0xC3; 16]);
        assert!(try_defer_page_inner(&shadow3, 0, PAGE, 0));
        assert!(try_defer_page_inner(&shadow3, PAGE, PAGE, 0));
        assert!(try_defer_page_inner(&shadow3, 2 * PAGE, PAGE, 0));

        let sent_before_retire = fetcher_sent_count();
        materialize_shadow(Arc::as_ptr(&shadow3) as usize);
        assert_eq!(
            fetcher_sent_count(),
            sent_before_retire + 1,
            "materialize_shadow must batch all 3 blocks into 1 DMA"
        );
        assert!(!has_block(va3 / PAGE) && !has_block(va3 / PAGE + 1) && !has_block(va3 / PAGE + 2));
        // SAFETY: all 3 pages are materialized to readable RAM.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(ptr3, 16), &[0xC1; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr3.add(PAGE), 16), &[0xC2; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr3.add(2 * PAGE), 16), &[0xC3; 16]);
        }

        for b in [
            va1 / PAGE,
            va1 / PAGE + 1,
            va2 / PAGE,
            va2 / PAGE + 1,
            va3 / PAGE,
            va3 / PAGE + 1,
            va3 / PAGE + 2,
        ] {
            clear_hot_for_test(b);
        }
        drop(shadow3);
        drop(shadow2);
        drop(shadow1);
        // SAFETY: cleanup.
        unsafe {
            free_watch_pages(ptr1);
            free_watch_pages(ptr2);
            free_watch_pages(ptr3);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_deferred_card_page_does_not_pin_card_sent_and_materializes_live() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let len = PAGE;
        let ptr = alloc_watch_pages(len);
        assert!(!ptr.is_null());
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);
        // SAFETY: `ptr` is valid writable `PAGE` bytes.
        unsafe {
            std::ptr::write_bytes(ptr, 0x11, len);
        }
        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Simulate a cold page deferred at an ancient submission watermark (`fetched_after = 1`).
        gpu_write_shadow(&shadow, 0, &[0x5A; 32]);
        assert!(try_defer_page_inner(&shadow, 0, len, 1));
        assert!(has_block(va / PAGE));

        // Advance `RELAY_FETCHED` far beyond `1` and verify `relay_fetched_value()` reflects it
        // (no longer pinned by `min_card_fetched_after`) and `card_dirty` prunes `Card::sent`.
        let base = crate::shadow::relay_fetched_value();
        let far = base.saturating_add(10_000);
        crate::shadow::relay_fetched(far);
        assert!(crate::shadow::relay_fetched_value() >= far);
        let _ = crate::shadow::card_dirty(far.saturating_add(1));
        assert_eq!(crate::shadow::card_sent_max(), 0);

        // Materializing the cold page fetches live GPU bytes using the current watermark.
        materialize_range(va, len);
        assert!(!has_block(va / PAGE));
        // SAFETY: page is materialized and readable.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(ptr, 32), &[0x5A; 32]);
        }

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_active_hot_block_does_not_decay_after_120_frames_and_reprobes_cleanly() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Promote the block across `promote_frames()` distinct frames.
        let threshold = promote_frames();
        for f in 0..threshold {
            advance_frames(1);
            gpu_write_shadow(&shadow, 0, &[0x50 + f as u8; 8]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, (f + 1) as u64));
            // SAFETY: CPU read fault increments hot counter.
            assert_eq!(unsafe { ptr.read() }, 0x50 + f as u8);
        }

        // Stay actively GPU-written for > HOT_DECAY_FRAMES (e.g. 200 frames):
        // must remain hot without thrashing every 120 frames.
        for f in 0..(HOT_DECAY_FRAMES + 80) {
            advance_frames(1);
            assert!(
                !try_defer_page_inner(&shadow, 0, PAGE, 100 + f),
                "actively GPU-written hot block must not decay at active frame {f}"
            );
        }

        // Simulate reaching the reprobe horizon while still active (stepping in
        // chunks <= HOT_DECAY_FRAMES so inactivity decay does not trigger).
        let step = HOT_DECAY_FRAMES / 2;
        let mut advanced = HOT_DECAY_FRAMES + 80;
        while advanced <= HOT_REPROBE_FRAMES {
            advance_frames(step as usize);
            advanced += step;
            let deferred = try_defer_page_inner(&shadow, 0, PAGE, 1000 + advanced);
            if advanced <= HOT_REPROBE_FRAMES {
                assert!(!deferred, "must stay hot before reprobe horizon");
            } else {
                assert!(deferred, "must allow one reprobe deferral past reprobe horizon");
            }
        }

        // One fault on the reprobed block must immediately re-promote it on the
        // very same frame so the next flight skips deferral again.
        // SAFETY: volatile read triggers VEH materialization and re-promotion.
        let got = unsafe { ptr.read_volatile() };
        assert_eq!(got, 0x50 + (threshold - 1) as u8);
        assert!(!has_block(block), "reprobe read must materialize deferred block");
        advance_frames(1);
        assert!(
            !try_defer_page_inner(&shadow, 0, PAGE, 9999),
            "single fault during reprobe must immediately re-promote hot block"
        );

        clear_hot_for_test(block);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn fast_copy_read_into_raw_and_unmapped_hole_respect_lazy_back_and_holes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // Reserve 3 pages, commit pages 0 and 2, leave page 1 uncommitted (hole).
        // SAFETY: reserving a fresh 3-page region for test.
        let base = unsafe { VirtualAlloc(ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as *mut u8;
        assert!(!base.is_null(), "VirtualAlloc reserve failed");
        // SAFETY: committing pages 0 and 2 inside `base`.
        let p0 = unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
        let p2 = unsafe {
            VirtualAlloc(base.add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READWRITE)
        } as *mut u8;
        assert_eq!(p0, base);
        assert_eq!(p2, unsafe { base.add(2 * PAGE) });
        // SAFETY: pages 0 and 2 are committed RW.
        unsafe {
            ptr::write_bytes(p0, 0x3A, PAGE);
            ptr::write_bytes(p2, 0x7C, PAGE);
        }

        // 1. `copy_mapped_raw_checked` across committed + hole + committed zeroes
        //    only the hole and preserves both committed pages.
        let mut dst = vec![0xEEu8; 3 * PAGE];
        assert!(crate::host::copy_mapped_raw_checked(
            dst.as_mut_ptr(),
            base,
            3 * PAGE,
            false
        ));
        assert!(dst[..PAGE].iter().all(|&b| b == 0x3A));
        assert!(dst[PAGE..2 * PAGE].iter().all(|&b| b == 0x00));
        assert!(dst[2 * PAGE..].iter().all(|&b| b == 0x7C));

        // 2. `read_into_raw` on mapped page succeeds; on uncommitted hole returns false.
        let mut buf = vec![0u8; 64];
        assert!(crate::host::read_into_raw(p0, &mut buf));
        assert!(buf.iter().all(|&b| b == 0x3A));
        assert!(!crate::host::read_into_raw(unsafe { base.add(PAGE) }, &mut buf));

        // 3. `read_into_raw` on a `lazy_back` `PAGE_NOACCESS` page returns false
        //    WITHOUT triggering `lazy_back::handle_fault` materialization.
        let va0 = p0 as usize;
        clear_hot_for_test(va0 / PAGE);
        if let Some((shadow, _side)) = make_card_shadow_at(va0 as u64, 1, 0x3A) {
            gpu_write_shadow(&shadow, 0, &[0x91; 16]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            assert!(has_block(va0 / PAGE));
            assert!(
                !crate::host::read_into_raw(p0, &mut buf),
                "read_into_raw must return false on deferred PAGE_NOACCESS"
            );
            assert!(
                has_block(va0 / PAGE),
                "read_into_raw must NOT materialize lazy_back block under caller locks"
            );
            // `read_into_fast` materializes and returns the GPU bytes.
            assert!(crate::host::read_into_fast(p0, &mut buf));
            assert_eq!(&buf[..16], &[0x91; 16]);
            assert!(!has_block(va0 / PAGE));
            clear_hot_for_test(va0 / PAGE);
            drop(shadow);
        }

        // SAFETY: releasing test reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }
}
