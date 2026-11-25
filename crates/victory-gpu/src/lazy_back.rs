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
//! - `watch::GUARDS` may briefly acquire `LAZY_STATE` (`has_block`), which is
//!   acyclic because `LAZY_STATE` never acquires `watch::GUARDS`.

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

/// Default number of distinct frames with CPU faults before a block promotes to hot.
const DEFAULT_PROMOTE_FRAMES: u32 = 3;
/// Distinct frames without a CPU fault after which a hot entry decays back to lazy.
const HOT_DECAY_FRAMES: u64 = 120;
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

/// Number of distinct frames a block must fault on before promoting to hot (`VICTORY_LAZY_PROMOTE_FRAMES`, default 3).
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
    materializing: bool,
}

#[derive(Clone, Copy)]
struct HotEntry {
    faults: u32,
    last_frame: u64,
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

struct MaterializeScope {
    block: usize,
    finished: bool,
}

impl MaterializeScope {
    fn enter(block: usize) -> Self {
        IN_MATERIALIZE.with(|v| v.borrow_mut().push(block));
        Self {
            block,
            finished: false,
        }
    }
}

impl Drop for MaterializeScope {
    fn drop(&mut self) {
        IN_MATERIALIZE.with(|v| {
            let mut list = v.borrow_mut();
            if let Some(pos) = list.iter().rposition(|&b| b == self.block) {
                list.remove(pos);
            }
        });
        if !self.finished {
            // Unwind safety: ensure armed pages do not stay stuck in `materializing` or `PAGE_NOACCESS`.
            let removed = {
                let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                remove_entry_locked(&mut table, self.block, true)
            };
            if let Some(entry) = removed {
                for &va in &entry.armed_vas {
                    let mut old = 0u32;
                    // SAFETY: restoring title view protection on unwind.
                    unsafe { VirtualProtect(va as *const _, PAGE, PAGE_READWRITE, &mut old) };
                }
                drop(entry);
            }
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

    fn fetch(&mut self, shadow: &Shadow, offset: usize, len: usize) -> windows::core::Result<Vec<u8>> {
        let host = crate::host::host().ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
        let needed = len.max(PAGE);
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

        // SAFETY: `shadow.buffer` is live (held by `Arc<Shadow>`) in `COMMON` state (implicit
        // promotion to `COPY_SOURCE` and decay back to `COMMON` at `ExecuteCommandLists`),
        // `rb_res` is live with capacity >= `len`, and `FETCHER` serializes this queue/list/fence.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.list
                .CopyBufferRegion(rb_res, 0, &shadow.buffer, offset as u64, len as u64);
            self.list.Close()?;
            let lists = [Some(self.list.cast::<ID3D12CommandList>()?)];
            self.queue.ExecuteCommandLists(&lists);
            self.sent += 1;
            self.queue.Signal(&self.fence, self.sent)?;
            self.fence.SetEventOnCompletion(self.sent, self.event)?;
            WaitForSingleObject(self.event, INFINITE);
            let mut out = vec![0u8; len];
            std::ptr::copy_nonoverlapping(*rb_ptr, out.as_mut_ptr(), len);
            Ok(out)
        }
    }
}

/// Leaf lock protecting the dedicated `COPY`-queue fetcher. Never acquired inside `LAZY_STATE`,
/// `watch::GUARDS`, `shadow::BUFFERS`, or any `Shadow` lock.
static FETCHER: Mutex<Option<LazyFetcher>> = Mutex::new(None);

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

fn remove_entry_locked(
    table: &mut LazyTable,
    primary_block: usize,
    record_recent: bool,
) -> Option<LazyEntry> {
    let entry = table.pages.remove(&primary_block)?;
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
    LAZY_COUNT.store(table.pages.len(), Ordering::Release);
    LAZY_CV.notify_all();
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
        if let Some(&hot) = table.hot.get(&primary_block) {
            if frame > hot.last_frame.saturating_add(HOT_DECAY_FRAMES) {
                table.hot.remove(&primary_block);
            } else if hot.faults >= promote_frames() {
                return false;
            }
        }
        if let Some(existing) = table.pages.get_mut(&primary_block) {
            if existing.materializing || !Arc::ptr_eq(&existing.shadow, shadow) {
                return false;
            }
            existing.fetched_after = existing.fetched_after.max(fetched_after);
            STATS.deferred_reused.fetch_add(1, Ordering::Relaxed);
            STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
            return true;
        }
        if table.alias_to_primary.contains_key(&primary_block) {
            return false;
        }
    }

    if crate::watch::is_pinned_block(primary_block) {
        return false;
    }

    let mut vas = crate::inplace::pages_of(primary_va);
    if vas.is_empty() {
        vas.push(primary_va);
    } else if !vas.contains(&primary_va) {
        vas.push(primary_va);
    }
    if vas.len() > crate::watch::MANY_VIEWS {
        return false;
    }
    for &va in &vas {
        if !va.is_multiple_of(PAGE) {
            return false;
        }
        let b = va / PAGE;
        if b != primary_block && crate::watch::is_pinned_block(b) {
            return false;
        }
    }

    crate::watch::ensure_handler();
    crate::watch::disarm_for_lazy(primary_block);

    let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if table.pages.contains_key(&primary_block) || table.alias_to_primary.contains_key(&primary_block) {
        return false;
    }
    if vas
        .iter()
        .any(|&va| table.alias_to_primary.contains_key(&(va / PAGE)))
    {
        return false;
    }

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
            materializing: false,
        },
    );
    EVER_ACTIVE.store(true, Ordering::Release);
    LAZY_COUNT.store(table.pages.len(), Ordering::Release);
    STATS.deferred_new.fetch_add(1, Ordering::Relaxed);
    STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
    true
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

/// Returns the minimum `fetched_after` watermark among currently deferred `card` entries, if any.
pub(crate) fn min_card_fetched_after() -> Option<u64> {
    if !active() {
        return None;
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table
        .pages
        .values()
        .filter(|e| e.shadow.carded())
        .map(|e| e.fetched_after)
        .min()
}

fn note_fault_hot_locked(table: &mut LazyTable, primary_block: usize) {
    let frame = crate::frame::current();
    let threshold = promote_frames();
    let entry = table.hot.entry(primary_block).or_insert(HotEntry {
        faults: 0,
        last_frame: frame,
    });
    if entry.faults == 0 {
        entry.faults = 1;
        entry.last_frame = frame;
    } else if frame > entry.last_frame.saturating_add(HOT_DECAY_FRAMES) {
        entry.faults = 1;
        entry.last_frame = frame;
    } else if frame != entry.last_frame {
        let was_below = entry.faults < threshold;
        entry.faults = entry.faults.saturating_add(1);
        entry.last_frame = frame;
        if was_below && entry.faults >= threshold {
            STATS.hot_promoted.fetch_add(1, Ordering::Relaxed);
            return;
        }
    }
    if entry.faults >= threshold && threshold == 1 {
        // First fault promotes immediately when `VICTORY_LAZY_PROMOTE_FRAMES=1`.
        STATS.hot_promoted.fetch_add(1, Ordering::Relaxed);
    }
}

/// Materializes `primary_block` outside `LAZY_STATE`.
/// `fault`: `None` for internal runtime materialization, `Some(false)` for CPU read fault,
/// `Some(true)` for CPU write fault. Returns `true` if `primary_block` was active (or waited
/// on another thread materializing it).
fn materialize_primary(primary_block: usize, fault: Option<bool>) -> bool {
    let (shadow, offset, len, fetched_after, armed_vas) = {
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
                    let info = (
                        entry.shadow.clone(),
                        entry.offset,
                        entry.len,
                        entry.fetched_after,
                        entry.armed_vas.clone(),
                    );
                    match fault {
                        Some(is_write) => {
                            note_fault_hot_locked(&mut table, primary_block);
                            if is_write {
                                STATS.write_faults.fetch_add(1, Ordering::Relaxed);
                            } else {
                                STATS.read_faults.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        None => {
                            STATS.internal_materializes.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    break info;
                }
                None => return waited,
            }
        }
    };

    let mut scope = MaterializeScope::enter(primary_block);
    let fetched = fetch_gpu_bytes(&shadow, offset, len);

    for &va in &armed_vas {
        let mut old = 0u32;
        // SAFETY: `va` was armed by us with `PAGE_NOACCESS`; restoring `PAGE_READWRITE`
        // before `card_merge_relay` / `relay_merge_rebar` so `read_into` and `write_mapped` succeed.
        unsafe { VirtualProtect(va as *const _, PAGE, PAGE_READWRITE, &mut old) };
    }

    match fetched {
        Ok(gpu_bytes) => {
            if shadow.carded() {
                let mut scratch = Vec::new();
                shadow.card_merge_relay(offset, &gpu_bytes, fetched_after, &mut scratch);
            } else if shadow.is_rebar() {
                shadow.relay_merge_rebar(&[(offset, offset + len)], &[&gpu_bytes]);
                shadow.mark_rebar_dirty_range(offset, len);
            }
            if fault == Some(true) {
                for &va in &armed_vas {
                    crate::watch::wrote(va, PAGE);
                }
            }
            STATS.materialized_bytes.fetch_add(len as u64, Ordering::Relaxed);
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                va = format_args!("{:#x}", primary_block * PAGE),
                "lazy copy-back DMA fetch failed"
            );
        }
    }

    let removed = {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        remove_entry_locked(&mut table, primary_block, true)
    };
    scope.finished = true;
    drop(scope);
    drop(removed);
    drop(shadow);
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
    if materialize_primary(primary_block, Some(is_write)) {
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

/// Materializes any deferred lazy blocks overlapping `[at, at + len)` before an internal
/// runtime read, kernel write, copy, or memory remap.
pub(crate) fn materialize_range(at: usize, len: usize) {
    if len == 0 || !active() {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / PAGE;
    let last = end.div_ceil(PAGE);
    loop {
        let next = {
            let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            let mut found = None;
            let mut wait_only = false;
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
                    found = Some(primary_block);
                }
                break;
            }
            if wait_only {
                drop(LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner()));
                continue;
            }
            found
        };
        let Some(primary_block) = next else {
            return;
        };
        materialize_primary(primary_block, None);
    }
}

/// Materializes all deferred lazy blocks owned by `shadow_ptr` before retirement or unregister.
pub(crate) fn materialize_shadow(shadow_ptr: usize) {
    if !active() {
        return;
    }
    loop {
        let next = {
            let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            let first = table
                .by_shadow
                .get(&shadow_ptr)
                .and_then(|set| set.iter().copied().find(|&b| !current_thread_materializing(b)));
            let Some(block) = first else {
                return;
            };
            if table.pages.get(&block).is_some_and(|e| e.materializing) {
                drop(LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner()));
                continue;
            }
            block
        };
        materialize_primary(next, None);
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
                if let Some(entry) = remove_entry_locked(&mut table, primary, true) {
                    dropped.push(entry);
                }
            } else {
                table.alias_to_primary.remove(&va_block);
                table.note_recent(va_block);
                if let Some(entry) = table.pages.get_mut(&primary) {
                    entry.armed_vas.retain(|&va| va / PAGE != va_block);
                }
            }
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
            if let Some(entry) = remove_entry_locked(&mut table, b, true) {
                dropped.push(entry);
            }
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
    let hot_promoted = STATS.hot_promoted.swap(0, Ordering::Relaxed) as f64;
    tracing::info!(
        "lazy copy-back (VICTORY_LAZY_BACK=1): deferred {deferred_new:.1} new / {deferred_reused:.1} reused ({skipped_mb:.1} MB/frame skipped); materialized {materialized_mb:.1} MB/frame (read faults {read_faults:.1}, write faults {write_faults:.1}, internal {internal:.1}); hot promoted {hot_promoted:.0}"
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

    fn make_card_shadow_at(title_start: u64, pages: usize, init_byte: u8) -> Option<(Arc<Shadow>, Box<Vec<u8>>)> {
        let len = pages * PAGE;
        let mut side = Box::new(vec![init_byte; len]);
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
}
