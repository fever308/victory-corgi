//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap
//! (unless `VICTORY_LATE_WAIT=1`, see [`late`]);
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

/// Maximum reusable readback buffer kept cached for ReBAR UAV copy-backs.
/// One-off readbacks larger than this are dropped right after use.
const REBAR_READBACK_KEEP_CAP: u64 = 256 << 20;

/// Minimum and maximum cached byte sizes for the ReBAR GPU hash-compare flags
/// (`DEFAULT` UAV + `READBACK`, 4 bytes per 64 KiB piece: 16 KiB = 4,096
/// pieces = 256 MiB; 4 MiB = 1,048,576 pieces = 64 GiB per submission).
const REBAR_FLAGS_MIN_BYTES: u64 = 16 << 10;
const REBAR_FLAGS_MAX_BYTES: u64 = 4 << 20;

const REBAR_HASH_SOURCE: &str = include_str!("../shaders/rebar_hash.hlsl");

struct RebarHashPipeline {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 root signatures and pipeline states are free-threaded.
unsafe impl Send for RebarHashPipeline {}
// SAFETY: as above.
unsafe impl Sync for RebarHashPipeline {}

fn rebar_hash_pipeline(device: &ID3D12Device) -> Option<&'static RebarHashPipeline> {
    static PIPELINE: OnceLock<Option<RebarHashPipeline>> = OnceLock::new();
    PIPELINE
        .get_or_init(|| match make_rebar_hash_pipeline(device) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "no ReBAR hash-compare pipeline; falling back to full-range readbacks"
                );
                None
            }
        })
        .as_ref()
}

fn make_rebar_hash_pipeline(device: &ID3D12Device) -> windows::core::Result<RebarHashPipeline> {
    let bytes = compile(
        REBAR_HASH_SOURCE,
        windows::core::s!("main"),
        windows::core::s!("cs_5_1"),
    )?;
    let bytes = &bytes[..];
    // SAFETY: plain object creation; the root signature is embedded in the shader.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(RebarHashPipeline { root, pso })
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RebarHashStats {
    pub(crate) compared_pieces: u64,
    pub(crate) changed_pieces: u64,
    pub(crate) post_readback_calls: u64,
    pub(crate) post_readback_bytes: u64,
}

/// Reusable command lists and readback/hash-flag buffers for DMAing changed
/// GPU-written ranges of ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) shadow buffers
/// into cached system RAM.
#[derive(Default)]
struct RebarReadback {
    list: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    buffer: Option<(ID3D12Resource, usize, u64)>,
    pre_hash_list: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    post_compare_list: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    flags_uav: Option<(ID3D12Resource, u64, u64)>,
    flags_readback: Option<(ID3D12Resource, usize, u64)>,
}

// SAFETY: D3D12 objects are free-threaded; mapped pointers are read only
// under `Host::rebar_readback` after the corresponding GPU fence has completed.
unsafe impl Send for RebarReadback {}

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    rebar_scratch: bool,
    rebar_shadow_budget: u64,
    rebar_readback: Mutex<RebarReadback>,
    queue: ID3D12CommandQueue,
    /// Dedicated DMA copy queue (`D3D12_COMMAND_LIST_TYPE_COPY`) for
    /// [`Host::relay_fetch_back`] (card-copy and ReBAR fetches), so readbacks
    /// of completed flights do not queue behind in-flight graphics work on
    /// `queue` (`VICTORY_RELAY_COPY_QUEUE=0` disables it).
    copy_queue: Option<ID3D12CommandQueue>,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
    #[cfg(test)]
    d3d12_errors: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_hash_compared_pieces: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_hash_changed_pieces: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_post_readback_calls: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    rebar_post_readback_bytes: std::sync::atomic::AtomicU64,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Warn once when pinned epochs hold back this many retired host objects
/// (for example, an unreset command list or a submission stalled behind an
/// unsignalled fence).
const RETIRED_WARN_THRESHOLD: usize = 4096;

/// Monotonic sequence bumped whenever a tracked host resource (`HostTexture`,
/// `Shadow`, `Staging`) is created. Starts at 1 so `0` means "unbounded /
/// existed before any sequence" (used by legacy [`retire`]).
static RESOURCE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Returns a strictly increasing birth sequence for a newly allocated host
/// resource. Active command lists whose last recorded command occurred before
/// this sequence cannot reference the resource and do not block its release.
pub(crate) fn birth_seq() -> u64 {
    RESOURCE_SEQ.fetch_add(1, Ordering::AcqRel) + 1
}

/// Sentinel work sequence shared by active [`Host::execute`] submissions so
/// any object retired while the GPU is executing stays alive until execution
/// finishes.
fn exec_work_seq() -> std::sync::Arc<std::sync::atomic::AtomicU64> {
    static SEQ: OnceLock<std::sync::Arc<std::sync::atomic::AtomicU64>> = OnceLock::new();
    SEQ.get_or_init(|| std::sync::Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX)))
        .clone()
}

struct ActiveEpoch {
    /// Highest [`RESOURCE_SEQ`] observed when this epoch's owner recorded work
    /// (or `u64::MAX` for an active [`Host::execute`] submission).
    work_seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Slab indices in [`RetireState::retired`] waiting on this epoch.
    waiting: Vec<usize>,
}

struct RetiredEntry {
    remaining: usize,
    object: Option<Retired>,
}

/// Host objects the title let go of while an open list or a queued/running
/// submission may still use them (a texture its descriptors name, a descriptor
/// slot's old resource, a buffer shadow, or texture staging). The title's
/// descriptor writes and releases reach the host at once, from its own threads,
/// while lists are recorded or submissions run for most of a second in the
/// world: freeing there was a GPU page fault on the freed texture
/// (DXGI_ERROR_DEVICE_HUNG, 2026-11-24, RX 7600).
///
/// Each retired object tracks only the specific active epochs whose recorded
/// `work_seq` is at or after the object's `born_seq` (or all active epochs when
/// `born_seq == 0`). When an epoch finishes, only the objects waiting on it
/// decrement their blocker count and drop when it hits zero — avoiding global
/// head-of-line blocking when an older unreset list or stalled wait stays
/// pinned while newer resources are created and freed.
///
/// Lock invariant: `RETIRE_STATE` is a strict leaf lock — never lock anything
/// else while holding it, and ALWAYS drop retired objects AFTER releasing it.
struct RetireState {
    next_epoch: u64,
    active_epochs: std::collections::BTreeMap<u64, ActiveEpoch>,
    retired: Vec<Option<RetiredEntry>>,
    free_slots: Vec<usize>,
    pending_count: usize,
    warned: bool,
}

impl RetireState {
    const fn new() -> Self {
        Self {
            next_epoch: 1,
            active_epochs: std::collections::BTreeMap::new(),
            retired: Vec::new(),
            free_slots: Vec::new(),
            pending_count: 0,
            warned: false,
        }
    }
}

static RETIRE_STATE: Mutex<RetireState> = Mutex::new(RetireState::new());

/// Live retirement telemetry: `(pending_objects, active_epochs, oldest_epoch)`.
fn retire_stats() -> (usize, usize, u64) {
    let state = RETIRE_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let oldest = state.active_epochs.keys().next().copied().unwrap_or(0);
    (state.pending_count, state.active_epochs.len(), oldest)
}

/// RAII guard pinning an active GPU execution epoch until the submission
/// finishes (or lands, for `late` flights).
struct ExecEpochGuard {
    #[cfg(test)]
    scope: Option<std::sync::Arc<Mutex<RetireState>>>,
    epoch: u64,
}

impl Drop for ExecEpochGuard {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(scope) = self.scope.as_deref() {
            release_epoch_in(scope, self.epoch);
            return;
        }
        release_epoch_in(&RETIRE_STATE, self.epoch);
    }
}

#[cfg(test)]
thread_local! {
    static RETIRE_SCOPE: std::cell::RefCell<Option<std::sync::Arc<Mutex<RetireState>>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct RetireTestScope {
    prev: Option<std::sync::Arc<Mutex<RetireState>>>,
}

#[cfg(test)]
impl Drop for RetireTestScope {
    fn drop(&mut self) {
        RETIRE_SCOPE.with(|cell| *cell.borrow_mut() = self.prev.take());
    }
}

#[cfg(test)]
fn current_retire_scope() -> Option<std::sync::Arc<Mutex<RetireState>>> {
    RETIRE_SCOPE.with(|cell| cell.borrow().clone())
}

/// Isolates retirement epochs and queued drops on the current test thread (and
/// any `HostList` created or acquired on it) from concurrent GPU tests.
#[cfg(test)]
pub(crate) fn isolated_retire_scope() -> RetireTestScope {
    let fresh = std::sync::Arc::new(Mutex::new(RetireState::new()));
    let prev = RETIRE_SCOPE.with(|cell| cell.borrow_mut().replace(fresh));
    RetireTestScope { prev }
}

/// A retired object, dropped on whatever thread finishes its last blocking
/// epoch.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// Acquires a monotonically increasing epoch (never 0) that keeps retired
/// objects alive until [`release_epoch_in`] is called with it.
fn acquire_epoch_in(
    retire_state: &Mutex<RetireState>,
    work_seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> u64 {
    let mut state = retire_state.lock().unwrap_or_else(|e| e.into_inner());
    let e = state.next_epoch;
    state.next_epoch += 1;
    state.active_epochs.insert(
        e,
        ActiveEpoch {
            work_seq,
            waiting: Vec::new(),
        },
    );
    e
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((now_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
    crate::tiles::give_back_let_go();
}

/// Releases `epoch` and drops every retired object whose last blocking epoch
/// was `epoch` (outside `retire_state`).
fn release_epoch_in(retire_state: &Mutex<RetireState>, epoch: u64) {
    let done: Vec<Retired> = {
        let mut state = retire_state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(active) = state.active_epochs.remove(&epoch) else {
            return;
        };
        let mut done = Vec::new();
        for idx in active.waiting {
            if let Some(entry) = state.retired.get_mut(idx).and_then(Option::as_mut) {
                entry.remaining -= 1;
                if entry.remaining == 0 {
                    if let Some(obj) = entry.object.take() {
                        done.push(obj);
                    }
                    state.retired[idx] = None;
                    state.free_slots.push(idx);
                    state.pending_count = state.pending_count.saturating_sub(1);
                }
            }
        }
        done
    };
    drop(done);
}

/// Objects [`retire_born`] dropped at once (no blocking epoch was active) and
/// kept until their blocking epochs finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

fn retire_in<T: 'static>(retire_state: &Mutex<RetireState>, object: T, born_seq: u64) {
    // Pair with `work_seq` updates in `HostList::mark_work` and `RESOURCE_SEQ`
    // increments in `birth_seq`.
    std::sync::atomic::fence(Ordering::SeqCst);
    let warn = {
        let mut state = retire_state.lock().unwrap_or_else(|e| e.into_inner());
        let slot = state.free_slots.pop().unwrap_or(state.retired.len());
        let mut blockers = 0usize;
        for entry in state.active_epochs.values_mut() {
            if born_seq == 0 || entry.work_seq.load(Ordering::Acquire) >= born_seq {
                entry.waiting.push(slot);
                blockers += 1;
            }
        }
        if blockers == 0 {
            if slot < state.retired.len() {
                state.free_slots.push(slot);
            }
            drop(state);
            RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
            drop(object);
            return;
        }
        let item = Some(RetiredEntry {
            remaining: blockers,
            object: Some(Retired(Box::new(object))),
        });
        if slot == state.retired.len() {
            state.retired.push(item);
        } else {
            state.retired[slot] = item;
        }
        state.pending_count += 1;
        RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
        if !state.warned && state.pending_count >= RETIRED_WARN_THRESHOLD {
            state.warned = true;
            let (oldest_epoch, oldest_work_seq) = state
                .active_epochs
                .iter()
                .next()
                .map_or((0, 0), |(&e, entry)| {
                    (e, entry.work_seq.load(Ordering::Relaxed))
                });
            Some((
                state.pending_count,
                oldest_epoch,
                oldest_work_seq,
                RESOURCE_SEQ.load(Ordering::Relaxed),
                state.active_epochs.len(),
                state.next_epoch.saturating_sub(1),
            ))
        } else {
            None
        }
    };
    if let Some((
        pending,
        oldest_epoch,
        oldest_work_seq,
        current_seq,
        active_epochs,
        latest_epoch,
    )) = warn
    {
        tracing::warn!(
            pending,
            oldest_epoch,
            oldest_work_seq,
            current_seq,
            active_epochs,
            latest_epoch,
            "retired host resource queue grew past threshold; an unreset command list or stalled wait may be pinning epochs"
        );
    }
}

/// Drops `object` once every active epoch that could have observed a resource
/// born at `born_seq` (`work_seq >= born_seq`, or all active epochs when
/// `born_seq == 0`) has finished: at once if none matches.
pub(crate) fn retire_born<T: 'static>(object: T, born_seq: u64) {
    #[cfg(test)]
    if let Some(scope) = current_retire_scope() {
        retire_in(&scope, object, born_seq);
        return;
    }
    retire_in(&RETIRE_STATE, object, born_seq);
}

/// Drops `object` once every epoch active now has finished: at once if none is.
#[cfg(test)]
pub(crate) fn retire<T: 'static>(object: T) {
    retire_born(object, 0);
}

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !trail_on() || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    CARD_CLOCK_US[2].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time. Returns the estimated card busy share in percent.
fn card_clock_line(frames: f64, ms: f64) -> f64 {
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    let busy = if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 };
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), the rest while the title \
         had not submitted; the submitting thread stood waiting for the card {stood:.1} ms",
        busy
    );
    busy
}

/// How many values [`profile_counters`] gives.
#[cfg(feature = "diagnostics")]
pub(crate) const PROFILE_COUNTERS: usize = 8;

/// For `crate::profile`: [`SUBMISSION_US`], [`COPY_BACK`] and [`PREPARING`]
/// as they stand.
#[cfg(feature = "diagnostics")]
pub(crate) fn profile_counters() -> Vec<u64> {
    let r = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
    vec![
        r(&SUBMISSION_US[0]),
        r(&SUBMISSION_US[1]),
        r(&COPY_BACK[0]),
        r(&COPY_BACK[1]),
        r(&PREPARING[0]),
        r(&PREPARING[1]),
        r(&PREPARING[2]),
        r(&PREPARING[3]),
    ]
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Phase breakdown of [`PREPARING`]`[0]` (`preparing: copies`) since the last
/// frame-rate window:
/// `[watch_us, refresh_copy_us, fill_us, fill_count, fill_bytes, title_copy_us, title_copy_count]`.
pub(crate) static PREPARING_COPIES: [std::sync::atomic::AtomicU64; 7] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 7];
pub(crate) const PREP_WATCH_US: usize = 0;
pub(crate) const PREP_REFRESH_COPY_US: usize = 1;
const PREP_FILL_US: usize = 2;
const PREP_FILL_COUNT: usize = 3;
const PREP_FILL_BYTES: usize = 4;
const PREP_TITLE_COPY_US: usize = 5;
const PREP_TITLE_COPY_COUNT: usize = 6;

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory; 631
/// lists were alive in run 65, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The host device's adapter name and dedicated video memory, when DXGI
/// named it (the frametime dump's hardware line).
pub(crate) fn adapter_summary() -> Option<(String, u64)> {
    let host = host()?;
    let adapter = host.adapter.as_ref()?;
    // SAFETY: a plain DXGI adapter query on an adapter the device named.
    let desc = unsafe {
        adapter
            .cast::<windows::Win32::Graphics::Dxgi::IDXGIAdapter1>()
            .ok()?
            .GetDesc1()
            .ok()?
    };
    let end = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    Some((
        String::from_utf16_lossy(&desc.Description[..end]),
        desc.DedicatedVideoMemory as u64,
    ))
}

/// The system's processor sets of group 0: each one's id, its logical
/// processor and its efficiency class (higher is faster).
fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe {
        GetSystemCpuSetInformation(
            std::ptr::null_mut(),
            0,
            &mut bytes,
            GetCurrentProcess(),
            0,
        )
    };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe {
        GetSystemCpuSetInformation(
            buffer.as_mut_ptr().cast(),
            bytes,
            &mut bytes,
            GetCurrentProcess(),
            0,
        )
    };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe {
            &*(buffer
                .as_ptr()
                .cast::<u8>()
                .add(at)
                .cast::<SYSTEM_CPU_SET_INFORMATION>())
        };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.LogicalProcessorIndex, set.EfficiencyClass));
        }
        at += info.Size as usize;
    }
    sets
}

/// Asks the system to run the calling thread on the fast cores only, where
/// the processor has two kinds (`VICTORY_FAST_CORES=0`: not asked).
pub(crate) fn keep_on_fast_cores(what: &str) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_FAST_CORES").is_none_or(|v| v != "0")) {
        return;
    }
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.2 == best).map(|c| c.0).collect();
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() || fast.len() == sets.len() {
        return;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), fast.as_ptr(), fast.len() as u32) };
    if ok == 0 {
        tracing::warn!(
            what,
            error = %std::io::Error::last_os_error(),
            "the thread could not be kept on the fast cores"
        );
    } else {
        tracing::info!(
            what,
            fast = fast.len(),
            of = sets.len(),
            "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)"
        );
    }
}

/// Asks the system to run the calling thread on the slow cores only, where
/// the processor has two kinds and at least eight slow logical processors
/// (a helper that must not take a fast core from the title's threads).
/// True if it was asked.
pub fn keep_on_slow_cores(what: &str) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    let slow: Vec<u32> = sets.iter().filter(|c| c.2 < best).map(|c| c.0).collect();
    if slow.len() < 8 {
        return false;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), slow.as_ptr(), slow.len() as u32) } != 0;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            what,
            slow = slow.len(),
            of = sets.len(),
            ok,
            "helper threads are kept on the slow cores: the first one"
        );
    });
    ok
}

/// The process's private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    crate::drvwatch::start();
    // PIX's capturer, before any device (`VICTORY_PIX=1`).
    crate::pix::load();
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
        }
        // Whether the card's memory can be written by the CPU directly (`GPU_UPLOAD` heaps,
        // which need Resizable BAR): what a write-combined title allocation could live in.
        gpu_upload_probe(&device);
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let copy_queue = if relay_copy_queue() {
            device
                .CreateCommandQueue::<ID3D12CommandQueue>(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })
                .map_err(|e| {
                    tracing::warn!(
                        error = %e,
                        "no copy queue for the relay's fetches: they go on the host's queue"
                    )
                })
                .ok()
        } else {
            None
        };
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
        let uma = device
            .CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&raw mut arch).cast(),
                size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
            )
            .is_ok()
            && arch.UMA.as_bool();
        let mut opt16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        let gpu_upload_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut opt16).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
            )
            .is_ok()
            && opt16.GPUUploadHeapSupported.as_bool();
        let desc = adapter.as_ref().and_then(|a| a.GetDesc1().ok());
        let local_vram_budget = adapter
            .as_ref()
            .and_then(query_local_vram_budget)
            .or_else(|| desc.map(|d| d.DedicatedVideoMemory as u64));
        let allow_rebar = !cfg!(feature = "diagnostics")
            || std::env::var_os("VICTORY_REBAR").is_none_or(|v| v != "0");
        let (rebar_scratch, rebar_shadow_budget) =
            rebar_policy(gpu_upload_supported, uma, local_vram_budget, allow_rebar);
        tracing::info!(
            gpu_upload_supported,
            uma,
            rebar_scratch,
            rebar_shadow_budget_mb = rebar_shadow_budget >> 20,
            "host D3D12 device ready"
        );
        Ok(Host {
            device,
            rebar_scratch,
            rebar_shadow_budget,
            rebar_readback: Mutex::default(),
            queue,
            copy_queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            adapter,
            #[cfg(test)]
            d3d12_errors: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_hash_compared_pieces: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_hash_changed_pieces: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_post_readback_calls: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            rebar_post_readback_bytes: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
pub(crate) fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices): keeps
    /// at most one warm chunk across resets and drops all chunks when parked in
    /// the pool.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// The list that runs those transitions just before this one, made
    /// when the list first has such transitions ([`lazy_prelude`]).
    prelude: OnceLock<Option<Prelude>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// Dispatches and buffer copies recorded since the reset: what may have
    /// written indirect arguments (`crate::indirect` clamps hoisted before).
    writes: std::sync::atomic::AtomicU64,
    /// Buffers for clamped indirect arguments when this list holds the
    /// clamps of the list after it (`clamp_room`).
    clamp_chunks: Mutex<ClampChunks>,
    /// The records its draws read from slots that its prelude fills
    /// (`crate::indirect::gather`).
    gather: Mutex<Gather>,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// The pair of timestamps its run is timed with since its last reset
    /// (`crate::cardmeter`), or none.
    meter: std::sync::atomic::AtomicU32,
    /// With `crate::cardmeter::pipelines`: the pair of the pipeline bound
    /// last, open until the next one is bound or the list closes.
    pipeline_meter: std::sync::atomic::AtomicU32,
    /// Active retirement epoch pinned while this list has recorded work or is
    /// queued/executing (0 when idle).
    epoch: std::sync::atomic::AtomicU64,
    /// Highest [`RESOURCE_SEQ`] observed when this list recorded work; shared
    /// with [`RetireState`] while `epoch != 0` so newer resources are not
    /// blocked by stale lists.
    work_seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Queued or executing submissions currently pinning this list.
    in_flight: std::sync::atomic::AtomicUsize,
    /// Serializes `in_flight` transitions with `epoch` updates so a concurrent
    /// `end_submit` never clears an epoch pinned by a new `begin_submit`.
    epoch_lock: Mutex<()>,
    /// Test-only isolated retirement state captured when the list is created or
    /// acquired so parallel GPU tests do not pin each other's drops.
    #[cfg(test)]
    retire_scope: Option<std::sync::Arc<Mutex<RetireState>>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
    /// A shader that can store to buffers was bound since the list's last
    /// wait for every earlier shader write (`crate::indirect`): an indirect
    /// call's arguments may be among what it wrote. Set when the list is
    /// made or reset (earlier lists' writes are not known here).
    buffer_writes: std::sync::atomic::AtomicBool,
    /// A shader that can write through any unordered access (buffers,
    /// textures, views picked from the heap by index) was bound since the
    /// list's last wait for every earlier shader write: the title's barrier
    /// groups wait only then (`crate::list`). Set when made or reset.
    shader_writes: std::sync::atomic::AtomicBool,
}

impl Drop for HostList {
    fn drop(&mut self) {
        self.release_epoch();
    }
}

/// A list's buffers for hoisted clamped indirect arguments with their
/// states, how many are in use since the reset, the bytes taken of the last
/// of those, whether the list has the barrier that precedes the clamps, and
/// whether it has the clamps' root signature, pipeline and budget bound.
#[derive(Default)]
struct ClampChunks {
    all: Vec<(ID3D12Resource, D3D12_RESOURCE_STATES)>,
    used: usize,
    at: u64,
    barrier: bool,
    bound: bool,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// What a list gathers for its prelude to clamp (`crate::indirect::gather`).
#[derive(Default)]
struct Gather {
    /// Something recorded in the list may write arguments: no record is
    /// gathered from here on (the ones gathered before are read before
    /// the list runs, which is before that).
    closed: bool,
    /// One set per argument buffer, kind and stride.
    sets: Vec<crate::indirect::GatherSet>,
    /// The budget words the gathered records are charged to.
    budget: Option<ID3D12Resource>,
    /// Scratch buffers of earlier recordings, kept for the next sets.
    spare: Vec<ID3D12Resource>,
}

/// The most argument buffers a list gathers from.
const GATHER_SETS: usize = 8;

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// Grand Theft Auto VI's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In Grand Theft Auto VI
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Runs of lists with indirect calls whose clamp budget was taken: runs
/// looked at, runs that spent all of it (every indirect draw and dispatch
/// after that point ran empty), and the most a run spent (`crate::indirect`).
static BUDGET_RUNS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// Since the last frame line: the largest group count in x an indirect
/// compute record asked for, and the records that asked for more than the
/// 65,535 one dispatch runs (cut to that: the groups past it do not run).
static COMPUTE_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping and its host GPU address.
struct Chunk {
    buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them.
const SPARE_MOST: u64 = 128 << 20;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Whether a depth plane's pass runs in the list ahead of a draw that
/// tests against the plane (`HostList::plane_ahead`); `VICTORY_PLANE_AHEAD=0`
/// leaves it for the list's close.
fn plane_ahead_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLANE_AHEAD").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data) = host.scratch_buffer(size)?;
    name(&buffer, || "list scratch".into());
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    Some(Chunk {
        buffer,
        data: data as usize,
        gpu,
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`SPARE_MOST`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                spare.give(chunk, size, SPARE_MOST)
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, set where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// The fetch inside flights ([`inflight_fetch`]), if it is on and was
    /// made.
    inflight: Option<CardInflight>,
}

/// What the fetch inside flights needs ([`inflight_fetch`], [`CARD_FETCH`]).
struct CardInflight {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// PC memory, mapped (buffer, mapping, bytes): dword 0 the counter of
    /// tickets, then one dword per dword of `CardWork::differs`, the entry
    /// of that piece.
    table: (ID3D12Resource, usize, u64),
    /// PC memory, mapped (buffer, mapping, slots of [`INFLIGHT_SLOT`]
    /// bytes): where the pieces come down.
    fetched: (ID3D12Resource, usize, u32),
    /// Every ticket below it belongs to a landed flight: its slot may be
    /// written over. Raised by the thread that lands flights; read when a
    /// compare's list is recorded (an older value only gives fewer slots).
    landed: u32,
    /// What fetches the old way brought (copy, offsets, and the highest
    /// send whose lists were surely on the queue ahead of the fetch: its
    /// order of the queue less one). A flight sent at or before that ran
    /// ahead of the fetch, so the CPU's side holds those bytes as the card
    /// had them after the flight, or newer: the flight's own slot is older
    /// there and is not merged there. Dropped once every flight sent up to
    /// it has landed.
    brought: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)>,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one 64-thread group per 64 KiB piece (cut at
/// multiples of 64 KiB of title memory, as `Shadow::card_piece` cuts them):
/// each lane strides through the piece 1024 bytes at a time for coalesced
/// 16-byte loads, then the group reduces the 64 partial hashes in shared
/// memory. `differs` gets a dword per piece, 1 where a byte differs. The
/// offsets are widened to 16 bytes; buffers are multiples of 64 KiB, and a
/// piece flagged for a byte just outside the range costs a fetch that finds
/// nothing.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Span : register(b0) {
    uint first;     // the first piece
    uint skew;      // where the copy starts in its piece 0
    uint size;      // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;    // the first piece's dword in `differs`, in bytes
    uint compare;   // 0: the hashes are stored; else compared with those stored (2: and then stored)
    uint mark_base; // base bit in `marks` (0xFFFFFFFFu = no view-mark filter)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
RWByteAddressBuffer marks : register(u2);
groupshared uint4 shared_h[64];
uint4 mix_pair(uint4 a, uint4 b) {
    uint4 x = (a ^ (b * uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu)))
            * uint4(0x01000193u, 0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u);
    return x.yzwx ^ (x >> 15);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint idx = gid.x;
    if (idx >= pieces) return;
    uint lane = gtid.x;
    uint piece = first + idx;
    bool active = true;
    if (compare != 0u && mark_base != 0xFFFFFFFFu) {
        uint bit = min(mark_base + piece, 4194303u);
        uint word = marks.Load((bit >> 5u) << 2u);
        active = ((word & (1u << (bit & 31u))) != 0u);
    }
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) ^ (lane * 0x9E3779B9u);
    if (active) {
        uint lo = piece * 65536u;
        lo = lo > skew ? lo - skew : 0u;
        uint hi = min((piece + 1u) * 65536u - skew, size);
        lo = lo & ~15u;
        hi = (hi + 15u) & ~15u;
        [loop]
        for (uint at = lo + lane * 16u; at < hi; at += 1024u) {
            h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            h = h.yzwx ^ (h >> 15);
        }
    }
    shared_h[lane] = h;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 32u; stride > 1u; stride >>= 1u) {
        if (active && lane < stride) {
            shared_h[lane] = mix_pair(shared_h[lane], shared_h[lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        if (!active) {
            differs.Store(out_at + idx * 4u, 0u);
        } else {
            uint4 final_h = mix_pair(shared_h[0], shared_h[1]);
            if (compare != 0u) {
                differs.Store(out_at + idx * 4u, any(final_h != hashes.Load4(piece * 16u)) ? 1u : 0u);
                // With the copy-back on its own thread: the next compare of the
                // piece finds what is written from here on.
                if (compare == 2u) hashes.Store4(piece * 16u, final_h);
            } else {
                hashes.Store4(piece * 16u, final_h);
            }
        }
    }
}
"#;

/// Dispatch [`CARD_COMPARE`] for `pieces` pieces (one thread group per piece),
/// chunking at D3D12's 65,535-group per-dimension limit if ever exceeded.
///
/// # Safety
/// The caller must have bound [`CARD_COMPARE`]'s root signature, pipeline,
/// SRV, and UAVs on `list`, and `list` must be open for recording.
unsafe fn dispatch_card_compare(
    list: &ID3D12GraphicsCommandList,
    first: u32,
    skew: u32,
    size: u32,
    pieces: u32,
    out_at_bytes: u32,
    compare: u32,
    mark_base: u32,
) {
    const MAX_GROUPS: u32 = 65_535;
    let mut done = 0u32;
    while done < pieces {
        let chunk = (pieces - done).min(MAX_GROUPS);
        let constants = [
            first + done,
            skew,
            size,
            chunk,
            out_at_bytes + done * 4,
            compare,
            mark_base,
        ];
        // SAFETY: caller guarantees `list` is open with CARD_COMPARE bound.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
            list.Dispatch(chunk, 1, 1);
        }
        done += chunk;
    }
}

/// Whether the pieces a flight's compare flags come down to the PC inside
/// the flight itself (`VICTORY_CARD_FETCH_INFLIGHT=1`; off by default, and
/// only with [`relay`]).
///
/// With the relay the thread that lands flights fetched the flagged pieces
/// with a list of its own on the host's queue, the one the submitting
/// thread keeps filling: the fetch ran behind every flight sent meanwhile.
/// Measured on an RTX 3070 Laptop: 140.9 ms a frame of that thread waiting
/// for fetches, for 63 pieces (3.2 MB) in about five fetches, some 28 ms a
/// fetch spent queued behind newer flights; and the submitting thread
/// waited 81.7 ms a frame in "title memory read" settles for that thread.
///
/// With the switch the compare's list does the fetch: right behind the
/// compare, [`CARD_FETCH`] copies each piece it flagged into a slot of a
/// buffer in the PC's memory ([`CardInflight`]), so when the flight's fence
/// is reached its bytes are there, and the landing merges them
/// (`Shadow::card_merge_relay`) with no list, signal or wait of its own.
/// The merge's "fetched after" is the flight's own send number
/// (`CardPlan::sent_as`): see [`Host::relay_fetch_back`] for why that is
/// exact and when a landing falls back to the old fetch.
fn inflight_fetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_CARD_FETCH_INFLIGHT").is_some_and(|v| v == "1"))
}

/// The bytes of the PC's memory pieces fetched inside flights go to
/// ([`inflight_fetch`]; `VICTORY_CARD_FETCH_INFLIGHT_MB`, 64): 1,024 slots
/// ([`inflight_slots`]). A flight whose flagged pieces find no free slot
/// lands the old way.
fn inflight_fetch_bytes() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_FETCH_INFLIGHT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64).max(1) << 20
    })
}

/// A slot of [`CARD_FETCH`]: a 64 KiB piece as the compare cuts it, widened
/// to 16 bytes at both ends (one more 16-byte step where the copy does not
/// start at a multiple of 16 in its piece).
pub(crate) const INFLIGHT_SLOT: u64 = (64 << 10) + 16;

/// The slots for `bytes` asked of [`inflight_fetch_bytes`]: one per 64 KiB,
/// rounded down to a power of two (at least 1, at most 2^15: the shader's
/// byte offsets into the slots are 32-bit), so that
/// `ticket % slots` runs on without a jump when the 32-bit ticket wraps
/// (2^32 is a multiple of it; with 1,023 slots it would jump by 4 and two
/// tickets in the air could share a slot). The buffer is `slots` times
/// [`INFLIGHT_SLOT`], a little over `bytes`.
pub(crate) fn inflight_slots(bytes: u64) -> u32 {
    let wanted = (bytes >> 16).clamp(1, 1 << 15);
    1u32 << (63 - wanted.leading_zeros())
}

/// [`CARD_FETCH`]'s table entry for a piece the compare did not flag.
pub(crate) const INFLIGHT_NONE: u32 = 0xffff_ffff;

/// Where piece `i` of `span` starts in its [`CARD_FETCH`] slot, as an
/// offset of the copy: the compare's lower bound of the piece, widened down
/// to 16 bytes (`CARD_COMPARE`, before clamping to the declared range).
/// What `Shadow::card_piece` gives for the piece lies at or above it, and
/// within [`INFLIGHT_SLOT`] of it.
pub(crate) fn inflight_piece_from(span: &crate::shadow::CardSpan, i: u32) -> usize {
    let lo = (span.first + i) as usize * 65536;
    let lo = lo.saturating_sub(span.skew as usize);
    lo & !15
}

/// Whether the ticket `k` of [`CARD_FETCH`]'s counter gets a slot, with
/// every ticket below `floor` landed and `slots` slots: the slot is
/// `k % slots`. The tickets of flights still in the air are at or above
/// `floor` (they are taken in the queue's order and flights land in that
/// order), so two that get a slot are fewer than `slots` apart and never
/// share one. The arithmetic of the shader, for the tests.
pub(crate) fn inflight_gets_slot(k: u32, floor: u32, slots: u32) -> bool {
    k.wrapping_sub(floor) < slots
}

/// The fetch inside a flight ([`inflight_fetch`]), run in the compare's list
/// right behind the compare of a declared range (a UAV barrier between):
/// one group per piece (looping where a range has more than 65,535); where
/// `differs` flags the piece, its first thread takes a ticket of the
/// counter (`table`'s dword 0) and, if [`inflight_gets_slot`], the group
/// copies the piece, cut and widened as the compare hashes it, into slot
/// `ticket % slots` of `fetched`. Every piece's entry in `table` (dword 1 +
/// its dword in `differs`) is written: the ticket, also one that got no
/// slot (the landing repeats the test with the plan's `landed`), or
/// [`INFLIGHT_NONE`].
pub(crate) const CARD_FETCH: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
#define SLOT 65552u
#define NONE 0xffffffffu
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint landed;   // every ticket below it is landed
    uint slots;    // slots of `fetched`
    uint groups;   // groups dispatched
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer differs : register(u0);
RWByteAddressBuffer table : register(u1);
RWByteAddressBuffer fetched : register(u2);
groupshared uint ticket;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    [loop]
    for (uint i = group.x; i < pieces; i += groups) {
        if (t == 0u) {
            uint entry = NONE;
            if (differs.Load(out_at + i * 4u) != 0u) {
                uint k;
                table.InterlockedAdd(0, 1u, k);
                entry = k;
            }
            table.Store(4u + out_at + i * 4u, entry);
            ticket = entry;
        }
        GroupMemoryBarrierWithGroupSync();
        uint k = ticket;
        GroupMemoryBarrierWithGroupSync();
        if (k != NONE && (k - landed) < slots) {
            uint piece = first + i;
            uint lo = piece * 65536u;
            lo = lo > skew ? lo - skew : 0u;
            uint hi = min((piece + 1u) * 65536u - skew, size);
            lo = lo & ~15u;
            hi = (hi + 15u) & ~15u;
            uint into = (k % slots) * SLOT;
            [loop]
            for (uint at = lo + t * 16u; at < hi; at += 1024u) {
                fetched.Store4(into + (at - lo), now.Load4(at));
            }
        }
    }
}
"#;

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists, and no timing so far names them: each walks its 64 KiB pieces
/// 16 bytes at a time, 4,096 steps one after the other, so a pass takes
/// the time of one piece however few its pieces are. Run 102 sent up 514
/// ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_REHASH_RUNS").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=0` disables;
/// `=1` forces; on by default on 4+ logical CPUs). Used by both synchronous
/// [`Host::card_back`] and the relay thread's [`Host::relay_fetch_back`] /
/// [`Host::relay_card_inflight`].
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_merge_enabled(
            std::env::var("VICTORY_WIDE_MERGE").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_merge_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

/// One fetched piece of a card-backed shadow ready to merge on the relay thread.
struct RelayCardPiece {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    lo: usize,
    hi: usize,
    /// Pointer to `hi - lo` readable fetched GPU bytes, valid for the duration
    /// of [`relay_merge_card_pieces`].
    gpu_ptr: usize,
    after: u64,
    track_walls: bool,
}

/// Merges fetched card-copy pieces into title memory via
/// [`crate::shadow::Shadow::card_merge_relay`], fanning out across
/// [`crate::shadow::wide_run`] when `parallel` is enabled and there are either
/// multiple distinct card shadows or at least 16 total pieces.
///
/// Any pieces whose title address ranges `[start + lo, start + hi)` overlap
/// (for example, two in-flight plans touching the same 64 KiB piece) are placed
/// in the same worker bucket in their original relative order so concurrent
/// workers only ever mutate strictly disjoint slices of `Shadow::data` and
/// title memory.
fn relay_merge_card_pieces(pieces: Vec<RelayCardPiece>, parallel: bool) -> usize {
    if pieces.is_empty() {
        return 0;
    }
    let run_seq = |items: &[RelayCardPiece]| -> usize {
        let mut title = Vec::new();
        let mut written = 0usize;
        for p in items {
            // SAFETY: caller guarantees `p.gpu_ptr` points to `p.hi - p.lo`
            // readable bytes that stay live until this function returns.
            let gpu = unsafe { std::slice::from_raw_parts(p.gpu_ptr as *const u8, p.hi - p.lo) };
            let merged = p.shadow.card_merge_relay(p.lo, gpu, p.after, &mut title);
            if p.track_walls {
                p.shadow.walls_merged(gpu.len(), merged);
            }
            written += merged;
        }
        written
    };

    let first_shadow = std::sync::Arc::as_ptr(&pieces[0].shadow) as usize;
    let multi_shadow = pieces
        .iter()
        .skip(1)
        .any(|p| std::sync::Arc::as_ptr(&p.shadow) as usize != first_shadow);
    if !parallel || pieces.len() < 2 || (!multi_shadow && pieces.len() < 16) {
        return run_seq(&pieces);
    }

    // Group overlapping title-address intervals `[start + lo, start + hi)`
    // into disjoint units while preserving original order inside each unit.
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_unstable_by_key(|&idx| {
        let p = &pieces[idx];
        ((p.shadow.span().0 as usize).saturating_add(p.lo), idx)
    });
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut cur_end = 0usize;
    for idx in order {
        let p = &pieces[idx];
        let base = p.shadow.span().0 as usize;
        let t_lo = base.saturating_add(p.lo);
        let t_hi = base.saturating_add(p.hi);
        let bytes = p.hi - p.lo;
        match groups.last_mut() {
            Some((g_bytes, g_indices)) if t_lo < cur_end => {
                g_indices.push(idx);
                *g_bytes += bytes;
                cur_end = cur_end.max(t_hi);
            }
            _ => {
                groups.push((bytes, vec![idx]));
                cur_end = t_hi;
            }
        }
    }
    if groups.len() < 2 {
        return run_seq(&pieces);
    }
    for (_, g_indices) in &mut groups {
        if g_indices.len() > 1 {
            g_indices.sort_unstable();
        }
    }

    // Pack disjoint groups into up to 16 balanced buckets (LPT by bytes) so
    // large single-shadow megabuffer piece lists spread evenly across workers
    // with dynamic work stealing.
    groups.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let num_buckets = groups.len().min(16);
    let mut bucket_load = vec![(0usize, 0usize); num_buckets];
    let mut bucket_indices: Vec<Vec<usize>> = (0..num_buckets).map(|_| Vec::new()).collect();
    for (g_bytes, g_indices) in groups {
        let best = bucket_load
            .iter()
            .enumerate()
            .min_by_key(|&(_, &load)| load)
            .map_or(0, |(i, _)| i);
        bucket_load[best].0 += g_bytes;
        bucket_load[best].1 += g_indices.len();
        bucket_indices[best].extend(g_indices);
    }

    let mut slots: Vec<Option<RelayCardPiece>> = pieces.into_iter().map(Some).collect();
    let written = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let jobs: Vec<Box<dyn FnOnce() + Send>> = bucket_indices
        .into_iter()
        .filter(|b| !b.is_empty())
        .map(|indices| {
            let bucket: Vec<RelayCardPiece> = indices
                .into_iter()
                .filter_map(|idx| slots[idx].take())
                .collect();
            let written = written.clone();
            Box::new(move || {
                let mut title = Vec::new();
                let mut local = 0usize;
                for p in bucket {
                    // SAFETY: caller guarantees `p.gpu_ptr` points to `p.hi - p.lo`
                    // readable bytes that stay live until `wide_run` returns.
                    let gpu =
                        unsafe { std::slice::from_raw_parts(p.gpu_ptr as *const u8, p.hi - p.lo) };
                    let merged = p.shadow.card_merge_relay(p.lo, gpu, p.after, &mut title);
                    if p.track_walls {
                        p.shadow.walls_merged(gpu.len(), merged);
                    }
                    local += merged;
                }
                written.fetch_add(local, Ordering::Relaxed);
            }) as Box<dyn FnOnce() + Send>
        })
        .collect();
    crate::shadow::wide_run(jobs);
    written.load(Ordering::Relaxed)
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
    /// Base bit in `SLOT_VIEW_WRITES` when all writes overlapping this range
    /// were marked by shaders (`VIEW_MARK_NONE` = no view-mark filter).
    mark_base: u32,
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// With [`relay`]: the number its sends were taken under
    /// ([`QUEUE_ORDER`]). Once its submission has run, every send up to it
    /// has run (`Host::relay_fetch_back` on a copy queue).
    sent_as: u64,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// Value of [`QUEUE_ORDER`] immediately after this submission's `sync`
    /// list was queued on `Host::queue`. Every send `<= order_after` has
    /// finished on the card once this flight completes.
    order_after: u64,
    /// Whether the compare's list fetches the flagged pieces itself
    /// ([`inflight_fetch`]): every compared range has its [`CARD_FETCH`].
    inflight: bool,
    /// The `CardInflight::landed` its [`CARD_FETCH`] was told: which of its
    /// tickets got a slot ([`inflight_gets_slot`]).
    inflight_landed: u32,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// `bool`: whether the shader marks written 64 KiB pages in the view-writes bitmap.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// The host pipelines [`HostList::bind_pipeline`] kept (by address), so a
    /// pipeline the list binds again is not kept or queried again.
    bound: std::collections::HashSet<usize>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Upload memory filled from title memory since the last title write
    /// (source, bytes): its host address, which a second fill would find
    /// holding the same bytes.
    filled: std::collections::HashMap<(usize, usize), u64>,
    /// Buffer ranges declared written by shaders since the last title copy
    /// (host copy, address, bytes, marked): declaring them again would
    /// change nothing.
    gpu_written: std::collections::HashSet<(usize, usize, usize, bool)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Title-written textures whose staging buffer is copied into the host
    /// texture in this list's prelude if refreshed or not uploaded yet.
    texture_uploads: Vec<(
        std::sync::Arc<crate::resource::HostTexture>,
        std::sync::Arc<crate::texture::Staging>,
    )>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch.
    backs: Vec<(ID3D12Resource, usize, [usize; 3])>,
    /// Buffer copies done once the list has run, after what its shaders
    /// wrote is in title memory ([`copy_after`]): destination, source,
    /// bytes.
    after: Vec<(usize, usize, usize)>,
    /// Title textures of the kinds copied back ([`crate::texback`]) that
    /// the list's shaders write, copied when the list closes.
    back_wanted: Vec<usize>,
    /// Those copies, whose texels go to title memory once the list has run.
    texture_backs: Vec<crate::texback::Back>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Deferred {
    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
    }
}

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1`; off by default, and off in the timing run, which
/// waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]).
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`, the default; `VICTORY_RELAY=0`
/// turns it off; it turns
/// [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says another stage,
/// and is off with it, in the timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// With [`relay`]: the thread that lands flights fetches card copies'
/// pieces on a copy queue of its own. On the host's queue a fetch ran
/// behind everything sent after the flights it lands (up to 128 lists of
/// later frames): run 20261124-1508, 9 fetches a frame waited 187 ms in all
/// while the card had finished the flights long before.
/// `VICTORY_RELAY_COPY_QUEUE=0` puts them on the host's queue again.
fn relay_copy_queue() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RELAY_COPY_QUEUE").is_none_or(|v| v != "0"))
}

/// Whether the host has `GPU_UPLOAD` heaps (video memory the CPU maps and writes over the
/// PCIe bus, with Resizable BAR on): logged always. `VICTORY_UPLOAD_PROBE=1` also measures
/// them against the plain upload heap (system memory): a 64 MiB buffer of each, written
/// whole by the CPU, read back whole by the CPU, and copied by the card into a buffer in
/// video memory, timed. The first step towards title memory the title writes and the card
/// reads without a copy of ours between (write-combined console allocations).
fn gpu_upload_probe(device: &ID3D12Device) {
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: the structure the feature asks for, with its size.
    let asked = unsafe {
        device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut o16).cast(), size_of_val(&o16) as u32)
    };
    let supported = asked.is_ok() && o16.GPUUploadHeapSupported.as_bool();
    tracing::info!(
        asked = asked.is_ok(),
        supported,
        "GPU upload heaps (video memory the CPU writes directly, Resizable BAR)"
    );
    if !supported || std::env::var_os("VICTORY_UPLOAD_PROBE").is_none_or(|v| v != "1") {
        return;
    }
    const BYTES: usize = 64 << 20;
    let buffer = |kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| -> Option<ID3D12Resource> {
        let props = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: BYTES as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: plain descriptions; the out pointer is ours.
        let r = unsafe { device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut made) };
        if let Err(e) = r {
            tracing::warn!(error = %e, kind = kind.0, "upload probe: buffer refused");
        }
        made
    };
    let source: Vec<u8> = (0..BYTES).map(|i| ((i * 2_654_435_761usize) >> 13) as u8).collect();
    let mut back = vec![0u8; BYTES];
    let mut results = Vec::new();
    for (name, kind) in [("upload heap (system memory)", D3D12_HEAP_TYPE_UPLOAD), ("GPU upload heap (video memory)", D3D12_HEAP_TYPE_GPU_UPLOAD)] {
        let state = if kind == D3D12_HEAP_TYPE_UPLOAD { D3D12_RESOURCE_STATE_GENERIC_READ } else { D3D12_RESOURCE_STATE_COMMON };
        let Some(b) = buffer(kind, state) else { continue };
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a CPU-visible buffer of ours, mapped whole.
        if unsafe { b.Map(0, None, Some(&mut ptr)) }.is_err() || ptr.is_null() {
            tracing::warn!(name, "upload probe: map refused");
            continue;
        }
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        let mbps = |ms: f64| (BYTES as f64 / (1 << 20) as f64) / (ms / 1e3);
        // SAFETY: `BYTES` mapped bytes; each copy covers them once.
        let write = {
            let t = std::time::Instant::now();
            for _ in 0..4 {
                unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), ptr.cast::<u8>(), BYTES) };
            }
            ms(t) / 4.0
        };
        let read = {
            let t = std::time::Instant::now();
            unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), back.as_mut_ptr(), BYTES) };
            ms(t)
        };
        let same = back == source;
        // SAFETY: mapped above.
        unsafe { b.Unmap(0, None) };
        results.push(format!(
            "{name}: CPU write {write:.1} ms ({:.0} MB/s), CPU read {read:.1} ms ({:.0} MB/s), contents {}",
            mbps(write),
            mbps(read),
            if same { "intact" } else { "DIFFERENT" }
        ));
    }
    for r in &results {
        tracing::info!("upload probe (VICTORY_UPLOAD_PROBE=1), 64 MiB: {r}");
    }
}

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Since the last frame-rate line, with [`inflight_fetch`]: pieces merged
/// from the slots their flights fetched them to, their bytes, landings that
/// went the old way because a plan's pieces overflowed the slots, and
/// landings that fetched the old way (for any reason, overflow too).
static INFLIGHT_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// After a fetch the old way taken with the queue's count at `after`, for
/// plans whose highest send is `landed`: up to which send the notes of
/// sends to card copies may go (`crate::shadow::relay_fetched`).
///
/// Without [`inflight_fetch`] every later merge is of a later fetch, whose
/// count is `after` or more: notes up to `after` are of no use, as before.
/// With it, a flight still in the air, sent before the fetch, merges its
/// own slot with "fetched after" its own send, which may be below `after`:
/// it needs the notes of the sends between, or it takes the title's bytes
/// that went up in them for its shaders' writes and puts its older card
/// bytes over them. (Flights 5, 6, 7 in the air; 5 lands the old way with
/// the count at 7; a title byte went up in send 7 into a piece 6's shaders
/// write; 6 lands from its slot, which holds that byte from before send 7.)
/// So the mark goes only to the landed plans' highest send: every flight
/// still to land was sent after it, and every later fetch the old way
/// takes a count at least that high. (`CardInflight::brought` is another
/// matter: the card's bytes the fetch put in the CPU's side, not the
/// title's.)
pub(crate) fn fallback_fetched_mark(inflight: bool, after: u64, landed: Option<u64>) -> u64 {
    match (inflight, landed) {
        (false, _) => after,
        (true, Some(landed)) => landed.min(after),
        (true, None) => 0,
    }
}

/// The landed mark of [`CARD_FETCH`]'s tickets after ticket `k` landed: one
/// past `k` if that is ahead of `landed`, in wrapping order (tickets in the
/// air are always within 2^31 of the mark), else `landed`.
pub(crate) fn inflight_raise(landed: u32, k: u32) -> u32 {
    let next = k.wrapping_add(1);
    if (next.wrapping_sub(landed) as i32) > 0 { next } else { landed }
}

/// With [`inflight_fetch`], after a landing merged (either way): the tickets
/// these plans' fetches took are landed, and their slots may be written
/// over. Plans land in the line's order and tickets are taken in the
/// queue's (a UAV barrier opens every compare's list), so every ticket of a
/// plan in the air is ahead of these.
///
/// The counter is never reset and wraps after 2^32 tickets: the mark is
/// raised in wrapping order ([`inflight_raise`]), and the slots are a power
/// of two, so `k % slots` runs on across the wrap. shortcut: ticket
/// 0xffff_ffff reads as [`INFLIGHT_NONE`]; a flagged piece that draws it
/// copies nothing and its landing goes the old way, once every 2^32.
fn inflight_tickets_landed(plans: &[&CardPlan]) {
    let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(work) = work.as_mut() else {
        return;
    };
    let dwords = (work.differs.2 / 4) as usize;
    let Some(inflight) = work.inflight.as_mut() else {
        return;
    };
    // SAFETY: the mapping of the tickets; these plans' fetches have run.
    let table = unsafe { std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords) };
    for plan in plans.iter().filter(|p| p.inflight) {
        for r in plan.ranges.iter().filter(|r| r.compared) {
            for p in 0..r.span.pieces {
                if let Some(&k) = table.get((r.at + p) as usize)
                    && k != INFLIGHT_NONE
                {
                    inflight.landed = inflight_raise(inflight.landed, k);
                }
            }
        }
    }
    // Every flight still to land was sent after these: what fetches the old
    // way brought up to them concerns none of them.
    if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
        inflight.brought.retain(|b| b.3 > last);
    }
}

/// Whether a fallback `relay_fetch_back` fetch taken at queue count `after`
/// can overlap any flight still in the air after landing plans up to `landed`.
/// Submissions up to `after - 1` have already had their lists queued; when
/// `after - 1 <= landed`, no later in-flight plan can ever match `b.3 >= sent_as`.
fn should_record_brought(after: u64, landed: Option<u64>) -> bool {
    after > 1 && landed.is_none_or(|last| after - 1 > last)
}

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range: all of them
/// flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: &[(usize, u32, bool)]) -> std::collections::HashSet<(usize, u32)> {
    flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect()
}

#[cfg(test)]
mod relay_tests {
    use super::{flagged_pieces, landable};

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(&flags);
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    })
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// `(address, end)`, and the subset declared without shader view-marks
/// (always compared unfiltered).
type Declared = (
    std::sync::Arc<crate::shadow::Shadow>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
);

/// Reusable pool of direct helper command lists and 512 KiB `READBACK` buffers
/// for capturing and clearing the GPU shader view-writes bitmap
/// (`SLOT_VIEW_WRITES`) on the GPU timeline at the end of each submission.
#[derive(Default)]
struct ViewMarksPool {
    lists: Vec<Prelude>,
    slots: Vec<(ID3D12Resource, usize, u64)>,
}

// SAFETY: D3D12 resources are free-threaded; mapped `READBACK` pointers are
// read only after their submission's GPU fence has completed.
unsafe impl Send for ViewMarksPool {}

static VIEW_MARKS_POOL: Mutex<ViewMarksPool> = Mutex::new(ViewMarksPool {
    lists: Vec::new(),
    slots: Vec::new(),
});

/// One submission's GPU-timeline snapshot of the shader view-writes bitmap.
struct ViewMarksCapture {
    list: Prelude,
    slot: (ID3D12Resource, usize, u64),
    words: usize,
}

// SAFETY: same as `ViewMarksPool`.
unsafe impl Send for ViewMarksCapture {}

impl ViewMarksCapture {
    fn words(&self) -> &[u32] {
        if self.words == 0 {
            return &[];
        }
        // SAFETY: submission fence completed and `slot` covers `words * 4` bytes.
        unsafe { std::slice::from_raw_parts(self.slot.1 as *const u32, self.words) }
    }

    fn recycle(self) {
        let mut pool = VIEW_MARKS_POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.lists.push(self.list);
        pool.slots.push(self.slot);
    }
}

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    marks: Option<ViewMarksCapture>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    exec_epoch: ExecEpochGuard,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    marks: Option<ViewMarksCapture>,
    _exec_epoch: ExecEpochGuard,
}

/// One thing in the line of what waits for the card, in order.
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence (its address) signalled behind the flights before it.
    Signal(usize, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders may write (address, end), in
    /// order and joined.
    written: Vec<(usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture).
static LATE_STATS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList]) -> Option<usize> {
    let written = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        line.written.clone()
    };
    blocked_by(&written, runs, true)
}

/// Diagnostics (experiment): which read of title memory a flight's shaders
/// may write makes a submission wait, and the written range it meets.
fn why_read_say(kind: &str, src: usize, len: usize, written: &[(usize, usize)]) {
    static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SAID.fetch_add(1, Ordering::Relaxed);
    if n >= 40 && !n.is_multiple_of(50) {
        return;
    }
    let at = written.partition_point(|r| r.1 <= src);
    let hit = written.get(at).copied().unwrap_or((0, 0));
    let line = line();
    let flights = line.line.iter().filter(|l| matches!(l, Late::Run(_))).count();
    let mut first = None;
    for (i, item) in line.line.iter().filter_map(|l| if let Late::Run(f) = l { Some(f) } else { None }).enumerate() {
        if item.back.iter().any(|(_, r, _)| r.iter().any(|&(a, b)| a < src + len && src < b)) {
            first.get_or_insert(i);
        }
    }
    drop(line);
    tracing::info!(
        "why-read #{n}: {kind} of {len} bytes at {src:#x} meets a written range {:#x}..{:#x} ({} KB) of {} ranges; flights in the air {flights}, the first that declared it #{first:?}",
        hit.0,
        hit.1,
        (hit.1 - hit.0) / 1024,
        written.len()
    );
}

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    if touches(written, src, src + len) || touches(written, dst, dst + len) {
                        // A copy that only READS where flights may write is
                        // not waited for either, and checked like a fill
                        // (`fill_ahead`): run 210 looked at 197 such waits
                        // (20,480 bytes out of the 174 MB buffer into a small
                        // buffer of its own, four a frame, 43 to 60 ms a
                        // frame) and neither side had other bytes after
                        // one. A copy INTO such a range still waits.
                        if !touches(written, dst, dst + len) && fill_ahead::on(src, len) {
                            if count {
                                fill_ahead::read(src, len);
                            }
                            continue;
                        }
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            why_read_say("copy", src, len, written);
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Fill(_, src, len) => {
                    if touches(written, src, src + len) {
                        // Not waited for, and checked when the flights land
                        // (`fill_ahead`).
                        if fill_ahead::on(src, len) {
                            if count {
                                fill_ahead::read(src, len);
                            }
                            continue;
                        }
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                            why_read_say("fill", src, len, written);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    if touches(written, at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Run(_) => return Some(WHY_RUN),
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(_) => {}
            }
        }
    }
    None
}

/// Fills of upload memory (a constant buffer a table names, copied into
/// the list's own memory when the list is submitted) that read title memory
/// flights "may write" are no longer waited for; what the flights then
/// write is checked against them, and fills that read the 64 KiB pieces
/// where a miss happened wait as before; after `HELD_MAX` such pieces,
/// every fill waits again. `VICTORY_FILL_WAIT=1`: they wait from the start.
///
/// A fill is only taken for missed when title memory still holds what the
/// fill read (its fingerprint at the fill): bytes the title rewrote after
/// the fill say nothing about what the fill saw, and comparing with them
/// used to turn the waiting back on for good on the first such rewrite (one
/// settle of about 100 ms a frame for the rest of the run).
mod fill_ahead {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    /// Fills are looked up by the 64 KiB of title memory they read.
    const PIECE: usize = 1 << 16;
    /// A fill is checked against flights that land this many frames after
    /// it at most (flights are settled at 256 lists in the air, and by
    /// every present that waits).
    const KEEP: u64 = 16;
    /// Pieces held back (fills reading them wait) before all fills wait.
    const HELD_MAX: usize = 8;

    /// More than `HELD_MAX` pieces had a miss: every fill waits.
    static WAIT: AtomicBool = AtomicBool::new(false);
    /// Whether `HELD` has any piece (a cheap look before the lock).
    static ANY_HELD: AtomicBool = AtomicBool::new(false);
    /// The pieces where a flight changed bytes a fill had read ahead of
    /// it: fills that read them wait.
    static HELD: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
    /// Since the last frame line: fills that read ahead; landings' writes
    /// that touched one's piece. Since the start: fills that read bytes a
    /// flight then changed; fills not compared because the title had
    /// rewritten their bytes since.
    static COUNT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

    /// The fingerprint of the title memory a fill reads.
    fn print(src: usize, len: usize) -> u64 {
        crate::texture::fingerprint(src as u64, len as u64, &[])
    }

    #[derive(Default)]
    struct Piece {
        /// The frame its old fills were last dropped in.
        pruned: u64,
        /// From, to, frame, and the number of the last submission started
        /// when it read: the flights in the air then have that number or
        /// a lower one, and only their writes can be ones it missed; and
        /// the fingerprint of the bytes it read.
        fills: Vec<(usize, usize, u64, u64, u64)>,
    }

    /// The lowest submission number among the flights the landing thread
    /// is landing now; 0: not known (a landing by another way), and then
    /// every write counts.
    static LANDING: AtomicU64 = AtomicU64::new(0);

    pub(super) fn landing(first: u64) {
        LANDING.store(first, Relaxed);
    }

    static READ: Mutex<Option<HashMap<usize, Piece>>> = Mutex::new(None);

    /// Whether a fill of `len` bytes at `src` may read ahead of flights.
    pub(super) fn on(src: usize, len: usize) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_FILL_WAIT").is_none_or(|v| v != "1")) || WAIT.load(Relaxed) {
            return false;
        }
        if len == 0 || !ANY_HELD.load(Relaxed) {
            return true;
        }
        let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let Some(held) = held.as_ref() else { return true };
        !(src / PIECE..=(src + len - 1) / PIECE).any(|piece| held.contains(&piece))
    }

    /// Fills that read `[from, to)` wait from now on (or every fill, past
    /// `HELD_MAX` pieces). True the first time anything is held.
    fn hold(from: usize, to: usize) -> bool {
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let held = held.get_or_insert_with(HashSet::new);
        let first = held.is_empty();
        held.extend(from / PIECE..=(to - 1) / PIECE);
        ANY_HELD.store(true, Relaxed);
        if held.len() > HELD_MAX && !WAIT.swap(true, Relaxed) {
            tracing::warn!(
                "fills of upload memory read ahead of flights that changed their bytes in {} pieces of 64 KiB: every fill \
                 waits for flights again from here on, as with VICTORY_FILL_WAIT=1",
                held.len()
            );
        }
        first
    }

    /// A fill reads `len` bytes at `src` ahead of flights that may write
    /// them.
    pub(super) fn read(src: usize, len: usize) {
        if len == 0 {
            return;
        }
        COUNT[0].fetch_add(1, Relaxed);
        let frame = crate::frame::current();
        let print = print(src, len);
        let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
        let map = read.get_or_insert_with(HashMap::new);
        for piece in src / PIECE..=(src + len - 1) / PIECE {
            let p = map.entry(piece).or_default();
            if p.pruned != frame {
                p.pruned = frame;
                p.fills.retain(|f| f.2 + KEEP >= frame);
            }
            p.fills.push((src, src + len, frame, super::STARTED.load(Relaxed), print));
        }
    }

    /// A landing is about to put `new` into title memory at `at`: bytes
    /// shaders wrote, merged into the title's.
    pub(crate) fn wrote(at: usize, new: &[u8]) {
        if new.is_empty() || WAIT.load(Relaxed) {
            return;
        }
        let frame = crate::frame::current();
        let landing = LANDING.load(Relaxed);
        let mut hits: Vec<(usize, usize, u64, u64)> = {
            let read = READ.lock().unwrap_or_else(|e| e.into_inner());
            let Some(map) = read.as_ref() else { return };
            let end = at + new.len();
            (at / PIECE..=(end - 1) / PIECE)
                .filter_map(|piece| map.get(&piece))
                .flat_map(|p| p.fills.iter().copied())
                .filter(|f| f.0 < end && f.1 > at && f.2 + KEEP >= frame && (landing == 0 || landing <= f.3))
                .map(|f| (f.0, f.1, f.2, f.4))
                .collect()
        };
        if hits.is_empty() {
            return;
        }
        // A fill that spans pieces is kept in each of them.
        hits.sort_unstable();
        hits.dedup();
        COUNT[1].fetch_add(1, Relaxed);
        let mut now = Vec::new();
        for (from, to, read_in, print_then) in hits {
            // The title rewrote the bytes since the fill (or a landing before
            // this one did, and was compared then): what is there now is not
            // what the fill read, so it cannot tell a miss.
            if print(from, to - from) != print_then {
                COUNT[3].fetch_add(1, Relaxed);
                continue;
            }
            let (a, b) = (from.max(at), to.min(at + new.len()));
            now.clear();
            now.resize(b - a, 0);
            // Title memory as it is: what the fill read. Not there: nothing
            // to say.
            if !super::read_into(a as *const u8, &mut now) || now[..] == new[a - at..b - at] {
                continue;
            }
            let n = COUNT[2].fetch_add(1, Relaxed) + 1;
            let first = now.iter().zip(&new[a - at..b - at]).position(|(x, y)| x != y).unwrap_or(0);
            if hold(from, to) {
                tracing::warn!(
                    "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                     them (first at {:#x}, landed in frame {frame}): the lists of that submission drew with the bytes from before. \
                     Fills that read its 64 KiB wait for flights from here on; past {HELD_MAX} such pieces, every fill does \
                     ({n} such fills)",
                    to - from,
                    a + first
                );
            }
        }
    }

    /// With the frame line.
    pub(super) fn line(frames: f64) {
        let [ahead, looked] = [0, 1].map(|i| COUNT[i].swap(0, Relaxed));
        if ahead == 0 && looked == 0 {
            return;
        }
        let kept = {
            let frame = crate::frame::current();
            let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
            read.as_mut().map_or(0, |map| {
                map.retain(|_, p| {
                    p.fills.retain(|f| f.2 + KEEP >= frame);
                    !p.fills.is_empty()
                });
                map.values().map(|p| p.fills.len()).sum::<usize>()
            })
        };
        tracing::info!(
            "fills of upload memory that read title memory flights may write, not waited for (VICTORY_FILL_WAIT=1 waits): {:.1} per \
             frame; {:.1} landings' writes per frame fell in the same 64 KiB as one and were compared with it; fills that read \
             bytes a flight then changed, since the start: {} ({} pieces of 64 KiB held back{}); fills not compared because the \
             title had rewritten their bytes since: {}; {kept} fills kept to check",
            ahead as f64 / frames,
            looked as f64 / frames,
            COUNT[2].load(Relaxed),
            HELD.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, HashSet::len),
            if WAIT.load(Relaxed) { ", every fill waits again" } else { "" },
            COUNT[3].load(Relaxed)
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn held(at: usize) -> bool {
            HELD.lock().unwrap().as_ref().is_some_and(|h| h.contains(&(at / PIECE)))
        }

        /// One test: the module's state is shared by the whole process.
        #[test]
        fn only_real_misses_hold_fills_back() {
            const LEN: usize = 480;
            // Three pieces of 64 KiB of their own, one per case.
            let mut memory = vec![0u8; 4 * PIECE];
            let start = memory.as_ptr() as usize;
            let base = start.next_multiple_of(PIECE) - start;
            let at = |i: usize| start + base + i * PIECE;
            let (rewritten, missed, same) = (at(0), at(1), at(2));
            for i in 0..3 {
                memory[base + i * PIECE..][..LEN].fill(1);
            }

            // The title rewrites the bytes after the fill, then a flight's
            // shaders land other bytes there: the fill read the 1s, nothing
            // can be said, and nothing is held back (this used to turn
            // waiting on for good).
            read(rewritten, LEN);
            memory[base..][..LEN].fill(2);
            wrote(rewritten, &[3; LEN]);
            assert!(!held(rewritten), "a rewrite by the title is not a miss");
            assert!(on(rewritten, LEN));

            // Nothing rewrote them and the shaders changed them: a miss;
            // fills of that piece wait, fills elsewhere do not.
            read(missed, LEN);
            wrote(missed, &[3; LEN]);
            assert!(held(missed), "a real miss holds its piece back");
            assert!(!on(missed, LEN));
            assert!(on(same, LEN));
            assert!(!WAIT.load(Relaxed), "one piece does not make every fill wait");

            // The shaders wrote what was there already: no miss.
            read(same, LEN);
            wrote(same, &[1; LEN]);
            assert!(!held(same));
            assert!(on(same, LEN));
        }
    }
}

pub(crate) use fill_ahead::wrote as shaders_write;

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> bool {
    use std::sync::Arc;
    // Per copy, the span of what is refreshed (a flush brings that much
    // up to date at most).
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step
                && shadow.writes_pending()
            {
                match spans.iter_mut().find(|s| Arc::ptr_eq(&s.0, shadow)) {
                    Some(s) => (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len)),
                    None => spans.push((shadow.clone(), *address, *address + *len)),
                }
            }
        }
    }
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there.
        // (`VICTORY_LATE_MERGE_RUNS`, experiment: 512 ran out once a frame
        // in gameplay, run 26, for constants the title writes all over a
        // 174 MB buffer its compute shaders also write.)
        static RUNS: OnceLock<usize> = OnceLock::new();
        let mut runs_left = *RUNS.get_or_init(|| {
            std::env::var("VICTORY_LATE_MERGE_RUNS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(512)
        });
        return spans.iter().any(|(shadow, from, to)| shadow.merge_under_writes(*from, *to - *from, &mut runs_left));
    }
    spans.iter().any(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from))
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: usize, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
/// Pure decision for [`late_met`]: `Some(true)` if the final queued write to
/// `[address, address + bytes)` satisfies `holds` with no conflicting shader,
/// timestamp, or partial write; `Some(false)` if definitely not met without
/// settling; `None` if an overlapping shader/timestamp/partial write requires
/// settling and re-checking memory.
fn eval_late_met(
    written: &[(usize, usize)],
    writes: impl IntoIterator<Item = (usize, crate::queue::Written, usize)>,
    address: usize,
    bytes: usize,
    holds: impl Fn(u64) -> bool,
) -> Option<bool> {
    let end = address.saturating_add(bytes);
    let mut unsure = touches(written, address, end);
    let mut last_exact: Option<bool> = None;
    for (at, what, n) in writes {
        let write_end = at.saturating_add(n);
        if at >= end || address >= write_end {
            continue;
        }
        match what {
            crate::queue::Written::Value(v) if at == address && n >= bytes => {
                last_exact = Some(holds(v));
            }
            _ => unsure = true,
        }
    }
    if unsure {
        None
    } else {
        Some(last_exact.unwrap_or(false))
    }
}

pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    let decided = {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() && !line.settling {
            return false;
        }
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().copied().chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((*at, *what, *n)),
            _ => None,
        }));
        eval_late_met(&line.written, writes, address, bytes, |v| wait.holds_for(v))
    };
    match decided {
        Some(true) => {
            LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
            true
        }
        Some(false) => false,
        None => {
            settle_now(WHY_VALUE);
            wait.met()
        }
    }
}

/// [`late_met`] without its settle: whether a bottom-of-pipe write of a
/// flight in the air, or of one being landed, is known to meet `wait`
/// without any conflicting shader, timestamp, or partial write. False says
/// nothing about what else the flights may write there.
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    let line = line();
    if line.line.is_empty() && line.in_hand.is_empty() && !line.settling {
        return false;
    }
    let writes = line.in_hand.iter().copied().chain(line.line.iter().filter_map(|item| match item {
        Late::Write(at, what, n) => Some((*at, *what, *n)),
        _ => None,
    }));
    if eval_late_met(&line.written, writes, address, bytes, |v| wait.holds_for(v)) == Some(true) {
        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
        return true;
    }
    false
}

/// Effect of a single host list's deferred writes on a memory-value wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteMatch {
    /// The list does not touch `[address, address + bytes)`.
    None,
    /// The list's final write to `[address, address + bytes)` is an exact
    /// constant value (with no overlapping shader, timestamp, or partial
    /// write); `true` means it satisfies the wait, `false` means it overwrites
    /// it with a non-matching value.
    Exact(bool),
    /// An overlapping shader, timestamp, or partial write makes the outcome
    /// unknown until the list has run.
    Unsure,
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let landed = std::mem::take(&mut line().landed);
    for (lists, order) in landed {
        crate::list::recycle(lists, &order);
    }
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    let Some(host) = host() else {
        return;
    };
    keep_on_fast_cores("victory-late-wait");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            land();
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        // SAFETY: our fence and event.
        unsafe {
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY=1)"
            );
        }
    });
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    let waiting = std::time::Instant::now();
    while !done(&line) {
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
}

/// Since the last frame-rate line: ReBAR relay fetches, bytes DMAed, and wait
/// microseconds on the relay thread's copy path.
static RELAY_REBAR_STATS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// What the thread that lands flights fetches card-copy and ReBAR pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), an optional shared `COPY` queue
/// (`Host::copy_queue`, [`relay_copy_queue`]), and the memory the pieces come
/// down to.
struct RelayFetch {
    list: Prelude,
    /// A copy queue of the fetch's own ([`relay_copy_queue`]); without one
    /// the fetch goes on the host's queue.
    copy: Option<ID3D12CommandQueue>,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
fn relay_worker() {
    let Some(host) = host() else {
        return;
    };
    keep_on_fast_cores("victory-relay");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                // SAFETY: the host's fence and this thread's event.
                let waited = unsafe {
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards).
fn relay_hand_back(back: &[Declared], bitmaps: &[&[u32]]) {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared, unmarked) in back {
        if shadow.carded() || shadow.is_rebar() {
            continue;
        }
        let spans = if !bitmaps.is_empty() && shadow.marked() {
            let (filtered, decl_pages, marked_pages) =
                shadow.filter_marked_spans(declared, unmarked, bitmaps);
            crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
            filtered
        } else {
            let mut spans: Vec<(usize, usize)> = Vec::new();
            join_ranges(&mut spans, declared.iter().copied());
            spans
        };
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    if work.is_empty() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    let scanning = std::time::Instant::now();
    // Looked through without a lock: a piece that changes meanwhile (a
    // later flight writing, the title's bytes going in) is found now or by
    // that flight's landing, and the merge looks again under the lock.
    let found = crate::shadow::find_all(&work);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    let (mut now, mut title) = (Vec::new(), Vec::new());
    for ((shadow, from, to), found) in work.iter().zip(found) {
        let mut written = 0;
        for (at, n) in found {
            written += shadow.relay_merge(at, n, &mut now, &mut title);
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::compared(*from, to - from, written);
        let _ = (from, to);
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "the relay per frame (VICTORY_RELAY=1): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it)",
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if inflight_fetch() {
            let i: Vec<f64> = INFLIGHT_STATS.iter().map(|i| i.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "card copies' pieces fetched inside their flights per frame (VICTORY_CARD_FETCH_INFLIGHT=1): {:.1} pieces \
                 ({:.2} MB) merged with no fetch of their own; {:.1} landings fetched the old way (the wait above), {:.1} of \
                 them because a flight's pieces overflowed the slots",
                i[0],
                i[1] / (1 << 20) as f64,
                i[3],
                i[2],
            );
        }
        let rr: [f64; 3] =
            [0, 1, 2].map(|i| RELAY_REBAR_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        if rr[0] > 0.0 {
            tracing::info!(
                "relay ReBAR fetches per frame ({}): {:.1} fetches, {:.1} MB, {:.1} ms waiting",
                if host().is_some_and(|h| h.copy_queue.is_some()) {
                    "COPY queue"
                } else {
                    "direct queue"
                },
                rr[0],
                rr[1] / (1 << 20) as f64,
                rr[2] / 1000.0,
            );
        }
    }
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB",
            bytes / (1 << 20) as f64
        );
    }
    fill_ahead::line(frames);
}

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// Collects the unique ranges the shaders of `runs` may write, per host copy
/// matching `include`, in recording order.
fn collect_declared_writes(
    runs: &[&HostList],
    include: impl Fn(&crate::shadow::Shadow) -> bool,
) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    let mut by_shadow: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut seen_ranges: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    let mut seen_unmarked: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len, marked) = step
                && include(shadow)
            {
                let ptr = std::sync::Arc::as_ptr(shadow) as usize;
                let range = (*address, *address + *len);
                let is_unmarked = !*marked || !shadow.marked();
                let new_range = seen_ranges.insert((ptr, range.0, range.1));
                let new_unmarked = is_unmarked && seen_unmarked.insert((ptr, range.0, range.1));
                if !new_range && !new_unmarked {
                    continue;
                }
                let idx = *by_shadow.entry(ptr).or_insert_with(|| {
                    let i = back.len();
                    back.push((shadow.clone(), Vec::new(), Vec::new()));
                    i
                });
                if new_range {
                    back[idx].1.push(range);
                }
                if new_unmarked {
                    back[idx].2.push(range);
                }
            }
        }
    }
    back
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    collect_declared_writes(runs, |_| true)
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; only with
/// `VICTORY_REFRESH_CHECK=1` as well: it costs about 100 ms a frame), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// `VICTORY_COPY_AFTER=1`: a buffer copy whose source a list's shaders were
/// declared to write earlier in the same list (`Step::GpuWrite`) is done
/// once the list has run and what they wrote is in title memory, instead
/// of at submission (where it copied the bytes from before the shaders
/// ran: a title that copies results into a read-back buffer got none).
pub(crate) fn copy_after() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_AFTER").is_some_and(|v| v == "1"))
}

/// Copies [`copy_after`] moved past their list's run, since the start.
pub(crate) static COPIES_AFTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    // Buffer copies out of what the list's shaders wrote ([`copy_after`]):
    // every caller has handed back what they wrote before this.
    for &(dst, src, len) in &list.deferred().after {
        copy_mapped(dst as *mut u8, src as *const u8, len, true);
        crate::watch::wrote(dst, len);
    }
    // Textures shaders wrote that the title reads on the CPU
    // ([`crate::texback`]).
    for back in &list.deferred().texture_backs {
        crate::texback::land(back);
    }
    for (buffer, at, [row, rows, pitch]) in &list.deferred().backs {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run; the title's destination holds every row (checked when the
        // copy was recorded).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                buffer.Unmap(0, None);
            }
        }
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared], bitmaps: &[&[u32]]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared, unmarked) in back {
            if shadow.carded() || shadow.is_rebar() {
                continue;
            }
            let spans = if !bitmaps.is_empty() && shadow.marked() {
                let (filtered, decl_pages, marked_pages) =
                    shadow.filter_marked_spans(declared, unmarked, bitmaps);
                crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                filtered
            } else {
                let mut sorted = declared.clone();
                sorted.sort_unstable();
                let mut spans: Vec<(usize, usize)> = Vec::new();
                for (from, to) in sorted {
                    match spans.last_mut() {
                        Some(last) if from <= last.1 => last.1 = last.1.max(to),
                        _ => spans.push((from, to)),
                    }
                }
                spans
            };
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared, _) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared, unmarked) in back {
        if shadow.carded() || shadow.is_rebar() {
            // Done before (`Host::card_back` / `Host::rebar_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        let spans = if !bitmaps.is_empty() && shadow.marked() {
            let (filtered, decl_pages, marked_pages) =
                shadow.filter_marked_spans(declared, unmarked, bitmaps);
            crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
            filtered
        } else {
            // Each byte compared once: the world declares dozens of views
            // of its 174 MB heap that overlap almost entirely, and comparing
            // every one whole was gigabytes a frame (0.3-0.5 s) for the few
            // MB shaders wrote. Only ranges that overlap or touch are
            // joined: what lies between two was not given to shaders (the
            // snapshot does not hold it).
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            spans
        };
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    #[cfg(test)]
    pub(crate) fn d3d12_errors(&self) -> u64 {
        self.d3d12_errors.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn rebar_hash_stats(&self) -> RebarHashStats {
        RebarHashStats {
            compared_pieces: self.rebar_hash_compared_pieces.load(Ordering::Relaxed),
            changed_pieces: self.rebar_hash_changed_pieces.load(Ordering::Relaxed),
            post_readback_calls: self.rebar_post_readback_calls.load(Ordering::Relaxed),
            post_readback_bytes: self.rebar_post_readback_bytes.load(Ordering::Relaxed),
        }
    }
    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    /// The host device (`crate::cardmeter`).
    pub(crate) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// Ticks per second of the direct queue's timestamp clock (`crate::cardmeter`).
    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: a plain query on our live command queue.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0)
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = OnceLock::new();
            if !lazy_prelude() {
                let _ = prelude.set(self.make_prelude());
            }
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                prelude,
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                writes: std::sync::atomic::AtomicU64::new(0),
                clamp_chunks: Mutex::default(),
                gather: Mutex::default(),
                _alive: Counted::new(),
                meter: std::sync::atomic::AtomicU32::new(u32::MAX),
                pipeline_meter: std::sync::atomic::AtomicU32::new(u32::MAX),
                epoch: std::sync::atomic::AtomicU64::new(0),
                work_seq: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                in_flight: std::sync::atomic::AtomicUsize::new(0),
                epoch_lock: Mutex::default(),
                #[cfg(test)]
                retire_scope: current_retire_scope(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                buffer_writes: std::sync::atomic::AtomicBool::new(true),
                shader_writes: std::sync::atomic::AtomicBool::new(true),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            CARD_CLOCK_US[0].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
            CARD_CLOCK_US[1].fetch_add(worked().saturating_sub(worked_then), Ordering::Relaxed);
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            D3DCompile(
                CARD_COMPARE.as_ptr().cast(),
                CARD_COMPARE.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                inflight: if inflight_fetch() { self.make_card_inflight(RESULTS) } else { None },
            })
        }
    }

    /// [`CardInflight`] for `results` bytes of the compare's results; `None`
    /// (and the old fetch) if the shader or the memory cannot be had.
    fn make_card_inflight(&self, results: u64) -> Option<CardInflight> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let made = (|| -> windows::core::Result<CardInflight> {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; the out
            // pointer is ours.
            unsafe {
                D3DCompile(
                    CARD_FETCH.as_ptr().cast(),
                    CARD_FETCH.len(),
                    windows::core::s!("card fetch in flight"),
                    None,
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let slots = inflight_slots(inflight_fetch_bytes());
            let (table, table_data) = self.shared_buffer(4 + results).ok_or_else(windows::core::Error::empty)?;
            name(&table, || "card fetch in flight: tickets".into());
            let fetched_bytes = u64::from(slots) * INFLIGHT_SLOT;
            let (fetched, fetched_data) = self.shared_buffer(fetched_bytes).ok_or_else(windows::core::Error::empty)?;
            name(&fetched, || "card fetch in flight: pieces".into());
            // SAFETY: the mapping holds the counter; nothing of ours ran
            // on the card with it yet.
            unsafe { table_data.cast::<u32>().write_volatile(0) };
            // SAFETY: the blob's own bytes; the root signature is inside
            // the shader.
            unsafe {
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
                let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
                Ok(CardInflight {
                    root,
                    pso,
                    table: (table, table_data as usize, 4 + results),
                    fetched: (fetched, fetched_data as usize, slots),
                    landed: 0,
                    brought: Vec::new(),
                })
            }
        })();
        match made {
            Ok(inflight) => {
                tracing::info!(
                    slots = inflight.fetched.2,
                    "card copies' flagged pieces come down inside their flights (VICTORY_CARD_FETCH_INFLIGHT=1)"
                );
                Some(inflight)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no fetch inside flights (VICTORY_CARD_FETCH_INFLIGHT=1): pieces are fetched the old way");
                None
            }
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    fn card_before(&self, runs: &[&HostList], flown: bool) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per copy, with the
        // subset declared without shader view-marks.
        let declared = collect_declared_writes(runs, |shadow| shadow.carded());
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let mut dirty = crate::shadow::card_dirty(sent_as);
        if declared.is_empty() && dirty.is_empty() {
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, _, _) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Sync);
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, size)) = shadow.card_sides() else {
                            continue;
                        };
                        for &(a, b) in ranges {
                            list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                            crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                        }
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        let Some(before) = before else {
                            continue;
                        };
                        // No twin (it cost the card the copy's size again,
                        // 399 MB in run 75): `before` holds a 128-bit hash
                        // of each 64 KiB piece. The pieces just sent up are
                        // hashed again, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        if !ranges.is_empty() {
                            list.ResourceBarrier(&[transition(
                                &shadow.buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            )]);
                        }
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&work.pso);
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, work.differs.0.GetGPUVirtualAddress());
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            ranges.iter().map(|&(a, b)| shadow.card_run(a, b)).collect()
                        };
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if rehash_runs() { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            dispatch_card_compare(
                                list,
                                run.first,
                                run.skew,
                                size as u32,
                                run.pieces,
                                0,
                                0,
                                victory_shader::emit::VIEW_MARK_NONE,
                            );
                        }
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        let order_after = order.as_deref().copied().unwrap_or(0);
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            sent_as,
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            order_after,
            inflight: false,
            inflight_landed: 0,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        let view_marks_gpu = crate::shadow::view_marks_on()
            .then(crate::shadow::view_writes)
            .flatten();
        for (shadow, ranges, unmarked) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                let has_unmarked = unmarked.iter().any(|&(u0, u1)| u0 < to && from < u1);
                let mark_base = if view_marks_gpu.is_some() && !has_unmarked {
                    shadow.mark_base().unwrap_or(victory_shader::emit::VIEW_MARK_NONE)
                } else {
                    victory_shader::emit::VIEW_MARK_NONE
                };
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                    mark_base,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Check);
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(
                        4,
                        view_marks_gpu.unwrap_or_else(|| work.differs.0.GetGPUVirtualAddress()),
                    );
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let Some((_, Some(before), size)) = r.shadow.card_sides() else {
                            continue;
                        };
                        list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        dispatch_card_compare(
                            list,
                            r.span.first,
                            r.span.skew,
                            size as u32,
                            r.span.pieces,
                            r.at * 4,
                            if flown { 2 } else { 1 },
                            r.mark_base,
                        );
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    // The fetch inside the flight (`inflight_fetch`): behind
                    // every compare of the list, the pieces they flagged go
                    // to slots in the PC's memory.
                    if flown && let Some(inflight) = &work.inflight {
                        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                        }]);
                        list.SetComputeRootSignature(&inflight.root);
                        list.SetPipelineState(&inflight.pso);
                        list.SetComputeRootUnorderedAccessView(2, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, inflight.table.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, inflight.fetched.0.GetGPUVirtualAddress());
                        let mut every = true;
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                every = false;
                                continue;
                            };
                            let groups = r.span.pieces.clamp(1, 65535);
                            let constants =
                                [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, inflight.landed, inflight.fetched.2, groups];
                            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.Dispatch(groups, 1, 1);
                        }
                        plan.inflight = every;
                        plan.inflight_landed = inflight.landed;
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
            if let Some(inflight) = &mut work.inflight {
                inflight.brought.clear();
            }
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_fetch_back`].
    /// Reuses `self.copy_queue` with a `D3D12_COMMAND_LIST_TYPE_COPY` list when
    /// available so card-copy and ReBAR readbacks run on the hardware DMA
    /// engine without waiting behind in-flight graphics work on `self.queue`.
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        // SAFETY: plain objects of our own.
        let copy: Option<(ID3D12CommandQueue, Prelude)> = relay_copy_queue()
            .then_some(self.copy_queue.as_ref())
            .flatten()
            .and_then(|queue| unsafe {
                (|| -> windows::core::Result<(ID3D12CommandQueue, Prelude)> {
                    let allocator: ID3D12CommandAllocator =
                        self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                    let list: ID3D12GraphicsCommandList =
                        self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
                    list.Close()?;
                    Ok((queue.clone(), (allocator, list)))
                })()
                .map_err(|e| tracing::warn!(error = %e, "no copy queue list for the relay's fetches: they go on the host's queue"))
                .ok()
            });
        let (copy, list) = match copy {
            Some((queue, list)) => (Some(queue), list),
            None => (None, self.make_prelude()?),
        };
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                copy,
                fence,
                event,
                sent: 0,
                buffer: None,
            })
        }
    }

    /// [`relay`], on the thread that lands flights: fetches both card-copy
    /// pieces flagged by `plans` and ReBAR UAV spans in `back` in a single
    /// DMA submission on `self.copy_queue` (falling back to `self.queue` when
    /// `VICTORY_RELAY_COPY_QUEUE=0` or no `COPY` queue is available), then
    /// merges GPU writes into title memory (`Shadow::card_merge_relay` /
    /// `Shadow::relay_merge_rebar`).
    ///
    /// With [`inflight_fetch`], card-copy pieces are first looked for where the
    /// flights' own compare lists put them ([`Host::relay_card_inflight`]);
    /// when that succeeds only ReBAR spans (if any) are DMAed here.
    fn relay_fetch_back(
        &self,
        plans: &[&CardPlan],
        back: &[Declared],
        bitmaps: &[&[u32]],
        fetch: &mut Option<RelayFetch>,
    ) {
        struct RebarEntry {
            shadow: std::sync::Arc<crate::shadow::Shadow>,
            spans: Vec<(usize, usize)>,
            offsets: Vec<u64>,
        }

        let mut card_done_inflight = false;
        if inflight_fetch() {
            let landed = self.relay_card_inflight(plans);
            inflight_tickets_landed(plans);
            if landed {
                card_done_inflight = true;
            }
        }
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut card_total = 0u64;
        if !card_done_inflight {
            let differs = {
                let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
                work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
            };
            if let Some((differs, dwords)) = differs {
                // SAFETY: the mapping of the results. The compares of these
                // plans have run, and their dwords are given to no other plan
                // before these are done with (`card_done`).
                let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
                let mut seen = std::collections::HashSet::new();
                for plan in plans {
                    let flags: Vec<(usize, u32, bool)> = plan
                        .ranges
                        .iter()
                        .flat_map(|r| {
                            (0..r.span.pieces).map(move |p| {
                                let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                                (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
                            })
                        })
                        .collect();
                    let fetched = flagged_pieces(&flags);
                    for r in &plan.ranges {
                        let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                        for p in 0..r.span.pieces {
                            if !fetched.contains(&(copy, r.span.first + p)) {
                                continue;
                            }
                            let (lo, hi) = r.shadow.card_piece(&r.span, p);
                            if lo < hi && seen.insert((copy, lo, hi)) {
                                pieces.push((r.shadow.clone(), lo, hi, card_total));
                                card_total += (hi - lo) as u64;
                            }
                        }
                        crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                    }
                }
            }
        }

        if !bitmaps.is_empty() {
            for plan in plans {
                for r in &plan.ranges {
                    if r.mark_base != victory_shader::emit::VIEW_MARK_NONE {
                        let (_, decl_pages, marked_pages) =
                            r.shadow.filter_marked_spans(&[r.declared], &[], bitmaps);
                        crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                    }
                }
            }
        }

        let mut rebar_entries: Vec<RebarEntry> = Vec::new();
        let mut total = card_total;
        for (shadow, declared, unmarked) in back {
            if !shadow.is_rebar() {
                continue;
            }
            let title_spans = if !bitmaps.is_empty() && shadow.marked() {
                let (filtered, decl_pages, marked_pages) =
                    shadow.filter_marked_spans(declared, unmarked, bitmaps);
                crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                filtered
            } else {
                declared.clone()
            };
            let ranges: Vec<(usize, usize)> = title_spans
                .iter()
                .map(|&(from, to)| shadow.inside(from, to.saturating_sub(from)))
                .filter(|&(from, to)| from < to)
                .collect();
            let spans = merge_ranges(ranges);
            if spans.is_empty() {
                continue;
            }
            let mut offsets = Vec::with_capacity(spans.len());
            for &(from, to) in &spans {
                offsets.push(total);
                total += (to - from) as u64;
            }
            rebar_entries.push(RebarEntry {
                shadow: shadow.clone(),
                spans,
                offsets,
            });
        }
        let rebar_total = total - card_total;
        let has_card = !pieces.is_empty();
        let has_rebar = !rebar_entries.is_empty();

        let completed_order = plans.iter().map(|p| p.order_after).max().unwrap_or(0);
        if !card_done_inflight && !has_card {
            if inflight_fetch() {
                // The next merge may be of a flight in the air now, with
                // bytes from when it ran: only sends up to these plans' are
                // older than it.
                if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
                    crate::shadow::relay_fetched(last);
                }
            } else {
                // Nothing to fetch for card copies: every send up to these flights is done.
                let after = if self.copy_queue.is_some() {
                    completed_order
                } else {
                    *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner())
                };
                crate::shadow::relay_fetched(after);
            }
        }
        if !has_card && !has_rebar {
            return;
        }
        if has_card && inflight_fetch() {
            INFLIGHT_STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let copy = f.copy.clone();
        // On a copy queue of its own the fetch runs as soon as it is sent:
        // the plans' submissions have run, and so every send up to the
        // latest plan's ([`CardPlan::sent_as`]); later sends may have run
        // or not, and their ranges are left out of the merge.
        let landed = plans.iter().map(|p| p.sent_as).max().unwrap_or(0);
        let waiting = std::time::Instant::now();
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (held by `pieces` and `rebar_entries`) are in COMMON state
        // (implicit promotion to COPY_SOURCE/COPY_DEST and decay back to
        // COMMON at ExecuteCommandLists boundaries); the fence and the event
        // are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in &pieces {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                for entry in &rebar_entries {
                    for (&(from, to), &at) in entry.spans.iter().zip(&entry.offsets) {
                        list.CopyBufferRegion(
                            &buffer,
                            at,
                            &entry.shadow.buffer,
                            from as u64,
                            (to - from) as u64,
                        );
                    }
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                if let Some(queue) = &copy {
                    queue.ExecuteCommandLists(&lists);
                    queue.Signal(&fence, sent)?;
                    fence.SetEventOnCompletion(sent, event)?;
                    WaitForSingleObject(event, INFINITE);
                    return Ok(landed);
                }
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        if has_card {
            // This fetch may bring bytes of flights in the air now: their own
            // slots are older than what it puts in the CPU's side there
            // (`relay_card_inflight`). Sends up to `after` are numbered, and the
            // submission numbered `after` may not have its lists on the queue
            // yet; every one before it has (the submission lock).
            let landed_max = plans.iter().map(|p| p.sent_as).max();
            if inflight_fetch()
                && should_record_brought(after, landed_max)
                && let Some(inflight) = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|w| w.inflight.as_mut())
            {
                inflight.brought.extend(pieces.iter().map(|(shadow, lo, hi, _)| (shadow.clone(), *lo, *hi, after - 1)));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(card_total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
            }
            let merge_pieces: Vec<RelayCardPiece> = pieces
                .into_iter()
                .map(|(shadow, lo, hi, at)| RelayCardPiece {
                    shadow,
                    lo,
                    hi,
                    gpu_ptr: data + at as usize,
                    after,
                    track_walls: true,
                })
                .collect();
            let written = relay_merge_card_pieces(merge_pieces, wide_merge());
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            // Sends up to `after` are of no use to any later merge; with the
            // fetch inside flights only those up to these plans' are
            // (`fallback_fetched_mark`).
            crate::shadow::relay_fetched(fallback_fetched_mark(inflight_fetch(), after, landed_max));
        }
        if has_rebar {
            RELAY_REBAR_STATS[0].fetch_add(1, Ordering::Relaxed);
            RELAY_REBAR_STATS[1].fetch_add(rebar_total, Ordering::Relaxed);
            RELAY_REBAR_STATS[2].fetch_add(us, Ordering::Relaxed);
            let mut written = 0usize;
            for entry in &rebar_entries {
                let slices: Vec<&[u8]> = entry
                    .spans
                    .iter()
                    .zip(&entry.offsets)
                    .map(|(&(from, to), &at)| {
                        // SAFETY: `f.buffer` holds `total` bytes and the DMA fence has completed.
                        unsafe {
                            std::slice::from_raw_parts((data + at as usize) as *const u8, to - from)
                        }
                    })
                    .collect();
                written += entry.shadow.relay_merge_rebar(&entry.spans, &slices);
                entry.shadow.mark_rebar_dirty_spans(&entry.spans);
            }
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    fn relay_rebar_back(&self, back: &[Declared], fetch: &mut Option<RelayFetch>) {
        self.relay_fetch_back(&[], back, &[], fetch);
    }

    /// [`Host::relay_fetch_back`] with [`inflight_fetch`]: the pieces these
    /// plans' compares flagged are taken from the slots their own compare
    /// lists copied them to ([`CARD_FETCH`]) and merged, plan by plan in the
    /// line's order, with no list or wait of ours. `false` (nothing merged)
    /// if any plan cannot land so; the caller then lands them all the old
    /// way, as without the switch.
    ///
    /// The merge's "fetched after" (`Shadow::card_merge_relay`) is the
    /// plan's own `sent_as`. `QUEUE_ORDER` is raised only in
    /// `Host::card_before`, under the submission lock, and that lock is held
    /// until the submission's lists and its compare's list are on the queue
    /// behind its sync list: so when the compare's list ran, the card had
    /// received exactly the sends 1..=`sent_as` of title bytes and no later
    /// one, which is what the merge's skip of later sends asks. (The old
    /// fetch, by contrast, ran whenever it reached the queue, and took the
    /// count at its own ExecuteCommandLists.)
    ///
    /// When a landing goes the old way instead, all of it does: a fetch the
    /// old way brings the card's bytes as they are now, newer than the slots
    /// of plans in the air then, and merging an older slot after it would
    /// put a flight's bytes back over a later flight's. For the same reason
    /// the bytes such a fetch brought (`CardInflight::brought`) are left
    /// out of the slots of flights whose lists were on the queue ahead of
    /// it: the CPU's side holds what the card held after them there. The
    /// rest of their slots is merged, newer than anything merged there
    /// before. (Left out whole, a flight sent before such a fetch would have
    /// had to land the old way, its fetch taking a later count still: one
    /// overflow kept every landing on the old way while flights were in
    /// the air.) Plans land in the line's order, each plan's slot as it was
    /// right behind its own lists, so every merge here is of bytes at least
    /// as new as the CPU's side holds.
    ///
    /// shortcut: all or nothing per landing, and a plan whose pieces overflow
    /// the slots lands the old way for all its pieces, not only the ones
    /// left out.
    fn relay_card_inflight(&self, plans: &[&CardPlan]) -> bool {
        use std::sync::Arc;
        let mut pieces: Vec<RelayCardPiece> = Vec::new();
        {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(work) = work.as_ref() else {
                return false;
            };
            let Some(inflight) = &work.inflight else {
                return false;
            };
            let dwords = (work.differs.2 / 4) as usize;
            // SAFETY: the mappings of the results and the tickets. The
            // compares and fetches of these plans have run, and their dwords
            // are given to no other plan before these are done with
            // (`card_done`).
            let (differs, table) = unsafe {
                (
                    std::slice::from_raw_parts(work.differs.1 as *const u32, dwords),
                    std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords),
                )
            };
            let slots = inflight.fetched.2;
            for plan in plans {
                if plan.ranges.is_empty() {
                    continue;
                }
                if !plan.inflight || plan.ranges.iter().any(|r| !r.compared) {
                    return false;
                }
                // Pieces (copy, number) the compare flagged, with their ticket.
                let mut tickets: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
                let mut flags: Vec<(usize, u32, bool)> = Vec::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let at = (r.at + p) as usize;
                        let (Some(&flag), Some(&entry)) = (differs.get(at), table.get(at)) else {
                            return false;
                        };
                        if flag != 0 {
                            if entry == INFLIGHT_NONE {
                                return false;
                            }
                            if !inflight_gets_slot(entry, plan.inflight_landed, slots) {
                                INFLIGHT_STATS[2].fetch_add(1, Ordering::Relaxed);
                                return false;
                            }
                            tickets.insert((copy, r.span.first + p), entry);
                        }
                        flags.push((copy, r.span.first + p, flag != 0));
                    }
                }
                let fetched = flagged_pieces(&flags);
                let mut seen = std::collections::HashSet::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece = r.span.first + p;
                        if !fetched.contains(&(copy, piece)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo >= hi || !seen.insert((copy, lo, hi)) {
                            continue;
                        }
                        let from = inflight_piece_from(&r.span, p);
                        let Some(&k) = tickets.get(&(copy, piece)) else {
                            return false;
                        };
                        if lo < from || (hi - from) as u64 > INFLIGHT_SLOT {
                            return false;
                        }
                        let slot = inflight.fetched.1 + (k % slots) as usize * INFLIGHT_SLOT as usize;
                        // Bytes a fetch the old way brought after this flight ran.
                        let newer: Vec<(usize, usize)> = inflight
                            .brought
                            .iter()
                            .filter(|b| b.3 >= plan.sent_as && Arc::ptr_eq(&b.0, &r.shadow) && b.1 < hi && b.2 > lo)
                            .map(|b| (b.1, b.2))
                            .collect();
                        for (a, b) in crate::shadow::outside(lo, hi, newer) {
                            pieces.push(RelayCardPiece {
                                shadow: r.shadow.clone(),
                                lo: a,
                                hi: b,
                                gpu_ptr: slot + (a - from),
                                after: plan.sent_as,
                                track_walls: false,
                            });
                        }
                    }
                }
            }
        }
        // Outside the lock: no slot of these plans is written over before
        // their tickets are landed (`inflight_tickets_landed`, after this).
        let piece_count = pieces.len() as u64;
        let bytes: u64 = pieces.iter().map(|p| (p.hi - p.lo) as u64).sum();
        let written = relay_merge_card_pieces(pieces, wide_merge());
        for plan in plans {
            for r in &plan.ranges {
                crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            }
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        INFLIGHT_STATS[0].fetch_add(piece_count, Ordering::Relaxed);
        INFLIGHT_STATS[1].fetch_add(bytes, Ordering::Relaxed);
        // The next merge is of a later plan, or a fetch the old way sent now.
        if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
            crate::shadow::relay_fetched(last);
        }
        true
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let plans: Vec<&CardPlan> = flights.iter().filter_map(|f| f.card.as_ref()).collect();
                let bitmaps: Vec<&[u32]> = flights
                    .iter()
                    .filter_map(|f| f.marks.as_ref().map(ViewMarksCapture::words))
                    .collect();
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    let flight_has_marks = flight.marks.is_some();
                    for (shadow, ranges, unmarked) in &flight.back {
                        let flight_unmarked = if flight_has_marks { unmarked } else { ranges };
                        match back.iter_mut().find(|(s, _, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all, all_unmarked)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                                for range in flight_unmarked {
                                    if !all_unmarked.contains(range) {
                                        all_unmarked.push(*range);
                                    }
                                }
                            }
                            None => back.push((
                                shadow.clone(),
                                ranges.clone(),
                                flight_unmarked.clone(),
                            )),
                        }
                    }
                }
                fill_ahead::landing(flights.iter().map(|f| f.number).min().unwrap_or(0));
                self.relay_fetch_back(&plans, &back, &bitmaps, fetch);
                relay_hand_back(&back, &bitmaps);
                fill_ahead::landing(0);
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges, _) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                COPY_BACK[0].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        let mut last = 0;
        let mut landed = Vec::new();
        for item in items {
            match item {
                Late::Run(flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    if let Some(marks) = flight.marks {
                        marks.recycle();
                    }
                    landed.push((flight.lists, flight.order));
                }
                Late::Write(at, what, bytes) => {
                    // Remove this write from `in_hand` before making it visible
                    // to the title, so a subsequent reset + new wait cannot
                    // falsely match the already-landed write.
                    {
                        let mut line = line();
                        if let Some(pos) = line
                            .in_hand
                            .iter()
                            .position(|&(a, _, b)| a == at && b == bytes)
                        {
                            line.in_hand.remove(pos);
                        }
                    }
                    what.write(at as *mut u8, bytes);
                }
                Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
            }
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
        }
        self.log_messages();
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r, _)| r.iter().copied()));
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in plan.ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                if r.compared && differs[(r.at + p) as usize] == 0 {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        let _ = self.submit(runs, false);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, true)
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges, _)| ranges.iter().copied()));
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                marks: sent.marks,
                _exec_epoch: sent.exec_epoch,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        land();
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges, _)| ranges.iter().copied()));
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                marks: sent.marks,
                _exec_epoch: sent.exec_epoch,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // SAFETY: the fence and event are ours, under the
                // submission lock; `submitted` is the last value signalled.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            working(false);
                            WaitForSingleObject(self.event, INFINITE);
                            stood(waiting);
                            working(true);
                        }
                        Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
                    }
                }
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let bitmaps: Vec<&[u32]> = items
                    .iter()
                    .filter_map(|item| match item {
                        Late::Run(flight) => flight.marks.as_ref().map(ViewMarksCapture::words),
                        _ => None,
                    })
                    .collect();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                        if !bitmaps.is_empty() {
                            for r in &card.ranges {
                                if r.mark_base != victory_shader::emit::VIEW_MARK_NONE {
                                    let (_, decl_pages, marked_pages) =
                                        r.shadow.filter_marked_spans(&[r.declared], &[], &bitmaps);
                                    crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                                }
                            }
                        }
                    }
                    let flight_has_marks = flight.marks.is_some();
                    for (shadow, ranges, unmarked) in &flight.back {
                        let flight_unmarked = if flight_has_marks { unmarked } else { ranges };
                        match back.iter_mut().find(|(s, _, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all, all_unmarked)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                                for range in flight_unmarked {
                                    if !all_unmarked.contains(range) {
                                        all_unmarked.push(*range);
                                    }
                                }
                            }
                            None => back.push((
                                shadow.clone(),
                                ranges.clone(),
                                flight_unmarked.clone(),
                            )),
                        }
                    }
                }
                self.rebar_back(submitted, &back, &bitmaps);
                hand_back(&back, &bitmaps);
                COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        if let Some(marks) = flight.marks {
                            marks.recycle();
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    fn submit<'a>(&'a self, runs: &[&HostList], late: bool) -> Option<(std::sync::MutexGuard<'a, u64>, Sent)> {
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let was_working = working(true);
        submitting(1, runs.len());
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if let Some(why) = late_blocked(runs) {
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                relay_wait(why, |line| blocked_by(&line.written, runs, false).is_none());
            } else {
                self.settle(&mut submitted, why);
            }
        }
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted.
        crate::descriptor::sync_heaps();
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run.
        crate::descriptor::make_waiting();
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        #[cfg(test)]
        let exec_scope = runs
            .iter()
            .find_map(|l| l.retire_scope.clone())
            .or_else(current_retire_scope);
        let exec_epoch = {
            #[cfg(test)]
            let exec_state: &Mutex<RetireState> = exec_scope.as_deref().unwrap_or(&RETIRE_STATE);
            #[cfg(not(test))]
            let exec_state: &Mutex<RetireState> = &RETIRE_STATE;
            let epoch = acquire_epoch_in(exec_state, exec_work_seq());
            ExecEpochGuard {
                #[cfg(test)]
                scope: exec_scope,
                epoch,
            }
        };
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            let mut spans: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && !shadow.unwatched()
                    {
                        let span = spans
                            .entry(std::sync::Arc::as_ptr(shadow) as usize)
                            .or_insert((*address, *address + *len));
                        *span = (span.0.min(*address), span.1.max(*address + *len));
                    }
                }
            }
            let ranges: Vec<(usize, usize)> = spans.into_values().collect();
            crate::watch::ask_once(&ranges);
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if behind && stale_under_flights(runs) {
                let settling = std::time::Instant::now();
                self.settle(&mut submitted, WHY_STALE);
                settled += settling.elapsed();
                crate::watch::ask_once(&ranges);
            }
        } else if behind && stale_under_flights(runs) {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        for list in runs {
            list.run_deferred(&mut batch, self, &mut submitted);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        crate::watch::ask_once_end();
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        let card = self.card_before(runs, flown);
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        let back = declared_writes(runs);
        let mut rebar_entries: Vec<RebarWriteEntry> = Vec::new();
        if !late {
            for (shadow, declared, unmarked) in &back {
                if !shadow.is_rebar() {
                    continue;
                }
                let ranges: Vec<(usize, usize)> = declared
                    .iter()
                    .map(|&(from, to)| shadow.inside(from, to.saturating_sub(from)))
                    .filter(|&(from, to)| from < to)
                    .collect();
                let declared_spans = merge_ranges(ranges);
                let piece_runs = spans_to_piece_runs(&declared_spans);
                let mark_base = if crate::shadow::view_marks_on() && unmarked.is_empty() {
                    shadow.mark_base()
                } else {
                    None
                };
                rebar_entries.push(RebarWriteEntry {
                    shadow: shadow.clone(),
                    declared_title: declared.clone(),
                    unmarked_title: unmarked.clone(),
                    declared_spans,
                    piece_runs,
                    compare_out_base: None,
                    compare_pieces: 0,
                    mark_base,
                });
            }
        }
        let mut rebar_hash_plan = if rebar_entries.is_empty() {
            RebarHashPlan::default()
        } else {
            self.prepare_rebar_hash_plan(&mut rebar_entries)
        };
        let (mut marks_list, marks_capture) = if !back.is_empty() {
            match self.record_view_marks_capture() {
                Some((cl, cap)) => (Some(cl), Some(cap)),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        let prepared = std::time::Instant::now();
        if late {
            if let Some(cl) = marks_list.take() {
                lists.push(Some(cl));
            }
            *submitted += 1;
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                self.queue.ExecuteCommandLists(&lists);
                self.queue.Signal(&self.fence, *submitted)
            };
            self.card_sent();
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                marks: marks_capture,
                writes,
                exec_epoch,
            };
            working(was_working);
            return Some((submitted, sent));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let timed = timing();
        #[cfg(not(feature = "diagnostics"))]
        let timed = false;
        let mut batched_lists: Vec<Option<ID3D12CommandList>> = Vec::new();
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timed {
            if let Some(pre) = rebar_hash_plan.pre_list.take() {
                let _ = self.submit_helper_list(&mut submitted, pre, "ReBAR pre-hash");
            }
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else if rebar_hash_plan.pre_list.is_some()
            || rebar_hash_plan.post_list.is_some()
            || marks_list.is_some()
        {
            batched_lists.reserve(lists.len() + 3);
            if let Some(pre) = rebar_hash_plan.pre_list.take() {
                batched_lists.push(Some(pre));
            }
            batched_lists.extend(lists.iter().cloned());
            if let Some(post) = rebar_hash_plan.post_list.take() {
                batched_lists.push(Some(post));
            }
            if let Some(cl) = marks_list.take() {
                batched_lists.push(Some(cl));
            }
            vec![&batched_lists[..]]
        } else {
            vec![&lists[..]]
        };
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                self.card_sent();
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        working(false);
                        WaitForSingleObject(self.event, INFINITE);
                        stood(started);
                        working(true);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let mut noted = 0.0;
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    noted += ms;
                    crate::copystat::command_took(&op, ms);
                }
                // The list's run as the wait saw it, less its noted
                // commands: its prelude, barriers, clears and copies, and
                // the wait itself. And the whole, to set beside the frame
                // line's time on the card (the rest of that is our passes
                // outside the title's lists).
                let wall = started.elapsed().as_secs_f64() * 1e3;
                crate::copystat::command_took("(title-lists beyond their noted commands)", (wall - noted).max(0.0));
                crate::copystat::command_took("(title-lists whole, as waited for)", wall);
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let mut post_compare_ok = true;
        if timed && let Some(post) = rebar_hash_plan.post_list {
            post_compare_ok = self.submit_helper_list(&mut submitted, post, "ReBAR post-compare");
        }
        let mut marks_capture_ok = true;
        if timed && let Some(cl) = marks_list.take() {
            marks_capture_ok = self.submit_helper_list(&mut submitted, cl, "view-marks capture");
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        let bitmaps: Vec<&[u32]> = if marks_capture_ok {
            marks_capture.iter().map(ViewMarksCapture::words).collect()
        } else {
            Vec::new()
        };
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
            if !bitmaps.is_empty() {
                for r in &card.ranges {
                    if r.mark_base != victory_shader::emit::VIEW_MARK_NONE {
                        let (_, decl_pages, marked_pages) =
                            r.shadow.filter_marked_spans(&[r.declared], &[], &bitmaps);
                        crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                    }
                }
            }
        }
        if !rebar_entries.is_empty() {
            self.finish_rebar_copy_back(
                &mut submitted,
                rebar_entries,
                if post_compare_ok {
                    rebar_hash_plan.total_flag_pieces
                } else {
                    0
                },
                &bitmaps,
            );
        }
        hand_back(&back, &bitmaps);
        if let Some(marks) = marks_capture {
            marks.recycle();
        }
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        drop(submitted);
        drop(exec_epoch);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                        #[cfg(test)]
                        self.d3d12_errors.fetch_add(1, Ordering::Relaxed);
                        if first_error.is_none() {
                            first_error = Some(text.to_owned());
                        }
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        self.twins_to_depth();
        self.textures_back();
        // The scratch buffers its prelude fills and leaves readable as
        // arguments: common again behind the list, as every scratch buffer.
        for set in self
            .gather
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sets
            .iter()
            .filter(|s| s.records() > 0)
        {
            // SAFETY: an open list and a live buffer.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    &set.scratch,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
        crate::cardmeter::end(
            &self.list,
            crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)),
        );
        crate::cardmeter::end_named(
            &self.list,
            crate::cardmeter::Span::from_raw(self.meter.swap(u32::MAX, Ordering::Relaxed)),
            &self.pipelines.lock().unwrap_or_else(|e| e.into_inner()),
        );
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    /// A shader that can store to buffers is bound ([`HostList::buffer_writes`]).
    pub(crate) fn buffers_written(&self) {
        self.buffer_writes.store(true, Ordering::Relaxed);
    }

    /// Whether a shader that can store to buffers was bound since the last
    /// call (the caller waits for every earlier shader write now).
    pub(crate) fn take_buffer_writes(&self) -> bool {
        let was = self.buffer_writes.swap(false, Ordering::Relaxed);
        if was {
            // The caller waits for every earlier shader write.
            self.shader_writes.store(false, Ordering::Relaxed);
        }
        was
    }

    /// A shader that can write through unordered access is bound
    /// ([`HostList::shader_writes`]).
    pub(crate) fn shaders_written(&self) {
        self.shader_writes.store(true, Ordering::Relaxed);
    }

    /// Whether a shader that can write through unordered access was bound
    /// since the last wait for every earlier shader write; the caller waits
    /// now if so (which serves buffer writes too).
    pub(crate) fn take_shader_writes(&self) -> bool {
        let was = self.shader_writes.swap(false, Ordering::Relaxed);
        if was {
            self.buffer_writes.store(false, Ordering::Relaxed);
        }
        was
    }

    fn retire_state(&self) -> &Mutex<RetireState> {
        #[cfg(test)]
        if let Some(scope) = self.retire_scope.as_deref() {
            return scope;
        }
        &RETIRE_STATE
    }

    fn ensure_epoch(&self, _guard: &std::sync::MutexGuard<'_, ()>) {
        if self.epoch.load(Ordering::Acquire) == 0 {
            let e = acquire_epoch_in(self.retire_state(), self.work_seq.clone());
            self.epoch.store(e, Ordering::Release);
        }
    }

    /// Pins a retirement epoch for this list once it records GPU work, and
    /// advances `work_seq` lock-free when new resources have been created
    /// since the last command on this list.
    pub(crate) fn mark_work(&self) {
        let seq = RESOURCE_SEQ.load(Ordering::Acquire);
        if self.epoch.load(Ordering::Acquire) != 0 {
            if self.work_seq.load(Ordering::Relaxed) < seq {
                self.work_seq.fetch_max(seq, Ordering::AcqRel);
            }
            return;
        }
        let guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.work_seq.fetch_max(seq, Ordering::AcqRel);
        self.ensure_epoch(&guard);
    }

    /// Pins an epoch and increments the in-flight submission count for this
    /// list before queuing it on a command queue timeline.
    #[allow(dead_code)]
    pub(crate) fn begin_submit(&self) {
        let guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        self.ensure_epoch(&guard);
    }

    /// Releases a queued submission pin and, if it was the last in-flight
    /// submission for this list, releases the list's retirement epoch.
    /// Keeps `work_seq` intact so a subsequent resubmission of the same closed
    /// list still protects resources recorded into it.
    #[allow(dead_code)]
    pub(crate) fn end_submit(&self) {
        let released = {
            let _guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
            if self.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.epoch.swap(0, Ordering::AcqRel)
            } else {
                0
            }
        };
        if released != 0 {
            release_epoch_in(self.retire_state(), released);
        }
    }

    /// Releases a queued submission pin for a cancelled (unexecuted) submission
    /// without clearing the list's recording epoch.
    #[allow(dead_code)]
    pub(crate) fn cancel_submit(&self) {
        let _guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    #[allow(dead_code)]
    pub(crate) fn has_epoch(&self) -> bool {
        self.epoch.load(Ordering::Acquire) != 0
    }

    fn release_epoch(&self) {
        self.release_epoch_inner(false);
    }

    fn release_epoch_inner(&self, clear_work_seq: bool) {
        let released = {
            let _guard = self.epoch_lock.lock().unwrap_or_else(|e| e.into_inner());
            if self.in_flight.load(Ordering::Acquire) == 0 {
                if clear_work_seq {
                    self.work_seq.store(0, Ordering::Relaxed);
                }
                self.epoch.swap(0, Ordering::AcqRel)
            } else {
                0
            }
        };
        if released != 0 {
            release_epoch_in(self.retire_state(), released);
        }
    }

    /// Clears per-recording tracking state shared between [`Self::reset`] and
    /// [`Self::release_for_pool`].
    fn clear_tracking(&self) {
        *self.deferred() = Deferred::default();
        self.writes.store(0, Ordering::Relaxed);
        {
            let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
            chunks.used = 0;
            chunks.at = 0;
            chunks.barrier = false;
            chunks.bound = false;
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            let Gather {
                closed,
                sets,
                spare,
                ..
            } = &mut *gather;
            *closed = false;
            for set in sets.drain(..) {
                if spare.len() < GATHER_SETS {
                    spare.push(set.into_scratch());
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Releases all retained memory and retirement pins before parking an idle,
    /// finished list in the idle list pool.
    pub(crate) fn release_for_pool(&self) {
        self.clear_tracking();
        {
            let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
            chunks.all.clear();
        }
        {
            let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            gather.spare.clear();
        }
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            give_chunks(std::mem::take(&mut scratch.chunks));
            (scratch.current, scratch.used) = (0, 0);
        }
        self.release_epoch_inner(true);
    }

    /// Counts a dispatch or a copy: what may write indirect arguments.
    pub(crate) fn count_write(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
    }

    /// Dispatches and copies since the reset.
    pub(crate) fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    /// Room for `bytes` of clamped indirect arguments in this list's
    /// buffers, writable by shaders from here on in the list: the buffer and
    /// the offset. None for more than a buffer holds.
    pub(crate) fn clamp_room(&self, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        const CHUNK: u64 = 256 << 10;
        let bytes = bytes.next_multiple_of(16);
        if bytes > CHUNK {
            return None;
        }
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        if chunks.used > 0 && chunks.at + bytes <= CHUNK {
            let at = chunks.at;
            chunks.at += bytes;
            return Some((chunks.all[chunks.used - 1].0.clone(), at));
        }
        // The next buffer, writable.
        if chunks.used == chunks.all.len() {
            let buffer = host()?.uav_buffer(CHUNK)?;
            name(&buffer, || "hoisted indirect arguments".to_string());
            chunks.all.push((buffer, D3D12_RESOURCE_STATE_COMMON));
        }
        let i = chunks.used;
        chunks.used += 1;
        chunks.at = bytes;
        let mut barriers = Vec::new();
        if !chunks.barrier {
            // What earlier lists' shaders wrote first.
            chunks.barrier = true;
            barriers.push(crate::indirect::uav_barrier());
        }
        let (buffer, state) = &mut chunks.all[i];
        if *state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            barriers.push(transition(buffer, *state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            *state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        let buffer = buffer.clone();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
        Some((buffer, 0))
    }

    /// Whether the list's clamps still have to bind their root signature,
    /// pipeline and budget: the first after the reset does, and the rest
    /// find them bound (nothing else records compute state in the list).
    pub(crate) fn clamp_unbound(&self) -> bool {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        !std::mem::replace(&mut chunks.bound, true)
    }

    /// The clamped arguments written so far become readable as indirect
    /// arguments by the lists after this one.
    pub(crate) fn clamp_end(&self) {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        let used = chunks.used;
        let barriers: Vec<_> = chunks.all[..used]
            .iter_mut()
            .filter(|(_, state)| *state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .map(|(buffer, state)| {
                *state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
                transition(
                    buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )
            })
            .collect();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.buffer_writes.store(true, Ordering::Relaxed);
        self.shader_writes.store(true, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self._alive.used();
        self.clear_tracking();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            } else if scratch.chunks.len() > 1 {
                // Keep at most one warm 4 MiB upload chunk so steady-state active
                // lists avoid re-allocating while multi-chunk spikes are freed.
                give_chunks(scratch.chunks.drain(1..).collect());
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        self.pipeline_meter.store(u32::MAX, Ordering::Relaxed);
        self.meter.store(
            crate::cardmeter::begin(&self.list, crate::cardmeter::Kind::Lists).raw(),
            Ordering::Relaxed,
        );
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
        self.release_epoch_inner(true);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    #[track_caller]
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted. A draw's depth target is given the
        // pass before the draw binds anything (`plane_ahead`); what is left
        // here are shader reads and copies of the plane.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            let n = crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
            // Which use it was, for the first few (diagnostics builds).
            #[cfg(feature = "diagnostics")]
            if n < 24 {
                tracing::info!(
                    resource = format_args!("{resource:?}"),
                    write,
                    at = %std::panic::Location::caller(),
                    "a depth plane used before its twin was drawn into it"
                );
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = n;
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// The colour twins of title depth texture `resource` (the twin of its depth plane, and
    /// the twin of its stencil plane or of the whole texture), each with the format a shader
    /// reads it in (R32_FLOAT, R16_UNORM, R8_UNORM), moved in this list to a state shaders
    /// read while `f` records its reads of them, and back to unordered access (the state they
    /// are kept in, and the one a dispatch about to run needs) after. False without twins.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn with_twins_readable(&self, resource: crate::object::This, mut f: impl FnMut(&ID3D12Resource, i32)) -> bool {
        const NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;
        const UNORDERED_ACCESS: u32 = 0x8;
        let Some(twin) = crate::resource::twin(resource) else { return false };
        let view = |t: &crate::resource::HostTexture| {
            // SAFETY: a live resource.
            match unsafe { t.texture.GetDesc() }.Format.0 {
                60..=62 => 61,
                53 | 55..=57 => 56,
                39..=42 => 41,
                _ => 0,
            }
        };
        let mut sides: Vec<(std::sync::Arc<crate::resource::HostTexture>, i32)> = Vec::new();
        if let Some(plane) = &twin.plane {
            sides.push((plane.clone(), view(plane)));
        }
        sides.push((twin.colour.clone(), view(&twin.colour)));
        sides.retain(|(_, v)| *v != 0);
        for (t, _) in &sides {
            self.settle(t, NON_PIXEL_SHADER_RESOURCE);
        }
        for (t, v) in &sides {
            f(&t.texture, *v);
        }
        for (t, _) in &sides {
            self.settle(t, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Before a draw that binds title depth texture `resource` as its
    /// target: if this list wrote the twin of the texture's depth plane
    /// since the plane was last given it, the twin is copied and drawn
    /// into the plane here, so that the draw tests against what compute
    /// wrote (Grand Theft Auto VI writes a material class into the depth
    /// plane through unordered access, then lights each class with draws
    /// that test the plane for equality). True if the pass was recorded:
    /// it left a pipeline, root signature, heaps, target, viewport,
    /// scissor and topology of ours set, and the caller sets the title's
    /// again (`State::replay`) before it binds anything.
    /// `VICTORY_PLANE_AHEAD=0` leaves the pass for the list's close.
    pub(crate) fn plane_ahead(&self, resource: crate::object::This) -> bool {
        if !plane_ahead_on() {
            return false;
        }
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        if twin.plane.is_none() {
            return false;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let Some(u) = twins.get_mut(&(std::sync::Arc::as_ptr(&twin) as usize)) else {
            return false;
        };
        if u.behind[0] != Some(true) {
            return false;
        }
        u.behind[0] = Some(false);
        u.late = false;
        let depth = u.depth.clone();
        drop(twins);
        self.twin_copy(&depth, &twin, false);
        self.pass_plane(&depth, &twin);
        crate::resource::PLANES[3].fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Before a draw or dispatch binds anything: every depth plane whose twin this list
    /// wrote since the plane was given it is given it now ([`HostList::plane_ahead`] for
    /// all of them), so that shaders reading the plane through a view see what compute
    /// wrote, as the console's do (the material classes Grand Theft Auto VI's lighting
    /// reads back). True if a pass was recorded: the caller sets the title's state again.
    pub(crate) fn planes_ahead(&self) -> bool {
        if !plane_ahead_on() {
            return false;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let due: Vec<_> = twins
            .values_mut()
            .filter(|u| u.twin.plane.is_some() && u.behind[0] == Some(true))
            .map(|u| {
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone())
            })
            .collect();
        drop(twins);
        for (depth, twin) in &due {
            self.twin_copy(depth, twin, false);
            self.pass_plane(depth, twin);
            crate::resource::PLANES[3].fetch_add(1, Ordering::Relaxed);
        }
        !due.is_empty()
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of Grand Theft Auto VI no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind. For
    /// title-written textures whose staging buffer was refreshed (or has
    /// not been uploaded yet) it copies the staging buffer into the host
    /// texture before the list uses it.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        // `run_deferred` has already refreshed each staging buffer, so we
        // know here exactly which textures need a GPU copy on this submission.
        let uploads: Vec<_> = {
            let deferred = self.deferred();
            deferred
                .texture_uploads
                .iter()
                .filter_map(|(copy, staging)| {
                    if staging.take_upload() {
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::TexUploads, 1);
                        Some((copy.clone(), staging.clone()))
                    } else {
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
                        None
                    }
                })
                .collect()
        };
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        for (copy, _) in &uploads {
            let key = std::sync::Arc::as_ptr(copy) as usize;
            if copying.contains_key(&key) {
                continue;
            }
            let real = copy.state.load(Ordering::Relaxed);
            copying.insert(key, COPY_DEST);
            if real != COPY_DEST {
                to_copy.push(transition(
                    &copy.texture,
                    host_state(real),
                    host_state(COPY_DEST),
                ));
                if !states.contains_key(&key) {
                    barriers.push(transition(
                        &copy.texture,
                        host_state(COPY_DEST),
                        host_state(real),
                    ));
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        // The records the list's draws read from slots (`crate::indirect::gather`).
        let gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        let gathered = gather.sets.iter().any(|s| s.records() > 0);
        if barriers.is_empty() && syncs.is_empty() && uploads.is_empty() && !gathered {
            return None;
        }
        let Some((allocator, list)) = self
            .prelude
            .get_or_init(|| host().and_then(|h| h.make_prelude()))
            .as_ref()
        else {
            fail(format_args!("no host list for a state prelude"));
        };
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                for (copy, staging) in &uploads {
                    staging.record_copy(list, &copy.texture);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                if gathered {
                    // The slots must hold clamped records before a draw
                    // reads them: a prelude that cannot record the passes
                    // is no prelude.
                    let filled = match &gather.budget {
                        Some(budget) => crate::indirect::record_gathers(list, &gather.sets, budget),
                        None => false,
                    };
                    if !filled {
                        fail(format_args!(
                            "a list's gathered indirect records could not be clamped in its prelude"
                        ));
                    }
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// From here on the list gathers no records (`crate::indirect::gather`):
    /// what is about to be recorded may write arguments.
    pub(crate) fn gather_close(&self) {
        self.gather.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
    }

    /// A slot for the record at `offset` of the host copy `src` in the
    /// scratch buffer the list's prelude fills (`crate::indirect::gather`,
    /// which checked the record): the buffer and the slot's place in it.
    pub(crate) fn gather_slot(
        &self,
        src: &ID3D12Resource,
        src_id: u64,
        offset: u32,
        (kind, stride): (u32, u32),
    ) -> Option<(ID3D12Resource, u64)> {
        let none = |why: usize| {
            crate::indirect::GATHER_NONE[why].fetch_add(1, Ordering::Relaxed);
            None
        };
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        if gather.closed {
            return none(1);
        }
        let found = gather.sets.iter().position(|s| {
            s.src_id == src_id
                && s.kind == kind
                && s.stride == stride
                && s.src.as_raw() == src.as_raw()
        });
        let at = match found {
            Some(at) => at,
            None => {
                if gather.sets.len() >= GATHER_SETS {
                    return none(2);
                }
                // The prelude zeroes the budget with WriteBufferImmediate.
                if self.list.cast::<ID3D12GraphicsCommandList2>().is_err() {
                    return none(4);
                }
                let Some(host) = host() else {
                    return none(4);
                };
                if gather.budget.is_none() {
                    let Some(budget) = host.uav_buffer(256) else {
                        return none(4);
                    };
                    name(&budget, || "indirect budget of gathered records".to_owned());
                    gather.budget = Some(budget);
                }
                let scratch = match gather.spare.pop() {
                    Some(scratch) => scratch,
                    None => {
                        let Some(scratch) = host.uav_buffer(
                            u64::from(crate::indirect::GATHER_MOST)
                                * u64::from(crate::indirect::GATHER_STRIDE),
                        ) else {
                            return none(4);
                        };
                        name(&scratch, || "indirect arguments, gathered".to_owned());
                        scratch
                    }
                };
                let Some(table) = self.scratch(u64::from(crate::indirect::GATHER_MOST) * 4) else {
                    gather.spare.push(scratch);
                    return none(4);
                };
                gather.sets.push(crate::indirect::GatherSet::new(
                    src.clone(),
                    src_id,
                    (kind, stride),
                    scratch,
                    table,
                ));
                gather.sets.len() - 1
            }
        };
        let set = &mut gather.sets[at];
        let Some(slot) = set.slot(offset) else {
            return none(3);
        };
        let dst_stride = if kind & crate::indirect::ROWS != 0 {
            crate::indirect::ROW_STRIDE
        } else {
            stride
        };
        Some((set.scratch.clone(), u64::from(slot) * u64::from(dst_stride)))
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            // `VICTORY_PIX_MARKERS=1`: the command's name as a marker on
            // the list, for a PIX capture's event list (without the notes
            // and query heaps of device-removed data).
            static MARK: OnceLock<bool> = OnceLock::new();
            if *MARK.get_or_init(|| std::env::var_os("VICTORY_PIX_MARKERS").is_some_and(|v| v == "1")) {
                let what = what();
                let text: Vec<u16> = what.chars().take(200).collect::<String>().encode_utf16().chain([0]).collect();
                // SAFETY: an open list; the marker is copied when recorded.
                unsafe { self.list.SetMarker(0, Some(text.as_ptr().cast()), 2 * text.len() as u32) };
                return;
            }
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.textures.is_empty()
                && deferred.texture_uploads.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty()
                && deferred.after.is_empty()
                && deferred.back_wanted.is_empty()
                && deferred.texture_backs.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        if !deferred.back_wanted.contains(&(resource as usize)) {
            deferred.back_wanted.push(resource as usize);
        }
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for resource in wanted {
            let resource = resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                continue;
            };
            self.settle(&copy, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, resource, &copy.texture) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// Notes a pipeline the list binds; called where it is bound.
    #[cfg(test)]
    pub(crate) fn used_pipeline(&self, name: &str) {
        {
            let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            if !pipelines.contains(name) {
                pipelines.insert(name.to_owned());
            }
        }
        if crate::cardmeter::pipelines() {
            // The pipeline before it ends here on the card's clock.
            crate::cardmeter::end(
                &self.list,
                crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)),
            );
            self.pipeline_meter.store(
                crate::cardmeter::begin_named(&self.list, name).raw(),
                Ordering::Relaxed,
            );
        }
    }

    /// Notes a pipeline the list binds and keeps it and its root signature
    /// until the list is reset, the first time only: the list keeps the
    /// first, so its address names the same pipeline from then on.
    pub(crate) fn bind_pipeline(&self, p: &crate::pipeline::host::HostPipeline) {
        let first = {
            let mut deferred = self.deferred();
            let first = deferred.bound.insert(p.pso.as_raw() as usize);
            if first {
                deferred.kept.push(p.pso.clone().into());
                deferred.kept.push(p.root.clone().into());
            }
            first
        };
        if first {
            let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            if !pipelines.contains(p.name.as_str()) {
                pipelines.insert(p.name.clone());
            }
        }
        if crate::cardmeter::pipelines() {
            // The pipeline before it ends here on the card's clock.
            crate::cardmeter::end(
                &self.list,
                crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)),
            );
            self.pipeline_meter.store(
                crate::cardmeter::begin_named(&self.list, &p.name).raw(),
                Ordering::Relaxed,
            );
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        TITLE_COPIES.fetch_add(1, Ordering::Relaxed);
        if bare() & 64 != 0 {
            return;
        }
        let mut deferred = self.deferred();
        // A copy out of memory this list's shaders write ([`copy_after`]):
        // done once the list has run and what they wrote is in title
        // memory, not at submission, when it would copy what was there
        // before.
        if copy_after() {
            let (from, to) = (src as usize, src as usize + len);
            let gpu_written = deferred.copies.iter().any(|step| {
                matches!(step, Step::GpuWrite(_, at, n, _) if *at < to && from < *at + *n)
            });
            if gpu_written {
                let n = COPIES_AFTER.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!(
                        copies = n,
                        dst = format_args!("{:#x}", dst as usize),
                        src = format_args!("{:#x}", src as usize),
                        len,
                        "a buffer copy out of memory the list's shaders write is done after the list (VICTORY_COPY_AFTER)"
                    );
                }
                deferred.after.push((dst as usize, src as usize, len));
                return;
            }
        }
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.filled.clear();
        deferred.gpu_written.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        if bare() & 64 != 0 {
            return;
        }
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// The host GPU address of `len` bytes of upload memory holding the title
    /// memory at `src` as of the list's submission: one copy for every ask
    /// until a title copy or run intervenes.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        if let Some(&gpu) = self.deferred().filled.get(&key) {
            return Some(gpu);
        }
        let (gpu, to) = self.scratch(len)?;
        self.defer_fill(to, src, len as usize);
        self.deferred().filled.insert(key, gpu);
        Some(gpu)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        if bare() & 4 != 0 {
            return;
        }
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Batched [`HostList::defer_refresh`]: takes the deferred lock once for
    /// all `(shadow, address, len)` ranges.
    pub(crate) fn defer_refresh_batch<'a>(
        &self,
        reserve: usize,
        items: impl IntoIterator<Item = (&'a std::sync::Arc<crate::shadow::Shadow>, u64, u64)>,
    ) {
        if bare() & 4 != 0 {
            return;
        }
        let mut deferred = self.deferred();
        if reserve > 0 {
            deferred.refreshed.reserve(reserve);
            deferred.copies.reserve(reserve);
        }
        for (shadow, address, len) in items {
            let clamped = shadow.clamp_len(address, len) as usize;
            let addr = address as usize;
            let key = (std::sync::Arc::as_ptr(shadow) as usize, addr, clamped);
            if deferred.refreshed.insert(key) {
                deferred
                    .copies
                    .push(Step::Refresh(shadow.clone(), addr, clamped));
            }
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    /// `marked`: shaders mark the 64 KiB pages they write in `SLOT_VIEW_WRITES`.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len, marked);
        if deferred.gpu_written.insert(key) {
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len, marked));
        }
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        self.deferred().backs.push((buffer, dst as usize, [row, rows, pitch]));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// How this list's deferred shader/bottom-of-pipe writes affect a wait on
    /// `[address, address + bytes)` (`crate::queue::Joined::meets`).
    pub(crate) fn will_write_state(
        &self,
        address: usize,
        bytes: usize,
        holds: impl Fn(u64) -> bool,
    ) -> WriteMatch {
        let end = address.saturating_add(bytes);
        let deferred = self.deferred();
        let shader_touches = deferred.copies.iter().any(|step| match step {
            Step::GpuWrite(_, at, len, _) => *at < end && address < at.saturating_add(*len),
            _ => false,
        });
        if shader_touches {
            return WriteMatch::Unsure;
        }
        let mut last_exact: Option<bool> = None;
        for &(at, what, n) in &deferred.writes {
            let write_end = at.saturating_add(n);
            if at >= end || address >= write_end {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if at == address && n >= bytes => {
                    last_exact = Some(holds(v));
                }
                _ => return WriteMatch::Unsure,
            }
        }
        match last_exact {
            Some(met) => WriteMatch::Exact(met),
            None => WriteMatch::None,
        }
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// `size` bytes of mapped upload memory, 512-byte aligned for a placed
    /// footprint, alive until the list is reset: the buffer, the offset in
    /// it and the CPU pointer.
    pub(crate) fn upload_scratch(&self, size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        self.scratch_in(size, 512, |buffer, at, _, data| (buffer.clone(), at, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        self.scratch_in(len, 256, |_, _, gpu, data| (gpu, data))
    }

    /// `len` bytes of upload memory at a multiple of `align` until the list
    /// is reset, handed to `f` with the buffer, the offset in it, the host GPU
    /// address and the CPU pointer.
    fn scratch_in<R>(
        &self,
        len: u64,
        align: u64,
        f: impl FnOnce(&ID3D12Resource, u64, u64, *mut u8) -> R,
    ) -> Option<R> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.scratch_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let made = f(&buffer, 0, gpu, data);
            self.deferred().hold_upload(buffer, len);
            return Some(made);
        }
        // The chunk in use while `len` at `align` fits in what is left of it,
        // else the next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len()
            && s.used.next_multiple_of(align) + len > s.chunks[s.current].1
        {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let chunk = &s.chunks[s.current].0;
        let at = s.used.next_multiple_of(align);
        let made = f(&chunk.buffer, at, chunk.gpu + at, (chunk.data + at as usize) as *mut u8);
        s.used = at + len;
        Some(made)
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all): taken, then zeroed.
            deferred.copies.push(Step::Host(Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                let spent = words.read_volatile();
                if spent > 0 {
                    BUDGET_RUNS[0].fetch_add(1, Ordering::Relaxed);
                    if spent >= crate::indirect::BUDGET {
                        BUDGET_RUNS[1].fetch_add(1, Ordering::Relaxed);
                    }
                    BUDGET_RUNS[2].fetch_max(u64::from(spent), Ordering::Relaxed);
                    BUDGET_RUNS[3].fetch_max(u64::from(words.add(3).read_volatile()), Ordering::Relaxed);
                }
                BUDGET_RUNS[4].fetch_add(u64::from(words.add(4).read_volatile()), Ordering::Relaxed);
                BUDGET_RUNS[5].fetch_max(u64::from(words.add(5).read_volatile()), Ordering::Relaxed);
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                // And of its indirect compute records (words 6 and 7): the largest x
                // asked for, and how many asked for more than a dispatch runs.
                let (widest, cut) = (words.add(6).read_volatile(), words.add(7).read_volatile());
                COMPUTE_GROUPS[0].fetch_max(u64::from(widest), Ordering::Relaxed);
                COMPUTE_GROUPS[1].fetch_add(u64::from(cut), Ordering::Relaxed);
                for k in 0..8 {
                    words.add(k).write_volatile(0);
                }
            })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
        deferred.filled.clear();
        deferred.gpu_written.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Copies `staging` into `copy` in this list's prelude when submitted,
    /// if `staging` has not been uploaded yet or was detiled by `refresh`.
    pub(crate) fn defer_texture_upload(
        &self,
        copy: std::sync::Arc<crate::resource::HostTexture>,
        staging: std::sync::Arc<crate::texture::Staging>,
    ) {
        self.deferred().texture_uploads.push((copy, staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch, host: &Host, submitted: &mut u64) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let mut started = std::time::Instant::now();
            let (copied, is_fill) = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written();
                    // `batch.title_written()` already timed and accounted its
                    // flushed refreshes in `PREPARING[0]`; measure only this
                    // copy step so refreshes are not double-counted.
                    started = std::time::Instant::now();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    (len, false)
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    crate::shadow::copy_mapped_wide(dst as *mut u8, src as *const u8, len);
                    (len, true)
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    // It wrote title memory, where is not known here: the
                    // submission's one asking ends, later refreshes ask anew.
                    if matches!(step, Step::Run(_)) {
                        crate::watch::ask_once_end();
                        crate::watch::ask_once_void();
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, _) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame).
                    batch.flush_one(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    shadow.writing(*address, *len, |buffer, size| {
                        host.readback_rebar_to_vec(submitted, buffer, size)
                    });
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        staging.refresh();
                    }
                    continue;
                }
            };
            let us = started.elapsed().as_micros() as u64;
            PREPARING[0].fetch_add(us, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
            if is_fill {
                PREPARING_COPIES[PREP_FILL_US].fetch_add(us, Ordering::Relaxed);
                PREPARING_COPIES[PREP_FILL_COUNT].fetch_add(1, Ordering::Relaxed);
                PREPARING_COPIES[PREP_FILL_BYTES].fetch_add(copied as u64, Ordering::Relaxed);
            } else {
                PREPARING_COPIES[PREP_TITLE_COPY_US].fetch_add(us, Ordering::Relaxed);
                PREPARING_COPIES[PREP_TITLE_COPY_COUNT].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH=0` disables it; `=1` forces it).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (four or more) are
/// spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_refresh_enabled(
            std::env::var("VICTORY_WIDE_REFRESH").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_refresh_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

#[cfg(test)]
mod wide_refresh_tests {
    use super::wide_refresh_enabled;

    #[test]
    fn parallel_refresh_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_refresh_enabled(None, Some(4)));
        assert!(wide_refresh_enabled(None, Some(16)));
        assert!(!wide_refresh_enabled(None, Some(2)));
        assert!(!wide_refresh_enabled(None, None));
        assert!(wide_refresh_enabled(Some("1"), Some(1)));
        assert!(!wide_refresh_enabled(Some("0"), Some(16)));
        assert!(wide_refresh_enabled(Some("auto"), Some(4)));
    }

    #[test]
    fn victory_copy_try_variants_preserve_bytes_and_recover_from_faults() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        super::install_fast_copy_handler();

        let variants: Vec<(&str, unsafe extern "C" fn(*mut u8, *const u8, usize) -> u32)> = {
            let mut v: Vec<(&str, unsafe extern "C" fn(*mut u8, *const u8, usize) -> u32)> =
                vec![("sse2", super::victory_copy_try_sse2)];
            if std::is_x86_feature_detected!("avx2") {
                v.push(("avx2", super::victory_copy_try_avx2));
            }
            v
        };

        // 1. Verify exact copies across lengths 0..=260, unaligned src/dst offsets,
        // and multi-kilobyte spans without clobbering sentinel guard bytes.
        let src: Vec<u8> = (0..8192usize)
            .map(|i| ((i.wrapping_mul(53) ^ (i >> 5) ^ 0x3c) & 0xff) as u8)
            .collect();
        let mut dst = vec![0xEEu8; 8192];
        for &(name, copy_fn) in &variants {
            for dst_off in [0usize, 1, 7, 15, 16, 17, 31, 32, 33] {
                for src_off in [0usize, 3, 15, 16, 31] {
                    for len in [
                        0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95,
                        127, 128, 129, 159, 160, 161, 255, 256, 257, 4097,
                    ] {
                        dst.fill(0xEE);
                        // SAFETY: disjoint buffers with `dst_off + len < 8192`.
                        let ok = unsafe {
                            copy_fn(
                                dst.as_mut_ptr().add(dst_off),
                                src.as_ptr().add(src_off),
                                len,
                            )
                        };
                        assert_eq!(ok, 1, "{name} copy failed for len={len}");
                        assert_eq!(
                            &dst[dst_off..dst_off + len],
                            &src[src_off..src_off + len],
                            "{name} mismatch dst_off={dst_off} src_off={src_off} len={len}"
                        );
                        assert!(
                            dst[..dst_off].iter().all(|&b| b == 0xEE)
                                && dst[dst_off + len..].iter().all(|&b| b == 0xEE),
                            "{name} overwrote guard bytes dst_off={dst_off} len={len}"
                        );
                    }
                }
            }
        }

        // 2. Verify fault recovery on an uncommitted page at multiple distances
        // (hits small tail, alignment prologue, and 128B/64B main vector loops).
        // SAFETY: reserve 2 pages, commit only the first.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 2 * 4096, MEM_RESERVE, PAGE_NOACCESS)
                .cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(
                VirtualAlloc(base.cast(), 4096, MEM_COMMIT, PAGE_READWRITE),
                base.cast()
            );
            base.write_bytes(0x42, 4096);
            let mut out = vec![0u8; 2 * 4096];
            for &(name, copy_fn) in &variants {
                for &(start_off, len) in &[
                    (4096usize, 16usize),
                    (4096 - 3, 16),
                    (4096 - 32, 64),
                    (4096 - 100, 256),
                    (0, 2 * 4096),
                ] {
                    let ok = copy_fn(out.as_mut_ptr().add(7), base.add(start_off), len);
                    assert_eq!(
                        ok, 0,
                        "{name} must return 0 on unmapped page (start_off={start_off}, len={len})"
                    );
                }
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn wide_merge_is_default_on_multicore_and_respects_disable() {
        use super::wide_merge_enabled;
        assert!(wide_merge_enabled(None, Some(4)));
        assert!(wide_merge_enabled(None, Some(16)));
        assert!(!wide_merge_enabled(None, Some(2)));
        assert!(!wide_merge_enabled(None, None));
        assert!(wide_merge_enabled(Some("1"), Some(1)));
        assert!(!wide_merge_enabled(Some("0"), Some(16)));
        assert!(wide_merge_enabled(Some("auto"), Some(4)));
    }

    #[test]
    fn wide_relay_card_merge_matches_sequential_across_pieces_and_shadows() {
        use super::{RelayCardPiece, relay_merge_card_pieces, wide_merge_enabled};
        use crate::shadow::Shadow;
        const PIECE: usize = 64 << 10;

        let run_case = |parallel: bool| -> Option<(usize, Vec<Vec<u8>>, Vec<Vec<u8>>)> {
            // 3 card shadows:
            // - Shadow 0: 24 pieces (1.5 MiB) -> tests partitioning one large shadow across workers (>= 16 pieces).
            // - Shadow 1: 4 pieces (256 KiB) with dirty & sent ranges + overlapping plan-ordered pieces.
            // - Shadow 2: 2 pieces (128 KiB) with unchanged and sparsely modified pieces.
            let specs: &[(usize, Vec<(usize, usize)>, Vec<(usize, usize, u64)>)] = &[
                (24 * PIECE, vec![], vec![(3 * PIECE + 128, 3 * PIECE + 256, 11)]),
                (
                    4 * PIECE,
                    vec![(PIECE + 64, PIECE + 192)],
                    vec![(16, 48, 8), (2 * PIECE + 32, 2 * PIECE + 96, 6)],
                ),
                (2 * PIECE, vec![], vec![]),
            ];

            let mut titles: Vec<Vec<u8>> = Vec::new();
            let mut sides: Vec<Vec<u8>> = Vec::new();
            let mut shadows: Vec<std::sync::Arc<Shadow>> = Vec::new();
            for (s_idx, (len, dirty, sent)) in specs.iter().enumerate() {
                let base: Vec<u8> = (0..*len)
                    .map(|i| (((s_idx + 1) * 31 + i * 13 + (i >> 8)) & 0xff) as u8)
                    .collect();
                let mut title = base.clone();
                let mut side = base;
                // Simulate newer title CPU writes already staged into `side` for dirty/later-sent ranges.
                for &(a, b) in dirty {
                    for off in a..b {
                        title[off] ^= 0x55;
                        side[off] = title[off];
                    }
                }
                for &(a, b, send_no) in sent {
                    if send_no > 6 {
                        for off in a..b {
                            title[off] ^= 0x33;
                            side[off] = title[off];
                        }
                    }
                }
                let shadow = Shadow::new_card_for_test(
                    side.as_mut_ptr(),
                    title.as_ptr() as u64,
                    *len as u64,
                    dirty.clone(),
                    sent.clone(),
                )?;
                titles.push(title);
                sides.push(side);
                shadows.push(shadow);
            }

            // Build fetched GPU buffers (owned here until `relay_merge_card_pieces` returns).
            let mut gpu_bufs: Vec<Vec<u8>> = Vec::new();
            let mut meta: Vec<(usize, usize, usize, u64)> = Vec::new();

            // Shadow 0: 24 disjoint pieces (even pieces modified by GPU, odd pieces unchanged except piece 3).
            for p in 0..24usize {
                let lo = p * PIECE;
                let hi = lo + PIECE;
                let mut buf = sides[0][lo..hi].to_vec();
                if p % 2 == 0 {
                    let start = 17 + (p * 19) % 512;
                    let len = 1024 + p * 37;
                    for b in &mut buf[start..start + len] {
                        *b ^= 0xA7;
                    }
                }
                if p == 3 {
                    // Card holds pre-send-11 bytes in [128..256), plus a real GPU write at [400..500).
                    for off in 128..256 {
                        buf[off] ^= 0x33;
                    }
                    for b in &mut buf[400..500] {
                        *b ^= 0xC9;
                    }
                }
                gpu_bufs.push(buf);
                meta.push((0, lo, hi, 10));
            }

            // Shadow 1:
            // - Piece 0: two overlapping plans in order (plan 1 after=7, plan 2 after=9) to verify order preservation.
            // - Piece 1: overlaps `dirty` [64..192) + GPU write at [300..420).
            // - Piece 2: sent at 6 <= after (10), GPU writes at [10..120).
            {
                let mut p0_plan1 = sides[1][0..PIECE].to_vec();
                // Pre-send-8 bytes at [16..48), GPU write from plan 1 at [100..200).
                for off in 16..48 {
                    p0_plan1[off] ^= 0x33;
                }
                for b in &mut p0_plan1[100..200] {
                    *b = 0x11;
                }
                gpu_bufs.push(p0_plan1);
                meta.push((1, 0, PIECE, 7));

                let mut p0_plan2 = sides[1][0..PIECE].to_vec();
                // Plan 2 ran after send 8 (`after=9`), and overwrote [120..240) with 0x22.
                for b in &mut p0_plan2[120..240] {
                    *b = 0x22;
                }
                gpu_bufs.push(p0_plan2);
                meta.push((1, 0, PIECE, 9));

                let mut p1 = sides[1][PIECE..2 * PIECE].to_vec();
                for off in 64..192 {
                    p1[off] ^= 0x55; // stale card bytes before dirty send
                }
                for b in &mut p1[300..420] {
                    *b ^= 0x6D;
                }
                gpu_bufs.push(p1);
                meta.push((1, PIECE, 2 * PIECE, 10));

                let mut p2 = sides[1][2 * PIECE..3 * PIECE].to_vec();
                for b in &mut p2[10..120] {
                    *b ^= 0x8E;
                }
                gpu_bufs.push(p2);
                meta.push((1, 2 * PIECE, 3 * PIECE, 10));
            }

            // Shadow 2: 1 unchanged piece + 1 sparse-diff piece.
            {
                gpu_bufs.push(sides[2][0..PIECE].to_vec());
                meta.push((2, 0, PIECE, 10));
                let mut p1 = sides[2][PIECE..2 * PIECE].to_vec();
                p1[511] ^= 0xF0;
                p1[PIECE - 7] ^= 0x0F;
                gpu_bufs.push(p1);
                meta.push((2, PIECE, 2 * PIECE, 10));
            }

            let pieces: Vec<RelayCardPiece> = meta
                .into_iter()
                .zip(gpu_bufs.iter())
                .map(|((s_idx, lo, hi, after), buf)| RelayCardPiece {
                    shadow: shadows[s_idx].clone(),
                    lo,
                    hi,
                    gpu_ptr: buf.as_ptr() as usize,
                    after,
                    track_walls: false,
                })
                .collect();

            let written = relay_merge_card_pieces(pieces, parallel);
            drop(shadows);
            Some((written, titles, sides))
        };

        let Some((seq_written, seq_titles, seq_sides)) =
            run_case(wide_merge_enabled(Some("0"), Some(16)))
        else {
            return;
        };
        let Some((par_written, par_titles, par_sides)) =
            run_case(wide_merge_enabled(None, Some(16)))
        else {
            return;
        };

        assert!(seq_written > 0, "expected non-zero merged bytes");
        assert_eq!(par_written, seq_written, "parallel written byte count must match sequential");
        assert_eq!(par_titles, seq_titles, "parallel merged title memory must match sequential");
        assert_eq!(par_sides, seq_sides, "parallel merged CPU staging must match sequential");
        // Verify plan 2's newer overlapping write won over plan 1's older write on Shadow 1 piece 0.
        assert!(par_titles[1][120..240].iter().all(|&b| b == 0x22));
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: std::collections::HashMap<usize, usize>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        match self.index.get(&key) {
            Some(&at) => {
                let r = &mut self.refreshes[at];
                (r.1, r.2) = (r.1.min(address), r.2.max(end));
            }
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Before shaders are given a range of `shadow`'s copy: its pending
    /// refresh is done, the others stay pending.
    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        let Some(at) = self.index.remove(&(std::sync::Arc::as_ptr(shadow) as usize)) else {
            return;
        };
        let (shadow, from, to) = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        Self::update(&shadow, from, to);
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
    }

    fn flush(&mut self) {
        self.index.clear();
        if self.refreshes.is_empty() {
            return;
        }
        // One write-watch asking for the copies that lie close together,
        // not one per copy. Skip when `ask_once` already covers the submission.
        let ahead = self.refreshes.len() > 1 && !crate::watch::ask_kept();
        if ahead {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = self
                .refreshes
                .iter()
                .filter(|r| !r.0.unwatched())
                .map(|r| (r.1, r.2))
                .collect();
            crate::watch::ask_ahead(&ranges);
            let us = started.elapsed().as_micros() as u64;
            PREPARING[0].fetch_add(us, Ordering::Relaxed);
            PREPARING_COPIES[PREP_WATCH_US].fetch_add(us, Ordering::Relaxed);
        }
        let started = std::time::Instant::now();
        let items: Vec<_> = self.refreshes.drain(..).collect();
        let copied = crate::shadow::refresh_batch(items, wide_refresh());
        let us = started.elapsed().as_micros() as u64;
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    if !title_dst && fast_fill() {
        // SAFETY: the destination is mapped writable memory of `len` bytes;
        // a fault on the source is caught (`fast_fill`), and `read_sparse` /
        // `read_pieces` handles any unmapped holes directly without repeating
        // a failing whole-range `ReadProcessMemory` syscall.
        if unsafe { victory_copy_try(dst, src, len) } != 0 {
            return;
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Fallbacks, 1);
        #[cfg(feature = "diagnostics")]
        let sparse = std::time::Instant::now();
        if len >= 64 << 10 {
            read_sparse(src, dst, len);
        } else {
            read_pieces(src, dst, len);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
}

// Non-overlapping vectorized copies that may fault on their source: both
// variants are pure Win64 ABI leaf functions (only volatile registers touched,
// stack untouched) so [`fast_fill`]'s vectored exception handler can redirect
// any faulting load inside `[start, end)` straight to the matching `fail` exit.
// Destination-aligned 128 B/iter AVX2 (or 64 B/iter SSE2) stores saturate both
// write-back staging RAM and write-combining ReBAR/upload VRAM far better than
// `rep movsb`.
std::arch::global_asm!(
    ".globl victory_copy_try_avx2",
    ".globl victory_copy_try_avx2_start",
    ".globl victory_copy_try_avx2_end",
    ".globl victory_copy_try_avx2_fail",
    "victory_copy_try_avx2:",
    "victory_copy_try_avx2_start:",
    "cmp r8, 128",
    "jb 2f",
    "vmovdqu ymm0, ymmword ptr [rdx]",
    "vmovdqu ymmword ptr [rcx], ymm0",
    "mov rax, rcx",
    "and rax, 31",
    "mov r9, 32",
    "sub r9, rax",
    "add rcx, r9",
    "add rdx, r9",
    "sub r8, r9",
    "cmp r8, 128",
    "jb 2f",
    ".p2align 4",
    "1:",
    "vmovdqu ymm0, ymmword ptr [rdx]",
    "vmovdqu ymm1, ymmword ptr [rdx + 32]",
    "vmovdqu ymm2, ymmword ptr [rdx + 64]",
    "vmovdqu ymm3, ymmword ptr [rdx + 96]",
    "vmovdqa ymmword ptr [rcx], ymm0",
    "vmovdqa ymmword ptr [rcx + 32], ymm1",
    "vmovdqa ymmword ptr [rcx + 64], ymm2",
    "vmovdqa ymmword ptr [rcx + 96], ymm3",
    "add rdx, 128",
    "add rcx, 128",
    "sub r8, 128",
    "cmp r8, 128",
    "jae 1b",
    "2:",
    "cmp r8, 32",
    "jb 4f",
    "3:",
    "vmovdqu ymm0, ymmword ptr [rdx]",
    "vmovdqu ymmword ptr [rcx], ymm0",
    "add rdx, 32",
    "add rcx, 32",
    "sub r8, 32",
    "cmp r8, 32",
    "jae 3b",
    "4:",
    "cmp r8, 16",
    "jb 5f",
    "vmovdqu xmm0, xmmword ptr [rdx]",
    "vmovdqu xmmword ptr [rcx], xmm0",
    "add rdx, 16",
    "add rcx, 16",
    "sub r8, 16",
    "5:",
    "cmp r8, 8",
    "jb 6f",
    "mov rax, qword ptr [rdx]",
    "mov qword ptr [rcx], rax",
    "add rdx, 8",
    "add rcx, 8",
    "sub r8, 8",
    "6:",
    "cmp r8, 4",
    "jb 7f",
    "mov eax, dword ptr [rdx]",
    "mov dword ptr [rcx], eax",
    "add rdx, 4",
    "add rcx, 4",
    "sub r8, 4",
    "7:",
    "test r8, r8",
    "jz 9f",
    "8:",
    "mov al, byte ptr [rdx]",
    "mov byte ptr [rcx], al",
    "inc rdx",
    "inc rcx",
    "dec r8",
    "jnz 8b",
    "9:",
    "victory_copy_try_avx2_end:",
    "vzeroupper",
    "mov eax, 1",
    "ret",
    "victory_copy_try_avx2_fail:",
    "vzeroupper",
    "xor eax, eax",
    "ret",

    ".globl victory_copy_try_sse2",
    ".globl victory_copy_try_sse2_start",
    ".globl victory_copy_try_sse2_end",
    ".globl victory_copy_try_sse2_fail",
    "victory_copy_try_sse2:",
    "victory_copy_try_sse2_start:",
    "cmp r8, 64",
    "jb 12f",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmmword ptr [rcx], xmm0",
    "mov rax, rcx",
    "and rax, 15",
    "mov r9, 16",
    "sub r9, rax",
    "add rcx, r9",
    "add rdx, r9",
    "sub r8, r9",
    "cmp r8, 64",
    "jb 12f",
    ".p2align 4",
    "11:",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqu xmm2, xmmword ptr [rdx + 32]",
    "movdqu xmm3, xmmword ptr [rdx + 48]",
    "movdqa xmmword ptr [rcx], xmm0",
    "movdqa xmmword ptr [rcx + 16], xmm1",
    "movdqa xmmword ptr [rcx + 32], xmm2",
    "movdqa xmmword ptr [rcx + 48], xmm3",
    "add rdx, 64",
    "add rcx, 64",
    "sub r8, 64",
    "cmp r8, 64",
    "jae 11b",
    "12:",
    "cmp r8, 16",
    "jb 14f",
    "13:",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmmword ptr [rcx], xmm0",
    "add rdx, 16",
    "add rcx, 16",
    "sub r8, 16",
    "cmp r8, 16",
    "jae 13b",
    "14:",
    "cmp r8, 8",
    "jb 15f",
    "mov rax, qword ptr [rdx]",
    "mov qword ptr [rcx], rax",
    "add rdx, 8",
    "add rcx, 8",
    "sub r8, 8",
    "15:",
    "cmp r8, 4",
    "jb 16f",
    "mov eax, dword ptr [rdx]",
    "mov dword ptr [rcx], eax",
    "add rdx, 4",
    "add rcx, 4",
    "sub r8, 4",
    "16:",
    "test r8, r8",
    "jz 18f",
    "17:",
    "mov al, byte ptr [rdx]",
    "mov byte ptr [rcx], al",
    "inc rdx",
    "inc rcx",
    "dec r8",
    "jnz 17b",
    "18:",
    "victory_copy_try_sse2_end:",
    "mov eax, 1",
    "ret",
    "victory_copy_try_sse2_fail:",
    "xor eax, eax",
    "ret",
);

unsafe extern "C" {
    fn victory_copy_try_avx2(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_copy_try_avx2_start: u8;
    static victory_copy_try_avx2_end: u8;
    static victory_copy_try_avx2_fail: u8;

    fn victory_copy_try_sse2(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_copy_try_sse2_start: u8;
    static victory_copy_try_sse2_end: u8;
    static victory_copy_try_sse2_fail: u8;
}

/// Copies `len` non-overlapping bytes from `src` to `dst`; returns `0` if a
/// source page faulted (some bytes may have been written). Only valid with
/// [`fast_fill`]'s vectored handler installed.
#[inline]
unsafe fn victory_copy_try(dst: *mut u8, src: *const u8, len: usize) -> u32 {
    static USE_AVX2: OnceLock<bool> = OnceLock::new();
    if *USE_AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2")) {
        // SAFETY: AVX2 availability verified once above.
        unsafe { victory_copy_try_avx2(dst, src, len) }
    } else {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { victory_copy_try_sse2(dst, src, len) }
    }
}

/// `VICTORY_FAST_COPY=1` (experiment): what is read from title memory into
/// host memory (fills of upload memory, refreshes of host copies) is copied
/// in this process, not by the kernel (`ReadProcessMemory`,
/// a system call for each of about 14,000 copies of a gameplay frame: 24
/// ms of the submitting thread, run 12). A page of the source that is not
/// there faults; a handler sends the copy to its failure exit and the
/// kernel's copy is used as before.
fn fast_fill() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_FAST_COPY").is_some_and(|v| v == "1");
        if on {
            install_fast_copy_handler();
        }
        on
    })
}

fn install_fast_copy_handler() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        unsafe {
            let r = &*(*info).ExceptionRecord;
            if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let context = &mut *(*info).ContextRecord;
            let (avx_s, avx_e) = (
                &raw const victory_copy_try_avx2_start as u64,
                &raw const victory_copy_try_avx2_end as u64,
            );
            if context.Rip >= avx_s && context.Rip < avx_e {
                context.Rip = &raw const victory_copy_try_avx2_fail as u64;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            let (sse_s, sse_e) = (
                &raw const victory_copy_try_sse2_start as u64,
                &raw const victory_copy_try_sse2_end as u64,
            );
            if context.Rip >= sse_s && context.Rip < sse_e {
                context.Rip = &raw const victory_copy_try_sse2_fail as u64;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        // SAFETY: the handler only inspects RIP within our leaf copy routines
        // and redirects a faulting source load to the matching failure exit.
        unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
    });
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let fast = fast_fill();
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if fast
            && (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0
            // SAFETY: destination is writable host memory; any concurrent
            // unmap on `at` faults into `on_fault` and falls through below.
            && unsafe { victory_copy_try(to, at, n) } != 0
        {
            // Fast userspace copy succeeded for this committed sub-region.
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable. When `VICTORY_FAST_COPY=1`, copies in-process with
/// [`victory_copy_try`] and falls back to the kernel read only on a fault,
/// preserving unmapped-page safety without a syscall per mapped page.
pub(crate) fn read_into_fast(src: *const u8, dst: &mut [u8]) -> bool {
    if dst.is_empty() {
        return true;
    }
    if fast_fill() {
        // SAFETY: `dst` is mapped writable memory of `dst.len()` bytes; a
        // fault on `src` is caught by `fast_fill`'s vectored handler and
        // returns 0.
        if unsafe { victory_copy_try(dst.as_mut_ptr(), src, dst.len()) } != 0 {
            return true;
        }
    }
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Title buffer copies recorded since the start, on any list: after one,
/// ranges brought up to date before it are due again (`draw`'s memo of
/// tables takes it for a change).
pub(crate) static TITLE_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `VICTORY_BARE=<bits>` (experiment): parts of the work left out, to find
/// what a frame's time is made of from the bottom up. The picture is wrong
/// or empty by design. 1: no mesh draws. 2: no compute dispatches. 4: no
/// refreshes of host copies from title memory. 8: nothing shaders write
/// is looked for or copied back. 16: no other draws. 32: no texture
/// uploads. 64: no fills of upload memory, no title buffer copies.
pub(crate) fn bare() -> u32 {
    static BITS: OnceLock<u32> = OnceLock::new();
    *BITS.get_or_init(|| {
        let bits = std::env::var("VICTORY_BARE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        if bits != 0 {
            tracing::warn!(bits, "VICTORY_BARE: parts of the work are left out; the picture is wrong by design");
        }
        bits
    })
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// The 4 KiB pages of title memory that shaders' writes were put back into
/// since the start (`VICTORY_INDIRECT_CPU`: an indirect record in such a
/// page is the card's to read, not the CPU's).
static SHADER_PAGES: std::sync::RwLock<Option<std::collections::HashSet<usize>>> = std::sync::RwLock::new(None);

fn shader_wrote(at: usize, len: usize) {
    if len == 0 || !crate::draw::indirect_cpu() {
        return;
    }
    let pages = at >> 12..=(at + len - 1) >> 12;
    {
        let known = SHADER_PAGES.read().unwrap_or_else(|e| e.into_inner());
        if known.as_ref().is_some_and(|k| pages.clone().all(|p| k.contains(&p))) {
            return;
        }
    }
    let mut known = SHADER_PAGES.write().unwrap_or_else(|e| e.into_inner());
    known.get_or_insert_with(Default::default).extend(pages);
}

/// Whether shaders' writes were ever put back into a page of `[at, at + len)`.
pub(crate) fn shader_written(at: usize, len: usize) -> bool {
    let known = SHADER_PAGES.read().unwrap_or_else(|e| e.into_inner());
    known.as_ref().is_some_and(|k| (at >> 12..=(at + len.max(1) - 1) >> 12).any(|p| k.contains(&p)))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    shader_wrote(dst as usize, len);
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

/// Queries the adapter's current local VRAM budget in bytes via DXGI.
fn query_local_vram_budget(adapter: &IDXGIAdapter3) -> Option<u64> {
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: valid adapter and output struct.
    unsafe {
        adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
            .ok()?;
    }
    (info.Budget > 0).then_some(info.Budget)
}

/// Decides `(rebar_scratch_enabled, rebar_shadow_budget_bytes)` for the active
/// adapter.
///
/// - Disabled when `allow_rebar` is false (`VICTORY_REBAR=0` in diagnostics
///   builds), when the adapter/driver does not support `D3D12_HEAP_TYPE_GPU_UPLOAD`,
///   or on UMA/APU adapters (where `L0` write-back `shared_buffer` already lives
///   in unified local memory without write-combining read penalties).
/// - On discrete GPUs with ReBAR support, scales the shadow VRAM budget by local
///   VRAM so low-VRAM cards (`<= 8 GiB`) stay safely on system-RAM shadows while
///   12–32+ GiB cards keep hot buffers resident in VRAM.
fn rebar_policy(
    gpu_upload_supported: bool,
    uma: bool,
    local_vram_budget: Option<u64>,
    allow_rebar: bool,
) -> (bool, u64) {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if !allow_rebar || !gpu_upload_supported || uma {
        return (false, 0);
    }
    let Some(budget) = local_vram_budget else {
        return (false, 0);
    };
    if budget <= 8 * GIB {
        (false, 0)
    } else if budget <= 14 * GIB {
        (true, 512 * MIB)
    } else if budget < 20 * GIB {
        (true, GIB)
    } else {
        (true, (budget / 16).clamp(GIB, 2 * GIB))
    }
}

/// Merges overlapping or adjacent half-open byte intervals `[from, to)` into
/// sorted disjoint intervals.
fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.retain(|&(from, to)| from < to);
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    merged
}

/// Converts sorted disjoint buffer-offset byte intervals `[from, to)` into
/// sorted, disjoint, non-adjacent 64 KiB piece runs `(first_piece, count)` so
/// each overlapping piece is hashed and compared at most once per submission.
fn spans_to_piece_runs(spans: &[(usize, usize)]) -> Vec<(u32, u32)> {
    use crate::watch::PAGE;
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &(from, to) in spans {
        if from >= to {
            continue;
        }
        let Ok(first) = u32::try_from(from / PAGE) else {
            continue;
        };
        let Ok(end) = u32::try_from(to.div_ceil(PAGE)) else {
            continue;
        };
        if first >= end {
            continue;
        }
        match runs.last_mut() {
            Some(last) if first <= last.0 + last.1 => {
                let new_end = (last.0 + last.1).max(end);
                last.1 = new_end - last.0;
            }
            _ => runs.push((first, end - first)),
        }
    }
    runs
}

/// Walks `piece_runs` paired with their per-piece GPU comparison `flags`,
/// coalescing contiguous flagged (`!= 0`) 64 KiB pieces into sorted disjoint
/// buffer-offset byte intervals clamped to `len`. Returns `(byte_runs, changed_pieces)`.
fn flagged_piece_byte_runs(
    piece_runs: &[(u32, u32)],
    flags: &[u32],
    len: usize,
) -> (Vec<(usize, usize)>, u64) {
    use crate::watch::PAGE;
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut cur: Option<(u32, u32)> = None;
    let mut idx = 0usize;
    let mut changed = 0u64;
    for &(first, count) in piece_runs {
        for off in 0..count {
            let flagged = flags.get(idx).copied().unwrap_or(1) != 0;
            idx += 1;
            if flagged {
                changed += 1;
                let piece = first + off;
                match cur {
                    Some((s, e)) if e == piece => cur = Some((s, piece + 1)),
                    Some((s, e)) => {
                        let lo = (s as usize) * PAGE;
                        let hi = ((e as usize) * PAGE).min(len);
                        if lo < hi {
                            out.push((lo, hi));
                        }
                        cur = Some((piece, piece + 1));
                    }
                    None => cur = Some((piece, piece + 1)),
                }
            } else if let Some((s, e)) = cur.take() {
                let lo = (s as usize) * PAGE;
                let hi = ((e as usize) * PAGE).min(len);
                if lo < hi {
                    out.push((lo, hi));
                }
            }
        }
    }
    if let Some((s, e)) = cur {
        let lo = (s as usize) * PAGE;
        let hi = ((e as usize) * PAGE).min(len);
        if lo < hi {
            out.push((lo, hi));
        }
    }
    (out, changed)
}

/// Intersects two sorted disjoint half-open interval lists `a` and `b`.
fn intersect_ranges(a: &[(usize, usize)], b: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo < hi {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

impl Host {
    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// Copies back GPU-written ranges of any ReBAR shadows in `back` via a
    /// batched `READBACK` DMA so write-combining VRAM is never read by the CPU.
    fn rebar_back(&self, submitted: &mut u64, back: &[Declared], bitmaps: &[&[u32]]) {
        struct RebarMerged {
            shadow: std::sync::Arc<crate::shadow::Shadow>,
            spans: Vec<(usize, usize)>,
            first: usize,
            end: usize,
        }
        let mut merged: Vec<RebarMerged> = Vec::new();
        let mut total_jobs = 0usize;
        for (shadow, declared, unmarked) in back {
            if !shadow.is_rebar() {
                continue;
            }
            let title_spans = if !bitmaps.is_empty() && shadow.marked() {
                let (filtered, decl_pages, marked_pages) =
                    shadow.filter_marked_spans(declared, unmarked, bitmaps);
                crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                filtered
            } else {
                declared.clone()
            };
            let ranges: Vec<(usize, usize)> = title_spans
                .iter()
                .map(|&(from, to)| shadow.inside(from, to.saturating_sub(from)))
                .filter(|&(from, to)| from < to)
                .collect();
            let spans = merge_ranges(ranges);
            let first = total_jobs;
            total_jobs += spans.len();
            merged.push(RebarMerged {
                shadow: shadow.clone(),
                spans,
                first,
                end: total_jobs,
            });
        }
        if merged.is_empty() {
            return;
        }
        if total_jobs == 0 {
            for entry in &merged {
                entry.shadow.clear_pending_rebar();
            }
            return;
        }
        let mut copy_jobs: Vec<(&ID3D12Resource, u64, u64)> = Vec::with_capacity(total_jobs);
        for entry in &merged {
            for &(from, to) in &entry.spans {
                copy_jobs.push((&entry.shadow.buffer, from as u64, (to - from) as u64));
            }
        }
        let ok = self.copy_rebar_readback(submitted, &copy_jobs, |slices| {
            for entry in &merged {
                let written = entry
                    .shadow
                    .copy_back_rebar(&entry.spans, &slices[entry.first..entry.end]);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        });
        if !ok {
            for entry in &merged {
                entry.shadow.clear_pending_rebar();
            }
        }
    }

    /// A mapped readback buffer of `size` bytes (`D3D12_HEAP_TYPE_READBACK`).
    fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let buffer = self.read_back_buffer(size)?;
        let mut data = std::ptr::null_mut();
        // SAFETY: freshly created readback buffer; persistent map for readback.
        unsafe { buffer.Map(0, None, Some(&mut data)).ok()? };
        Some((buffer, data.cast()))
    }

    /// Fallback fence wait when event-based waiting fails.
    fn poll_fence(&self, target: u64) {
        // SAFETY: valid fence handle.
        while unsafe { self.fence.GetCompletedValue() } < target {
            if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                break;
            }
            std::thread::yield_now();
        }
    }

    /// Whether allocating `extra_bytes` of local VRAM stays under 85% of the
    /// adapter's current local VRAM budget (leaves headroom for textures and
    /// render targets).
    fn has_vram_headroom(&self, extra_bytes: u64) -> bool {
        let Some([budget, used, ..]) = self.video_memory() else {
            return true;
        };
        if budget == 0 {
            return false;
        }
        let limit = (budget / 100).saturating_mul(85);
        used.saturating_add(extra_bytes) <= limit
    }

    /// Byte budget reserved for ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) shadow
    /// buffers on this adapter (0 when ReBAR shadows are disabled).
    pub(crate) fn rebar_shadow_budget(&self) -> u64 {
        self.rebar_shadow_budget
    }

    /// A buffer in `D3D12_HEAP_TYPE_GPU_UPLOAD` (CPU-writable local VRAM via
    /// Resizable BAR), starting in `COMMON` and mapped for sequential CPU
    /// writes. Callers MUST NOT read from the returned pointer on the CPU.
    fn gpu_upload_buffer(
        &self,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-writable VRAM buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        error = %e,
                        size,
                        "GPU upload heap allocation failed; falling back to system memory"
                    );
                }
                None
            }
        }
    }

    /// A scratch buffer for write-only per-draw host data: placed in
    /// `D3D12_HEAP_TYPE_GPU_UPLOAD` when ReBAR is enabled and VRAM has
    /// headroom, otherwise falls back to standard `D3D12_HEAP_TYPE_UPLOAD`.
    pub(crate) fn scratch_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if self.rebar_scratch
            && self.has_vram_headroom(size)
            && let Some(buf) = self.gpu_upload_buffer(size, D3D12_RESOURCE_FLAG_NONE)
        {
            return Some(buf);
        }
        self.upload_buffer(size)
    }

    /// Attempts to allocate a UAV-capable ReBAR shadow buffer in local VRAM.
    /// Returns `None` if ReBAR shadows are disabled, VRAM headroom is low, or
    /// allocation fails (callers fall back to [`Host::shared_buffer`]).
    pub(crate) fn try_rebar_shadow_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if self.rebar_shadow_budget == 0 || !self.has_vram_headroom(size) {
            return None;
        }
        self.gpu_upload_buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    }
}

/// One unique ReBAR shadow with declared `GpuWrite` ranges in a submission.
struct RebarWriteEntry {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    declared_title: Vec<(usize, usize)>,
    unmarked_title: Vec<(usize, usize)>,
    declared_spans: Vec<(usize, usize)>,
    piece_runs: Vec<(u32, u32)>,
    /// Starting index in the GPU hash-compare `flags` array if GPU hashing ran
    /// for this entry; `None` falls back to reading back all `declared_spans`.
    compare_out_base: Option<usize>,
    compare_pieces: usize,
    /// Base bit in `SLOT_VIEW_WRITES` when all writes to this shadow in this
    /// submission were shader-marked (`None` = no view-mark filter).
    mark_base: Option<u32>,
}

/// Optional pre-submission rehash and post-submission hash-compare command
/// lists recorded for ReBAR UAV shadows in one submission.
#[derive(Default)]
struct RebarHashPlan {
    pre_list: Option<ID3D12CommandList>,
    post_list: Option<ID3D12CommandList>,
    total_flag_pieces: usize,
}

impl Host {
    /// Records a tiny direct command list that copies the active prefix of the
    /// GPU shader view-writes bitmap (`SLOT_VIEW_WRITES`) into a pooled
    /// `READBACK` slot and zeroes the GPU bitmap back to 0 on the GPU timeline
    /// before the next submission runs.
    fn record_view_marks_capture(&self) -> Option<(ID3D12CommandList, ViewMarksCapture)> {
        if !crate::shadow::view_marks_on() {
            return None;
        }
        self.record_view_marks_capture_inner()
    }

    fn record_view_marks_capture_inner(&self) -> Option<(ID3D12CommandList, ViewMarksCapture)> {
        self.record_view_marks_capture_words(0)
    }

    fn record_view_marks_capture_words(
        &self,
        min_words: usize,
    ) -> Option<(ID3D12CommandList, ViewMarksCapture)> {
        let (uav, _gpu, zero) = crate::shadow::view_writes_resources()?;
        let bitmap_bytes = u64::from(victory_shader::emit::VIEW_WRITE_BITS / 8);
        let used_words = (crate::shadow::marks_end() as usize)
            .div_ceil(32)
            .max(min_words)
            .min((victory_shader::emit::VIEW_WRITE_BITS / 32) as usize);
        let copy_bytes = ((used_words as u64) * 4).max(4);
        let (list, slot) = {
            let mut pool = VIEW_MARKS_POOL.lock().unwrap_or_else(|e| e.into_inner());
            (pool.lists.pop(), pool.slots.pop())
        };
        let list = list.or_else(|| self.make_prelude())?;
        let slot = match slot {
            Some(s) if s.2 >= bitmap_bytes => s,
            _ => {
                let (rb, ptr) = self.readback_buffer(bitmap_bytes)?;
                name(&rb, || "shader view-writes readback".into());
                (rb, ptr as usize, bitmap_bytes)
            }
        };
        let (allocator, cmd) = &list;
        // SAFETY: `list` is recycled only after its previous submission finished;
        // `uav` enters and exits in `UNORDERED_ACCESS`; `zero` is `UPLOAD`
        // (`GENERIC_READ`); `slot.0` is `READBACK` (`COPY_DEST`).
        let recorded: windows::core::Result<ID3D12CommandList> = unsafe {
            (|| {
                allocator.Reset()?;
                cmd.Reset(allocator, None)?;
                cmd.ResourceBarrier(&[
                    crate::indirect::uav_barrier(),
                    transition(
                        uav,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ]);
                cmd.CopyBufferRegion(&slot.0, 0, uav, 0, copy_bytes);
                cmd.ResourceBarrier(&[transition(
                    uav,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                cmd.CopyBufferRegion(uav, 0, zero, 0, copy_bytes);
                if copy_bytes < bitmap_bytes {
                    cmd.CopyBufferRegion(uav, bitmap_bytes - 4, zero, bitmap_bytes - 4, 4);
                }
                cmd.ResourceBarrier(&[transition(
                    uav,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                cmd.Close()?;
                cmd.cast()
            })()
        };
        match recorded {
            Ok(cl) => {
                crate::shadow::record_view_marks_capture();
                Some((
                    cl,
                    ViewMarksCapture {
                        list,
                        slot,
                        words: used_words,
                    },
                ))
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not record shader view-writes capture");
                let mut pool = VIEW_MARKS_POOL.lock().unwrap_or_else(|e| e.into_inner());
                pool.lists.push(list);
                pool.slots.push(slot);
                None
            }
        }
    }

    /// Ensures a reusable closed direct command list exists in `slot`.
    fn ensure_direct_list<'a>(
        &self,
        slot: &'a mut Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    ) -> windows::core::Result<&'a (ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        if slot.is_none() {
            // SAFETY: valid device creating a direct allocator and closed list.
            let pair = unsafe {
                let allocator: ID3D12CommandAllocator = self
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None,
                )?;
                list.Close()?;
                (allocator, list)
            };
            *slot = Some(pair);
        }
        Ok(slot.as_ref().unwrap())
    }

    /// Submits a single closed helper command list on `self.queue` and waits
    /// on `self.fence` (`submitted` must be held by the caller).
    fn submit_helper_list(&self, submitted: &mut u64, cmd: ID3D12CommandList, label: &str) -> bool {
        *submitted += 1;
        // SAFETY: `cmd` is closed; `fence` and `event` are owned by `self` and
        // serialized under `submitted`.
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            match self.queue.Signal(&self.fence, *submitted) {
                Ok(()) => match self.fence.SetEventOnCompletion(*submitted, self.event) {
                    Ok(()) => {
                        let waited = WaitForSingleObject(self.event, INFINITE);
                        if waited != WAIT_OBJECT_0 {
                            self.poll_fence(*submitted);
                        }
                    }
                    Err(_) => self.poll_fence(*submitted),
                },
                Err(e) => {
                    tracing::warn!(error = %e, "{label} fence signal failed");
                    return false;
                }
            }
        }
        true
    }

    /// Builds optional pre-submission hash-update and post-submission
    /// hash-compare command lists for UAV-written ReBAR shadows so copy-back
    /// only DMAs 64 KiB pieces whose GPU hash actually changed. Falls back
    /// cleanly to full-range readbacks if the compute pipeline or buffers are
    /// unavailable.
    fn prepare_rebar_hash_plan(&self, entries: &mut [RebarWriteEntry]) -> RebarHashPlan {
        let Some(pipe) = rebar_hash_pipeline(&self.device) else {
            return RebarHashPlan::default();
        };
        struct PreparedHash {
            entry_idx: usize,
            hashes_gpu: u64,
            aligned_len: u32,
            pre_runs: Vec<(u32, u32)>,
            out_base: usize,
            pieces: usize,
        }
        let mut prepared: Vec<PreparedHash> = Vec::with_capacity(entries.len());
        let mut total_flag_pieces = 0usize;
        for (entry_idx, entry) in entries.iter().enumerate() {
            let entry_pieces: usize = entry.piece_runs.iter().map(|&(_, c)| c as usize).sum();
            if entry_pieces == 0 {
                continue;
            }
            let Some(crate::shadow::PreparedRebarHashes {
                hashes_gpu,
                aligned_len,
                pre_runs,
            }) = entry.shadow.prepare_rebar_hashes(|n| {
                self.default_buffer_in_state(n, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            })
            else {
                continue;
            };
            let Some(next_total) = total_flag_pieces.checked_add(entry_pieces) else {
                entry.shadow.mark_rebar_dirty_runs(&pre_runs);
                continue;
            };
            if (next_total as u64).saturating_mul(4) > REBAR_FLAGS_MAX_BYTES {
                entry.shadow.mark_rebar_dirty_runs(&pre_runs);
                continue;
            }
            prepared.push(PreparedHash {
                entry_idx,
                hashes_gpu,
                aligned_len,
                pre_runs,
                out_base: total_flag_pieces,
                pieces: entry_pieces,
            });
            total_flag_pieces = next_total;
        }
        if prepared.is_empty() {
            return RebarHashPlan::default();
        }
        let restore_dirty = |entries: &[RebarWriteEntry], prepared: &[PreparedHash]| {
            for p in prepared {
                if !p.pre_runs.is_empty() {
                    entries[p.entry_idx]
                        .shadow
                        .mark_rebar_dirty_runs(&p.pre_runs);
                }
            }
        };
        let mut rb = self
            .rebar_readback
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let need_bytes = ((total_flag_pieces as u64) * 4).max(4);
        let need_flags_alloc = rb
            .flags_uav
            .as_ref()
            .is_none_or(|(_, _, c)| *c < need_bytes)
            || rb
                .flags_readback
                .as_ref()
                .is_none_or(|(_, _, c)| *c < need_bytes);
        if need_flags_alloc {
            let cap = need_bytes
                .next_power_of_two()
                .clamp(REBAR_FLAGS_MIN_BYTES, REBAR_FLAGS_MAX_BYTES);
            let Some(uav) =
                self.default_buffer_in_state(cap, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            else {
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            };
            let Some((rb_buf, rb_ptr)) = self.readback_buffer(cap) else {
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            };
            name(&uav, || format!("ReBAR hash flags UAV {cap:#x}"));
            name(&rb_buf, || format!("ReBAR hash flags readback {cap:#x}"));
            // SAFETY: live D3D12 buffer.
            let uav_gpu = unsafe { uav.GetGPUVirtualAddress() };
            rb.flags_uav = Some((uav, uav_gpu, cap));
            rb.flags_readback = Some((rb_buf, rb_ptr as usize, cap));
        }
        let (flags_uav_res, flags_uav_gpu, _) = rb.flags_uav.as_ref().unwrap().clone();
        let flags_rb_res = rb.flags_readback.as_ref().unwrap().0.clone();

        // Record pre-submission hash pass for shadows that just created `hashes`
        // or received CPU writes (`Shadow::copy` / `copy_back_rebar`) since last hash.
        let pre_list = if prepared.iter().any(|p| !p.pre_runs.is_empty()) {
            let Ok((allocator, list)) = self.ensure_direct_list(&mut rb.pre_hash_list) else {
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            };
            let to_srv: Vec<D3D12_RESOURCE_BARRIER> = prepared
                .iter()
                .filter(|p| !p.pre_runs.is_empty())
                .map(|p| {
                    transition(
                        &entries[p.entry_idx].shadow.buffer,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )
                })
                .collect();
            let to_common: Vec<D3D12_RESOURCE_BARRIER> = prepared
                .iter()
                .filter(|p| !p.pre_runs.is_empty())
                .map(|p| {
                    transition(
                        &entries[p.entry_idx].shadow.buffer,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )
                })
                .collect();
            // SAFETY: `submitted` is held exclusively so no prior pre-hash list
            // is in flight; all shadow buffers enter in COMMON and return to
            // COMMON; `hashes` and `flags_uav` live in UNORDERED_ACCESS.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    list.ResourceBarrier(&to_srv);
                    list.SetComputeRootSignature(&pipe.root);
                    list.SetPipelineState(&pipe.pso);
                    list.SetComputeRootUnorderedAccessView(3, flags_uav_gpu);
                    list.SetComputeRootUnorderedAccessView(4, flags_uav_gpu);
                    for p in prepared.iter().filter(|p| !p.pre_runs.is_empty()) {
                        list.SetComputeRootShaderResourceView(
                            1,
                            entries[p.entry_idx].shadow.base_gpu(),
                        );
                        list.SetComputeRootUnorderedAccessView(2, p.hashes_gpu);
                        for &(first, count) in &p.pre_runs {
                            let constants = [
                                first,
                                p.aligned_len,
                                0u32,
                                0u32,
                                count,
                                victory_shader::emit::VIEW_MARK_NONE,
                                0u32,
                            ];
                            list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                            list.Dispatch(count, 1, 1);
                        }
                    }
                    list.ResourceBarrier(&to_common);
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(cmd) => Some(cmd),
                Err(e) => {
                    tracing::warn!(error = %e, "could not record ReBAR pre-hash list");
                    restore_dirty(entries, &prepared);
                    return RebarHashPlan::default();
                }
            }
        } else {
            None
        };

        // Record post-submission hash-compare pass over all declared piece runs,
        // copying the tiny `flags` dword array to `flags_readback`.
        let Ok((allocator, list)) = self.ensure_direct_list(&mut rb.post_compare_list) else {
            restore_dirty(entries, &prepared);
            return RebarHashPlan::default();
        };
        let mut before_barriers = Vec::with_capacity(prepared.len() + 1);
        before_barriers.push(crate::indirect::uav_barrier());
        for p in &prepared {
            before_barriers.push(transition(
                &entries[p.entry_idx].shadow.buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            ));
        }
        let mut after_barriers = Vec::with_capacity(prepared.len() + 1);
        after_barriers.push(transition(
            &flags_uav_res,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        ));
        for p in &prepared {
            after_barriers.push(transition(
                &entries[p.entry_idx].shadow.buffer,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            ));
        }
        // SAFETY: `submitted` is held exclusively so no prior post-compare list
        // is in flight; `HostList::close` returned every shadow buffer to
        // COMMON; `flags_uav` transitions UAV -> COPY_SOURCE -> UAV and
        // `flags_rb_res` stays permanently in COPY_DEST covering `total_flag_pieces * 4`.
        let recorded = unsafe {
            (|| -> windows::core::Result<ID3D12CommandList> {
                allocator.Reset()?;
                list.Reset(allocator, None)?;
                list.ResourceBarrier(&before_barriers);
                let view_marks_gpu = crate::shadow::view_marks_on()
                    .then(crate::shadow::view_writes)
                    .flatten();
                list.SetComputeRootSignature(&pipe.root);
                list.SetPipelineState(&pipe.pso);
                list.SetComputeRootUnorderedAccessView(3, flags_uav_gpu);
                list.SetComputeRootUnorderedAccessView(
                    4,
                    view_marks_gpu.unwrap_or(flags_uav_gpu),
                );
                for p in &prepared {
                    let entry = &entries[p.entry_idx];
                    list.SetComputeRootShaderResourceView(1, entry.shadow.base_gpu());
                    list.SetComputeRootUnorderedAccessView(2, p.hashes_gpu);
                    let mark_base = if view_marks_gpu.is_some() {
                        entry.mark_base.unwrap_or(victory_shader::emit::VIEW_MARK_NONE)
                    } else {
                        victory_shader::emit::VIEW_MARK_NONE
                    };
                    let mark_skew = u32::from((entry.shadow.span().0 & 0xFFFF) != 0);
                    let mut out_idx = p.out_base;
                    for &(first, count) in &entry.piece_runs {
                        let constants = [
                            first,
                            p.aligned_len,
                            (out_idx * 4) as u32,
                            1u32,
                            count,
                            mark_base,
                            mark_skew,
                        ];
                        list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                        list.Dispatch(count, 1, 1);
                        out_idx += count as usize;
                    }
                }
                list.ResourceBarrier(&after_barriers);
                list.CopyBufferRegion(
                    &flags_rb_res,
                    0,
                    &flags_uav_res,
                    0,
                    (total_flag_pieces as u64) * 4,
                );
                list.ResourceBarrier(&[transition(
                    &flags_uav_res,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                list.Close()?;
                list.cast()
            })()
        };
        let post_list = match recorded {
            Ok(cmd) => Some(cmd),
            Err(e) => {
                tracing::warn!(error = %e, "could not record ReBAR post-compare list");
                restore_dirty(entries, &prepared);
                return RebarHashPlan::default();
            }
        };
        drop(rb);
        for p in prepared {
            entries[p.entry_idx].compare_out_base = Some(p.out_base);
            entries[p.entry_idx].compare_pieces = p.pieces;
        }
        RebarHashPlan {
            pre_list,
            post_list,
            total_flag_pieces,
        }
    }

    /// Reads GPU hash-compare flags (if run), coalesces contiguous flagged
    /// 64 KiB pieces intersected with declared UAV spans, and DMAs only those
    /// changed runs into `READBACK` for [`crate::shadow::Shadow::copy_back_rebar`].
    fn finish_rebar_copy_back(
        &self,
        submitted: &mut u64,
        entries: Vec<RebarWriteEntry>,
        total_flag_pieces: usize,
        bitmaps: &[&[u32]],
    ) {
        struct PendingCopy {
            shadow: std::sync::Arc<crate::shadow::Shadow>,
            spans: Vec<(usize, usize)>,
            first: usize,
            end: usize,
        }
        let mut pending: Vec<PendingCopy> = Vec::with_capacity(entries.len());
        let mut total_jobs = 0usize;
        let mut compared_total = 0u64;
        let mut changed_total = 0u64;
        {
            let rb = self
                .rebar_readback
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let flags_all: Option<&[u32]> = if total_flag_pieces > 0 {
                rb.flags_readback.as_ref().and_then(|(_, ptr, cap)| {
                    ((total_flag_pieces as u64) * 4 <= *cap).then(|| {
                        // SAFETY: submission fence completed and `ptr` is mapped
                        // for `*cap >= total_flag_pieces * 4` bytes of `READBACK`.
                        unsafe {
                            std::slice::from_raw_parts(
                                (*ptr as *const u8).cast::<u32>(),
                                total_flag_pieces,
                            )
                        }
                    })
                })
            } else {
                None
            };
            for entry in entries {
                let spans = if let (Some(base), Some(all)) = (entry.compare_out_base, flags_all)
                    && base + entry.compare_pieces <= all.len()
                {
                    if !bitmaps.is_empty() && entry.mark_base.is_some() {
                        let (_, decl_pages, marked_pages) = entry.shadow.filter_marked_spans(
                            &entry.declared_title,
                            &entry.unmarked_title,
                            bitmaps,
                        );
                        crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                    }
                    let slice = &all[base..base + entry.compare_pieces];
                    let (flagged, changed) = flagged_piece_byte_runs(
                        &entry.piece_runs,
                        slice,
                        entry.shadow.len() as usize,
                    );
                    compared_total += entry.compare_pieces as u64;
                    changed_total += changed;
                    intersect_ranges(&flagged, &entry.declared_spans)
                } else if !bitmaps.is_empty() && entry.shadow.marked() {
                    let (filtered_title, decl_pages, marked_pages) =
                        entry.shadow.filter_marked_spans(
                            &entry.declared_title,
                            &entry.unmarked_title,
                            bitmaps,
                        );
                    crate::shadow::record_view_marks_filter(decl_pages, marked_pages);
                    let ranges: Vec<(usize, usize)> = filtered_title
                        .iter()
                        .map(|&(from, to)| entry.shadow.inside(from, to.saturating_sub(from)))
                        .filter(|&(from, to)| from < to)
                        .collect();
                    let filtered = merge_ranges(ranges);
                    entry.shadow.mark_rebar_dirty_spans(&filtered);
                    filtered
                } else {
                    // Fallback: read back the entire merged declared spans and
                    // mark them dirty in any existing hash buffer.
                    entry.shadow.mark_rebar_dirty_spans(&entry.declared_spans);
                    entry.declared_spans
                };
                let first = total_jobs;
                total_jobs += spans.len();
                pending.push(PendingCopy {
                    shadow: entry.shadow,
                    spans,
                    first,
                    end: total_jobs,
                });
            }
        }
        #[cfg(test)]
        if compared_total > 0 {
            self.rebar_hash_compared_pieces
                .fetch_add(compared_total, Ordering::Relaxed);
            self.rebar_hash_changed_pieces
                .fetch_add(changed_total, Ordering::Relaxed);
        }
        let _ = (compared_total, changed_total);
        if total_jobs == 0 {
            for entry in &pending {
                entry.shadow.clear_pending_rebar();
            }
            return;
        }
        let mut copy_jobs: Vec<(&ID3D12Resource, u64, u64)> = Vec::with_capacity(total_jobs);
        let mut copy_bytes = 0u64;
        for entry in &pending {
            for &(from, to) in &entry.spans {
                let len = (to - from) as u64;
                copy_bytes += len;
                copy_jobs.push((&entry.shadow.buffer, from as u64, len));
            }
        }
        #[cfg(test)]
        {
            self.rebar_post_readback_calls
                .fetch_add(1, Ordering::Relaxed);
            self.rebar_post_readback_bytes
                .fetch_add(copy_bytes, Ordering::Relaxed);
        }
        let _ = copy_bytes;
        let ok = self.copy_rebar_readback(submitted, &copy_jobs, |slices| {
            for entry in &pending {
                if entry.first < entry.end {
                    let written = entry
                        .shadow
                        .copy_back_rebar(&entry.spans, &slices[entry.first..entry.end]);
                    COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                } else {
                    entry.shadow.clear_pending_rebar();
                }
            }
        });
        if !ok {
            for entry in &pending {
                entry.shadow.clear_pending_rebar();
            }
        }
    }

    /// DMAs `copies` (`(src_buffer, src_offset, len)`) from ReBAR buffers into
    /// a cached system-RAM readback buffer on `self.queue`, waits for `submitted`
    /// (already locked by the caller), and invokes `f` with the resulting slices.
    fn copy_rebar_readback(
        &self,
        submitted: &mut u64,
        copies: &[(&ID3D12Resource, u64, u64)],
        f: impl FnOnce(&[&[u8]]),
    ) -> bool {
        if copies.is_empty() {
            f(&[]);
            return true;
        }
        let mut total: u64 = 0;
        let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(copies.len());
        for &(_, _, len) in copies {
            let Some(end) = total.checked_add(len) else {
                return false;
            };
            offsets.push((total as usize, len as usize));
            total = end;
        }
        if total == 0 {
            let empty: Vec<&[u8]> = vec![&[]; copies.len()];
            f(&empty);
            return true;
        }
        let mut rb = self
            .rebar_readback
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Err(e) = self.ensure_direct_list(&mut rb.list) {
            tracing::warn!(error = %e, "could not create ReBAR readback command list");
            return false;
        }
        let mut transient: Option<(ID3D12Resource, *const u8)> = None;
        let (dst_res, dst_ptr): (ID3D12Resource, *const u8) = if total > REBAR_READBACK_KEEP_CAP {
            let Some((buf, ptr)) = self.readback_buffer(total) else {
                return false;
            };
            transient = Some((buf.clone(), ptr));
            (buf, ptr)
        } else {
            let need_new = rb.buffer.as_ref().is_none_or(|(_, _, cap)| *cap < total);
            if need_new {
                let cap = total
                    .next_power_of_two()
                    .clamp(1 << 20, REBAR_READBACK_KEEP_CAP);
                let Some((buf, ptr)) = self.readback_buffer(cap) else {
                    return false;
                };
                name(&buf, || format!("ReBAR readback {cap:#x}"));
                rb.buffer = Some((buf, ptr as usize, cap));
            }
            let (buf, ptr, _) = rb.buffer.as_ref().unwrap();
            (buf.clone(), *ptr as *const u8)
        };
        let (allocator, list) = rb.list.as_ref().unwrap();
        // SAFETY: `submitted` is held exclusively so no previous readback is in
        // flight; all source buffers are in COMMON state (implicitly promoting
        // to COPY_SOURCE and decaying back to COMMON on completion), and
        // `dst_res` is a READBACK buffer in COPY_DEST state covering `total`.
        let recorded = unsafe {
            (|| -> windows::core::Result<()> {
                allocator.Reset()?;
                list.Reset(allocator, None)?;
                for (&(src, src_off, len), &(dst_off, _)) in copies.iter().zip(&offsets) {
                    if len > 0 {
                        list.CopyBufferRegion(&dst_res, dst_off as u64, src, src_off, len);
                    }
                }
                list.Close()?;
                Ok(())
            })()
        };
        if let Err(e) = recorded {
            tracing::warn!(error = %e, "could not record ReBAR readback list");
            drop(transient);
            return false;
        }
        let Ok(cmd): windows::core::Result<ID3D12CommandList> = list.cast() else {
            drop(transient);
            return false;
        };
        if !self.submit_helper_list(submitted, cmd, "ReBAR readback") {
            drop(transient);
            return false;
        }
        let slices: Vec<&[u8]> = offsets
            .iter()
            .map(|&(off, len)| {
                // SAFETY: GPU copy completed and `dst_ptr` covers `total` bytes.
                unsafe { std::slice::from_raw_parts(dst_ptr.add(off), len) }
            })
            .collect();
        f(&slices);
        drop(transient);
        true
    }

    /// Reads the entire ReBAR shadow buffer into a cached system-RAM `Vec<u8>`
    /// once, on its first UAV write, without ever reading write-combining memory
    /// from the CPU.
    fn readback_rebar_to_vec(
        &self,
        submitted: &mut u64,
        buffer: &ID3D12Resource,
        size: u64,
    ) -> Option<Vec<u8>> {
        if size == 0 {
            return Some(Vec::new());
        }
        let mut out = None;
        let ok = self.copy_rebar_readback(submitted, &[(buffer, 0, size)], |slices| {
            out = Some(slices[0].to_vec());
        });
        ok.then_some(out).flatten()
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        crate::frametime::tick();
        crate::cardmeter::collect();
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        crate::fpsbox::set(fps);
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [
            prep_watch_ms,
            prep_refresh_copy_ms,
            prep_fill_ms,
            prep_fill_count,
            prep_fill_mb,
            prep_title_copy_ms,
            prep_title_copy_count,
        ] = [0, 1, 2, 3, 4, 5, 6].map(|i| {
            let scale = match i {
                PREP_WATCH_US | PREP_REFRESH_COPY_US | PREP_FILL_US | PREP_TITLE_COPY_US => 1e3,
                PREP_FILL_BYTES => 1e6,
                _ => 1.0,
            };
            PREPARING_COPIES[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame"
        );
        let [budget_runs, budget_over, budget_most, budget_record, mesh_big, mesh_big_most] =
            [0, 1, 2, 3, 4, 5].map(|i| BUDGET_RUNS[i].swap(0, Ordering::Relaxed));
        tracing::info!(
            "mesh records asking for more than 65535 groups (under 2^24): {mesh_big}, the largest {mesh_big_most}; direct mesh dispatches run as rows: {}",
            crate::draw::take_wide_mesh()
        );
        tracing::info!(
            "indirect clamp budget: {budget_runs} list runs used it, {budget_over} spent all of it (their later indirect calls ran empty), the most spent {budget_most} of {}, the costliest record {budget_record}",
            crate::indirect::BUDGET
        );
        let compute_widest = COMPUTE_GROUPS[0].swap(0, Ordering::Relaxed);
        let compute_cut = COMPUTE_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect compute dispatches: the largest group count in x asked for {compute_widest} (run: at most 65535),              {compute_cut:.1} records per frame asked for more and were cut to that"
        );
        {
            let [texture_us, slots] = [0, 1].map(|i| crate::draw::HEAP_PASS_PARTS[i].swap(0, Ordering::Relaxed));
            tracing::info!(
                "the pass over the heap per frame (the frame line's \"heap buffers ... ms\" is both parts): {:.1} ms in its                  pass over textures picked by index; {:.0} buffer slots looked at by its pass over buffer views",
                texture_us as f64 / 1e3 / frames,
                slots as f64 / frames
            );
        }
        crate::shadow::exact_line(frames);
        crate::descriptor::sync_line(frames);
        late_line(frames);
        let gpu_pct = card_clock_line(frames, ms);
        // (kept for the frametime dump below)
        let (card_passes, card_compares, texture_refresh_ms);
        crate::shadow::back_line(frames);
        crate::shadow::view_marks_line(frames);
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            (card_passes, card_compares) = (passes, compares);
            if passes + compares > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        {
            let [own, asking, still, all] =
                [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
            texture_refresh_ms = all;
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it); the rest is reading, hashing and detiling"
                );
            }
            let [bare, turn, alias, written] =
                [0, 1, 2, 3].map(|i| crate::texture::HASH_WHY[i].swap(0, Ordering::Relaxed) as f64 / 1048576.0 / frames);
            if bare + turn + alias + written > 0.0 {
                tracing::info!(
                    "textures hashed per frame, by why: {bare:.1} MB with no watch or guard over them, {turn:.1} MB guarded and \
                     unwritten on their turn (every {} frames, VICTORY_TEXTURE_HASH_EVERY), {alias:.1} MB whose pages show at \
                     another address too, {written:.1} MB the guard saw written",
                    crate::texture::hash_every()
                );
            }
            let [through, left] = [0, 1].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there)"
                );
            }
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        // Passes over the heap's buffer views for shaders that pick them by index.
        let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_ms =
            crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        // Copies between depth textures and their colour twins, each way.
        let [twin_to_depth, twin_to_colour] =
            [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Depth planes drawn from their twins.
        let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        // Copies in the card's memory (`crate::shadow::cards`).
        let card = crate::shadow::card_stats();
        #[cfg(feature = "diagnostics")]
        let slow = crate::watch::slow_asks();
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        // `VICTORY_FRAMETIME_DUMP`: one row per summary interval, with the
        // buckets above and the worst/mean frame measured at each present.
        let (frame_ms_max, frame_ms_avg, _timed) = crate::frametime::take_interval();
        crate::frametime::record(&crate::frametime::Row {
            frame: crate::queue::frame_count(),
            frames,
            fps,
            ms,
            frame_ms_max,
            frame_ms_avg,
            cpu_prepare_ms: cpu,
            gpu_ms: gpu,
            gpu_pct,
            copies_ms: copy_ms,
            copies_mb: copy_mb,
            textures_ms: texture_ms,
            textures_mb: texture_mb,
            copyback_ms: back_ms,
            copyback_mb: back_mb,
            draws,
            dropped,
            making: crate::pipeline::making() as f64,
            indirect,
            heap_passes: picked,
            heap_ms: picked_ms,
            texture_refresh_ms,
            card_passes,
            card_compares,
            card_sent_mb: card[0] as f64 / frames / 1e6,
            card_ranges: card[1] as f64 / frames,
            card_pieces: card[2] as f64 / frames,
            card_fetched: card[3] as f64 / frames,
            card_fetched_mb: card[4] as f64 / frames / 1e6,
            card_waits: card[5] as f64 / frames,
            card_wait_ms: card[6] as f64 / frames / 1e3,
            twins_into_depth: twin_to_depth,
            twins_into_colour: twin_to_colour,
            depth_planes: planes,
            ram_mb: crate::frametime::private_mb(),
            cpu_pct: crate::frametime::cpu_pct(),
        });
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "the title's barrier groups since the start: {} waited for every earlier shader write, {} left out (no shader that writes through unordered access since the last wait; left out only under VICTORY_BARRIER_GROUPS=skip)",
                crate::list::GROUPS_WAITED.load(Ordering::Relaxed),
                crate::list::GROUPS_LEFT_OUT.load(Ordering::Relaxed)
            );
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made); \
                 neutral textures shown: {} slots in all, {} of them written with a texture not uploaded yet (VICTORY_FALLBACK_TEXTURES=0: null)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed),
                crate::descriptor::host::FALLBACK_SLOTS.load(Ordering::Relaxed),
                crate::descriptor::host::EMPTY_AT_WRITE.load(Ordering::Relaxed)
            );
            crate::tiles::status();
            if crate::drvwatch::errors() > 0 {
                tracing::warn!(
                    "errors the graphics driver reported to the system since the start: {} (the paths under suspicion are off)",
                    crate::drvwatch::errors()
                );
            }
            {
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} gaps joined at capacity), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
            }
            {
                let mode = crate::watch::page_writes();
                let [writes, bytes, guarded, plain] = [0, 1, 2, 3].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                let [kept, gaps, slots, uploaded] =
                    [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                tracing::info!(
                    "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the                      title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to                      plain addresses finished ({}); {kept} ranges under texture views kept for the texture pass ({gaps} gaps joined at capacity);                      {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({})                      (VICTORY_PAGE_WRITES: unset or guards opens the guarded blocks, 0 counts only, 1, slots, all)",
                    bytes >> 20,
                    if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                    if mode & 4 != 0 { "told as well" } else { "counted only" },
                    if mode & 2 != 0 { "made due again" } else { "left as they were" }
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                     ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
            }
            crate::watch::asks_say(frames);
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms;                  texture slots written so far: {placed} placed, {objects} of objects; \
                 textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                 {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                refused = format!(
                    "{}; depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass, {} passes ahead of a draw",
                    crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                    crate::resource::TWINS[0].load(Ordering::Relaxed),
                    crate::resource::TWINS[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[2].load(Ordering::Relaxed),
                    crate::resource::PLANES[3].load(Ordering::Relaxed)
                ),
                making = crate::pipeline::making(),
                placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
            );
            tracing::info!(
                "preparing copies breakdown per frame: watch {prep_watch_ms:.1} ms, \
                 refresh-copy {prep_refresh_copy_ms:.1} ms, fills {prep_fill_ms:.1} ms \
                 ({prep_fill_count:.0} fills, {prep_fill_mb:.1} MB), \
                 title-copies {prep_title_copy_ms:.1} ms ({prep_title_copy_count:.0})"
            );
            let (buf_n, buf_b, loose_n, loose_b, loose_pin, rebar_b) = crate::shadow::cache_stats();
            let (stg_n, stg_b, stg_pin) = crate::texture::cache_stats();
            let (mem_n, mem_b) = crate::memory::cache_stats();
            let (ret_pending, ret_epochs, ret_oldest) = retire_stats();
            let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
            tracing::info!(
                "gpu memory caches: buffers={:.0} MiB ({buf_n}), loose={:.0} MiB ({loose_n}, {loose_pin} pinned), \
                 staging={:.0} MiB ({stg_n}, {stg_pin} pinned), raw_mem={:.0} MiB ({mem_n}), \
                 rebar={:.0} MiB, retired={ret_pending} (epochs={ret_epochs} oldest={ret_oldest})",
                mib(buf_b),
                mib(loose_b),
                mib(stg_b),
                mib(mem_b),
                mib(rebar_b),
            );
            let [remap_n, remap_wait_us, remap_hold_us] =
                [0, 1, 2].map(|i| crate::watch::REMAPS[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if remap_n > 0.0 {
                tracing::info!(
                    "changes of title memory's mapping per frame: {remap_n:.0}, {:.1} ms of them waiting for reads in place, \
                     {:.1} ms under way ({})",
                    remap_wait_us / 1000.0,
                    remap_hold_us / 1000.0,
                    if crate::watch::by_range() {
                        "each holds the bytes it changes; VICTORY_MAPPING_RANGES=0 holds all of title memory"
                    } else {
                        "all of title memory held for each; VICTORY_MAPPING_RANGES=0"
                    }
                );
            }
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
                // The debugger's overlay: the same figures, short.
                if crate::debugger::on() {
                    use std::fmt::Write as _;
                    let mut t = format!(
                        "{fps:.1} fps  {ms:.0} ms/frame  {draws:.0} draws\n\
                         preparing {cpu:.0} ms: copies {copy_ms:.0} ms {copy_mb:.0} MB, textures {texture_ms:.0} ms {texture_mb:.0} MB\n\
                         GPU wait {gpu:.0} ms   copying back {back_ms:.0} ms {back_mb:.0} MB\n\
                         dropped {dropped:.0}/frame (no host pipeline), {indirect:.0} indirect\n"
                    );
                    for (why, n) in drops.iter().take(6) {
                        let why: String = why.chars().take(70).collect();
                        let _ = writeln!(t, "  {:>6.1}  {why}", *n as f64 / frames);
                    }
                    let _ = write!(
                        t,
                        "textures by index: {} uploaded ({} MB), {} waiting, {} unmapped\n\
                         texture slots: {} placed, {} of objects\n\
                         views kept {}  fallback slots {}  empty at write {}",
                        crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                        crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                        crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                        crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                        crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                        crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed),
                        crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed),
                        crate::descriptor::host::FALLBACK_SLOTS.load(Ordering::Relaxed),
                        crate::descriptor::host::EMPTY_AT_WRITE.load(Ordering::Relaxed),
                    );
                    crate::debugger::stats(t);
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            #[cfg(feature = "diagnostics")]
            let read_now = crate::capture::wanted(frame) || crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            let list = &copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            // The frame rate, into the corner of what is shown (not of the
            // title's own frame, which the lines below read back).
            crate::fpsbox::draw(self, list, &back, (width, height, format));
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() { None } else { crate::capture::probe(&self.device, list, texture) };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            copy.close();
            match slot {
                Some(i) => {
                    let flown = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                }
                None => self.execute(&[copy]),
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            crate::frame::before_present();
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            // `VICTORY_PROFILE=<dir>`: a row per frame, taken around the
            // frame lines (which zero the counters it reads).
            #[cfg(feature = "diagnostics")]
            crate::profile::frame_begins();
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            crate::profile::frame_ends();
            crate::texdebug::frame();
            #[cfg(feature = "diagnostics")]
            crate::copystat::BY_LINE_FRAMES.fetch_add(1, Ordering::Relaxed);
            crate::pix::frame();
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        self.default_buffer_in_state(size, D3D12_RESOURCE_STATE_COMMON)
    }

    /// A GPU-only default-heap buffer that shaders can write (unordered
    /// access). Starts in the common state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.default_buffer_in_state(size, D3D12_RESOURCE_STATE_COMMON)
    }

    /// A GPU-only default-heap UAV buffer created in `initial_state`.
    pub(crate) fn default_buffer_in_state(
        &self,
        size: u64,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                initial_state,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

#[cfg(test)]
pub(crate) struct DropCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl DropCounter {
    pub(crate) fn new() -> (Self, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (Self(drops.clone()), drops)
    }
}

#[cfg(test)]
impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod retire_tests {
    use super::*;

    #[test]
    fn retire_drops_immediately_when_no_epoch_is_active() {
        let _retire = isolated_retire_scope();
        let (counter, drops) = DropCounter::new();
        retire(counter);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn idle_list_does_not_block_retirement() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let idle = host.new_list().expect("host list");
        let (counter, drops) = DropCounter::new();
        retire(counter);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(idle);
    }

    #[test]
    fn open_marked_list_pins_retired_objects_across_other_executions() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let open = host.new_list().expect("open list");
        open.mark_work();

        let (c1, d1) = DropCounter::new();
        retire(c1);
        assert_eq!(d1.load(Ordering::SeqCst), 0);

        // Submitting and executing a DIFFERENT list must not drop objects
        // retired while `open` is still active.
        let other = host.new_list().expect("other list");
        other.mark_work();
        other.close();
        host.execute(&[&other]);
        other.reset();
        drop(other);
        assert_eq!(
            d1.load(Ordering::SeqCst),
            0,
            "executing another list must not release an open list's epoch"
        );

        // Closing and resetting the open list releases its epoch.
        open.close();
        open.reset();
        assert_eq!(d1.load(Ordering::SeqCst), 1);

        // Marking work again and dropping the list also releases its epoch.
        open.mark_work();
        let (c2, d2) = DropCounter::new();
        retire(c2);
        assert_eq!(d2.load(Ordering::SeqCst), 0);
        drop(open);
        assert_eq!(d2.load(Ordering::SeqCst), 1);

        // A closed list pinned via begin_submit stays active across execution
        // until end_submit.
        let submitted = host.new_list().expect("submitted list");
        submitted.mark_work();
        submitted.begin_submit();
        let (c3, d3) = DropCounter::new();
        retire(c3);
        submitted.close();
        host.execute(&[&submitted]);
        assert_eq!(d3.load(Ordering::SeqCst), 0);
        submitted.end_submit();
        assert_eq!(d3.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn active_execution_epoch_pins_objects_retired_during_execute() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let drops_during = drops.clone();
        // Use an unmarked deferred step without `begin_submit()` so `list`
        // itself pins no epoch; only `Host::execute`'s `exec_epoch` keeps the
        // object alive until execution finishes.
        list.defer_host(move || {
            let counter = DropCounter(drops_during.clone());
            retire(counter);
            assert_eq!(drops_during.load(Ordering::SeqCst), 0);
        });
        assert!(!list.has_epoch());
        list.close();
        host.execute(&[&list]);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stale_pinned_list_does_not_block_newer_born_retirements() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        // Simulate an early command list (e.g. oldest_epoch=2) that recorded
        // work once and stays unreset indefinitely.
        let stale = host.new_list().expect("stale list");
        stale.mark_work();

        // A resource born after `stale` last recorded work must drop
        // immediately once no newer epoch is active, without head-of-line
        // blocking behind `stale`.
        let born1 = birth_seq();
        let (c1, d1) = DropCounter::new();
        retire_born(c1, born1);
        assert_eq!(
            d1.load(Ordering::SeqCst),
            1,
            "stale pinned epoch must not block a resource created after its last work"
        );

        // A resource born before a newer active list must wait for that newer
        // list, then drop as soon as the newer list finishes — still while
        // `stale` remains pinned.
        let born2 = birth_seq();
        let active = host.new_list().expect("active list");
        active.mark_work();
        let (c2, d2) = DropCounter::new();
        retire_born(c2, born2);
        assert_eq!(d2.load(Ordering::SeqCst), 0);
        active.close();
        active.reset();
        assert_eq!(
            d2.load(Ordering::SeqCst),
            1,
            "resource must drop once its own blocking epoch finishes"
        );

        // If the older open list subsequently records new work after `born3`,
        // its `work_seq` advances and `c3` MUST wait for `stale` to finish.
        let born3 = birth_seq();
        stale.mark_work();
        let (c3, d3) = DropCounter::new();
        retire_born(c3, born3);
        assert_eq!(
            d3.load(Ordering::SeqCst),
            0,
            "an open list that records work after birth_seq must pin that resource"
        );
        stale.close();
        stale.reset();
        assert_eq!(d3.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn resubmitted_closed_list_preserves_work_seq_until_second_completion() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let born = birth_seq();
        let list = host.new_list().expect("reused list");
        list.mark_work();
        list.close();

        // First submission completes and releases the epoch, but keeps
        // `work_seq` because the list was not reset.
        list.begin_submit();
        host.execute(&[&list]);
        list.end_submit();
        assert!(!list.has_epoch());

        // Second submission re-acquires an epoch preserving `work_seq >= born`.
        list.begin_submit();
        let (c, d) = DropCounter::new();
        retire_born(c, born);
        assert_eq!(
            d.load(Ordering::SeqCst),
            0,
            "resubmitted closed list must keep resources from its recording alive"
        );
        host.execute(&[&list]);
        assert_eq!(d.load(Ordering::SeqCst), 0);
        list.end_submit();
        assert_eq!(d.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn host_list_reset_trims_excess_scratch_chunks() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");

        // Allocate 3 x 3 MiB so 3 distinct 4 MiB scratch chunks are created.
        for _ in 0..3 {
            let (gpu, ptr) = list.scratch(3 << 20).expect("scratch allocation");
            assert_ne!(gpu, 0);
            assert!(!ptr.is_null());
        }
        assert!(
            list.scratch
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .chunks
                .len()
                >= 2,
            "multi-chunk scratch spike should allocate at least two chunks"
        );

        let errors_before = host.d3d12_errors();
        list.close();
        list.reset();

        {
            let scratch = list.scratch.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                scratch.chunks.len() <= 1,
                "reset should keep at most one warm scratch chunk (or 0 when pooled)"
            );
            assert_eq!((scratch.current, scratch.used), (0, 0));
        }

        // Subsequent allocations succeed cleanly with zero D3D12 errors.
        let (gpu, ptr) = list.scratch(4096).expect("warm scratch allocation");
        assert_ne!(gpu, 0);
        assert!(!ptr.is_null());
        assert_eq!(
            list.scratch
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .chunks
                .len(),
            1
        );

        list.close();
        host.execute(&[&list]);
        assert_eq!(host.d3d12_errors(), errors_before);
    }

    #[test]
    fn upload_scratch_aligns_placements_and_falls_back_when_oversized() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let errors_before = host.d3d12_errors();

        // First 256-byte scratch leaves `used = 256`, which is not 512-byte aligned.
        let (gpu0, _) = list.scratch(256).expect("initial scratch");
        assert_eq!(gpu0 % 256, 0);

        let (buf1, at1, ptr1) = list.upload_scratch(128).expect("aligned sub-allocated upload");
        assert_eq!(at1, 512, "texture placement offset must round up to 512 bytes");
        assert_eq!(at1 % 512, 0);
        assert!(!ptr1.is_null());
        assert_eq!(
            list.deferred().uploads.len(),
            0,
            "sub-allocated staging must not allocate a dedicated upload buffer"
        );
        // SAFETY: `buf1` is a live resource and `ptr1` points to 256+ writable bytes.
        unsafe {
            assert_eq!(buf1.GetGPUVirtualAddress() + at1, gpu0 + at1);
            ptr1.write_bytes(0x5a, 128);
        }

        // Oversized copy falls back to a dedicated upload buffer at offset 0.
        let (buf_big, at_big, ptr_big) = list
            .upload_scratch(SCRATCH_CHUNK + 512)
            .expect("oversized fallback upload");
        assert_eq!(at_big, 0);
        assert!(!ptr_big.is_null());
        assert_ne!(buf_big.as_raw(), buf1.as_raw());
        assert_eq!(list.deferred().uploads.len(), 1);
        // SAFETY: `ptr_big` is mapped for at least `SCRATCH_CHUNK + 512` bytes.
        unsafe { ptr_big.write_bytes(0x3c, 16) };

        list.close();
        host.execute(&[&list]);
        assert_eq!(host.d3d12_errors(), errors_before);
    }

    #[test]
    fn release_for_pool_clears_scratch_uploads_and_tracked_resources_immediately() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");

        // Track a title texture on the list, then release the title object so
        // only the list's tracked state and epoch keep HostTexture alive.
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let iid = ID3D12Resource::IID;
        let mut tex = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                0,
                null.cast(),
                (&raw const iid).cast(),
                &mut tex,
            ),
            crate::object::S_OK
        );
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        list.transition(tex, SHADER_RESOURCE);
        let weak_tex = std::sync::Arc::downgrade(
            &crate::resource::host_copy(tex).expect("tracked host texture"),
        );
        let release =
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(tex, 0x10);
        assert_eq!(release(tex), 0);
        assert!(weak_tex.upgrade().is_some());

        // Populate scratch chunks, uploads, kept objects, deferred work,
        // pipelines, and a pinned retirement epoch.
        assert!(list.scratch(3 << 20).is_some());
        assert!(list.scratch(3 << 20).is_some());
        let (upload_res, _) = list.upload(host, 4096).expect("list upload");
        list.keep(&upload_res);
        drop(upload_res);
        list.mark_work();
        list.used_pipeline("recycle_test_pipeline");
        let (deferred_counter, deferred_drops) = DropCounter::new();
        list.defer_host(move || {
            let _ = &deferred_counter;
        });
        let (retired_counter, retired_drops) = DropCounter::new();
        retire(retired_counter);
        assert!(list.has_epoch());
        assert_eq!(retired_drops.load(Ordering::SeqCst), 0);

        let errors_before = host.d3d12_errors();
        list.close();
        host.execute(&[&list]);

        assert!(weak_tex.upgrade().is_some());
        assert_eq!(deferred_drops.load(Ordering::SeqCst), 0);
        assert_eq!(retired_drops.load(Ordering::SeqCst), 0);
        assert_eq!(
            list.scratch
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .chunks
                .len(),
            2
        );

        list.release_for_pool();

        assert!(
            weak_tex.upgrade().is_none(),
            "tracked HostTexture must drop immediately when list is released for pool"
        );
        assert_eq!(
            deferred_drops.load(Ordering::SeqCst),
            1,
            "deferred closures must drop immediately when list is released for pool"
        );
        assert_eq!(
            retired_drops.load(Ordering::SeqCst),
            1,
            "list epoch must release immediately when list is released for pool"
        );
        assert!(
            list.scratch
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .chunks
                .is_empty(),
            "pooled list must hold 0 scratch chunks"
        );
        assert!(
            list.states
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        );
        assert!(
            list.pipelines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        );
        assert!(!list.has_epoch());
        {
            let deferred = list.deferred();
            assert!(deferred.uploads.is_empty());
            assert!(deferred.kept.is_empty());
            assert!(deferred.copies.is_empty());
        }

        // Verify D3D12 reset + recording + execution succeed cleanly with zero
        // debug-layer errors.
        list.reset();
        let (gpu, ptr) = list.scratch(4096).expect("scratch after pool reuse");
        assert_ne!(gpu, 0);
        assert!(!ptr.is_null());
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.d3d12_errors(), errors_before);
    }

    #[test]
    fn fill_scratch_deduplicates_until_copy_or_defer_run() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let errors_before = host.d3d12_errors();
        let mut src = [0x11u8; 64];
        let mut dst = [0u8; 64];

        let gpu1 = list.fill_scratch(src.as_ptr(), 64).expect("first fill");
        let gpu2 = list.fill_scratch(src.as_ptr(), 64).expect("dedup fill");
        assert_eq!(gpu1, gpu2);
        assert_eq!(
            list.deferred()
                .copies
                .iter()
                .filter(|s| matches!(s, Step::Fill(..)))
                .count(),
            1
        );

        // Intervening title copy invalidates `filled` cache.
        list.defer_copy(dst.as_mut_ptr(), src.as_ptr(), 4);
        let gpu3 = list.fill_scratch(src.as_ptr(), 64).expect("post-copy fill");
        assert_ne!(gpu3, gpu1);

        // Intervening deferred run also invalidates `filled` cache.
        list.defer(|| {});
        src[0] = 0x22;
        let gpu4 = list.fill_scratch(src.as_ptr(), 64).expect("post-run fill");
        assert_ne!(gpu4, gpu3);

        list.close();
        host.execute(&[&list]);
        assert_eq!(host.d3d12_errors(), errors_before);
    }

    #[test]
    fn defer_gpu_write_deduplicates_preserving_marked_and_clears_on_copy() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let errors_before = host.d3d12_errors();
        let mut backing = vec![0u8; 4096];
        let start = backing.as_mut_ptr() as u64;
        let addr = start as usize;
        crate::shadow::register(addr, start, backing.len() as u64);
        let shadow = crate::shadow::find(start, backing.len() as u64).expect("registered shadow");

        list.defer_gpu_write(&shadow, addr, 256, true);
        list.defer_gpu_write(&shadow, addr, 256, true);
        // Same range with `marked = false` must not be dropped by the `marked = true` entry.
        list.defer_gpu_write(&shadow, addr, 256, false);

        let gpu_write_count = list
            .deferred()
            .copies
            .iter()
            .filter(|s| matches!(s, Step::GpuWrite(..)))
            .count();
        assert_eq!(gpu_write_count, 2);

        let declared = declared_writes(&[&list]);
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].1, vec![(addr, addr + 256)]);
        assert_eq!(declared[0].2, vec![(addr, addr + 256)]);

        let mut dst = [0u8; 16];
        list.defer_copy(dst.as_mut_ptr(), backing.as_ptr(), 16);
        list.defer_gpu_write(&shadow, addr, 256, true);
        let gpu_write_count_after = list
            .deferred()
            .copies
            .iter()
            .filter(|s| matches!(s, Step::GpuWrite(..)))
            .count();
        assert_eq!(gpu_write_count_after, 3);

        list.close();
        list.reset();
        drop(declared);
        drop(shadow);
        crate::shadow::unregister(addr, start);
        assert_eq!(host.d3d12_errors(), errors_before);
    }

    #[test]
    fn clamp_unbound_returns_true_once_until_reset() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let list = host.new_list().expect("host list");
        let errors_before = host.d3d12_errors();

        assert!(list.clamp_unbound());
        assert!(!list.clamp_unbound());
        assert!(!list.clamp_unbound());

        list.close();
        list.reset();
        assert!(list.clamp_unbound());
        assert!(!list.clamp_unbound());

        list.close();
        host.execute(&[&list]);
        assert_eq!(host.d3d12_errors(), errors_before);
    }
}

#[cfg(test)]
mod batch {
    // Batch::texture_due lives here; `mod tests` is already taken above
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        use super::vram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }

    #[test]
    fn should_record_brought_only_when_later_flights_exist() {
        use super::should_record_brought;
        // Single/last flight overflow (after == 5, landed == Some(5)): after - 1 = 4 <= 5.
        assert!(!should_record_brought(5, Some(5)));
        // Next submission numbered 6 has not queued lists yet (after == 6, landed == Some(5)): after - 1 = 5 <= 5.
        assert!(!should_record_brought(6, Some(5)));
        // True overlap with flight 6 in the air (after == 7, landed == Some(5)): after - 1 = 6 > 5.
        assert!(should_record_brought(7, Some(5)));
        // Degenerate `after <= 1` never records.
        assert!(!should_record_brought(0, None));
        assert!(!should_record_brought(1, None));
        assert!(should_record_brought(2, None));
    }

    #[test]
    fn eval_late_met_uses_final_write_and_honors_unsure() {
        use super::eval_late_met;
        use crate::queue::Written;
        let addr = 0x1000usize;
        let eq_one = |v: u64| v == 1;

        // Later overwrite (1 -> 0) invalidates an earlier matching write.
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(1), 4), (addr, Written::Value(0), 4)],
                addr,
                4,
                eq_one,
            ),
            Some(false)
        );
        // Later matching write (0 -> 1) satisfies the wait without settling.
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(0), 4), (addr, Written::Value(1), 4)],
                addr,
                4,
                eq_one,
            ),
            Some(true)
        );
        // An overlapping in-flight shader write forces settling (`None`) even
        // when a bottom-of-pipe write matches.
        assert_eq!(
            eval_late_met(
                &[(addr, addr + 4)],
                [(addr, Written::Value(1), 4)],
                addr,
                4,
                eq_one,
            ),
            None
        );
        // A partial/unaligned overlapping write or timestamp forces settling (`None`).
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(1), 4), (addr + 2, Written::Value(0), 4)],
                addr,
                4,
                eq_one,
            ),
            None
        );
        assert_eq!(
            eval_late_met(
                &[],
                [(addr, Written::Value(1), 4), (addr, Written::GpuTimestamp, 8)],
                addr,
                4,
                eq_one,
            ),
            None
        );
        // Unrelated writes return `Some(false)` without settling.
        assert_eq!(
            eval_late_met(
                &[(addr + 16, addr + 32)],
                [(addr + 8, Written::Value(1), 4)],
                addr,
                4,
                eq_one,
            ),
            Some(false)
        );
    }
}

#[cfg(test)]
mod rebar_tests {
    use super::{
        REBAR_HASH_SOURCE, compile, flagged_piece_byte_runs, host, intersect_ranges, merge_ranges,
        rebar_hash_pipeline, rebar_policy, spans_to_piece_runs, transition,
    };
    use crate::watch::PAGE;

    #[test]
    fn rebar_policy_handles_all_hardware_tiers_and_fallbacks() {
        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;

        // Non-ReBAR, WARP, UMA/APU, or explicitly disabled always stay off.
        assert_eq!(rebar_policy(false, false, Some(32 * GIB), true), (false, 0));
        assert_eq!(rebar_policy(true, true, Some(64 * GIB), true), (false, 0));
        assert_eq!(rebar_policy(true, false, Some(32 * GIB), false), (false, 0));
        assert_eq!(rebar_policy(true, false, None, true), (false, 0));

        // Low-VRAM discrete GPUs (<= 8 GiB) stay on system-RAM shadows.
        assert_eq!(rebar_policy(true, false, Some(4 * GIB), true), (false, 0));
        assert_eq!(rebar_policy(true, false, Some(8 * GIB), true), (false, 0));

        // 12 GiB tier: conservative 512 MiB shadow budget.
        assert_eq!(
            rebar_policy(true, false, Some(12 * GIB), true),
            (true, 512 * MIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(14 * GIB), true),
            (true, 512 * MIB)
        );

        // 16 GiB tier: 1 GiB shadow budget.
        assert_eq!(rebar_policy(true, false, Some(16 * GIB), true), (true, GIB));

        // 20–32+ GiB tier: 1/16 of local VRAM budget clamped to 1–2 GiB to bound WDDM commit.
        assert_eq!(
            rebar_policy(true, false, Some(20 * GIB), true),
            (true, 1280 * MIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(24 * GIB), true),
            (true, 1536 * MIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(32 * GIB), true),
            (true, 2 * GIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(48 * GIB), true),
            (true, 2 * GIB)
        );
    }

    #[test]
    fn merge_ranges_coalesces_overlapping_and_adjacent_spans() {
        assert_eq!(merge_ranges(vec![]), vec![]);
        assert_eq!(merge_ranges(vec![(10, 10), (20, 15)]), vec![]);
        assert_eq!(
            merge_ranges(vec![(100, 200), (0, 50), (50, 120), (300, 400)]),
            vec![(0, 200), (300, 400)]
        );
        assert_eq!(
            merge_ranges(vec![(0, 100), (10, 40), (20, 90)]),
            vec![(0, 100)]
        );
    }

    #[test]
    fn piece_runs_and_flagged_intersection_dedupe_and_clamp() {
        // Two disjoint spans inside piece 0 plus a span in piece 1 coalesce into
        // one contiguous run `(0, 2)`, while piece 4 stays separate.
        let declared = vec![
            (0x100, 0x200),
            (0x8000, 0x9000),
            (PAGE + 16, PAGE + 64),
            (4 * PAGE, 5 * PAGE + 128),
        ];
        let runs = spans_to_piece_runs(&declared);
        assert_eq!(runs, vec![(0, 2), (4, 2)]);

        // Flag pieces 0, 4, and 5 (skip piece 1); clamp end to buffer length.
        let len = 5 * PAGE + 256;
        let (flagged, changed) = flagged_piece_byte_runs(&runs, &[1, 0, 1, 1], len);
        assert_eq!(changed, 3);
        assert_eq!(flagged, vec![(0, PAGE), (4 * PAGE, len)]);

        // Intersecting with declared spans yields only the declared bytes in
        // flagged pieces (both disjoint spans inside piece 0 are preserved).
        assert_eq!(
            intersect_ranges(&flagged, &declared),
            vec![(0x100, 0x200), (0x8000, 0x9000), (4 * PAGE, 5 * PAGE + 128)]
        );

        // Zero flagged pieces produces zero readback spans.
        let (none, changed) = flagged_piece_byte_runs(&runs, &[0, 0, 0, 0], len);
        assert_eq!(changed, 0);
        assert!(none.is_empty());
        assert!(intersect_ranges(&none, &declared).is_empty());
    }

    #[test]
    fn rebar_hash_hlsl_compiles_and_builds_on_host() {
        compile(
            REBAR_HASH_SOURCE,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .expect("rebar_hash HLSL compiles");
        if let Some(h) = host() {
            assert!(rebar_hash_pipeline(&h.device).is_some());
        }
    }

    #[test]
    fn rebar_hash_shader_detects_single_byte_change_and_swap_on_gpu() {
        use windows::Win32::Graphics::Direct3D12::*;

        let Some(h) = host() else {
            return; // no host GPU here
        };
        let Some(pipe) = rebar_hash_pipeline(&h.device) else {
            panic!("rebar_hash compute pipeline failed to build on a live host");
        };
        let errors_before = h.d3d12_errors();
        let total_bytes = (2 * PAGE) as u64;
        let (src_buf, src_ptr) = h.shared_buffer(total_bytes).expect("shared src");
        let hashes_buf = h
            .default_buffer_in_state(32, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .expect("hashes UAV");
        let flags_uav = h
            .default_buffer_in_state(16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .expect("flags UAV");
        let (flags_rb, flags_ptr) = h.readback_buffer(16).expect("flags readback");

        // Seed two 64 KiB pieces with distinct non-repeating synthetic bytes.
        // SAFETY: `src_ptr` is mapped for `2 * PAGE` bytes.
        let src_slice = unsafe { std::slice::from_raw_parts_mut(src_ptr, 2 * PAGE) };
        for (i, b) in src_slice.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(37) ^ (i >> 8)) & 0xFF) as u8;
        }

        // SAFETY: live D3D12 resources.
        let (src_gpu, hashes_gpu, flags_gpu) = unsafe {
            (
                src_buf.GetGPUVirtualAddress(),
                hashes_buf.GetGPUVirtualAddress(),
                flags_uav.GetGPUVirtualAddress(),
            )
        };

        let run_pass = |mode: u32| -> [u32; 2] {
            let list = h.new_list().expect("host list");
            let l = &list.list;
            let constants = [
                0u32,
                total_bytes as u32,
                0u32,
                mode,
                2u32,
                victory_shader::emit::VIEW_MARK_NONE,
                0u32,
            ];
            // SAFETY: open list with live buffers and valid state transitions.
            unsafe {
                l.ResourceBarrier(&[transition(
                    &src_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                l.SetComputeRootSignature(&pipe.root);
                l.SetPipelineState(&pipe.pso);
                l.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, src_gpu);
                l.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                l.SetComputeRootUnorderedAccessView(3, flags_gpu);
                l.SetComputeRootUnorderedAccessView(4, flags_gpu);
                l.Dispatch(2, 1, 1);
                if mode != 0 {
                    l.ResourceBarrier(&[
                        transition(
                            &flags_uav,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        ),
                        transition(
                            &src_buf,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COMMON,
                        ),
                    ]);
                    l.CopyBufferRegion(&flags_rb, 0, &flags_uav, 0, 8);
                    l.ResourceBarrier(&[transition(
                        &flags_uav,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                } else {
                    l.ResourceBarrier(&[transition(
                        &src_buf,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            }
            list.close();
            h.execute(&[&list]);
            list.reset();
            // SAFETY: fence completed and `flags_ptr` covers 16 bytes of READBACK.
            let out = unsafe { std::slice::from_raw_parts(flags_ptr.cast::<u32>(), 2) };
            [out[0], out[1]]
        };

        // 1. Compute initial hashes.
        let _ = run_pass(0);
        // 2. Compare unchanged -> both flags must be 0.
        assert_eq!(run_pass(1), [0, 0]);
        // 3. Flip a single byte in piece 1 -> only piece 1 flagged (and hash updated).
        src_slice[PAGE + 12345] ^= 0x01;
        assert_eq!(run_pass(1), [0, 1]);
        // 4. Re-compare without further writes -> stored hash was updated, so [0, 0].
        assert_eq!(run_pass(1), [0, 0]);
        // 5. Swap two 16-byte chunks at different lanes in piece 0 -> non-commutative
        //    reduction must detect the swap on piece 0 only.
        for i in 0..16 {
            src_slice.swap(i, 16 + i);
        }
        assert_eq!(run_pass(1), [1, 0]);
        assert_eq!(run_pass(1), [0, 0]);
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during rebar_hash test"
        );
    }

    #[test]
    fn relay_rebar_back_uses_dedicated_copy_queue_cleanly() {
        let Some(h) = host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let mut title = vec![0x22u8; 4096];
        let start = title.as_mut_ptr() as u64;
        let len = title.len() as u64;
        crate::shadow::register(start as usize, start, len);
        let shadow = crate::shadow::find(start, len).expect("registered shadow");
        if !shadow.is_rebar() {
            // Non-ReBAR hardware / WARP: still verify make_relay_fetch wires up
            // the host's COPY queue cleanly when enabled.
            let fetch = h.make_relay_fetch().expect("relay fetch");
            if super::relay_copy_queue() {
                assert!(
                    fetch.copy.is_some(),
                    "expected dedicated D3D12 COPY queue on host device"
                );
            }
            drop(shadow);
            crate::shadow::unregister(start as usize, start);
            h.log_messages();
            assert_eq!(h.d3d12_errors(), errors_before);
            return;
        }
        // Seed initial title bytes into shadow + mirror via a sync upload.
        let list = h.new_list().expect("host list");
        shadow.refresh(&list, start, len);
        list.close();
        h.execute(&[&list]);
        list.reset();

        // Populate ReBAR mirror and simulate a GPU write at offset 64..68 (via
        // a Direct-queue CopyBufferRegion from an upload buffer) plus a
        // concurrent CPU write at offset 256.
        shadow.writing(start as usize, len as usize, |_, size| {
            Some(vec![0x22u8; size as usize])
        });
        let (src_buf, src_ptr) = h.upload_buffer(256).expect("upload buffer");
        // SAFETY: mapped 256-byte upload buffer.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0xCC, 4);
            let l = &list.list;
            l.CopyBufferRegion(&shadow.buffer, 64, &src_buf, 0, 4);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();

        title[256] = 0x55;
        let mut fetch = None;
        h.relay_rebar_back(
            &[(
                shadow.clone(),
                vec![(start as usize, start as usize + len as usize)],
                vec![(start as usize, start as usize + len as usize)],
            )],
            &mut fetch,
        );
        h.log_messages();
        if super::relay_copy_queue() {
            assert!(
                fetch.as_ref().and_then(|f| f.copy.as_ref()).is_some(),
                "relay_rebar_back must use the dedicated D3D12 COPY queue"
            );
        }
        assert_eq!(
            &title[64..68],
            &[0xCC; 4],
            "GPU bytes must land via COPY queue"
        );
        assert_eq!(title[256], 0x55, "concurrent CPU write must be preserved");
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during COPY-queue relay ReBAR fetch"
        );
        shadow.copy_back_done(&[(start as usize, start as usize + len as usize)]);
        drop(shadow);
        crate::shadow::unregister(start as usize, start);
    }

    #[test]
    fn view_marks_gpu_capture_clear_and_rebar_hash_filter() {
        use windows::Win32::Graphics::Direct3D12::*;

        // 1. Both CARD_COMPARE and rebar_hash HLSL compile with marks UAV bindings.
        compile(
            super::CARD_COMPARE,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .expect("CARD_COMPARE HLSL compiles");

        let Some(h) = host() else {
            return; // no host GPU here
        };
        let Some(pipe) = rebar_hash_pipeline(&h.device) else {
            panic!("rebar_hash compute pipeline failed to build on a live host");
        };
        let errors_before = h.d3d12_errors();

        // 2. Verify rebar_hash skips unmarked pieces and checks both adjacent bits
        //    when mark_skew != 0 (unaligned shadow start).
        let total_bytes = (2 * PAGE) as u64;
        let (src_buf, src_ptr) = h.shared_buffer(total_bytes).expect("shared src");
        let hashes_buf = h
            .default_buffer_in_state(32, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .expect("hashes UAV");
        let flags_uav = h
            .default_buffer_in_state(16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .expect("flags UAV");
        let marks_uav = h
            .default_buffer_in_state(64, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .expect("marks UAV");
        let (marks_up, marks_up_ptr) = h.upload_buffer(64).expect("marks upload");
        let (flags_rb, flags_ptr) = h.readback_buffer(16).expect("flags readback");

        // SAFETY: `src_ptr` maps `2 * PAGE` bytes.
        let src_slice = unsafe { std::slice::from_raw_parts_mut(src_ptr, 2 * PAGE) };
        for (i, b) in src_slice.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(29) ^ (i >> 7)) & 0xFF) as u8;
        }
        // Set bit 6 (= mark_base 5 + piece 1); leave bit 5 (= mark_base 5 + piece 0) clear.
        // SAFETY: `marks_up_ptr` maps 64 bytes.
        unsafe {
            std::ptr::write_bytes(marks_up_ptr, 0, 64);
            *marks_up_ptr.cast::<u32>() = 1u32 << 6;
        }
        {
            let list = h.new_list().expect("seed marks list");
            // SAFETY: open list, valid transitions.
            unsafe {
                list.list.ResourceBarrier(&[transition(
                    &marks_uav,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                list.list.CopyBufferRegion(&marks_uav, 0, &marks_up, 0, 64);
                list.list.ResourceBarrier(&[transition(
                    &marks_uav,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            list.close();
            h.execute(&[&list]);
            list.reset();
        }

        // SAFETY: live D3D12 resources.
        let (src_gpu, hashes_gpu, flags_gpu, marks_gpu) = unsafe {
            (
                src_buf.GetGPUVirtualAddress(),
                hashes_buf.GetGPUVirtualAddress(),
                flags_uav.GetGPUVirtualAddress(),
                marks_uav.GetGPUVirtualAddress(),
            )
        };

        let run_pass = |mode: u32, mark_base: u32, mark_skew: u32| -> [u32; 2] {
            let list = h.new_list().expect("host list");
            let l = &list.list;
            let constants = [0u32, total_bytes as u32, 0u32, mode, 2u32, mark_base, mark_skew];
            // SAFETY: open list with live buffers and valid state transitions.
            unsafe {
                l.ResourceBarrier(&[transition(
                    &src_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                l.SetComputeRootSignature(&pipe.root);
                l.SetPipelineState(&pipe.pso);
                l.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, src_gpu);
                l.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                l.SetComputeRootUnorderedAccessView(3, flags_gpu);
                l.SetComputeRootUnorderedAccessView(4, marks_gpu);
                l.Dispatch(2, 1, 1);
                if mode != 0 {
                    l.ResourceBarrier(&[
                        transition(
                            &flags_uav,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        ),
                        transition(
                            &src_buf,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COMMON,
                        ),
                    ]);
                    l.CopyBufferRegion(&flags_rb, 0, &flags_uav, 0, 8);
                    l.ResourceBarrier(&[transition(
                        &flags_uav,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                } else {
                    l.ResourceBarrier(&[transition(
                        &src_buf,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            }
            list.close();
            h.execute(&[&list]);
            list.reset();
            // SAFETY: fence completed and `flags_ptr` covers 16 bytes of READBACK.
            let out = unsafe { std::slice::from_raw_parts(flags_ptr.cast::<u32>(), 2) };
            [out[0], out[1]]
        };

        // Seed initial hashes, then flip a byte in BOTH pieces.
        let _ = run_pass(0, victory_shader::emit::VIEW_MARK_NONE, 0);
        src_slice[123] ^= 0xFF;
        src_slice[PAGE + 456] ^= 0xFF;

        // With mark_base = 5 and skew = 0: only bit 6 (piece 1) is set, so piece 0
        // early-exits with flag 0 while piece 1 detects the change and updates its hash.
        assert_eq!(run_pass(1, 5, 0), [0, 1]);
        // With mark_skew = 1: piece 0 also checks bit 5 + 1 = 6 (which is set), so piece 0
        // now hashes and detects its change while piece 1's updated hash matches.
        assert_eq!(run_pass(1, 5, 1), [1, 0]);
        assert_eq!(run_pass(1, 5, 1), [0, 0]);

        // 3. Verify GPU-timeline bitmap capture + clear round-trip on the real
        //    SLOT_VIEW_WRITES resource (`view_writes_resources`).
        let (vw_uav, _, _) = crate::shadow::view_writes_resources().expect("view_writes bitmap");
        let (seed_up, seed_ptr) = h.upload_buffer(16).expect("seed upload");
        // SAFETY: `seed_ptr` maps 16 bytes.
        unsafe {
            let words = std::slice::from_raw_parts_mut(seed_ptr.cast::<u32>(), 4);
            words[0] = 0xA55A_1234;
            words[1] = 0x0000_00F0;
            words[2] = 0;
            words[3] = 0;
        }
        {
            let list = h.new_list().expect("seed view_writes list");
            // SAFETY: `vw_uav` is always in UNORDERED_ACCESS between submissions.
            unsafe {
                list.list.ResourceBarrier(&[transition(
                    vw_uav,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                list.list.CopyBufferRegion(vw_uav, 0, &seed_up, 0, 16);
                list.list.ResourceBarrier(&[transition(
                    vw_uav,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            list.close();
            h.execute(&[&list]);
            list.reset();
        }
        let (cl1, cap1) = h
            .record_view_marks_capture_words(2)
            .expect("first view-marks capture");
        {
            let mut submitted = h.submitted.lock().unwrap_or_else(|e| e.into_inner());
            assert!(h.submit_helper_list(&mut submitted, cl1, "test view-marks capture 1"));
        }
        assert_eq!(&cap1.words()[..2], &[0xA55A_1234, 0x0000_00F0]);
        cap1.recycle();

        // Second capture must read back all zeros, proving the GPU-timeline clear ran.
        let (cl2, cap2) = h
            .record_view_marks_capture_words(2)
            .expect("second view-marks capture");
        {
            let mut submitted = h.submitted.lock().unwrap_or_else(|e| e.into_inner());
            assert!(h.submit_helper_list(&mut submitted, cl2, "test view-marks capture 2"));
        }
        assert_eq!(&cap2.words()[..2], &[0, 0]);
        cap2.recycle();

        h.log_messages();
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during view-marks GPU test"
        );
    }
}
