//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (CPU-writable VRAM via Resizable BAR when supported with
//! headroom, or write-back system memory otherwise), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Total VRAM bytes currently allocated to ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// shadow buffers across live [`Shadow`] instances.
static REBAR_SHADOW_BYTES: AtomicU64 = AtomicU64::new(0);

/// Atomically reserves `size` bytes from `budget`, returning `false` if doing
/// so would exceed `budget`.
fn try_reserve_rebar(used: &AtomicU64, size: u64, budget: u64) -> bool {
    if budget == 0 || size == 0 || size > budget {
        return false;
    }
    used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        let next = cur.checked_add(size)?;
        (next <= budget).then_some(next)
    })
    .is_ok()
}

/// Maximum 64 KiB pieces hashed on the GPU for one ReBAR shadow
/// (`65535 * 64 KiB`, ~4 GiB): keeps 32-bit byte offsets in `rebar_hash.hlsl`
/// overflow-free and fits in a single 1D D3D12 `Dispatch(pieces, 1, 1)`.
pub(crate) const REBAR_HASH_MAX_PIECES: u32 = 65535;

/// Compact bitset of 64 KiB pieces whose VRAM contents were modified on the CPU
/// since their GPU hash was last computed.
struct RebarDirtyBits {
    pieces: u32,
    words: Vec<u64>,
    any: bool,
}

impl RebarDirtyBits {
    fn new(pieces: u32) -> Self {
        Self {
            pieces,
            words: vec![0; (pieces as usize).div_ceil(64)],
            any: false,
        }
    }

    /// Marks pieces `[first, first + count)` (clamped to `self.pieces`) dirty.
    fn mark_pieces(&mut self, first: u32, count: u32) {
        let start = first.min(self.pieces);
        let end = start.saturating_add(count).min(self.pieces);
        if start >= end {
            return;
        }
        self.any = true;
        let first_word = (start / 64) as usize;
        let last_word = ((end - 1) / 64) as usize;
        let start_bit = start % 64;
        let end_bit = ((end - 1) % 64) + 1;
        if first_word == last_word {
            let mask = if end_bit == 64 {
                !0u64 << start_bit
            } else {
                ((1u64 << (end_bit - start_bit)) - 1) << start_bit
            };
            self.words[first_word] |= mask;
        } else {
            self.words[first_word] |= !0u64 << start_bit;
            for w in &mut self.words[first_word + 1..last_word] {
                *w = !0u64;
            }
            let tail_mask = if end_bit == 64 {
                !0u64
            } else {
                (1u64 << end_bit) - 1
            };
            self.words[last_word] |= tail_mask;
        }
    }

    /// Marks every 64 KiB piece overlapping buffer byte range `[off, off + len)`.
    fn mark_byte_range(&mut self, off: usize, len: usize) {
        if len == 0 || self.pieces == 0 {
            return;
        }
        let end_byte = off.saturating_add(len);
        let first = (off / PAGE).min(self.pieces as usize) as u32;
        let end = end_byte.div_ceil(PAGE).min(self.pieces as usize) as u32;
        if first < end {
            self.mark_pieces(first, end - first);
        }
    }

    /// Clears all dirty bits and returns coalesced `(first_piece, count)` runs.
    fn take_runs(&mut self) -> Vec<(u32, u32)> {
        if !self.any {
            return Vec::new();
        }
        self.any = false;
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let mut cur: Option<(u32, u32)> = None;
        for (wi, word) in self.words.iter_mut().enumerate() {
            let mut w = std::mem::replace(word, 0);
            let base = (wi as u32) * 64;
            while w != 0 {
                let tz = w.trailing_zeros();
                let shifted = w >> tz;
                let ones = shifted.trailing_ones();
                let run_start = base + tz;
                let run_end = (run_start + ones).min(self.pieces);
                if run_start < run_end {
                    match cur {
                        Some((s, e)) if e == run_start => cur = Some((s, run_end)),
                        Some((s, e)) => {
                            runs.push((s, e - s));
                            cur = Some((run_start, run_end));
                        }
                        None => cur = Some((run_start, run_end)),
                    }
                }
                if tz + ones >= 64 {
                    break;
                }
                w &= !(((1u64 << ones) - 1) << tz);
            }
        }
        if let Some((s, e)) = cur {
            runs.push((s, e - s));
        }
        runs
    }
}

/// Prepared GPU hash state for one ReBAR shadow before a submission.
pub(crate) struct PreparedRebarHashes {
    pub(crate) hashes_gpu: u64,
    pub(crate) aligned_len: u32,
    pub(crate) pre_runs: Vec<(u32, u32)>,
}

/// GPU-side 128-bit per-64-KiB-piece hashes for a UAV-written ReBAR shadow.
///
/// Invariant: for every piece whose bit in `dirty` is clear, `buffer[piece]`
/// holds the 128-bit hash of `Shadow::buffer`'s bytes in that piece.
struct RebarHashes {
    _buffer: ID3D12Resource,
    gpu: u64,
    dirty: RebarDirtyBits,
}

/// Extra state for a [`Shadow`] backed by CPU-writable VRAM
/// (`D3D12_HEAP_TYPE_GPU_UPLOAD`, Resizable BAR).
///
/// `Shadow::data` is a write-combining pointer into VRAM and MUST NEVER be read
/// on the CPU. Read-only buffers keep `mirror = None` and `hashes = None` (zero
/// duplicate RAM or hash VRAM). On the first UAV write, `mirror` is lazily
/// populated via a one-time GPU DMA and kept synced on subsequent
/// `Shadow::copy` / `copy_back_rebar` calls; `hashes` is lazily created as a
/// `DEFAULT` UAV buffer (`16` bytes per 64 KiB piece) so post-submission
/// copy-back only DMAs 64 KiB pieces whose GPU hash actually changed.
struct RebarShadow {
    bytes: u64,
    mirror: Mutex<Option<Vec<u8>>>,
    /// Leaf lock: never acquired while holding `mirror`, `pending`, or `copied`,
    /// and never held across D3D12 or kernel calls.
    hashes: Mutex<Option<RebarHashes>>,
}

impl RebarShadow {
    fn has_mirror(&self) -> bool {
        self.mirror.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    fn mark_dirty_range(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut guard = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.dirty.mark_byte_range(off, len);
        }
    }

    fn mark_dirty_segs(&self, start: usize, segs: &[CopySeg]) {
        if segs.is_empty() {
            return;
        }
        let mut guard = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for seg in segs {
                if seg.len > 0 {
                    h.dirty.mark_byte_range(seg.src - start, seg.len);
                }
            }
        }
    }
}

/// Our own `MapViewOfFile` view backing a `VICTORY_STRETCH` in-place heap.
/// Unmapped on drop after `Shadow::buffer` and `Shadow::place` have released
/// the placed resource and `ID3D12Heap`.
struct StretchView {
    addr: usize,
    section: usize,
}

impl Drop for StretchView {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        // SAFETY: our own view mapped in `through_another_view`, no longer held
        // by any D3D12 resource or heap (`buffer` and `place` drop earlier).
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.addr as *mut _,
            });
        }
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Monotonic resource creation sequence for epoch-based retirement.
    born_seq: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read). Also used as the CPU-side lock for card copies
    /// (`RwLock` so parallel relay merges on disjoint 64 KiB pieces take
    /// shared read locks while `merge_under_writes` takes an exclusive write
    /// lock).
    before: RwLock<Option<Vec<u8>>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    /// Present when `buffer` lives in ReBAR `D3D12_HEAP_TYPE_GPU_UPLOAD` VRAM.
    rebar: Option<RebarShadow>,
    /// Monotonic tick of the last loose-cache access (0 for registered buffers).
    last_used: AtomicU64,
    /// First bit in the shader view-writes bitmap (`VICTORY_VIEW_MARKS=1`), one
    /// bit per 64 KiB title page this copy touches; `None` when view marks are
    /// off, for in-place buffers, or if the bitmap is full.
    marks: Option<u32>,
    /// Our own `MapViewOfFile` view backing a `VICTORY_STRETCH` in-place heap;
    /// must remain the last field so `buffer` and `place` drop before unmapping.
    stretch_view: Option<StretchView>,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` (1) and `VICTORY_CARD_BUFFERS_MB` (1024).
/// Run 69's gameplay has ten title buffers of 16 MiB or more, 494 MB, and
/// 1,650 MB of the card's budget free. With 16 and 512 (the old defaults)
/// run 20261124-235840 still snapshotted and compared 350-400 MB a frame
/// of shader-written ranges on the CPU; with 1 and 1024 (run 20261125-
/// 002550) 26 MB, the card finding the changed pieces itself.
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default) << 20
        };
        // From 1 MiB up and 1,024 MB in all by default (it was 16 and 512):
        // the shader writes of the smaller buffers are then found by the
        // card's hashes too, and with that the first room's walls were
        // drawn in every run.
        (mb("VICTORY_CARD_BUFFERS_MIN_MB", 1).max(1 << 20), mb("VICTORY_CARD_BUFFERS_MB", 1024))
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (on by default, `VICTORY_CARD_ROOM_FOR_BUFFERS=0` turns it off; see
/// below).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    card_limits().1.saturating_sub(CARD_BYTES[0].load(Ordering::Relaxed) + CARD_REFUSED[1].load(Ordering::Relaxed))
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|w| w.strong_count() > 0).count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none). Run
/// 20261125-002550 ran with 1024 (3.0 fps became 4.8 with title buffers'
/// copies on the card as well), but walls went missing in runs with it on
/// and stayed in run 20261125-010141 with it off at the same frame rate
/// (4.4-5.7 fps): off until that is understood.
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while a fifth of the card's budget is free, as title buffers do
/// (`card_for`): textures keep their room, and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 10 * 8
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow
            .card
            .as_ref()
            .map(|c| {
                let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let ranges = std::mem::take(&mut *dirty);
                if relay() {
                    let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.retain(|s| s.2 > fetched);
                    sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
                }
                ranges
            })
            .unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
        true
    });
    out
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// The fetch inside flights (`CARD_FETCH` in host.rs) copies each piece
    /// from `inflight_piece_from` on: what the merge takes of the piece lies
    /// in that slot.
    #[test]
    fn a_piece_fetched_in_flight_fits_its_slot() {
        let slot = crate::host::INFLIGHT_SLOT as usize;
        for start in [0usize, 4096, 7, PIECE - 16, PIECE - 3, 3 * PIECE + 0x1f09, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    let at = crate::host::inflight_piece_from(&span, i);
                    assert_eq!(at % 16, 0);
                    assert!(lo >= at && hi - at <= slot, "{start:#x} {from} {to} piece {i}: {lo}..{hi} from {at}");
                }
            }
        }
    }

    /// The slots are a power of two, one per 64 KiB asked, within the
    /// shader's 32-bit offsets.
    #[test]
    fn inflight_slots_are_a_power_of_two() {
        use crate::host::{INFLIGHT_SLOT, inflight_slots};
        assert_eq!(inflight_slots(64 << 20), 1024);
        assert_eq!(inflight_slots(100 << 20), 1024);
        assert_eq!(inflight_slots(1), 1);
        assert_eq!(inflight_slots(1 << 20), 16);
        let most = inflight_slots(u64::MAX);
        assert!(most.is_power_of_two() && u64::from(most) * INFLIGHT_SLOT <= 1 << 32);
    }

    /// Tickets drawn and landed in order, the mark raised as the landing
    /// raises it (every ticket a landed flight drew), through the counter's
    /// wrap with the slots the default gives: no two tickets in the air
    /// that got a slot share one, slots are still given after the wrap, and
    /// the mark follows the counter.
    #[test]
    fn inflight_tickets_never_share_a_slot_across_the_wrap() {
        use crate::host::{inflight_gets_slot, inflight_raise, inflight_slots};
        let slots = inflight_slots(64 << 20);
        let mut landed = u32::MAX - 3000;
        let mut next = landed;
        // Flights of 1 to 700 flagged pieces, up to three in the air:
        // (tickets drawn, those that got a slot).
        let mut air: std::collections::VecDeque<(Vec<u32>, Vec<u32>)> = Default::default();
        let mut after_wrap = 0;
        for flight in 0..40u32 {
            let drawn: Vec<u32> = (0..1 + flight * 37 % 700).map(|i| next.wrapping_add(i)).collect();
            next = next.wrapping_add(drawn.len() as u32);
            let slotted: Vec<u32> = drawn.iter().copied().filter(|&k| inflight_gets_slot(k, landed, slots)).collect();
            after_wrap += slotted.iter().filter(|&&k| k < 1 << 20).count();
            air.push_back((drawn, slotted));
            let mut used: Vec<u32> = air.iter().flat_map(|a| a.1.iter()).map(|k| k % slots).collect();
            let n = used.len();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used.len(), n, "flight {flight}: two tickets in the air share a slot");
            if air.len() == 3
                && let Some((drawn, _)) = air.pop_front()
            {
                for k in drawn {
                    landed = inflight_raise(landed, k);
                }
            }
        }
        while let Some((drawn, _)) = air.pop_front() {
            for k in drawn {
                landed = inflight_raise(landed, k);
            }
        }
        assert!(next < 1 << 20, "the counter wrapped");
        assert!(after_wrap > 0, "slots after the wrap");
        assert_eq!(landed, next);
        // Raised in wrapping order, never back.
        assert_eq!(inflight_raise(u32::MAX - 1, u32::MAX), 0);
        assert_eq!(inflight_raise(5, 2), 5);
        assert_eq!(inflight_raise(u32::MAX - 1, 3), 4);
        assert!(!inflight_gets_slot(3, 4, 8));
    }

    /// Flights 5, 6, 7 in the air with the fetch inside flights: 5 lands
    /// the old way with the queue's count at 7. The title's byte that went
    /// up in send 7 must still be told apart when 6 lands from its slot
    /// (its merge leaves out sends after 6), so send 7's note must outlive
    /// the mark (`card_dirty` keeps notes above it, `card_merge_relay`
    /// leaves out those above its "fetched after").
    #[test]
    fn a_fetch_the_old_way_keeps_the_notes_flights_in_the_air_need() {
        use crate::host::fallback_fetched_mark;
        let notes = [5u64, 6, 7];
        let mark = fallback_fetched_mark(true, 7, Some(5));
        let kept: Vec<u64> = notes.iter().copied().filter(|&s| s > mark).collect();
        assert!(kept.iter().any(|&s| s > 6), "send 7's note was dropped before flight 6 merged");
        assert_eq!(mark, 5);
        // Without the switch every later merge fetches after 7: as before.
        assert_eq!(fallback_fetched_mark(false, 7, Some(5)), 7);
        assert_eq!(fallback_fetched_mark(true, 7, None), 0);
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (on by default, as `VICTORY_RELAY=1`; `2` for its second stage,
/// `VICTORY_RELAY=0` turns it off; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Pending ranges that can already have GPU writes from before this
    /// submission. A range counted for both this submission and an older
    /// one is still protected, even when its last declaration is current.
    fn older_ranges(&self, preparing: u64, from: usize, to: usize) -> Vec<(usize, usize)> {
        joined(
            self.0
                .iter()
                .filter(|&&((a, b), count, last)| a < to && b > from && (last != preparing || count > 1))
                .map(|&((a, b), _, _)| (a, b))
                .collect(),
        )
    }

    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
pub(crate) fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// The runs of 4-byte words of title memory (cut on its addresses;
/// `read` and `merged` are of the memory at `address`) that hold a byte in
/// which `merged` differs from `read`, as (offset, bytes) in order to
/// `put`. A shader stores words, so these are what the card wrote; every
/// other byte of `merged` is the title's as it was read.
fn word_runs(address: usize, read: &[u8], merged: &[u8], mut put: impl FnMut(usize, usize)) {
    let n = read.len().min(merged.len());
    let mut i = 0;
    while i < n {
        // Equal bytes are passed over, 32 at a time while they last.
        match first_difference(&read[i..n], &merged[i..n]) {
            Some(d) => i += d,
            None => break,
        }
        // From the word that holds this byte (or the stretch's first
        // byte) to the end of the last word in a row that holds one.
        let first = i - ((address + i) & 3).min(i);
        let mut end = first;
        while end < n {
            let word = (((address + end) | 3) + 1 - address).min(n);
            if read[end..word] == merged[end..word] {
                break;
            }
            end = word;
        }
        put(first, end - first);
        i = end;
    }
}

/// Whether the merges that write a stretch of title memory back
/// ([`Shadow::relay_merge`], [`Shadow::card_merge`],
/// [`Shadow::card_merge_relay`]) look, right before that write, for title
/// bytes it is about to put back to older values (`VICTORY_MERGE_RACE=1`;
/// off by default). Logging only: [`merge_race`].
fn merge_race_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_RACE").is_some_and(|v| v == "1"))
}

/// [`merge_race`]'s sums. Of frame `frame`, and in `total` since the
/// start: stretches written back, their bytes, title bytes put back to
/// older values, bytes the title and the card both changed, stretches with
/// either, the runs a write of the card's words alone makes
/// ([`word_runs`]) and their bytes. `said`: the frame's first stretches
/// with either. `quiet`: frames since the last line.
struct MergeRace {
    frame: u64,
    now: [u64; 7],
    said: Vec<String>,
    total: [u64; 7],
    quiet: u64,
}

static MERGE_RACE: Mutex<MergeRace> =
    Mutex::new(MergeRace { frame: 0, now: [0; 7], said: Vec::new(), total: [0; 7], quiet: 0 });

/// With [`merge_race_on`], right before a merge writes `merged` over title
/// memory at `address` (a stretch of the copy of the title buffer `copy`:
/// start and bytes): title memory is read once more and compared with
/// `read`, what the merge read of it before. A byte that changed in
/// between and that the write-back stores with its value as first read
/// (the card did not change it) is a title byte put back to an older
/// value; one the write-back stores with the card's byte is one both
/// changed. Only bytes the write-back stores are looked at. Summed per
/// frame; a line for every frame that has any, and one every 100 frames
/// whatever it holds, so that "none" is said, not assumed.
///
/// A lower bound: a title write that lands after this read (while the
/// write-back itself runs) is not seen, nor one that wrote a byte and then
/// its old value again. Costs one more read and two compares of every
/// stretch written back, on the thread that merges.
fn merge_race(read: Option<&[u8]>, copy: (u64, u64), address: usize, merged: &[u8], which: &'static str) {
    if !merge_race_on() {
        return;
    }
    let Some(read) = read else {
        return;
    };
    let n = read.len().min(merged.len());
    let (read, merged) = (&read[..n], &merged[..n]);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    word_runs(address, read, merged, |off, len| runs.push((off, len)));
    let whole = [(0, n)];
    let looked: &[(usize, usize)] = &whole;
    let mut again = vec![0u8; n];
    let (mut put_back, mut both, mut first) = (0u64, 0u64, None);
    if crate::host::read_into(address as *const u8, &mut again) {
        for &(off, len) in looked {
            for i in off..off + len {
                if again[i] == read[i] || again[i] == merged[i] {
                    continue;
                }
                if merged[i] == read[i] {
                    put_back += 1;
                    first.get_or_insert(i);
                } else {
                    both += 1;
                }
            }
        }
    }
    let frame = crate::frame::current();
    let mut race = MERGE_RACE.lock().unwrap_or_else(|e| e.into_inner());
    if race.frame != frame {
        race.quiet += 1;
        if race.now[4] > 0 || race.quiet >= 100 {
            tracing::info!(
                frame = race.frame,
                stretches = race.now[0],
                stretch_bytes = race.now[1],
                put_back = race.now[2],
                both_changed = race.now[3],
                stretches_hit = race.now[4],
                word_runs = race.now[5],
                word_run_bytes = race.now[6],
                stretches_since_start = race.total[0],
                stretch_bytes_since_start = race.total[1],
                put_back_since_start = race.total[2],
                both_changed_since_start = race.total[3],
                stretches_hit_since_start = race.total[4],
                word_runs_since_start = race.total[5],
                word_run_bytes_since_start = race.total[6],
                first = ?race.said,
                "merge race: title bytes that changed between a merge's read of title memory and its write-back of what shaders wrote, in this frame (VICTORY_MERGE_RACE=1)"
            );
            race.quiet = 0;
        }
        race.frame = frame;
        race.now = [0; 7];
        race.said.clear();
    }
    let hit = u64::from(put_back + both > 0);
    let run_bytes: u64 = runs.iter().map(|r| r.1 as u64).sum();
    let add = [1, n as u64, put_back, both, hit, runs.len() as u64, run_bytes];
    for (i, v) in add.iter().enumerate() {
        race.now[i] += v;
        race.total[i] += v;
    }
    if hit > 0 && race.said.len() < 8 {
        race.said.push(format!(
            "{which}: copy {:#x}+{:#x} stretch {address:#x}+{n:#x}: {put_back} put back (first at {:#x}), {both} both changed",
            copy.0,
            copy.1,
            address + first.unwrap_or(0)
        ));
    }
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

std::thread_local! {
    /// Set while a thread is executing a job inside [`wide`] so nested callers
    /// never re-enter the shared worker pool and deadlock.
    static IN_WIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct InWideGuard(bool);

impl InWideGuard {
    fn enter() -> Self {
        Self(IN_WIDE.with(|flag| flag.replace(true)))
    }
}

impl Drop for InWideGuard {
    fn drop(&mut self) {
        IN_WIDE.with(|flag| flag.set(self.0));
    }
}

fn in_wide() -> bool {
    IN_WIDE.with(|flag| flag.get())
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        let n = std::thread::available_parallelism().map_or(4, |n| n.get() / 2).clamp(2, 6);
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                let workers = WORKERS.get().expect("made above");
                loop {
                    let job = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let _guard = InWideGuard::enter();
                    job();
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            let _guard = InWideGuard::enter();
            job();
        }
        return;
    }
    let workers = workers();
    let left = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => {
                let _guard = InWideGuard::enter();
                job();
            }
            None => break,
        }
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// Whether large shadow refreshes and staging fills parallelize their dirty
/// copy chunks across worker threads (`VICTORY_WIDE_COPY=0` disables it;
/// `=1` forces it; on by default on 4+ logical CPUs).
pub(crate) fn wide_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_copy_enabled(
            std::env::var("VICTORY_WIDE_COPY").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_copy_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

/// Minimum total dirty bytes to split across worker threads in [`run_copy_segs`].
const WIDE_COPY_MIN: usize = 1 << 20;
/// Maximum target byte size per parallel copy chunk.
const WIDE_COPY_CHUNK: usize = 512 << 10;

/// One contiguous byte range to copy from title memory into a shadow copy
/// (and optionally through a cached ReBAR RAM mirror into WC VRAM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopySeg {
    src: usize,
    dst: usize,
    /// Non-zero for mirrored ReBAR shadows: copy `src -> mirror`, then
    /// `mirror -> dst` (WC VRAM).
    mirror: usize,
    len: usize,
}

/// Slices `segs` into buckets of at most `chunk_bytes` each, splitting long
/// runs cleanly without losing or duplicating a single byte.
fn chunk_copy_segs(segs: &[CopySeg], chunk_bytes: usize) -> Vec<Vec<CopySeg>> {
    let chunk_bytes = chunk_bytes.max(1);
    let mut buckets: Vec<Vec<CopySeg>> = Vec::new();
    let mut cur: Vec<CopySeg> = Vec::new();
    let mut bytes = 0usize;
    for &seg in segs {
        let mut at = 0usize;
        while at < seg.len {
            let n = (chunk_bytes - bytes).min(seg.len - at);
            cur.push(CopySeg {
                src: seg.src + at,
                dst: seg.dst + at,
                mirror: if seg.mirror != 0 { seg.mirror + at } else { 0 },
                len: n,
            });
            at += n;
            bytes += n;
            if bytes >= chunk_bytes {
                buckets.push(std::mem::take(&mut cur));
                bytes = 0;
            }
        }
    }
    if !cur.is_empty() {
        buckets.push(cur);
    }
    buckets
}

#[inline]
fn execute_copy_seg(seg: CopySeg) {
    if seg.len == 0 {
        return;
    }
    if seg.mirror != 0 {
        crate::host::copy_mapped(seg.mirror as *mut u8, seg.src as *const u8, seg.len, false);
        // SAFETY: `seg` lies inside the shadow's mirror and WC VRAM mapping
        // (`off + len <= self.len`), and chunks are strictly disjoint.
        unsafe {
            std::ptr::copy_nonoverlapping(
                seg.mirror as *const u8,
                seg.dst as *mut u8,
                seg.len,
            );
        }
    } else {
        crate::host::copy_mapped(seg.dst as *mut u8, seg.src as *const u8, seg.len, false);
    }
}

/// Executes copy jobs on the warm `workers()` pool when called from outside
/// [`wide`], or via a bounded `std::thread::scope` if called re-entrantly from
/// inside a `wide` worker so the pool can never deadlock.
fn run_copy_jobs(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    if !in_wide() {
        wide(jobs);
        return;
    }
    // Re-entrant call from inside a `wide` worker: never push back into the
    // pool queue; fan out with a small scoped group instead.
    let threads = jobs.len().min(4);
    let queue = Mutex::new(std::collections::VecDeque::from(jobs));
    std::thread::scope(|s| {
        for _ in 1..threads {
            let queue = &queue;
            s.spawn(move || {
                while let Some(job) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front() {
                    job();
                }
            });
        }
        while let Some(job) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front() {
            job();
        }
    });
}

fn run_copy_segs(segs: &[CopySeg]) {
    let total: usize = segs.iter().map(|s| s.len).sum();
    if total == 0 {
        return;
    }
    if !wide_copy() || total < WIDE_COPY_MIN {
        for &seg in segs {
            execute_copy_seg(seg);
        }
        return;
    }
    let chunk_bytes = (total / 4).clamp(256 << 10, WIDE_COPY_CHUNK);
    let buckets = chunk_copy_segs(segs, chunk_bytes);
    if buckets.len() < 2 {
        for &seg in segs {
            execute_copy_seg(seg);
        }
        return;
    }
    let jobs: Vec<Job> = buckets
        .into_iter()
        .map(|bucket| {
            Box::new(move || {
                for seg in bucket {
                    execute_copy_seg(seg);
                }
            }) as Job
        })
        .collect();
    run_copy_jobs(jobs);
}

/// Copies `len` bytes from title memory `src` into writable host memory `dst`,
/// splitting large fills across worker threads when [`wide_copy`] is enabled.
pub(crate) fn copy_mapped_wide(dst: *mut u8, src: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    if !wide_copy() || len < WIDE_COPY_MIN {
        crate::host::copy_mapped(dst, src, len, false);
        return;
    }
    run_copy_segs(&[CopySeg {
        src: src as usize,
        dst: dst as usize,
        mirror: 0,
        len,
    }]);
}

/// Dirty byte ranges discovered by [`Shadow::plan_update`] before copying.
#[derive(Default)]
struct UpdatePlan {
    runs: Vec<(usize, usize)>,
    busy_ranges: Vec<(usize, usize)>,
    watched_ranges: Vec<(usize, usize)>,
    #[cfg(feature = "diagnostics")]
    unwatched_ranges: Vec<(usize, usize)>,
}

/// Two-phase refresh for a [`crate::host::Batch`] flush:
/// 1. Discover stale byte runs for each shadow (in parallel across shadows when
///    `parallel_plan` is enabled and there are at least 4 shadows), releasing
///    each shadow's `copied` lock immediately after planning.
/// 2. Flatten all non-mirrored dirty runs across the entire batch into balanced
///    copy chunks and execute them in one flat [`wide`] pass, then apply
///    per-shadow metadata updates (`changed`, `card_changed`, ReBAR dirty bits)
///    and copy any mirrored ReBAR shadows.
///
/// This eliminates both straggler single-thread bottlenecks (one huge 50–174 MB
/// buffer stalling a multi-shadow flush) and nested `wide` re-entrancy.
pub(crate) fn refresh_batch(
    items: Vec<(Arc<Shadow>, usize, usize)>,
    parallel_plan: bool,
) -> usize {
    if items.is_empty() {
        return 0;
    }
    if items.len() == 1 {
        let (shadow, from, to) = items.into_iter().next().unwrap();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, started.elapsed().as_micros() as u64, 0);
        return copied;
    }

    #[cfg(feature = "diagnostics")]
    let batch_started = std::time::Instant::now();
    let plan_started = std::time::Instant::now();

    let count = items.len();
    let planned: Vec<(Arc<Shadow>, usize, usize, UpdatePlan)> =
        if parallel_plan && count >= 4 && !in_wide() {
            let slots: Arc<Vec<Mutex<Option<(Arc<Shadow>, usize, usize, UpdatePlan)>>>> =
                Arc::new((0..count).map(|_| Mutex::new(None)).collect());
            let jobs: Vec<Job> = items
                .into_iter()
                .enumerate()
                .map(|(i, (shadow, from, to))| {
                    let slots = slots.clone();
                    Box::new(move || {
                        let plan = shadow.plan_update(from, to - from);
                        *slots[i].lock().unwrap_or_else(|e| e.into_inner()) =
                            Some((shadow, from, to, plan));
                    }) as Job
                })
                .collect();
            wide(jobs);
            slots
                .iter()
                .filter_map(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).take())
                .collect()
        } else {
            items
                .into_iter()
                .map(|(shadow, from, to)| {
                    let plan = shadow.plan_update(from, to - from);
                    (shadow, from, to, plan)
                })
                .collect()
        };
    crate::host::PREPARING_COPIES[crate::host::PREP_WATCH_US]
        .fetch_add(plan_started.elapsed().as_micros() as u64, Ordering::Relaxed);

    let copy_started = std::time::Instant::now();
    let mut all_segs: Vec<CopySeg> = Vec::new();
    let mut non_mirrored: Vec<(usize, usize, usize)> = Vec::new();
    let mut mirrored: Vec<usize> = Vec::new();
    let mut total_copied = 0usize;

    for (idx, (shadow, _, _, plan)) in planned.iter().enumerate() {
        if plan.runs.is_empty() {
            continue;
        }
        if shadow.rebar.as_ref().is_some_and(RebarShadow::has_mirror) {
            mirrored.push(idx);
            continue;
        }
        let seg_lo = all_segs.len();
        let bytes = shadow.push_copy_segs(&plan.runs, &mut all_segs);
        let seg_hi = all_segs.len();
        if bytes > 0 {
            non_mirrored.push((idx, seg_lo, seg_hi));
            total_copied += bytes;
        }
    }

    if !all_segs.is_empty() {
        run_copy_segs(&all_segs);
        for &(idx, seg_lo, seg_hi) in &non_mirrored {
            let shadow = &planned[idx].0;
            shadow.finish_copy_segs(shadow.start as usize, &all_segs[seg_lo..seg_hi]);
        }
    }

    for idx in mirrored {
        let (shadow, _, _, plan) = &planned[idx];
        total_copied += shadow.copy_runs(&plan.runs);
    }

    for (shadow, _, _, plan) in &planned {
        for &(a, b) in &plan.busy_ranges {
            total_copied += shadow.update_busy(a, b);
        }
    }

    let copy_us = copy_started.elapsed().as_micros() as u64;
    crate::host::PREPARING_COPIES[crate::host::PREP_REFRESH_COPY_US]
        .fetch_add(copy_us, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    let batch_us = batch_started.elapsed().as_micros() as u64;

    for (shadow, from, to, plan) in &planned {
        for &(a, b) in &plan.watched_ranges {
            shadow.refresh_check(a, b);
        }
        #[cfg(feature = "diagnostics")]
        {
            for &(a, b) in &plan.unwatched_ranges {
                crate::copystat::whole(a, b - a, copy_us);
            }
            crate::copystat::range(*from, to.saturating_sub(*from), batch_us, 0);
        }
        let _ = (from, to);
    }

    total_copied
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// An older flight may have written bytes between title changes. Only
/// send exact changes there; otherwise a changed page slice goes up as one
/// range so adjacent slices can join into a single upload.
fn card_busy_runs(title: &[u8], held: &[u8], exact: bool) -> impl Iterator<Item = (usize, usize)> + use<> {
    let (whole, fragments) = if exact {
        (None, changed_runs(title, held))
    } else {
        ((title != held).then_some((0, title.len())), Vec::new())
    };
    whole.into_iter().chain(fragments)
}

/// `VICTORY_LATE_MERGE_GAP` (experiment, 0): changed runs of a page of a
/// copy in the card's memory that lie at most this many bytes apart go up
/// as one (each run is a copy of its own on the card: 88,948 a frame for
/// 1 MB in gameplay, run 35, the title's constants differing a few bytes
/// at a time). The equal bytes between them go up with them: a byte a
/// flight's shader wrote there meanwhile would be put back to what the
/// title has.
fn merge_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_LATE_MERGE_GAP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// `runs` (in order) with those at most `gap` bytes apart joined.
fn join_runs(runs: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    if gap == 0 {
        return runs;
    }
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, n) in runs {
        match out.last_mut() {
            Some(last) if at <= last.0 + last.1 + gap => last.1 = at + n - last.0,
            _ => out.push((at, n)),
        }
    }
    out
}

#[cfg(test)]
mod merge_tests {
    use super::{PIECE, back_piece, changed_runs};

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
        if let Some(rebar) = &self.rebar {
            REBAR_SHADOW_BYTES.fetch_sub(rebar.bytes, Ordering::Relaxed);
        }
        if let Some(base) = self.marks {
            marks().give_back(base, mark_pages(self.start, self.len));
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        loose
            .entries
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    for s in loose.entries.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.iter().filter(|&&at| at != 0).count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut guard = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if guard.entries.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        let idle: Vec<(u64, u64)> = guard
            .entries
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        idle.into_iter()
            .filter_map(|key| {
                let s = guard.entries.remove(&key)?;
                guard.total_bytes = guard.total_bytes.saturating_sub(s.len);
                Some(s)
            })
            .collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .entries
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new_inner(placed.buffer, start as *mut u8, placed.gpu, start, len, None, false);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Clamps `len` to the bytes remaining in this copy from `address` on.
    #[inline]
    pub(crate) fn clamp_len(&self, address: u64, len: u64) -> u64 {
        len.min((self.start + self.len).saturating_sub(address))
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = self.clamp_len(address, len);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        rebar_bytes: Option<u64>,
    ) -> Shadow {
        Self::new_inner(buffer, data, gpu, start, len, rebar_bytes, view_marks_on())
    }

    fn new_inner(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        rebar_bytes: Option<u64>,
        allocate_marks: bool,
    ) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        let mark_base = allocate_marks
            .then(|| marks().take(mark_pages(start, len)))
            .flatten();
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            born_seq: crate::host::birth_seq(),
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: RwLock::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            place: None,
            rebar: rebar_bytes.map(|bytes| RebarShadow {
                bytes,
                mirror: Mutex::new(None),
                hashes: Mutex::new(None),
            }),
            last_used: AtomicU64::new(0),
            marks: mark_base,
            stretch_view: None,
        }
    }

    /// First bit in the shader view-writes bitmap for this copy, if allocated.
    pub(crate) fn mark_base(&self) -> Option<u32> {
        self.marks
    }

    /// The view-mark table entry `(base_bit, phase)` for an unordered buffer
    /// view of this copy starting at title address `start`.
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through unordered buffer views of this copy are marked
    /// in the shader view-writes bitmap.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// Filters `declared_title_ranges` down to only the 64 KiB title pages
    /// marked in any of `bitmaps` (union with `unmarked_title_ranges`),
    /// returning `(filtered_title_ranges, declared_pages, marked_pages)`.
    pub(crate) fn filter_marked_spans(
        &self,
        declared_title_ranges: &[(usize, usize)],
        unmarked_title_ranges: &[(usize, usize)],
        bitmaps: &[&[u32]],
    ) -> (Vec<(usize, usize)>, u64, u64) {
        filter_title_ranges_by_marks(
            self.start,
            self.len,
            self.marks,
            declared_title_ranges,
            unmarked_title_ranges,
            bitmaps,
        )
    }

    fn touch(&self) {
        self.last_used.store(
            LOOSE_TICK.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    /// Whether this copy lives in ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) VRAM.
    pub(crate) fn is_rebar(&self) -> bool {
        self.rebar.is_some()
    }

    /// Logical byte length of the title buffer.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Base host GPU virtual address of the shadow buffer.
    pub(crate) fn base_gpu(&self) -> u64 {
        self.gpu
    }

    /// Lazily allocates the per-64-KiB-piece hash UAV buffer for a ReBAR shadow
    /// and returns [`PreparedRebarHashes`], where `pre_runs` are the
    /// `(first_piece, piece_count)` runs that must be hashed before the
    /// submission executes (all pieces on initial creation, or only
    /// CPU-dirtied pieces thereafter). Returns `None` if this shadow is not
    /// eligible for GPU hashing or allocation fails.
    pub(crate) fn prepare_rebar_hashes(
        &self,
        alloc: impl FnOnce(u64) -> Option<ID3D12Resource>,
    ) -> Option<PreparedRebarHashes> {
        let rebar = self.rebar.as_ref()?;
        if self.len == 0 {
            return None;
        }
        let pieces = u32::try_from(self.len.div_ceil(PAGE as u64)).ok()?;
        if pieces > REBAR_HASH_MAX_PIECES {
            return None;
        }
        let aligned_len = u32::try_from(self.len.next_multiple_of(16)).ok()?;
        if rebar
            .mirror
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            return None;
        }
        {
            let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(h) = guard.as_mut() {
                return Some(PreparedRebarHashes {
                    hashes_gpu: h.gpu,
                    aligned_len,
                    pre_runs: h.dirty.take_runs(),
                });
            }
        }
        // Allocate outside the leaf lock.
        let buffer = alloc(u64::from(pieces) * 16)?;
        crate::host::name(&buffer, || {
            format!("ReBAR hashes {:#x}+{:#x}", self.start, self.len)
        });
        // SAFETY: live D3D12 buffer resource.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            return Some(PreparedRebarHashes {
                hashes_gpu: h.gpu,
                aligned_len,
                pre_runs: h.dirty.take_runs(),
            });
        }
        *guard = Some(RebarHashes {
            _buffer: buffer,
            gpu,
            dirty: RebarDirtyBits::new(pieces),
        });
        Some(PreparedRebarHashes {
            hashes_gpu: gpu,
            aligned_len,
            pre_runs: vec![(0, pieces)],
        })
    }

    /// Marks `(first_piece, piece_count)` runs dirty in this ReBAR shadow's
    /// hash state (e.g. if a pre-hash recording failed after draining dirty
    /// bits).
    pub(crate) fn mark_rebar_dirty_runs(&self, runs: &[(u32, u32)]) {
        let Some(rebar) = &self.rebar else {
            return;
        };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for &(first, count) in runs {
                h.dirty.mark_pieces(first, count);
            }
        }
    }

    /// Marks all 64 KiB pieces overlapping buffer-offset `spans` dirty in this
    /// ReBAR shadow's hash state (used when a submission falls back to
    /// full-range readback without running the post-compare shader).
    pub(crate) fn mark_rebar_dirty_spans(&self, spans: &[(usize, usize)]) {
        let Some(rebar) = &self.rebar else {
            return;
        };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for &(from, to) in spans {
                if from < to {
                    h.dirty.mark_byte_range(from, to - from);
                }
            }
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    #[track_caller]
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.gpu_writes_marked(list, address, len, false);
    }

    /// [`Shadow::gpu_writes`], with `marked` indicating that the shader marks
    /// the 64 KiB pages it writes through unordered buffer views in the
    /// view-writes bitmap (`VICTORY_VIEW_MARKS=1`).
    #[track_caller]
    pub(crate) fn gpu_writes_marked(
        self: &Arc<Self>,
        list: &HostList,
        address: u64,
        len: u64,
        marked: bool,
    ) {
        if crate::host::bare() & 8 != 0 {
            return;
        }
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        let len = len.min(self.left(address));
        if len >= 32 << 20 {
            // Experiment: who declares the big ranges.
            static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 12 || n.is_multiple_of(2000) {
                let (start, span) = self.span();
                tracing::info!(
                    "big-write #{n}: {} MB declared written at {address:#x} (copy {start:#x}+{span:#x}) from {}",
                    len >> 20,
                    std::panic::Location::caller()
                );
            }
        }
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(
        &self,
        address: usize,
        len: usize,
        rebar_init: impl FnOnce(&ID3D12Resource, u64) -> Option<Vec<u8>>,
    ) {
        if self.place.is_some() {
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        if let Some(rebar) = &self.rebar {
            pending.push(range);
            drop(pending);
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if mirror.is_none() {
                *mirror = rebar_init(&self.buffer, self.len);
            }
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.write().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
    }

    /// Clears pending UAV-write ranges for a ReBAR shadow if readback failed.
    pub(crate) fn clear_pending_rebar(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// [`relay`] counterpart to [`Shadow::copy_back_rebar`]: merges GPU bytes
    /// DMAed by the relay thread (`gpu_slices`, for sorted disjoint buffer-offset
    /// `spans`) into title memory and the cached RAM mirror under `rebar.mirror`'s
    /// lock. Never writes back to WC VRAM (a later flight may be writing the GPU
    /// buffer concurrently) and never clears `pending` (handled per-flight by
    /// [`Shadow::copy_back_done`]).
    pub(crate) fn relay_merge_rebar(&self, spans: &[(usize, usize)], gpu_slices: &[&[u8]]) -> usize {
        const PARALLEL_THRESHOLD: usize = 4 << 20;
        const PER_THREAD: usize = 4 << 20;

        let Some(rebar) = &self.rebar else {
            return 0;
        };
        let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mirror) = mirror_guard.as_mut() else {
            return 0;
        };
        let mirror_len = mirror.len();
        let total_bytes: usize = spans
            .iter()
            .zip(gpu_slices)
            .filter(|&(&(from, to), &gpu_span)| {
                from < to && to <= mirror_len && gpu_span.len() == to - from
            })
            .map(|(&(from, to), _)| to - from)
            .sum();
        if total_bytes == 0 {
            return 0;
        }

        let start = self.start as usize;
        let apply_subrange =
            |sub_at: usize, gpu: &[u8], old: &mut [u8], title: &mut Vec<u8>| -> usize {
                let sub_n = gpu.len();
                title.clear();
                title.resize(sub_n, 0);
                if crate::host::read_into((start + sub_at) as *const u8, title) {
                    let as_read = merge_race_on().then(|| title.clone());
                    if merge_piece(gpu, old, title) {
                        crate::host::shaders_write(start + sub_at, &title[..]);
                        #[cfg(feature = "diagnostics")]
                        back_note(start + sub_at, &title[..sub_n], self.start, self.len);
                        if self.put_back_own(start + sub_at, &title[..]) {
                            return sub_n;
                        }
                        crate::watch::writing(start + sub_at, sub_n);
                        merge_race(
                            as_read.as_deref(),
                            (self.start, self.len),
                            start + sub_at,
                            &title[..],
                            "rebar relay",
                        );
                        crate::host::write_mapped(
                            (start + sub_at) as *mut u8,
                            title.as_ptr(),
                            sub_n,
                        );
                        return sub_n;
                    }
                } else {
                    old.copy_from_slice(gpu);
                }
                0
            };

        let mut written = 0;
        let mut title = Vec::new();
        if total_bytes < PARALLEL_THRESHOLD {
            for (&(from, to), &gpu_span) in spans.iter().zip(gpu_slices) {
                if from >= to || to > mirror_len || gpu_span.len() != to - from {
                    continue;
                }
                let mut at = from;
                while at < to {
                    let n = (PAGE - (start + at) % PAGE).min(to - at);
                    let gpu = &gpu_span[at - from..at - from + n];
                    let old = &mut mirror[at..at + n];
                    if gpu != old
                        && let Some(first) = first_difference(gpu, old)
                    {
                        let last = last_difference(gpu, old).unwrap_or(first);
                        written += apply_subrange(
                            at + first,
                            &gpu[first..=last],
                            &mut old[first..=last],
                            &mut title,
                        );
                    }
                    at += n;
                }
            }
            return written;
        }

        let threads = total_bytes.div_ceil(PER_THREAD).clamp(2, 8);
        let target = total_bytes.div_ceil(threads);
        let mut buckets: Vec<Vec<(usize, usize, usize, usize)>> = Vec::with_capacity(threads);
        let mut cur_bucket: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut cur_bytes = 0usize;
        for (slice_idx, (&(from, to), &gpu_span)) in spans.iter().zip(gpu_slices).enumerate() {
            if from >= to || to > mirror_len || gpu_span.len() != to - from {
                continue;
            }
            let mut at = from;
            while at < to {
                if buckets.len() + 1 >= threads {
                    cur_bucket.push((slice_idx, from, at, to));
                    break;
                }
                let room = target.saturating_sub(cur_bytes).max(1);
                let next = if to - at <= room {
                    to
                } else {
                    let raw = at + room;
                    let rem = (start + raw) % PAGE;
                    if rem == 0 {
                        raw
                    } else {
                        raw.saturating_add(PAGE - rem).min(to)
                    }
                };
                cur_bucket.push((slice_idx, from, at, next));
                cur_bytes += next - at;
                at = next;
                if cur_bytes >= target && buckets.len() + 1 < threads {
                    buckets.push(std::mem::take(&mut cur_bucket));
                    cur_bytes = 0;
                }
            }
        }
        if !cur_bucket.is_empty() {
            buckets.push(cur_bucket);
        }

        let diffs = {
            let mirror_ro: &[u8] = mirror;
            let scan_bucket = |bucket: Vec<(usize, usize, usize, usize)>|
             -> Vec<(usize, usize, usize, usize)> {
                let mut found = Vec::new();
                for (slice_idx, span_from, seg_from, seg_to) in bucket {
                    let gpu_span = gpu_slices[slice_idx];
                    let mut at = seg_from;
                    while at < seg_to {
                        let n = (PAGE - (start + at) % PAGE).min(seg_to - at);
                        let span_off = at - span_from;
                        let gpu = &gpu_span[span_off..span_off + n];
                        let old = &mirror_ro[at..at + n];
                        if gpu != old
                            && let Some(first) = first_difference(gpu, old)
                        {
                            let last = last_difference(gpu, old).unwrap_or(first);
                            found.push((
                                at + first,
                                last + 1 - first,
                                slice_idx,
                                span_off + first,
                            ));
                        }
                        at += n;
                    }
                }
                found
            };
            std::thread::scope(|s| {
                let mut iter = buckets.into_iter();
                let first = iter.next().unwrap_or_default();
                let scan_bucket = &scan_bucket;
                let handles: Vec<_> = iter
                    .map(|bucket| s.spawn(move || scan_bucket(bucket)))
                    .collect();
                let mut out = scan_bucket(first);
                for handle in handles {
                    match handle.join() {
                        Ok(part) => out.extend(part),
                        Err(panic) => std::panic::resume_unwind(panic),
                    }
                }
                out
            })
        };

        for (sub_at, sub_n, slice_idx, slice_off) in diffs {
            let gpu = &gpu_slices[slice_idx][slice_off..slice_off + sub_n];
            let old = &mut mirror[sub_at..sub_at + sub_n];
            written += apply_subrange(sub_at, gpu, old, &mut title);
        }
        written
    }

    /// ReBAR counterpart to [`Shadow::copy_back`]: merges post-submission GPU
    /// bytes (`gpu_slices`, already DMAed into cached `READBACK` RAM) for
    /// sorted disjoint buffer-offset `spans` against the cached RAM mirror,
    /// without ever reading write-combining VRAM on the CPU.
    pub(crate) fn copy_back_rebar(&self, spans: &[(usize, usize)], gpu_slices: &[&[u8]]) -> usize {
        let Some(rebar) = &self.rebar else {
            return 0;
        };
        let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let mut vram_dirty: Vec<(usize, usize)> = Vec::new();
        let start = self.start as usize;
        if let Some(mirror) = mirror_guard.as_mut() {
            let mut title = None;
            for (&(from, to), &gpu_span) in spans.iter().zip(gpu_slices) {
                if from >= to || to > mirror.len() || gpu_span.len() != to - from {
                    continue;
                }
                let mut synced = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PAGE - (start + at) % PAGE).min(to - at);
                    let gpu = &gpu_span[at - from..at - from + n];
                    let old = &mut mirror[at..at + n];
                    if gpu != old {
                        let title_buf = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n];
                        if crate::host::read_into((start + at) as *const u8, title_buf) {
                            let mut changed = false;
                            let mut host_differs = false;
                            for ((t, &g), o) in title_buf.iter_mut().zip(gpu).zip(old.iter_mut()) {
                                if g != *o && *t != g {
                                    *t = g;
                                    changed = true;
                                }
                                if *t != g {
                                    host_differs = true;
                                }
                                *o = *t;
                            }
                            let mut used_own = false;
                            if changed {
                                crate::host::shaders_write(start + at, title_buf);
                                #[cfg(feature = "diagnostics")]
                                back_note(start + at, title_buf, self.start, self.len);
                                used_own = self.put_back_own(start + at, title_buf);
                                if !used_own {
                                    crate::watch::writing(start + at, n);
                                    crate::host::write_mapped(
                                        (start + at) as *mut u8,
                                        title_buf.as_ptr(),
                                        n,
                                    );
                                }
                                written += n;
                            }
                            if host_differs {
                                // Only push back to WC VRAM when a concurrent CPU
                                // write altered bytes in the same page; record the
                                // range so its GPU piece hash is refreshed next.
                                // SAFETY: `at + n <= self.len`, write-only WC store.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        title_buf.as_ptr(),
                                        self.data.add(at),
                                        n,
                                    );
                                }
                                vram_dirty.push((at, n));
                            }
                            if !(used_own && !host_differs) {
                                synced.push((start + at) / PAGE);
                            }
                        } else {
                            // Unmapped in title memory: VRAM holds `gpu`, keep mirror in sync.
                            old.copy_from_slice(gpu);
                        }
                    }
                    at += n;
                }
                let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
                if first < last
                    && !synced.is_empty()
                    && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                    && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
                {
                    let base = start / PAGE;
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                        copied[page - base] = now;
                    }
                }
            }
        }
        drop(mirror_guard);
        for (at, n) in vram_dirty {
            rebar.mark_dirty_range(at, n);
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        written
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.read().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                let at = at + first;
                crate::host::shaders_write(start + at, &title[..n.min(title.len())]);
                #[cfg(feature = "diagnostics")]
                back_note(start + at, &title[..n], self.start, self.len);
                // Only the GPU's bytes (upstream #2009): the rest of the
                // stretch is the title's as read, and a title write landing
                // there since stays.
                let mut opened = false;
                for &(off, len) in &runs {
                    if !self.put_back_own(start + at + off, &title[off..off + len]) {
                        if !opened {
                            crate::watch::writing(start + at, n);
                            opened = true;
                        }
                        crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                    }
                    written += len;
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.write().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = merge_race_on().then(|| title.clone());
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        #[cfg(feature = "diagnostics")]
        back_note(start + at, &title[..n], self.start, self.len);
        if self.put_back_own(start + at, &title[..]) {
            return n;
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "pc relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    /// With `crate::inplace::back_own` (`VICTORY_BACK_OWN_VIEW=1`): stores
    /// what a merge made of title memory `[at, at + bytes.len())` (inside
    /// one 64 KiB page, as every merged stretch is) through a view of our
    /// own, tells the journal, and notes the page current for this copy if
    /// it was current before: the copy holds the same bytes (the merge put
    /// them into both), so the next refresh need not take the page again.
    /// The guards are not touched: a title write to the page still faults
    /// and is reported at a later asking, which every copy then takes,
    /// this one too. False with nothing done if the switch is off or the
    /// memory cannot be written that way (the caller writes as before).
    fn put_back_own(&self, at: usize, bytes: &[u8]) -> bool {
        /// Since the start: stretches and bytes stored this way, pages
        /// noted current for their copy, stretches written the old way;
        /// and with `VICTORY_BACK_OWN_CHECK=1`: stores the title's own
        /// address did not show afterwards, pages noted current whose
        /// CPU side in the copy differs from title memory, and those of
        /// them whose guard was clean (a hole in the rule: a title write
        /// on its way leaves the guard open).
        static OWN_WRITES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
        static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let check = *CHECK
            .get_or_init(|| std::env::var_os("VICTORY_BACK_OWN_CHECK").is_some_and(|v| v == "1"));
        if !crate::inplace::back_own() {
            return false;
        }
        let Some(shown) = crate::inplace::write_own(at, bytes) else {
            OWN_WRITES[3].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        if check {
            // The title's own address must show what went in through ours.
            let mut back = vec![0u8; bytes.len()];
            if crate::host::read_into(at as *const u8, &mut back) && back != bytes {
                OWN_WRITES[4].fetch_add(1, Ordering::Relaxed);
            }
        }
        let page = at / PAGE;
        let others: Vec<usize> = shown.iter().map(|&a| a / PAGE).filter(|&p| p != page).collect();
        let (before, now) = crate::watch::wrote_own(page, &others);
        // A placed texture there changed with no descriptor written (what
        // `crate::watch::writing` tells the texture table on the old way).
        crate::descriptor::host::remapped(at, bytes.len());
        let base = self.start as usize / PAGE;
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(taken) = page.checked_sub(base).and_then(|i| copied.get_mut(i))
                && *taken != 0
                && *taken >= before
            {
                *taken = (*taken).max(now);
                OWN_WRITES[2].fetch_add(1, Ordering::Relaxed);
                // A copy in the card's memory only: its CPU side is never
                // written by the card, so it must equal title memory now.
                // (ReBAR shadows keep WC VRAM in `self.data`, never CPU-read.)
                if check && self.card.is_some() {
                    let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                    let (lo, hi) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
                    if lo < hi {
                        let mut title = vec![0u8; hi - lo];
                        // SAFETY: inside the copy's mapping; read only.
                        let held = unsafe {
                            std::slice::from_raw_parts(
                                self.data.add(lo - start).cast_const(),
                                hi - lo,
                            )
                        };
                        if crate::host::read_into(lo as *const u8, &mut title) && title != held {
                            OWN_WRITES[5].fetch_add(1, Ordering::Relaxed);
                            if crate::watch::clean(page) {
                                OWN_WRITES[6].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        }
        OWN_WRITES[1].fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let n = OWN_WRITES[0].fetch_add(1, Ordering::Relaxed) + 1;
        if n % 16384 == 0 {
            tracing::info!(
                "what shaders wrote, stored through views of our own (VICTORY_BACK_OWN_VIEW=1), since the start: {n} stretches {} MB, {} pages noted current for the copy that wrote them (not copied in again), {} stretches written the old way; with VICTORY_BACK_OWN_CHECK=1: {} stores the title's address did not show, {} pages noted current whose CPU side differs from title memory, {} of those with the guard clean",
                OWN_WRITES[1].load(Ordering::Relaxed) >> 20,
                OWN_WRITES[2].load(Ordering::Relaxed),
                OWN_WRITES[3].load(Ordering::Relaxed),
                OWN_WRITES[4].load(Ordering::Relaxed),
                OWN_WRITES[5].load(Ordering::Relaxed),
                OWN_WRITES[6].load(Ordering::Relaxed)
            );
        }
        true
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        if !pending.is_empty() || self.card.is_some() || self.rebar.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.write().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Batched [`Shadow::changed`] for `segs`: acquires `self.kept` once.
    fn changed_segs(&self, start: usize, segs: &[CopySeg]) {
        if !keeps() || segs.is_empty() {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for seg in segs {
            if seg.len == 0 {
                continue;
            }
            let off = seg.src - start;
            for piece in self.pieces(off, off + seg.len) {
                kept.0[piece] = false;
            }
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => !stale_pages_in(base, &copied, a, e, &written).is_empty(),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let mut title = vec![0u8; PAGE];
        for (a, e) in busy {
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                return true;
            };
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied, a, e, &written)
            };
            for page in stale {
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    continue;
                }
                let n = hi - lo;
                let title = &mut title[..n];
                let at = lo - start;
                if let Some(rebar) = &self.rebar {
                    let mut dirty_runs = Vec::new();
                    {
                        let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
                        if !crate::host::read_into(lo as *const u8, title) {
                            continue;
                        }
                        MERGED[0].fetch_add(1, Ordering::Relaxed);
                        let Some(held_all) = mirror.as_mut() else {
                            return true;
                        };
                        if at + n > held_all.len() {
                            return true;
                        }
                        let held = &mut held_all[at..at + n];
                        let runs = changed_runs(title, held);
                        if runs.len() > *runs_left {
                            return true;
                        }
                        *runs_left -= runs.len();
                        for (r, m) in runs {
                            // SAFETY: inside the ReBAR WC VRAM mapping (`at + r + m <= self.len`).
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    title.as_ptr().add(r),
                                    self.data.add(at + r),
                                    m,
                                );
                            }
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            dirty_runs.push((at + r, m));
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                    }
                    for (off, len) in dirty_runs {
                        rebar.mark_dirty_range(off, len);
                    }
                } else {
                    // The lock of the copy's reference (the snapshot, or the
                    // CPU's side of a copy in the card's memory), from before
                    // title memory is read until the title's bytes are in: with
                    // the copy-back on its own thread ([`relay`]) that thread
                    // puts what the card wrote into title memory and into the
                    // reference under the same lock, and a page read before
                    // such a merge and compared after it would take the card's
                    // bytes for the title's.
                    let mut before = self.before.write().unwrap_or_else(|e| e.into_inner());
                    // A page that is not there is left as it is, and looked at
                    // again next time.
                    if !crate::host::read_into(lo as *const u8, title) {
                        continue;
                    }
                    MERGED[0].fetch_add(1, Ordering::Relaxed);
                    if self.card.is_some() {
                        // SAFETY: inside the mapping of the copy's CPU side,
                        // which only the submitting thread writes.
                        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                        let runs = join_runs(changed_runs(title, held), merge_gap());
                        if runs.len() > *runs_left {
                            return true;
                        }
                        *runs_left -= runs.len();
                        for (r, m) in runs {
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            self.card_changed(at + r, at + r + m);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                    } else {
                        let Some(snapshot) = before.as_mut() else {
                            return true;
                        };
                        // SAFETY: the snapshot's allocation holds the copy's
                        // bytes at the copy's offsets (`writing`), and these
                        // are in a declared range.
                        let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                        for (r, m) in changed_runs(title, held) {
                            // SAFETY: inside the copy's mapping. The card may be
                            // writing the copy beside these bytes; they are
                            // written here and nothing around them.
                            unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                    }
                    drop(before);
                }
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(at) = copied.get_mut(page - base) {
                        *at = now;
                    }
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let plan_started = std::time::Instant::now();
        let plan = self.plan_update(address, len);
        crate::host::PREPARING_COPIES[crate::host::PREP_WATCH_US]
            .fetch_add(plan_started.elapsed().as_micros() as u64, Ordering::Relaxed);
        if plan.runs.is_empty() && plan.watched_ranges.is_empty() && plan.busy_ranges.is_empty() {
            return 0;
        }
        let copy_started = std::time::Instant::now();
        let mut copied = self.copy_runs(&plan.runs);
        for &(from, to) in &plan.busy_ranges {
            copied += self.update_busy(from, to);
        }
        for &(from, to) in &plan.watched_ranges {
            self.refresh_check(from, to);
        }
        let copy_us = copy_started.elapsed().as_micros() as u64;
        crate::host::PREPARING_COPIES[crate::host::PREP_REFRESH_COPY_US]
            .fetch_add(copy_us, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        if !plan.unwatched_ranges.is_empty() {
            for &(from, to) in &plan.unwatched_ranges {
                crate::copystat::whole(from, to - from, copy_us);
            }
        }
        copied
    }

    /// Computes the disjoint dirty byte runs of `[address, address + len)`
    /// outside pending shader writes and advances `self.copied` for whole
    /// stale pages, releasing `self.copied`'s lock before any memory copy.
    /// Ranges under pending shader writes are collected in `plan.busy_ranges`
    /// so [`Shadow::update_busy`] can copy changed title bytes under them
    /// without disturbing untouched bytes (`fcdefcfc`).
    fn plan_update(&self, address: usize, len: usize) -> UpdatePlan {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return UpdatePlan::default();
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return UpdatePlan::default();
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        let mut plan = UpdatePlan::default();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them, what the title
            // wrote since it was taken (`update_busy`).
            busy.sort_unstable();
            let mut at = from;
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    self.plan_update_free(at, a, &mut plan);
                }
                if e > a.max(at) {
                    plan.busy_ranges.push((a.max(at), e));
                }
                at = at.max(e);
            }
            if at < to {
                self.plan_update_free(at, to, &mut plan);
            }
            return plan;
        }
        self.plan_update_free(from, to, &mut plan);
        plan
    }

    /// [`Shadow::update`] of `[from, to)` where an earlier list declared
    /// shader writes, whether it is in flight or still being prepared.
    /// Copies title changes into the reference (`rebar.mirror`, card CPU
    /// staging, or PC `before` snapshot) so copy-back does not mistake them
    /// for GPU writes.
    /// Where an earlier flight's shaders may have written, copy only the
    /// changed bytes. A card-backed page slice named only by the submission
    /// being prepared can be uploaded as a single range before its lists run.
    fn update_busy(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return 0;
        }
        let Some((now, written)) = crate::watch::writes(from, to, true) else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let stale: Vec<usize> = {
            let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            stale_pages_in(base, &copied, from, to, &written)
        };
        if stale.is_empty() {
            return 0;
        }
        // Complete any deferred snapshots (`wide_back`) before comparing
        // against `self.before` for a PC shadow.
        if self.rebar.is_none() && self.card.is_none() && wide_back() {
            take_snapshots();
        }
        // Preparation is serialized, so previously submitted ranges can
        // only disappear after this snapshot (the relay landed them).
        // Without relay bookkeeping in late mode, assume every busy range
        // might still have an earlier writer.
        let protected = if !self.carded() || !crate::host::late() {
            Vec::new()
        } else if relay() {
            let preparing = PREPARED.load(Ordering::Relaxed);
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).older_ranges(preparing, from, to)
        } else {
            vec![(from, to)]
        };
        let mut protected = protected.into_iter().peekable();
        let mut title = vec![0u8; PAGE];
        let mut copied_bytes = 0usize;
        for page in stale {
            let (lo, hi) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if lo >= hi {
                continue;
            }
            let n = hi - lo;
            let title = &mut title[..n];
            let at = lo - start;
            let mut page_ok = false;
            if let Some(rebar) = &self.rebar {
                let mut dirty_runs = Vec::new();
                {
                    let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
                    if !crate::host::read_into(lo as *const u8, title) {
                        continue;
                    }
                    if let Some(held_all) = mirror.as_mut()
                        && at + n <= held_all.len()
                    {
                        let held = &mut held_all[at..at + n];
                        for (r, m) in changed_runs(title, held) {
                            // SAFETY: inside the ReBAR WC VRAM mapping (`at + r + m <= self.len`).
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    title.as_ptr().add(r),
                                    self.data.add(at + r),
                                    m,
                                );
                            }
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            dirty_runs.push((at + r, m));
                            copied_bytes += m;
                        }
                        page_ok = true;
                    }
                }
                for (off, len) in dirty_runs {
                    rebar.mark_dirty_range(off, len);
                }
            } else {
                let mut before = self.before.write().unwrap_or_else(|e| e.into_inner());
                if !crate::host::read_into(lo as *const u8, title) {
                    continue;
                }
                if self.card.is_some() {
                    // SAFETY: inside the copy's CPU-side staging mapping.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    while protected.peek().is_some_and(|&(_, b)| b <= lo) {
                        protected.next();
                    }
                    let exact = protected.peek().is_some_and(|&(a, _)| a < hi);
                    for (r, m) in card_busy_runs(title, held, exact) {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        self.card_changed(at + r, at + r + m);
                        copied_bytes += m;
                    }
                    page_ok = true;
                } else if let Some(snapshot) = before.as_mut()
                    && at + n <= snapshot.capacity()
                {
                    // SAFETY: `pooled` allocates capacity >= `self.len` with
                    // `len() == 0`; `writing` (and `take_snapshots`) populated
                    // the snapshot bytes for every declared range.
                    let held = unsafe {
                        std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n)
                    };
                    for (r, m) in changed_runs(title, held) {
                        // SAFETY: inside the copy's mapping (`at + r + m <= self.len`).
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                title.as_ptr().add(r),
                                self.data.add(at + r),
                                m,
                            );
                        }
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        copied_bytes += m;
                    }
                    page_ok = true;
                }
            }
            if page_ok && (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(slot) = copied.get_mut(page - base) {
                    *slot = now;
                }
            }
        }
        copied_bytes
    }

    /// [`Shadow::plan_update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn plan_update_free(&self, from: usize, to: usize, plan: &mut UpdatePlan) {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let before = plan.runs.len();
                self.collect_stale(from, to, now, &written, &mut plan.runs);
                plan.watched_ranges.push((from, to));
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    let copied: usize = plan.runs[before..].iter().map(|&(a, b)| b - a).sum();
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                let _ = before;
                return;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        plan.runs.push((from, to));
        #[cfg(feature = "diagnostics")]
        plan.unwatched_ranges.push((from, to));
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Collects contiguous stale byte ranges `(lo, hi)` of `[from, to)`
    /// written after their last copy (`written`: last writes by page, in page
    /// order; `now`: this asking), advances `self.copied` for whole pages,
    /// and releases `self.copied`'s lock before any copy runs.
    fn collect_stale(
        &self,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        out: &mut Vec<(usize, usize)>,
    ) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let mut run = None;
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                let lo = (first * PAGE).max(from);
                let hi = (page * PAGE).min(to);
                if lo < hi {
                    out.push((lo, hi));
                }
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            let lo = (first * PAGE).max(from);
            let hi = ((last + 1) * PAGE).min(to);
            if lo < hi {
                out.push((lo, hi));
            }
        }
    }

    /// Converts disjoint title-address `runs` (clamped to the buffer) into
    /// [`CopySeg`] entries targeting `self.data`, returning the total bytes.
    fn push_copy_segs(&self, runs: &[(usize, usize)], out: &mut Vec<CopySeg>) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let mut total = 0usize;
        for &(raw_from, raw_to) in runs {
            let (from, to) = (raw_from.max(start), raw_to.min(end));
            if from >= to {
                continue;
            }
            let off = from - start;
            let len = to - from;
            out.push(CopySeg {
                src: from,
                dst: (self.data as usize) + off,
                mirror: 0,
                len,
            });
            total += len;
        }
        total
    }

    /// Copies disjoint title-memory `runs`, cut to the buffer, into the copy
    /// (in parallel across worker threads when [`wide_copy`] is enabled and
    /// total bytes reach [`WIDE_COPY_MIN`]).
    fn copy_runs(&self, runs: &[(usize, usize)]) -> usize {
        if runs.is_empty() {
            return 0;
        }
        let start = self.start as usize;
        let mut segs = Vec::with_capacity(runs.len());
        let total = self.push_copy_segs(runs, &mut segs);
        if total == 0 {
            return 0;
        }
        if let Some(rebar) = &self.rebar {
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(m) = mirror.as_mut() {
                let m_ptr = m.as_mut_ptr() as usize;
                let m_len = m.len();
                if segs.iter().all(|s| (s.src - start) + s.len <= m_len) {
                    // Copy once from guarded title memory into cached RAM, then
                    // push straight to write-combining VRAM so both see identical bytes.
                    for s in &mut segs {
                        s.mirror = m_ptr + (s.src - start);
                    }
                    run_copy_segs(&segs);
                    drop(mirror);
                    rebar.mark_dirty_segs(start, &segs);
                    return total;
                }
            }
        }
        run_copy_segs(&segs);
        self.finish_copy_segs(start, &segs);
        total
    }

    /// Post-copy metadata invalidation for non-mirrored `segs`, acquiring each
    /// per-shadow lock at most once.
    fn finish_copy_segs(&self, start: usize, segs: &[CopySeg]) {
        if segs.is_empty() {
            return;
        }
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed_segs(start, segs);
        self.card_changed_segs(start, segs);
        if let Some(rebar) = &self.rebar {
            rebar.mark_dirty_segs(start, segs);
        }
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
    }

    /// Batched [`Shadow::card_changed`] for `segs`: acquires `card.dirty` once.
    fn card_changed_segs(&self, start: usize, segs: &[CopySeg]) {
        let Some(card) = &self.card else {
            return;
        };
        if segs.is_empty() {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        for seg in segs {
            if seg.len == 0 {
                continue;
            }
            let from = seg.src - start;
            let to = from + seg.len;
            match dirty.last_mut() {
                Some(last) if from <= last.1 && to >= last.0 => {
                    *last = (last.0.min(from), last.1.max(to));
                }
                _ => dirty.push((from, to)),
            }
        }
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            let as_read = merge_race_on().then(|| title.clone());
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                crate::host::shaders_write(start + at, &title[..]);
                #[cfg(feature = "diagnostics")]
                back_note(start + at, &title[..n], self.start, self.len);
                if !self.put_back_own(start + at, &title[..]) {
                    crate::watch::writing(start + at, n);
                    merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card");
                    crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                }
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.read().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it concurrently from the submitting thread
        // (`merge_under_writes`) takes a write lock on `self.before`, and
        // parallel relay workers operate on strictly disjoint `[at, at + len)`
        // ranges.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                .filter(|&(a, b)| a < at + n && b > at)
                .collect()
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = (mapped && merge_race_on()).then(|| title.clone());
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        #[cfg(feature = "diagnostics")]
        back_note(start + at, &title[..n], self.start, self.len);
        if self.put_back_own(start + at, &title[..]) {
            return n;
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    #[cfg(test)]
    pub(crate) fn new_card_for_test(
        side_data: *mut u8,
        title_start: u64,
        len: u64,
        dirty: Vec<(usize, usize)>,
        sent: Vec<(usize, usize, u64)>,
    ) -> Option<Arc<Shadow>> {
        let h = host()?;
        let (buffer, _) = h.shared_buffer(64)?;
        let (staging, _) = h.shared_buffer(64)?;
        let mut shadow = Shadow::new(buffer, side_data, 0, title_start, len, None);
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: Mutex::new(dirty),
            sent: Mutex::new(sent),
            // 0: no card bytes counted (`Drop` subtracts `card.size`).
            size: 0,
        });
        Some(Arc::new(shadow))
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
// (A lock for readers and writers: every draw looks its buffers up, from
// five recording threads at once, and nearly every look finds the copy
// made; under one mutex they stood in line for it.)
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let before = buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
    // A buffer made again at the same place and address: its old copy may
    // still be named by lists and descriptors, as one the title released.
    if let Some(shadow) = before.and_then(|e| e.shadow) {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let shadow = {
        let mut map = buffers();
        map.remove(&(start, resource)).and_then(|e| e.shadow)
    };
    if let Some(shadow) = shadow {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut guard = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = guard
            .entries
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| {
            let s = guard.entries.remove(&key)?;
            guard.total_bytes = guard.total_bytes.saturating_sub(s.len);
            Some(s)
        }));
    }
    {
        let mut stretches = STRETCHES.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<u64> = stretches
            .range(..end)
            .filter(|(_, s)| s.start + s.len > at)
            .map(|(k, _)| *k)
            .collect();
        gone.extend(keys.into_iter().filter_map(|k| stretches.remove(&k)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
    n
}

/// Retires any active `VICTORY_STRETCH` window opened on `section` when the
/// runtime reports that `section`'s handle number has been reused for a new
/// section (`crate::inplace::title_memory_section`).
pub(crate) fn stretch_section_reused(section: usize) {
    let gone: Vec<Arc<Shadow>> = {
        let mut stretches = STRETCHES.write().unwrap_or_else(|e| e.into_inner());
        if stretches.is_empty() {
            return;
        }
        let keys: Vec<u64> = stretches
            .iter()
            .filter(|(_, s)| s.stretch_view.as_ref().is_some_and(|v| v.section == section))
            .map(|(&k, _)| k)
            .collect();
        keys.into_iter().filter_map(|k| stretches.remove(&k)).collect()
    };
    for shadow in gone {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.entries.keys().map(|&(start, len)| (start, len, true)));
    let stretches = STRETCHES.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(stretches.values().map(|s| (s.start, s.len, true)));
    spans
}

/// Allocates a shadow backing buffer: prefers ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// VRAM within the adapter's budget, falling back cleanly to system-RAM
/// [`Host::shared_buffer`] when ReBAR is unavailable, disabled, or full.
fn alloc_shadow_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8, Option<u64>)> {
    let h = host()?;
    let budget = h.rebar_shadow_budget();
    if try_reserve_rebar(&REBAR_SHADOW_BYTES, size, budget) {
        if let Some((buffer, data)) = h.try_rebar_shadow_buffer(size) {
            return Some((buffer, data, Some(size)));
        }
        REBAR_SHADOW_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    let (buffer, data) = h.shared_buffer(size)?;
    Some((buffer, data, None))
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    // The copy is made already, nearly always: found under the readers'
    // lock.
    {
        let map = BUFFERS.read().unwrap_or_else(|e| e.into_inner());
        let found = map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end);
        match found {
            Some((_, entry)) => {
                if let Some(shadow) = &entry.shadow {
                    return Some(shadow.clone());
                }
            }
            // In no title buffer: a window, which has a table of its own
            // (and must not wait for this one's writers).
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
        }
    }
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && size >= card_limits().0;
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the CPU's staging side stays in system memory.
        let (buffer, data, card, rebar_bytes) = match card_for(size) {
            Some(on_card) => {
                let (staging, data) = host()?.shared_buffer(size)?;
                crate::host::name(&staging, || format!("title buffer {:#x}+{len:#x}", key.0));
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    size,
                };
                (on_card, data, Some(card), None)
            }
            None => {
                let (buffer, data, rebar_bytes) = alloc_shadow_buffer(size)?;
                crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
                (buffer, data, None, rebar_bytes)
            }
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                rebar = rebar_bytes.is_some(),
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                rebar = rebar_bytes.is_some(),
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len, rebar_bytes);
        shadow.card = card;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches, bounded by a RAM-scaled
/// LRU cache.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Monotonic access clock for LRU eviction of loose shadow windows.
static LOOSE_TICK: AtomicU64 = AtomicU64::new(1);

fn total_physical_ram_bytes() -> Option<u64> {
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: `status` is initialized with `dwLength` set to its byte size as
    // required by `GlobalMemoryStatusEx`.
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut status) };
    (ok != 0 && status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
}

fn loose_budget_for_ram(total_ram_bytes: Option<u64>) -> u64 {
    match total_ram_bytes {
        None => 256 * MIB,
        Some(ram) if ram <= 16 * GIB => 128 * MIB,
        Some(ram) if ram <= 32 * GIB => 256 * MIB,
        Some(_) => 512 * MIB,
    }
}

fn loose_budget() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let ram = total_physical_ram_bytes();
        let budget = loose_budget_for_ram(ram);
        tracing::info!(
            total_ram_mib = ram.map(|b| b / MIB),
            budget_mib = budget / MIB,
            "loose shadow cache budget"
        );
        budget
    })
}

struct LooseCache {
    entries: BTreeMap<(u64, u64), Arc<Shadow>>,
    total_bytes: u64,
}

impl LooseCache {
    const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Evicts least-recently-used idle windows while `total_bytes > budget`,
    /// never evicting `keep`, entries still held outside the cache
    /// (`Arc::strong_count > 1`), or entries with pending GPU copy-back ranges.
    fn evict_over_budget(&mut self, keep: (u64, u64), budget: u64) -> Vec<Arc<Shadow>> {
        if self.total_bytes <= budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, (u64, u64))> = self
            .entries
            .iter()
            .filter_map(|(&key, shadow)| {
                if key == keep || Arc::strong_count(shadow) != 1 {
                    return None;
                }
                if !shadow
                    .pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
                {
                    return None;
                }
                Some((shadow.last_used.load(Ordering::Relaxed), key))
            })
            .collect();
        candidates.sort_unstable_by_key(|&(tick, _)| tick);
        let mut evicted = Vec::new();
        for (_, key) in candidates {
            if self.total_bytes <= budget {
                break;
            }
            if let Some(shadow) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(shadow.len);
                evicted.push(shadow);
            }
        }
        if !evicted.is_empty() {
            tracing::debug!(
                evicted = evicted.len(),
                total_bytes = self.total_bytes,
                budget,
                "evicted idle loose shadow windows"
            );
        }
        evicted
    }
}

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    cover(loose.entries.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Copies of title memory outside buffers, by (start, length).
// (Readers and writers: nearly every ask finds its window made.)
static LOOSE: std::sync::RwLock<LooseCache> = std::sync::RwLock::new(LooseCache::new());

/// Contiguous section-view stretches mapped once in place (`VICTORY_STRETCH`),
/// keyed by start address. Kept separate from [`LOOSE`] so multi-megabyte
/// zero-copy streaming stretches do not evict or count against the host RAM
/// copy budget.
static STRETCHES: std::sync::RwLock<BTreeMap<u64, Arc<Shadow>>> =
    std::sync::RwLock::new(BTreeMap::new());

/// Maximum reach of one stretch window (aligned down/up to 1 GiB).
const STRETCH_REACH: usize = 1 << 30;
/// Stretches shorter than this are copied: each window is a host heap.
const STRETCH_LEAST: usize = 4 << 20;

/// Whether contiguous 64 KiB section stretches are opened in place
/// (`VICTORY_STRETCH=0` turns it off).
fn stretch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STRETCH").is_none_or(|v| v != "0"))
}

/// Active stretch count and total bytes covered (`VICTORY_STRETCH`).
pub(crate) fn stretch_stats() -> (usize, u64) {
    let map = STRETCHES.read().unwrap_or_else(|e| e.into_inner());
    let bytes = map.values().map(|s| s.len).sum();
    (map.len(), bytes)
}

fn note_window_use(shadow: &Shadow, address: u64, end: u64, unsized_ask: bool) {
    shadow.touch();
    shadow.used.store(crate::host::moment(), Ordering::Relaxed);
    shadow.asked[0].fetch_min(address, Ordering::Relaxed);
    shadow.asked[1].fetch_max(end, Ordering::Relaxed);
    if unsized_ask {
        shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
    }
}

fn stretch_find_in(
    stretches: &std::sync::RwLock<BTreeMap<u64, Arc<Shadow>>>,
    address: u64,
    end: u64,
    unsized_ask: bool,
) -> Option<Arc<Shadow>> {
    let map = stretches.read().unwrap_or_else(|e| e.into_inner());
    let s = map
        .range(..=address)
        .rev()
        .take(8)
        .map(|(_, s)| s)
        .find(|s| s.start + s.len >= end)?;
    note_window_use(s, address, end, unsized_ask);
    Some(s.clone())
}

/// A window on title memory the title shows page by page: Grand Theft Auto
/// VI maps its streaming heaps one 64 KiB view at a time, every view its
/// own allocation, which the host cannot open as one heap, though the pages
/// follow one another in their section (1,200 views for 75 MiB, one stretch
/// of one section). The stretch is mapped once more, whole, at an address of
/// ours, and the host heap is opened there: the same pages, so nothing is
/// copied, and the title's own views stay as they are. When the title maps
/// other pages there, the window goes as any other ([`in_place_let_go`]) and
/// our view is unmapped on drop ([`StretchView`]).
fn through_another_view(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    address: u64,
    end: u64,
) -> Option<Shadow> {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
    };
    let (start, stop, section, offset) =
        crate::inplace::stretch(address as usize, end as usize, STRETCH_REACH)?;
    let len = stop - start;
    if len < STRETCH_LEAST {
        return None;
    }
    // SAFETY: a section the runtime told us of, alive while it has views;
    // the stretch is inside it.
    let view = unsafe {
        MapViewOfFile(
            section as _,
            FILE_MAP_WRITE,
            (offset >> 32) as u32,
            offset as u32,
            len,
        )
    };
    if view.Value.is_null() {
        tracing::warn!(
            error = %std::io::Error::last_os_error(),
            start = format_args!("{start:#x}"),
            len,
            "a stretch of title memory could not be mapped again"
        );
        return None;
    }
    // Read-only guards would refuse the GPU's writes.
    crate::watch::writing(start, len);
    let mut shadow = match crate::inplace::place_stretch(device, view.Value as usize, len as u64) {
        Ok(placed) => Shadow::placed(placed, start as u64, len as u64, true),
        Err(error) => {
            tracing::warn!(
                %error,
                start = format_args!("{start:#x}"),
                len,
                "a stretch of title memory refused by the host GPU"
            );
            // SAFETY: our view, used by nothing.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view.Value,
                });
            }
            return None;
        }
    };
    shadow.stretch_view = Some(StretchView {
        addr: view.Value as usize,
        section,
    });
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count <= 32 || count.is_power_of_two() {
        tracing::info!(
            count,
            start = format_args!("{start:#x}"),
            len,
            "title memory in place through a view of ours"
        );
    }
    Some(shadow)
}

fn stretch_create_in(
    stretches: &std::sync::RwLock<BTreeMap<u64, Arc<Shadow>>>,
    address: u64,
    end: u64,
    unsized_ask: bool,
    enabled: bool,
) -> Option<Arc<Shadow>> {
    if !enabled {
        return None;
    }
    let h = host()?;
    let (window, retired) = {
        let mut map = stretches.write().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map
            .range(..=address)
            .rev()
            .take(8)
            .map(|(_, s)| s)
            .find(|s| s.start + s.len >= end)
        {
            note_window_use(s, address, end, unsized_ask);
            return Some(s.clone());
        }
        let window = Arc::new(through_another_view(&h.device, address, end)?);
        note_window_use(&window, address, end, unsized_ask);
        let (w_start, w_end) = (window.start, window.start + window.len);
        let overlap: Vec<u64> = map
            .range(..w_end)
            .filter(|(_, s)| s.start + s.len > w_start)
            .map(|(&k, _)| k)
            .collect();
        let retired: Vec<Arc<Shadow>> = overlap.into_iter().filter_map(|k| map.remove(&k)).collect();
        map.insert(w_start, window.clone());
        (window, retired)
    };
    for old in retired {
        let born = old.born_seq;
        crate::host::retire_born(old, born);
    }
    Some(window)
}

fn loose_pick(
    map: &BTreeMap<(u64, u64), Arc<Shadow>>,
    address: u64,
    end: u64,
    unsized_ask: bool,
    any: bool,
) -> (u64, u64) {
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    if unsized_ask && tails() {
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
    } else if any && fine() && !unsized_ask {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        holder(there, address, end).unwrap_or(named)
    } else {
        named
    }
}

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    if stretch_enabled() {
        if let Some(s) = stretch_find_in(&STRETCHES, address, end, unsized_ask) {
            return Some(s);
        }
        // Check LOOSE read-only first so stretches that the host GPU refused
        // and fell back to LOOSE are not retried on every ask.
        {
            let guard = LOOSE.read().unwrap_or_else(|e| e.into_inner());
            if let Some(shadow) = guard.entries.get(&loose_pick(&guard.entries, address, end, unsized_ask, any)) {
                note_window_use(shadow, address, end, unsized_ask);
                return Some(shadow.clone());
            }
        }
        if let Some(s) = stretch_create_in(&STRETCHES, address, end, unsized_ask, true) {
            return Some(s);
        }
    }
    loose_in(&LOOSE, address, end, unsized_ask, any, loose_budget())
}

fn loose_in(
    cache: &std::sync::RwLock<LooseCache>,
    address: u64,
    end: u64,
    unsized_ask: bool,
    any: bool,
    budget: u64,
) -> Option<Arc<Shadow>> {
    // Made already, nearly always: found beside the other threads that ask.
    {
        let guard = cache.read().unwrap_or_else(|e| e.into_inner());
        if let Some(shadow) = guard.entries.get(&loose_pick(&guard.entries, address, end, unsized_ask, any)) {
            note_window_use(shadow, address, end, unsized_ask);
            return Some(shadow.clone());
        }
    }
    let (shadow, evicted) = {
        let mut guard = cache.write().unwrap_or_else(|e| e.into_inner());
        let (start, len) = loose_pick(&guard.entries, address, end, unsized_ask, any);
        let key = (start, len);
        if let Some(shadow) = guard.entries.get(&key) {
            note_window_use(shadow, address, end, unsized_ask);
            return Some(shadow.clone());
        }
        // In place, if asked for and the window's memory can be opened: the
        // window is the title's own memory, with nothing to copy.
        let shadow = if let Some(placed) = crate::inplace::place_for(start, len, true) {
            Arc::new(Shadow::placed(placed, start, len, true))
        } else {
            // Keep unbounded loose 4 MiB windows in system memory (unless
            // `window_card_for` places them on the card) so they cannot starve
            // registered title buffers out of the bounded ReBAR budget.
            let (buffer, data) = host()?.shared_buffer(len)?;
            crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
            // With room for windows on the card (`window_card_most`): shaders get a
            // buffer there, and the one just made is the CPU's side.
            let (buffer, card) = match window_card_for(len) {
                Some(on_card) => {
                    crate::host::name(&on_card, || {
                        format!("title memory {start:#x}+{len:#x} on the card")
                    });
                    let card = Card {
                        staging: buffer,
                        before: Mutex::default(),
                        dirty: Mutex::default(),
                        sent: Mutex::default(),
                        size: len,
                    };
                    (on_card, Some(card))
                }
                None => (buffer, None),
            };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            tracing::debug!(
                start = format_args!("{start:#x}"),
                len,
                "loose memory copied to the host"
            );
            let mut shadow = Shadow::new(buffer, data, gpu, start, len, None);
            shadow.window = true;
            shadow.card = card;
            let shadow = Arc::new(shadow);
            if shadow.card.is_some() {
                CARDED
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(Arc::downgrade(&shadow));
            }
            shadow
        };
        note_window_use(&shadow, address, end, unsized_ask);
        guard.entries.insert(key, shadow.clone());
        guard.total_bytes = guard.total_bytes.saturating_add(len);
        let evicted = guard.evict_over_budget(key, budget);
        (shadow, evicted)
    };
    for old in evicted {
        let born = old.born_seq;
        crate::host::retire_born(old, born);
    }
    Some(shadow)
}

/// Live byte and entry counts across registered buffers, loose windows, and
/// ReBAR shadow reservations for periodic memory diagnostics.
pub(crate) fn cache_stats() -> (usize, u64, usize, u64, usize, u64) {
    let (buffers_count, buffers_bytes) = {
        let map = buffers();
        let mut count = 0usize;
        let mut bytes = 0u64;
        for entry in map.values() {
            if let Some(s) = &entry.shadow {
                count += 1;
                bytes = bytes.saturating_add(s.len);
            }
        }
        (count, bytes)
    };
    let (loose_count, loose_bytes, loose_pinned) = {
        let cache = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let pinned = cache
            .entries
            .values()
            .filter(|s| Arc::strong_count(s) > 1)
            .count();
        (cache.entries.len(), cache.total_bytes, pinned)
    };
    let rebar_bytes = REBAR_SHADOW_BYTES.load(Ordering::Relaxed);
    (
        buffers_count,
        buffers_bytes,
        loose_count,
        loose_bytes,
        loose_pinned,
        rebar_bytes,
    )
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `own_bytes`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, address, len, None));
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_OWN_COPY=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

/// Whether shader view-marks 64 KiB page filtering is enabled (`VICTORY_VIEW_MARKS=1`).
pub(crate) fn view_marks_on() -> bool {
    victory_shader::emit::view_marks_enabled()
}

/// Number of 64 KiB title pages touched by `[start, start + len)`.
fn mark_pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// Free-list allocator for runs of bits in the shader view-writes bitmap
/// (`victory_shader::emit::VIEW_WRITE_BITS`), one bit per 64 KiB title page.
#[derive(Default)]
struct Marks {
    /// Free bit runs below `end`, keyed by first bit -> run length.
    free: BTreeMap<u32, u32>,
    /// High-water mark of bits handed out; bits at or past `end` are unused.
    end: u32,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// Allocates a contiguous run of `n` bits (the last bit of `VIEW_WRITE_BITS`
    /// is reserved as an overflow sink and never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end.checked_add(n).is_some_and(|e| e < victory_shader::emit::VIEW_WRITE_BITS) {
            let at = self.end;
            self.end += n;
            return Some(at);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                "no bits left in the view-writes bitmap (VICTORY_VIEW_MARKS=1): new copies fall back to full declared-range compare"
            )
        });
        None
    }

    /// Returns a run of `n` bits at `at`, coalescing with adjacent free runs.
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// High-water mark of allocated bits in the view-writes bitmap.
pub(crate) fn marks_end() -> u32 {
    marks().end
}

/// GPU-side 64 KiB page bitmap (`SLOT_VIEW_WRITES`, `DEFAULT` heap in
/// `UNORDERED_ACCESS` state) plus a matching zero-filled `UPLOAD` buffer for
/// fast GPU-timeline clears after each submission's snapshot copy.
struct ViewWritesBitmap {
    uav: ID3D12Resource,
    gpu: u64,
    zero: ID3D12Resource,
}

// SAFETY: D3D12 resources are free-threaded; GPU accesses are strictly
// ordered on the direct command queue.
unsafe impl Send for ViewWritesBitmap {}
// SAFETY: as above.
unsafe impl Sync for ViewWritesBitmap {}

fn view_writes_bitmap() -> Option<&'static ViewWritesBitmap> {
    static BITMAP: OnceLock<Option<ViewWritesBitmap>> = OnceLock::new();
    BITMAP
        .get_or_init(|| {
            let h = host()?;
            let bytes = u64::from(victory_shader::emit::VIEW_WRITE_BITS / 8);
            let uav = h.default_buffer_in_state(
                bytes,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            crate::host::name(&uav, || "shader view-writes bitmap".into());
            // SAFETY: live D3D12 buffer.
            let gpu = unsafe { uav.GetGPUVirtualAddress() };
            let (zero, data) = h.upload_buffer(bytes)?;
            crate::host::name(&zero, || "shader view-writes clear source".into());
            // SAFETY: mapped upload buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            Some(ViewWritesBitmap { uav, gpu, zero })
        })
        .as_ref()
}

/// GPU virtual address of the shader view-writes bitmap (`SLOT_VIEW_WRITES`),
/// allocated lazily when `VICTORY_VIEW_MARKS=1`.
pub(crate) fn view_writes() -> Option<u64> {
    view_writes_bitmap().map(|b| b.gpu)
}

/// `(uav_buffer, gpu_va, zero_upload_buffer)` for capturing and clearing the
/// shader view-writes bitmap on the GPU timeline.
pub(crate) fn view_writes_resources()
-> Option<(&'static ID3D12Resource, u64, &'static ID3D12Resource)> {
    view_writes_bitmap().map(|b| (&b.uav, b.gpu, &b.zero))
}

/// Pure helper: intersects `declared_title_ranges` (title addresses `[from, to)`)
/// with the 64 KiB title pages whose bit is set in any of `bitmaps`, unioned with
/// `unmarked_title_ranges` (which are always kept unfiltered).
///
/// Returns `(filtered_title_ranges, declared_pages, marked_pages)`.
pub(crate) fn filter_title_ranges_by_marks(
    start: u64,
    len: u64,
    mark_base: Option<u32>,
    declared_title_ranges: &[(usize, usize)],
    unmarked_title_ranges: &[(usize, usize)],
    bitmaps: &[&[u32]],
) -> (Vec<(usize, usize)>, u64, u64) {
    let (s_start, s_end) = (start as usize, start.saturating_add(len) as usize);
    let clamp_merge = |ranges: &[(usize, usize)]| -> Vec<(usize, usize)> {
        let clamped: Vec<(usize, usize)> = ranges
            .iter()
            .filter_map(|&(a, b)| {
                let lo = a.clamp(s_start, s_end);
                let hi = b.clamp(lo, s_end);
                (lo < hi).then_some((lo, hi))
            })
            .collect();
        if clamped.is_empty() {
            Vec::new()
        } else {
            joined(clamped)
        }
    };
    let declared = clamp_merge(declared_title_ranges);
    if declared.is_empty() {
        return (Vec::new(), 0, 0);
    }
    let mut declared_pages = 0u64;
    let mut last_counted_page = usize::MAX;
    for &(from, to) in &declared {
        let p0 = from / PAGE;
        let p1 = to.div_ceil(PAGE);
        for p in p0..p1 {
            if p != last_counted_page {
                declared_pages += 1;
                last_counted_page = p;
            }
        }
    }
    let Some(base) = mark_base else {
        return (declared, declared_pages, declared_pages);
    };
    if bitmaps.is_empty() {
        return (declared, declared_pages, declared_pages);
    }
    let unmarked = clamp_merge(unmarked_title_ranges);
    let base_title_page = (start >> 16) as usize;
    let max_k = mark_pages(start, len);
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut push_range = |lo: usize, hi: usize| {
        if lo >= hi {
            return;
        }
        match out.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    };
    let mut marked_pages = 0u64;
    let mut last_marked_page = usize::MAX;
    for &(from, to) in &declared {
        let p0 = from / PAGE;
        let p1 = to.div_ceil(PAGE);
        for p in p0..p1 {
            let page_lo = p * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            let lo = from.max(page_lo);
            let hi = to.min(page_hi);
            let k = p.saturating_sub(base_title_page) as u32;
            let bit = base.saturating_add(k);
            let is_marked = k < max_k
                && bitmaps.iter().any(|words| {
                    let wi = (bit / 32) as usize;
                    wi < words.len() && (words[wi] >> (bit & 31)) & 1 != 0
                });
            if is_marked {
                if p != last_marked_page {
                    marked_pages += 1;
                    last_marked_page = p;
                }
                push_range(lo, hi);
            } else if !unmarked.is_empty() {
                for &(u0, u1) in &unmarked {
                    if u1 <= lo {
                        continue;
                    }
                    if u0 >= hi {
                        break;
                    }
                    push_range(lo.max(u0), hi.min(u1));
                }
            }
        }
    }
    (out, declared_pages, marked_pages)
}

/// View-marks telemetry since the last frame line:
/// `[0]`: total declared 64 KiB pages evaluated,
/// `[1]`: marked 64 KiB pages (bit set by shaders),
/// `[2]`: GPU bitmap captures recorded.
static VIEW_MARKS_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

pub(crate) fn record_view_marks_filter(declared_pages: u64, marked_pages: u64) {
    VIEW_MARKS_STATS[0].fetch_add(declared_pages, Ordering::Relaxed);
    VIEW_MARKS_STATS[1].fetch_add(marked_pages, Ordering::Relaxed);
}

pub(crate) fn record_view_marks_capture() {
    VIEW_MARKS_STATS[2].fetch_add(1, Ordering::Relaxed);
}

/// Logs shader view-marks filtering telemetry alongside the frame-rate line.
pub(crate) fn view_marks_line(frames: f64) {
    if !view_marks_on() || frames <= 0.0 {
        return;
    }
    let [declared, marked, captures] =
        [0, 1, 2].map(|i| VIEW_MARKS_STATS[i].swap(0, Ordering::Relaxed));
    if declared == 0 && captures == 0 {
        return;
    }
    let skipped = declared.saturating_sub(marked);
    let pct = if declared > 0 {
        skipped as f64 * 100.0 / declared as f64
    } else {
        0.0
    };
    tracing::info!(
        "view marks per frame (VICTORY_VIEW_MARKS=1): {:.1} marked of {:.1} declared 64 KiB pages ({pct:.1}% filtered out), {:.1} bitmap captures",
        marked as f64 / frames,
        declared as f64 / frames,
        captures as f64 / frames,
    );
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }
}

#[cfg(test)]
mod card_merge_tests {
    use super::{Card, Shadow};
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;

    /// A plain buffer on the software adapter: what a copy's `buffer` and
    /// its card's `staging` are; the merge never touches them.
    fn warp_buffer() -> Option<ID3D12Resource> {
        // SAFETY: plain DXGI and D3D12 creation calls on the software adapter.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).ok()?;
            let device = device?;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 64 << 10,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .ok()?;
            buffer
        }
    }

    /// The mixed landing the fetch inside flights must survive, on a real
    /// copy in the card's memory and its merge (`Shadow::card_merge_relay`):
    /// flights 5, 6, 7 in the air; 5 lands the old way with the queue's
    /// count at 7; the title wrote a byte that went up in send 7 into a
    /// piece flight 6's shaders write; 6 then lands from its own slot, which
    /// holds that byte as it was before send 7. The notes of sends are
    /// dropped as `card_dirty` drops them, below the mark the old-way fetch
    /// sets (`crate::host::fallback_fetched_mark`): with the mark at the
    /// landed plans' send the title's byte survives and the shader's write
    /// lands; with the mark at the fetch's count (as before the fix) the
    /// title's byte is put back to the card's older one.
    #[test]
    fn a_title_byte_sent_after_a_flight_survives_its_slot() {
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let (title_byte, shader_byte) = (100usize, 200usize);
        let content: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        let land = |mark: u64| -> Vec<u8> {
            let mut title = content.clone();
            let mut side = content.clone();
            // The title's byte, refreshed into the CPU's side and sent up in send 7.
            title[title_byte] ^= 0x5a;
            side[title_byte] = title[title_byte];
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64, None);
            shadow.card = Some(Card {
                staging: staging.clone(),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                sent: Mutex::new(vec![(title_byte, title_byte + 1, 7)]),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
            });
            // As `card_dirty` with the mark set by the old-way fetch.
            if let Some(card) = &shadow.card {
                card.sent.lock().unwrap().retain(|s| s.2 > mark);
            }
            // Flight 6's slot: the card's bytes behind its lists, before send 7.
            let mut slot = content.clone();
            slot[shader_byte] ^= 0xc3;
            let written = shadow.card_merge_relay(0, &slot, 6, &mut Vec::new());
            assert!(written > 0);
            drop(shadow);
            title
        };
        let fixed = land(crate::host::fallback_fetched_mark(true, 7, Some(5)));
        assert_eq!(fixed[title_byte], content[title_byte] ^ 0x5a, "the title's byte was lost");
        assert_eq!(fixed[shader_byte], content[shader_byte] ^ 0xc3, "the shader's write did not land");
        // The mark at the fetch's count: the note of send 7 is gone and the
        // older card byte is taken for a shader's write.
        let before = land(7);
        assert_eq!(before[title_byte], content[title_byte]);
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    all.extend(
        STRETCHES
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// `VICTORY_BACK_LOG=1` (diagnostics): once the ray-tracing structure
/// builder runs, each run of GPU bytes put into title memory is logged
/// (address, length, how many are 0xff, the host copy it came from), and
/// each builder dispatch with its root views: to find stores that land on
/// the title's own objects.
#[cfg(feature = "diagnostics")]
pub(crate) fn back_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BACK_LOG").is_some_and(|v| v == "1"))
}

#[cfg(feature = "diagnostics")]
static BACK_LINES: AtomicU64 = AtomicU64::new(0);

/// [`back_log`]: one run of title memory about to take `bytes` (the GPU's
/// merged in), with how many of them are 0xff, from the copy of
/// `[start, start + len)`.
#[cfg(feature = "diagnostics")]
fn back_note(address: usize, bytes: &[u8], start: u64, len: u64) {
    if !back_log() || !crate::pipeline::builder_on() || BACK_LINES.fetch_add(1, Ordering::Relaxed) >= 20000 {
        return;
    }
    let ff = bytes.iter().filter(|&&b| b == 0xff).count();
    tracing::info!(
        target: "backlog",
        address = format_args!("{address:#x}"),
        len = bytes.len(),
        ff,
        buffer = format_args!("{start:#x}+{len:#x}"),
        "gpu bytes put into title memory"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebar_budget_reservation_enforces_cap_and_releases() {
        let used = AtomicU64::new(0);
        assert!(!try_reserve_rebar(&used, 64, 0));
        assert!(!try_reserve_rebar(&used, 0, 1024));
        assert!(!try_reserve_rebar(&used, 2048, 1024));
        assert!(try_reserve_rebar(&used, 600, 1024));
        assert_eq!(used.load(Ordering::Relaxed), 600);
        assert!(!try_reserve_rebar(&used, 500, 1024));
        assert!(try_reserve_rebar(&used, 424, 1024));
        assert_eq!(used.load(Ordering::Relaxed), 1024);
    }

    #[cfg(windows)]
    #[test]
    fn rebar_shadow_gpu_write_round_trips_and_preserves_concurrent_cpu_writes() {
        let Some(h) = host() else {
            return;
        };
        if h.rebar_shadow_budget() == 0 {
            // Requires a discrete GPU with D3D12 GPU upload heap (ReBAR) support.
            return;
        }
        let mut backing = vec![0x11u8; 4096];
        let start = backing.as_ptr() as u64;
        let resource = 0xdead_beef_0003_usize;
        register(resource, start, backing.len() as u64);

        let shadow = find(start, backing.len() as u64).expect("shadow");
        assert!(
            shadow.is_rebar(),
            "expected ReBAR-backed shadow on ReBAR adapter"
        );

        let list = h.new_list().expect("host list");
        // Seed initial title bytes, declare overlapping GPU writes, and overwrite
        // bytes 16..32 on the GPU via CopyBufferRegion while the CPU concurrently
        // writes byte 64 in the same page.
        shadow.refresh(&list, start, backing.len() as u64);
        shadow.gpu_writes(&list, start, 128);
        shadow.gpu_writes(&list, start + 16, 128);

        let (src_buf, src_ptr) = h.upload_buffer(16).expect("upload");
        // SAFETY: `src_ptr` is mapped for 16 bytes.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0xAA, 16);
        }
        // SAFETY: live buffers and valid 16-byte copy inside both.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, 16, &src_buf, 0, 16);
        }
        // Simulate a concurrent CPU write to byte 64 while the GPU runs.
        let concurrent_addr = backing.as_mut_ptr() as usize;
        list.defer_host(move || {
            // SAFETY: `backing` stays alive for the duration of this test.
            unsafe {
                *(concurrent_addr as *mut u8).add(64) = 0x77;
            }
        });
        list.close();
        h.execute(&[&list]);

        assert!(backing[..16].iter().all(|&b| b == 0x11));
        assert!(backing[16..32].iter().all(|&b| b == 0xAA));
        assert_eq!(backing[64], 0x77);
        assert!(backing[65..128].iter().all(|&b| b == 0x11));
        // Also verify the initial 0x11 bytes ACTUALLY arrived in VRAM (not zeroed
        // by ReadProcessMemory failing on a PCIe BAR destination!).
        let rebar_mirror = shadow
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .expect("mirror populated after gpu_writes");
        assert!(
            rebar_mirror[..16].iter().all(|&b| b == 0x11),
            "initial title bytes did not reach ReBAR VRAM! got: {:02x?}",
            &rebar_mirror[..16]
        );

        list.reset();
        drop(shadow);
        unregister(resource, start);
    }

    #[test]
    fn rebar_dirty_bits_marks_and_coalesces_across_word_boundaries() {
        let mut bits = RebarDirtyBits::new(140);
        assert!(bits.take_runs().is_empty());

        // Single piece, cross-word range (62..67), and tail clamped at 140.
        bits.mark_pieces(3, 1);
        bits.mark_pieces(62, 5);
        bits.mark_pieces(138, 10);
        assert_eq!(bits.take_runs(), vec![(3, 1), (62, 5), (138, 2)]);
        assert!(bits.take_runs().is_empty());

        // Byte range straddling a 64 KiB boundary marks both pieces.
        bits.mark_byte_range(PAGE - 8, 16);
        bits.mark_byte_range(5 * PAGE, PAGE);
        assert_eq!(bits.take_runs(), vec![(0, 2), (5, 1)]);
        assert!(bits.take_runs().is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn rebar_gpu_hash_compare_skips_unchanged_pieces_and_merges_concurrent_writes() {
        let Some(h) = host() else {
            return; // no host GPU here
        };
        if h.rebar_shadow_budget() == 0 {
            // Requires a discrete GPU with D3D12 GPU upload heap (ReBAR) support.
            return;
        }
        let errors_before = h.d3d12_errors();
        let mut backing = vec![0x11u8; 4 * PAGE];
        let start = backing.as_ptr() as u64;
        let resource = 0xdead_beef_0004_usize;
        register(resource, start, backing.len() as u64);

        let shadow = find(start, backing.len() as u64).expect("shadow");
        assert!(
            shadow.is_rebar(),
            "expected ReBAR-backed shadow on ReBAR adapter"
        );

        let (src_aa, ptr_aa) = h.upload_buffer(16).expect("upload AA");
        let (src_bb, ptr_bb) = h.upload_buffer(16).expect("upload BB");
        let (src_11, ptr_11) = h.upload_buffer(16).expect("upload 11");
        // SAFETY: each upload buffer is mapped for 16 bytes.
        unsafe {
            std::ptr::write_bytes(ptr_aa, 0xAA, 16);
            std::ptr::write_bytes(ptr_bb, 0xBB, 16);
            std::ptr::write_bytes(ptr_11, 0x11, 16);
        }

        // Submission 1: declare the entire 256 KiB buffer as UAV-written, but
        // only modify piece 1 (+32..+48) and piece 3 (+64..+80) on the GPU
        // while the CPU concurrently writes piece 1 (+128) and piece 0 (+128).
        let stats0 = h.rebar_hash_stats();
        let list = h.new_list().expect("host list");
        shadow.refresh(&list, start, backing.len() as u64);
        shadow.gpu_writes(&list, start, backing.len() as u64);
        // SAFETY: live buffers and in-bounds 16-byte copies.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, (PAGE + 32) as u64, &src_aa, 0, 16);
            list.list
                .CopyBufferRegion(&shadow.buffer, (3 * PAGE + 64) as u64, &src_bb, 0, 16);
        }
        let base_addr = backing.as_mut_ptr() as usize;
        list.defer_host(move || {
            // SAFETY: `backing` stays alive for the duration of this test.
            unsafe {
                *(base_addr as *mut u8).add(128) = 0x55;
                *(base_addr as *mut u8).add(PAGE + 128) = 0x77;
            }
        });
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats1 = h.rebar_hash_stats();
        assert_eq!(stats1.compared_pieces - stats0.compared_pieces, 4);
        assert_eq!(
            stats1.changed_pieces - stats0.changed_pieces,
            2,
            "pieces 0 and 2 were untouched by the GPU and must be skipped"
        );
        assert_eq!(stats1.post_readback_calls - stats0.post_readback_calls, 1);
        assert_eq!(
            stats1.post_readback_bytes - stats0.post_readback_bytes,
            (2 * PAGE) as u64,
            "only the 2 changed 64 KiB pieces should be DMAed back"
        );
        assert_eq!(backing[128], 0x55);
        assert!(backing[PAGE + 32..PAGE + 48].iter().all(|&b| b == 0xAA));
        assert_eq!(backing[PAGE + 128], 0x77);
        assert!(backing[2 * PAGE..3 * PAGE].iter().all(|&b| b == 0x11));
        assert!(
            backing[3 * PAGE + 64..3 * PAGE + 80]
                .iter()
                .all(|&b| b == 0xBB)
        );

        // Submission 2: speculative UAV over all 256 KiB with zero GPU writes.
        // Piece 1 was pushed back to VRAM at the end of submission 1 (due to
        // the concurrent 0x77 write), so pre-hash must re-hash piece 1 and
        // post-compare must report 0 changed pieces and 0 readback DMAs.
        shadow.refresh(&list, start, backing.len() as u64);
        shadow.gpu_writes(&list, start, backing.len() as u64);
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats2 = h.rebar_hash_stats();
        assert_eq!(stats2.compared_pieces - stats1.compared_pieces, 4);
        assert_eq!(stats2.changed_pieces - stats1.changed_pieces, 0);
        assert_eq!(
            stats2.post_readback_calls - stats1.post_readback_calls,
            0,
            "0 changed pieces must trigger 0 readback DMAs"
        );
        assert_eq!(stats2.post_readback_bytes - stats1.post_readback_bytes, 0);
        assert_eq!(backing[128], 0x55);
        assert_eq!(backing[PAGE + 128], 0x77);

        // Submission 3: GPU overwrites byte PAGE + 128 back to 0x11 (replacing
        // the concurrent CPU 0x77 write). Only piece 1 must be flagged & DMAed.
        shadow.refresh(&list, start, backing.len() as u64);
        shadow.gpu_writes(&list, start, backing.len() as u64);
        // SAFETY: live buffers and in-bounds 16-byte copy.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, (PAGE + 128) as u64, &src_11, 0, 16);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats3 = h.rebar_hash_stats();
        assert_eq!(stats3.compared_pieces - stats2.compared_pieces, 4);
        assert_eq!(stats3.changed_pieces - stats2.changed_pieces, 1);
        assert_eq!(
            stats3.post_readback_bytes - stats2.post_readback_bytes,
            PAGE as u64
        );
        assert!(backing[PAGE + 128..PAGE + 144].iter().all(|&b| b == 0x11));

        // Submission 4: CPU modifies piece 2 and refreshes it, then declares a
        // speculative UAV write with no GPU writes. Pre-hash must refresh piece
        // 2's hash so post-compare sees 0 changes and 0 readback DMAs.
        backing[2 * PAGE + 16..2 * PAGE + 32].fill(0xCC);
        shadow.refresh(&list, start, backing.len() as u64);
        shadow.gpu_writes(&list, start, backing.len() as u64);
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats4 = h.rebar_hash_stats();
        assert_eq!(stats4.compared_pieces - stats3.compared_pieces, 4);
        assert_eq!(stats4.changed_pieces - stats3.changed_pieces, 0);
        assert_eq!(stats4.post_readback_calls - stats3.post_readback_calls, 0);
        assert_eq!(stats4.post_readback_bytes - stats3.post_readback_bytes, 0);
        assert!(
            backing[2 * PAGE + 16..2 * PAGE + 32]
                .iter()
                .all(|&b| b == 0xCC)
        );
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during ReBAR hash-compare test"
        );

        drop(shadow);
        unregister(resource, start);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_drops_shadow_immediately_when_idle() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0101_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);
        unregister(resource, start);
        assert_eq!(weak.strong_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_retires_shadow_until_active_epoch_finishes() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0102_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);

        let list = host.new_list().expect("host list");
        list.mark_work();

        unregister(resource, start);
        assert_eq!(
            weak.strong_count(),
            1,
            "unregistered shadow must wait for active epochs that recorded after its birth"
        );

        list.close();
        list.reset();
        assert_eq!(weak.strong_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_drops_new_shadow_despite_older_pinned_list() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let stale = host.new_list().expect("stale list");
        stale.mark_work();

        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0103_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);

        unregister(resource, start);
        assert_eq!(
            weak.strong_count(),
            0,
            "shadow created after stale list's last work must not leak behind it"
        );
        stale.close();
        stale.reset();
    }

    #[test]
    fn loose_budget_scales_across_ram_tiers() {
        assert_eq!(loose_budget_for_ram(None), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(8 * GIB)), 128 * MIB);
        assert_eq!(loose_budget_for_ram(Some(16 * GIB)), 128 * MIB);
        assert_eq!(loose_budget_for_ram(Some(16 * GIB + 1)), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(32 * GIB)), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(32 * GIB + 1)), 512 * MIB);
        assert_eq!(loose_budget_for_ram(Some(64 * GIB)), 512 * MIB);
    }

    #[cfg(windows)]
    #[test]
    fn loose_evicts_lru_idle_windows_over_budget_and_skips_in_use_or_pending() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let cache = std::sync::RwLock::new(LooseCache::new());
        let budget = 2 * WINDOW;
        let base = 0x1000_0000_0000_u64;
        let win = |i: u64| base + i * WINDOW;

        let weak0 = Arc::downgrade(
            &loose_in(&cache, win(0), win(0) + 64, false, false, budget).expect("win 0"),
        );
        let weak1 = Arc::downgrade(
            &loose_in(&cache, win(1), win(1) + 64, false, false, budget).expect("win 1"),
        );
        assert_eq!(weak0.strong_count(), 1);
        assert_eq!(weak1.strong_count(), 1);

        // Touch window 0 so window 1 becomes the least-recently-used entry.
        drop(loose_in(&cache, win(0), win(0) + 64, false, false, budget).expect("touch win 0"));

        // Inserting window 2 exceeds the 2-window budget and immediately drops
        // idle LRU window 1 through `retire`.
        let weak2 = Arc::downgrade(
            &loose_in(&cache, win(2), win(2) + 64, false, false, budget).expect("win 2"),
        );
        assert_eq!(weak1.strong_count(), 0, "idle LRU window 1 must be evicted");
        assert_eq!(weak0.strong_count(), 1, "MRU window 0 must stay cached");
        assert_eq!(weak2.strong_count(), 1, "new window 2 must stay cached");
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.total_bytes, budget);
            assert_eq!(guard.entries.len(), 2);
        }

        // Inserting window 3 evicts idle LRU window 0.
        let weak3 = Arc::downgrade(
            &loose_in(&cache, win(3), win(3) + 64, false, false, budget).expect("win 3"),
        );
        {
            let guard = cache.read().unwrap();
            assert!(!guard.entries.contains_key(&(win(0), WINDOW)));
            assert_eq!(guard.total_bytes, budget);
        }
        assert_eq!(weak0.strong_count(), 0, "idle LRU window 0 must be evicted");

        // Pin window 2 via an external Arc (strong_count > 1) and window 3 via
        // a pending GPU UAV copy-back range; neither may be evicted.
        let held2 = loose_in(&cache, win(2), win(2) + 64, false, false, budget).expect("hold win 2");
        {
            let s3 = loose_in(&cache, win(3), win(3) + 64, false, false, budget)
                .expect("mark win 3 pending");
            s3.pending
                .lock()
                .unwrap()
                .push((win(3) as usize, win(3) as usize + 64));
        }
        let weak4 = Arc::downgrade(
            &loose_in(&cache, win(4), win(4) + 64, false, false, budget).expect("win 4"),
        );
        assert_eq!(
            weak2.strong_count(),
            2,
            "in-use window 2 must not be evicted"
        );
        assert_eq!(
            weak3.strong_count(),
            1,
            "window 3 with pending copy-back must not be evicted"
        );
        assert_eq!(weak4.strong_count(), 1);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 3);
            assert_eq!(guard.total_bytes, 3 * WINDOW);
        }

        // Once window 2 is released and window 3's pending copy-back clears,
        // the next insert drains both stale windows back down to budget.
        drop(held2);
        weak3
            .upgrade()
            .expect("win 3 still live")
            .pending
            .lock()
            .unwrap()
            .clear();
        let weak5 = Arc::downgrade(
            &loose_in(&cache, win(5), win(5) + 64, false, false, budget).expect("win 5"),
        );
        assert_eq!(weak2.strong_count(), 0);
        assert_eq!(weak3.strong_count(), 0);
        assert_eq!(weak4.strong_count(), 1);
        assert_eq!(weak5.strong_count(), 1);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 2);
            assert_eq!(guard.total_bytes, budget);
        }

        // Re-accessing previously evicted window 1 cleanly recreates its shadow.
        let recreated = loose_in(&cache, win(1) + 128, win(1) + 256, false, false, budget)
            .expect("recreate win 1");
        assert_eq!(recreated.start, win(1));
        assert_eq!(recreated.len, WINDOW);
        assert_ne!(recreated.gpu(win(1) + 128), 0);
        assert_eq!(weak4.strong_count(), 0);
        assert_eq!(weak5.strong_count(), 1);
        drop(recreated);
        // Evicted window while an active list recorded after its birth stays
        // retired until that list resets.
        let list = host.new_list().expect("host list");
        list.mark_work();
        drop(loose_in(&cache, win(6), win(6) + 64, false, false, budget).expect("win 6"));
        assert_eq!(weak5.strong_count(), 1, "evicted window waits for active epoch");
        list.close();
        list.reset();
        assert_eq!(weak5.strong_count(), 0);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 2);
            assert_eq!(guard.total_bytes, budget);
            assert!(guard.entries.contains_key(&(win(1), WINDOW)));
            assert!(guard.entries.contains_key(&(win(6), WINDOW)));
        }
    }

    #[test]
    fn relay_merge_rebar_merges_gpu_writes_without_touching_concurrent_cpu_writes() {
        let Some(host) = host() else {
            return;
        };
        let mut title = vec![0x11u8; 4096];
        let start = title.as_mut_ptr() as u64;
        let Some((buffer, data)) = host.shared_buffer(4096) else {
            return;
        };
        // SAFETY: live buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, start, 4096, Some(0));
        // Seed mirror with initial bytes [0x11; 4096] and a pending range.
        {
            let rebar = shadow.rebar.as_ref().expect("rebar state");
            *rebar.mirror.lock().unwrap() = Some(vec![0x11u8; 4096]);
            shadow
                .pending
                .lock()
                .unwrap()
                .push((start as usize, start as usize + 4096));
        }
        // Concurrent CPU write at byte 200; GPU write at byte 100.
        title[200] = 0x77;
        let mut gpu_slice = vec![0x11u8; 4096];
        gpu_slice[100] = 0xAA;

        let written = shadow.relay_merge_rebar(&[(0, 4096)], &[&gpu_slice]);
        assert!(written > 0);
        assert_eq!(title[100], 0xAA, "GPU write must land in title memory");
        assert_eq!(title[200], 0x77, "concurrent CPU write must be preserved");
        assert_eq!(
            shadow.pending.lock().unwrap().len(),
            1,
            "relay_merge_rebar must leave pending cleanup to copy_back_done"
        );
        // Mirror now holds 0xAA at byte 100, so a repeat merge with the same
        // GPU bytes writes nothing.
        assert_eq!(shadow.relay_merge_rebar(&[(0, 4096)], &[&gpu_slice]), 0);
    }

    #[test]
    fn relay_merge_rebar_parallel_scan_merges_large_buffer_across_chunks() {
        let Some(host) = host() else {
            return;
        };
        const LEN: usize = 8 << 20;
        let mut title = vec![0x11u8; LEN];
        let start = title.as_mut_ptr() as u64;
        let Some((buffer, data)) = host.shared_buffer(LEN as u64) else {
            return;
        };
        // SAFETY: live buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, start, LEN as u64, Some(0));
        {
            let rebar = shadow.rebar.as_ref().expect("rebar state");
            *rebar.mirror.lock().unwrap() = Some(vec![0x11u8; LEN]);
            shadow
                .pending
                .lock()
                .unwrap()
                .push((start as usize, start as usize + LEN));
        }

        // Concurrent CPU writes: two in GPU-written pages and one in an untouched page.
        let mid = 4 << 20;
        title[200] = 0x77;
        title[mid + 32] = 0x88;
        title[(2 << 20) + 500] = 0x99;

        // GPU writes across multiple PAGE chunks and straddling the 4 MiB worker cut.
        let mut gpu_slice = vec![0x11u8; LEN];
        gpu_slice[100..108].fill(0xAA);
        gpu_slice[PAGE - 4..PAGE + 4].fill(0xBB);
        gpu_slice[mid - 8..mid + 8].fill(0xCC);
        gpu_slice[LEN - 16..LEN].fill(0xDD);

        let written = shadow.relay_merge_rebar(&[(0, LEN)], &[&gpu_slice]);
        assert!(written > 0);
        assert!(title[100..108].iter().all(|&b| b == 0xAA));
        assert!(title[PAGE - 4..PAGE + 4].iter().all(|&b| b == 0xBB));
        assert!(title[mid - 8..mid + 8].iter().all(|&b| b == 0xCC));
        assert!(title[LEN - 16..LEN].iter().all(|&b| b == 0xDD));
        assert_eq!(title[200], 0x77, "concurrent CPU write in chunk 0 must be preserved");
        assert_eq!(title[mid + 32], 0x88, "concurrent CPU write in chunk 1 must be preserved");
        assert_eq!(
            title[(2 << 20) + 500],
            0x99,
            "concurrent CPU write in untouched page must be preserved"
        );
        assert_eq!(title[0], 0x11);
        assert_eq!(title[108], 0x11);
        assert_eq!(title[mid + 8], 0x11);
        assert_eq!(title[LEN - 17], 0x11);
        assert_eq!(
            shadow.pending.lock().unwrap().len(),
            1,
            "relay_merge_rebar must leave pending cleanup to copy_back_done"
        );
        // Repeat merge with identical GPU bytes finds nothing to write.
        assert_eq!(shadow.relay_merge_rebar(&[(0, LEN)], &[&gpu_slice]), 0);

        // Multi-span parallel path (>= 4 MiB across disjoint spans).
        let span0 = (0usize, mid);
        let span1 = (mid + PAGE, LEN);
        gpu_slice[3 * PAGE + 16] = 0xEE;
        gpu_slice[(6 << 20) + 42] = 0xEF;
        let spans = [span0, span1];
        let slices = [&gpu_slice[span0.0..span0.1], &gpu_slice[span1.0..span1.1]];
        let written2 = shadow.relay_merge_rebar(&spans, &slices);
        assert!(written2 > 0);
        assert_eq!(title[3 * PAGE + 16], 0xEE);
        assert_eq!(title[(6 << 20) + 42], 0xEF);
        assert_eq!(title[200], 0x77);
        assert_eq!(title[(2 << 20) + 500], 0x99);
        assert_eq!(shadow.relay_merge_rebar(&spans, &slices), 0);
    }

    #[test]
    fn wide_copy_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_copy_enabled(None, Some(4)));
        assert!(wide_copy_enabled(None, Some(16)));
        assert!(!wide_copy_enabled(None, Some(2)));
        assert!(!wide_copy_enabled(None, None));
        assert!(wide_copy_enabled(Some("1"), Some(1)));
        assert!(!wide_copy_enabled(Some("0"), Some(16)));
        assert!(wide_copy_enabled(Some("auto"), Some(4)));
    }

    #[test]
    fn chunk_copy_segs_preserves_boundaries_and_mirror_offsets() {
        let segs = [
            CopySeg {
                src: 0x1000,
                dst: 0x2000,
                mirror: 0x3000,
                len: 700_000,
            },
            CopySeg {
                src: 0x1000 + 800_000,
                dst: 0x2000 + 800_000,
                mirror: 0x3000 + 800_000,
                len: 123,
            },
            CopySeg {
                src: 0x1000 + 900_000,
                dst: 0x2000 + 900_000,
                mirror: 0,
                len: 1_500_000,
            },
        ];
        let buckets = chunk_copy_segs(&segs, 256 << 10);
        assert!(buckets.len() >= 8);
        let mut flat: Vec<CopySeg> = Vec::new();
        for bucket in &buckets {
            let sum: usize = bucket.iter().map(|s| s.len).sum();
            assert!(sum > 0 && sum <= 256 << 10);
            flat.extend_from_slice(bucket);
        }
        // Reassemble adjacent sub-segments and verify exact equality with input.
        let mut merged: Vec<CopySeg> = Vec::new();
        for seg in flat {
            match merged.last_mut() {
                Some(last)
                    if last.src + last.len == seg.src
                        && last.dst + last.len == seg.dst
                        && ((last.mirror == 0 && seg.mirror == 0)
                            || (last.mirror != 0 && last.mirror + last.len == seg.mirror)) =>
                {
                    last.len += seg.len;
                }
                _ => merged.push(seg),
            }
        }
        assert_eq!(merged, segs);
    }

    #[cfg(windows)]
    #[test]
    fn parallel_shadow_update_preserves_boundaries_unmapped_pages_and_card_dirty() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let Some(h) = host() else {
            return;
        };
        const LEN: usize = 4 << 20; // 4 MiB (> WIDE_COPY_MIN)
        // SAFETY: reserve a 4 MiB address range and commit parts around a hole.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>()
        };
        assert!(!base.is_null());
        let mut expected = vec![0u8; LEN];
        // Leave [1 MiB, 1 MiB + 128 KiB) unmapped (hole) and keep [2 MiB, 2 MiB + 128 KiB)
        // as a pending shader-write range that `update` must skip completely.
        let hole = (1 << 20, (1 << 20) + 2 * PAGE);
        let pending = (2 << 20, (2 << 20) + 2 * PAGE);
        for &(from, to) in &[(0usize, hole.0), (hole.1, LEN)] {
            // SAFETY: committing subranges inside our reserved allocation.
            let p = unsafe {
                VirtualAlloc(base.add(from).cast(), to - from, MEM_COMMIT, PAGE_READWRITE)
            };
            assert_eq!(p, unsafe { base.add(from).cast() });
            // Fill with a non-repeating byte pattern so any offset slip is caught.
            for off in from..to {
                let v = ((off.wrapping_mul(31) ^ (off >> 12)) & 0xff) as u8;
                // SAFETY: committed above.
                unsafe { *base.add(off) = v };
                expected[off] = v;
            }
        }

        let Some((staging, data)) = h.shared_buffer(LEN as u64) else {
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
            return;
        };
        let Some((on_card, _)) = h.shared_buffer(LEN as u64) else {
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
            return;
        };
        // Pre-fill shadow staging with a sentinel so untouched regions are verifiable.
        // SAFETY: `data` maps `LEN` bytes.
        unsafe { std::ptr::write_bytes(data, 0xA5, LEN) };
        let gpu_va = unsafe { on_card.GetGPUVirtualAddress() };
        let mut shadow = Shadow::new(on_card, data, gpu_va, base as u64, LEN as u64, None);
        shadow.card = Some(Card {
            staging,
            before: Mutex::default(),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            size: LEN as u64,
        });
        shadow
            .pending
            .lock()
            .unwrap()
            .push((base as usize + pending.0, base as usize + pending.1));

        // Refresh [37, LEN - 41): tests unaligned ends, hole zeroing, pending gap,
        // and multi-chunk parallel copy.
        let req_lo = 37usize;
        let req_hi = LEN - 41;
        let copied = shadow.update(base as usize + req_lo, req_hi - req_lo);
        assert_eq!(copied, (pending.0 - req_lo) + (req_hi - pending.1));

        // SAFETY: `data` maps `LEN` bytes and `update` has finished all workers.
        let got = unsafe { std::slice::from_raw_parts(data, LEN) };
        assert!(
            got[..req_lo].iter().all(|&b| b == 0xA5),
            "bytes before refreshed start must stay untouched"
        );
        assert!(
            got[req_hi..].iter().all(|&b| b == 0xA5),
            "bytes after refreshed end must stay untouched"
        );
        assert!(
            got[pending.0..pending.1].iter().all(|&b| b == 0xA5),
            "pending shader-write range must stay untouched"
        );
        assert_eq!(got[req_lo..hole.0], expected[req_lo..hole.0]);
        assert!(
            got[hole.0..hole.1].iter().all(|&b| b == 0),
            "unmapped hole must be zeroed"
        );
        assert_eq!(got[hole.1..pending.0], expected[hole.1..pending.0]);
        assert_eq!(got[pending.1..req_hi], expected[pending.1..req_hi]);

        let dirty = shadow.card.as_ref().unwrap().dirty.lock().unwrap().clone();
        assert_eq!(dirty, vec![(req_lo, pending.0), (pending.1, req_hi)]);

        // SAFETY: freeing our test allocation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[cfg(windows)]
    #[test]
    fn parallel_shadow_update_rebar_mirror_syncs_mirror_and_vram() {
        let Some(h) = host() else {
            return;
        };
        const LEN: usize = 3 << 20; // 3 MiB (> WIDE_COPY_MIN)
        let title: Vec<u8> = (0..LEN)
            .map(|i| ((i.wrapping_mul(17) ^ (i >> 8)) & 0xff) as u8)
            .collect();
        let start = title.as_ptr() as u64;
        let Some((buffer, data)) = h.shared_buffer(LEN as u64) else {
            return;
        };
        // SAFETY: `data` maps `LEN` bytes.
        unsafe { std::ptr::write_bytes(data, 0x5A, LEN) };
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, start, LEN as u64, Some(0));
        {
            let rebar = shadow.rebar.as_ref().expect("rebar state");
            *rebar.mirror.lock().unwrap() = Some(vec![0x5Au8; LEN]);
        }

        // Disjoint runs totaling > 2 MiB that straddle chunk cuts.
        let runs = [
            (start as usize + 19, start as usize + (1 << 20) + 123),
            (start as usize + (1 << 20) + PAGE, start as usize + LEN - 29),
        ];
        let copied = shadow.copy_runs(&runs);
        let expected_copied: usize = runs.iter().map(|&(a, b)| b - a).sum();
        assert_eq!(copied, expected_copied);

        let mirror = shadow
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        // SAFETY: `data` maps `LEN` bytes.
        let vram = unsafe { std::slice::from_raw_parts(data, LEN) };
        let r0 = (runs[0].0 - start as usize, runs[0].1 - start as usize);
        let r1 = (runs[1].0 - start as usize, runs[1].1 - start as usize);
        assert_eq!(mirror[r0.0..r0.1], title[r0.0..r0.1]);
        assert_eq!(vram[r0.0..r0.1], title[r0.0..r0.1]);
        assert_eq!(mirror[r1.0..r1.1], title[r1.0..r1.1]);
        assert_eq!(vram[r1.0..r1.1], title[r1.0..r1.1]);
        assert!(mirror[..r0.0].iter().all(|&b| b == 0x5A));
        assert!(vram[..r0.0].iter().all(|&b| b == 0x5A));
        assert!(mirror[r0.1..r1.0].iter().all(|&b| b == 0x5A));
        assert!(vram[r0.1..r1.0].iter().all(|&b| b == 0x5A));
        assert!(mirror[r1.1..].iter().all(|&b| b == 0x5A));
        assert!(vram[r1.1..].iter().all(|&b| b == 0x5A));
    }

    #[cfg(windows)]
    #[test]
    fn wide_refresh_reentrancy_and_batch_do_not_deadlock() {
        let Some(h) = host() else {
            return;
        };
        const BIG: usize = 2 << 20;
        const SMALL: usize = 64 << 10;
        let mut backing: Vec<Vec<u8>> = Vec::new();
        let mut shadows: Vec<Arc<Shadow>> = Vec::new();
        for i in 0..8usize {
            let len = if i < 4 { BIG } else { SMALL };
            let buf: Vec<u8> = (0..len)
                .map(|j| (((i + 1) * 29 + j * 7 + (j >> 9)) & 0xff) as u8)
                .collect();
            let start = buf.as_ptr() as u64;
            let Some((res, data)) = h.shared_buffer(len as u64) else {
                return;
            };
            let gpu_va = unsafe { res.GetGPUVirtualAddress() };
            let shadow = Arc::new(Shadow::new(res, data, gpu_va, start, len as u64, None));
            backing.push(buf);
            shadows.push(shadow);
        }

        // 1. Direct re-entrancy stress: invoke multi-MB `Shadow::update` from
        // inside `wide_run` workers concurrently. Must never deadlock `workers()`.
        let jobs: Vec<Job> = shadows
            .iter()
            .map(|s| {
                let s = s.clone();
                Box::new(move || {
                    let n = s.update(s.start as usize, s.len as usize);
                    assert_eq!(n, s.len as usize);
                }) as Job
            })
            .collect();
        wide_run(jobs);
        for (shadow, src) in shadows.iter().zip(&backing) {
            // SAFETY: `shadow.data` maps `shadow.len` bytes.
            let got = unsafe { std::slice::from_raw_parts(shadow.data, src.len()) };
            assert_eq!(got, src.as_slice());
        }

        // 2. Two-phase `refresh_batch` mixing big + small shadows (parallel plan
        // + single flat parallel copy pass).
        for shadow in &shadows {
            // SAFETY: clear staging before batch refresh.
            unsafe { std::ptr::write_bytes(shadow.data, 0, shadow.len as usize) };
        }
        let batch_items: Vec<(Arc<Shadow>, usize, usize)> = shadows
            .iter()
            .map(|s| (s.clone(), s.start as usize, (s.start + s.len) as usize))
            .collect();
        let expected_total: usize = backing.iter().map(Vec::len).sum();
        assert_eq!(refresh_batch(batch_items, true), expected_total);
        for (shadow, src) in shadows.iter().zip(&backing) {
            // SAFETY: `shadow.data` maps `shadow.len` bytes.
            let got = unsafe { std::slice::from_raw_parts(shadow.data, src.len()) };
            assert_eq!(got, src.as_slice());
        }
    }

    #[test]
    fn view_marks_runs_are_reused_and_joined() {
        assert_eq!(mark_pages(0, 0), 1);
        assert_eq!(mark_pages(0, 1), 1);
        assert_eq!(mark_pages(0, PAGE as u64), 1);
        assert_eq!(mark_pages(0, PAGE as u64 + 1), 2);
        // Unaligned start crossing a 64 KiB boundary spans 2 pages.
        assert_eq!(mark_pages((PAGE as u64) - 4, 8), 2);
        assert_eq!(mark_pages((PAGE as u64) - 4, 4), 1);

        let mut pool = Marks {
            free: BTreeMap::new(),
            end: 0,
        };
        let a = pool.take(4).expect("run a");
        let b = pool.take(6).expect("run b");
        let c = pool.take(2).expect("run c");
        assert_eq!((a, b, c), (0, 4, 10));
        assert_eq!(pool.end, 12);

        // Give back outer runs, then middle: all three coalesce into 0..12.
        pool.give_back(a, 4);
        pool.give_back(c, 2);
        assert_eq!(pool.free.len(), 2);
        pool.give_back(b, 6);
        assert_eq!(pool.free.len(), 1);
        assert_eq!(pool.free.get(&0), Some(&12));

        // Reusing a slice leaves the remainder in the free list.
        assert_eq!(pool.take(5), Some(0));
        assert_eq!(pool.free.get(&5), Some(&7));
        assert_eq!(pool.take(7), Some(5));
        assert!(pool.free.is_empty());

        // The last bit (`VIEW_WRITE_BITS - 1`) is reserved as an overflow sink.
        let mut full = Marks {
            free: BTreeMap::new(),
            end: victory_shader::emit::VIEW_WRITE_BITS - 2,
        };
        assert_eq!(full.take(2), None);
        assert_eq!(full.take(1), Some(victory_shader::emit::VIEW_WRITE_BITS - 2));
        assert_eq!(full.take(1), None);
    }

    #[test]
    fn view_marks_filter_spans_respects_marked_unmarked_and_multi_flight() {
        // Shadow starting unaligned inside title page 0x10 (+0x1234) covering 5 title pages (0x10..=0x14).
        let base_page = 0x10usize;
        let start = (base_page * PAGE + 0x1234) as u64;
        let len = (4 * PAGE) as u64; // ends at 0x14 * PAGE + 0x1234
        let mark_base = 30u32; // crosses a 32-bit word boundary (bits 30..=34)

        let declared = [(start as usize, (start + len) as usize)];

        // Flight 1 marks page +0 (bit 30) and page +3 (bit 33); flight 2 marks page +1 (bit 31).
        let mut bm1 = [0u32; 2];
        bm1[0] |= 1 << 30;
        bm1[1] |= 1 << (33 - 32);
        let mut bm2 = [0u32; 2];
        bm2[0] |= 1 << 31;

        // Multi-flight union: pages +0 and +1 merge into one span; page +3 is separate.
        let (filtered, decl_pages, marked_pages) =
            filter_title_ranges_by_marks(start, len, Some(mark_base), &declared, &[], &[&bm1, &bm2]);
        assert_eq!(decl_pages, 5);
        assert_eq!(marked_pages, 3);
        assert_eq!(
            filtered,
            vec![
                (start as usize, (base_page + 2) * PAGE),
                ((base_page + 3) * PAGE, (base_page + 4) * PAGE),
            ]
        );

        // Unmarked range inside page +2 (not set in bitmap) is preserved without inflating marked_pages.
        let unmarked = [((base_page + 2) * PAGE + 0x400, (base_page + 2) * PAGE + 0x800)];
        let (filtered_with_unmarked, decl2, marked2) = filter_title_ranges_by_marks(
            start,
            len,
            Some(mark_base),
            &declared,
            &unmarked,
            &[&bm1],
        );
        assert_eq!(decl2, 5);
        assert_eq!(marked2, 2);
        assert_eq!(
            filtered_with_unmarked,
            vec![
                (start as usize, (base_page + 1) * PAGE),
                ((base_page + 2) * PAGE + 0x400, (base_page + 2) * PAGE + 0x800),
                ((base_page + 3) * PAGE, (base_page + 4) * PAGE),
            ]
        );

        // Zero bits set -> empty filtered spans (0% readback!).
        let zero = [0u32; 2];
        let (none, decl3, marked3) =
            filter_title_ranges_by_marks(start, len, Some(mark_base), &declared, &[], &[&zero]);
        assert_eq!(decl3, 5);
        assert_eq!(marked3, 0);
        assert!(none.is_empty());

        // Missing mark_base or empty bitmaps -> safe fallback to full declared ranges.
        let (fb1, d1, m1) =
            filter_title_ranges_by_marks(start, len, None, &declared, &[], &[&zero]);
        assert_eq!((fb1, d1, m1), (declared.to_vec(), 5, 5));
        let (fb2, d2, m2) =
            filter_title_ranges_by_marks(start, len, Some(mark_base), &declared, &[], &[]);
        assert_eq!((fb2, d2, m2), (declared.to_vec(), 5, 5));
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders (`fcdefcfc`): the refresh
    /// before the submission runs still brings it to the copy, a shader's
    /// write still goes back, and a title write made after the refresh is not
    /// undone.
    #[cfg(windows)]
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let Some(h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        // SAFETY: fresh write-watched allocation freed at the end of the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize
        };
        assert_ne!(p, 0);
        let Some((buffer, data)) = h.shared_buffer(len as u64) else {
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            return;
        };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu_va, p as u64, len as u64, None);
        let title = |at: usize| (p + at) as *mut u8;
        let host_byte = |at: usize| unsafe { copy.data.add(at).read_volatile() };

        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, |_, _| None);
        // SAFETY: bytes inside our write-watched allocation.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host_byte(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        // SAFETY: inside `copy.data` mapping.
        unsafe { copy.data.add(10).write_volatile(0xaa) };
        // SAFETY: inside our write-watched allocation.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        copy.copy_back_done(&[(p, p + len)]);
        // SAFETY: inside our write-watched allocation.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
    }

    /// Shaders only declared during preparation have not run when the card
    /// uploads the refresh. Diffing individual bytes in this case produces
    /// thousands of needless CopyBufferRegion calls for strided constants.
    #[cfg(windows)]
    #[test]
    fn card_busy_refresh_coalesces_current_submission_constants() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = 16 * PAGE;
        // SAFETY: fresh write-watched allocation, released below. Use high
        // addresses so other tests' fresh allocations do not reuse the
        // address of a page still recorded in the watch journal.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize
        };
        assert_ne!(p, 0);
        let mut held = vec![0u8; len];
        let Some(copy) =
            Shadow::new_card_for_test(held.as_mut_ptr(), p as u64, len as u64, Vec::new(), Vec::new())
        else {
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            return; // no host GPU here
        };
        let card = copy.card.as_ref().unwrap();
        copy.update(p, len);
        card.dirty.lock().unwrap().clear();
        copy.writing(p, len, |_, _| None);

        let (lo, hi) = (19, len - 23);
        for i in (lo..hi).step_by(16) {
            // SAFETY: inside the write-watched allocation.
            unsafe { ((p + i) as *mut u8).write_volatile(0x55) };
        }
        // A write outside the refresh must not be uploaded.
        unsafe { ((p + lo - 1) as *mut u8).write_volatile(0xaa) };
        let copied = copy.update(p + lo, hi - lo);
        let uploads = joined(std::mem::take(&mut *card.dirty.lock().unwrap()));
        drop(copy);
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };

        assert!(held[lo..hi].iter().step_by(16).all(|&byte| byte == 0x55));
        assert_eq!(held[lo - 1], 0);
        assert_eq!(uploads.len(), 1, "expected one contiguous upload, got {} fragments", uploads.len());
        assert_eq!(uploads, [(lo, hi)]);
        assert_eq!(copied, hi - lo);
    }

    #[test]
    fn card_busy_refresh_preserves_bytes_from_older_flights() {
        let mut flights = Flights::default();
        let shared = (108, 120);
        let older_only = (128, 133);
        flights.declare(shared, 1);
        flights.declare(older_only, 1);
        flights.declare((140, 150), 1);
        flights.declare(shared, 2);
        flights.declare((100, 140), 2);
        let protected = flights.older_ranges(2, 100, 140);
        assert_eq!(protected, [(108, 120), (128, 133)]);

        let held = [0u8; 40];
        let mut title = held;
        let mut device = held;
        // Simulate two GPU writes since staging last matched the device,
        // with new title bytes on either side of each write.
        for at in [9, 30] {
            device[at] = 0xaa;
            title[at - 1] = 0x55;
            title[at + 1] = 0x55;
        }
        for (at, len) in card_busy_runs(&title, &held, !protected.is_empty()) {
            device[at..at + len].copy_from_slice(&title[at..at + len]);
        }
        assert_eq!(device[8..11], [0x55, 0xaa, 0x55]);
        assert_eq!(device[29..32], [0x55, 0xaa, 0x55]);

        // Landing the older flight frees both ranges for batching, even
        // though the current submission still declares the shared range.
        flights.done(&[shared, older_only]);
        assert!(flights.older_ranges(2, 100, 140).is_empty());
    }

    /// ReBAR mirror counterpart to `title_writes_under_a_busy_range_are_copied`:
    /// title writes under a pending shader-write range update both ReBAR VRAM
    /// and the RAM mirror without overwriting concurrent GPU writes beside them.
    #[cfg(windows)]
    #[test]
    fn title_writes_under_a_busy_range_are_copied_for_rebar_mirror() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let Some(h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        // SAFETY: fresh write-watched allocation freed at the end of the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize
        };
        assert_ne!(p, 0);
        let Some((buffer, data)) = h.shared_buffer(len as u64) else {
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            return;
        };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let copy = Shadow::new(buffer, data, gpu_va, p as u64, len as u64, Some(0));

        copy.update(p, len);
        copy.writing(p, len, |_, size| Some(vec![0u8; size as usize]));
        // Simulate an in-flight shader write to VRAM at PAGE + 24 (not in mirror).
        // SAFETY: inside `copy.data` mapping.
        unsafe { copy.data.add(PAGE + 24).write_volatile(0xbb) };
        // Title writes a constant at PAGE + 20.
        // SAFETY: inside our write-watched allocation.
        unsafe { ((p + PAGE + 20) as *mut u8).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        // SAFETY: inside `copy.data` mapping.
        unsafe {
            assert_eq!(copy.data.add(PAGE + 20).read_volatile(), 0x55);
            assert_eq!(
                copy.data.add(PAGE + 24).read_volatile(),
                0xbb,
                "concurrent GPU byte beside the title write must stay untouched"
            );
        }
        let mirror = copy
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        assert_eq!(mirror[PAGE + 20], 0x55, "mirror must stay in sync with title write");
        assert_eq!(mirror[PAGE + 24], 0x00, "mirror must keep pre-shader byte for copy-back");
        copy.clear_pending_rebar();
        // SAFETY: freeing our test allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[cfg(windows)]
    #[test]
    fn reregistering_a_title_buffer_retires_its_old_shadow_by_epoch() {
        let Some(h) = host() else {
            return;
        };
        const LEN: u64 = 2 << 20;
        let resource = 0xdead_beef_0004usize;
        let start = 0x79f0_0000_0000u64;

        register(resource, start, LEN);
        let first = find(start, 1).expect("shadow created for registered buffer");
        assert_eq!(first.left(start + 0x10_0000), LEN - 0x10_0000);
        let weak = Arc::downgrade(&first);
        let born = first.born_seq;
        drop(first);

        let list = h.new_list().expect("host list");
        list.mark_work();
        assert!(list.has_epoch());

        // Re-register at the exact same address and resource while `list` holds an epoch.
        register(resource, start, LEN);
        assert_eq!(
            weak.strong_count(),
            1,
            "replaced Shadow (born_seq={born}) must be retired, not dropped while an active list epoch may name it"
        );

        list.close();
        list.reset();
        assert_eq!(
            weak.strong_count(),
            0,
            "retired Shadow must drop once its blocking epoch finishes"
        );

        unregister(resource, start);
    }

    #[cfg(windows)]
    #[test]
    fn stretch_opens_contiguous_section_views_and_unmaps_cleanly() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_COMMIT, MEM_FREE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE,
            VirtualQuery,
        };
        if host().is_none() {
            return;
        }
        const LEN: usize = STRETCH_LEAST;
        const PAGES: usize = LEN / PAGE;
        // SAFETY: committed pagefile section closed at the end of the test.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        let section_id = section as usize;
        let title_base = 0x79e0_0000_0000usize;

        crate::inplace::title_memory_section(section_id, LEN as u64, true);
        for k in 0..PAGES {
            crate::inplace::title_memory_view(
                title_base + k * PAGE,
                PAGE,
                section_id,
                (k * PAGE) as u64,
            );
        }

        // `VICTORY_STRETCH=0` opt-out skips stretch creation.
        assert_eq!(
            stretch_enabled(),
            std::env::var_os("VICTORY_STRETCH").is_none_or(|v| v != "0")
        );
        assert!(
            stretch_create_in(
                &STRETCHES,
                title_base as u64,
                title_base as u64 + 64,
                false,
                false,
            )
            .is_none()
        );
        assert_eq!(stretch_stats(), (0, 0));

        // Contiguous 64 KiB views open one in-place stretch heap over our own view.
        let s = stretch_create_in(
            &STRETCHES,
            title_base as u64 + 128,
            title_base as u64 + 256,
            false,
            true,
        )
        .expect("contiguous 4 MiB section views should open a stretch");
        assert!(s.place.is_some());
        assert_eq!(s.span(), (title_base as u64, LEN as u64));
        let view_addr = s
            .stretch_view
            .as_ref()
            .expect("stretch must track its mapped view")
            .addr;
        assert_eq!(stretch_stats(), (1, LEN as u64));

        // Later asks anywhere in the stretch reuse the same `Arc<Shadow>`.
        let hit = stretch_find_in(
            &STRETCHES,
            (title_base + 2 * (1 << 20)) as u64,
            (title_base + 2 * (1 << 20) + 64) as u64,
            false,
        )
        .expect("subsequent ask inside stretch must hit");
        assert!(Arc::ptr_eq(&s, &hit));
        drop(hit);

        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: querying our live mapped view.
        assert_ne!(
            unsafe { VirtualQuery(view_addr as *const _, &mut info, size_of_val(&info)) },
            0
        );
        assert_eq!(info.State, MEM_COMMIT);

        let weak = Arc::downgrade(&s);
        drop(s);

        // Path A: remapping a single 64 KiB page inside the 4 MiB stretch lets go
        // of the stretch and unmaps our view cleanly once dropped.
        crate::watch::title_memory_remapped(title_base + PAGE, PAGE);
        assert_eq!(stretch_stats(), (0, 0));
        assert_eq!(weak.strong_count(), 0, "remapped stretch must drop and unmap");

        let mut after: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: querying after `StretchView::drop` unmapped `view_addr`.
        if unsafe { VirtualQuery(view_addr as *const _, &mut after, size_of_val(&after)) } != 0 {
            assert!(
                after.State == MEM_FREE || after.AllocationBase as usize != view_addr,
                "stretch view must be unmapped after drop"
            );
        }

        // Path B: re-register views and verify section-handle reuse retires the
        // active stretch cleanly and unmaps its view too.
        for k in 0..PAGES {
            crate::inplace::title_memory_view(
                title_base + k * PAGE,
                PAGE,
                section_id,
                (k * PAGE) as u64,
            );
        }
        let s2 = stretch_create_in(
            &STRETCHES,
            title_base as u64,
            title_base as u64 + 64,
            false,
            true,
        )
        .expect("recreated stretch");
        let view_addr2 = s2.stretch_view.as_ref().unwrap().addr;
        let weak2 = Arc::downgrade(&s2);
        drop(s2);
        assert_eq!(stretch_stats(), (1, LEN as u64));

        crate::inplace::title_memory_section(section_id, LEN as u64, true);
        assert_eq!(stretch_stats(), (0, 0));
        assert_eq!(
            weak2.strong_count(),
            0,
            "section handle reuse must retire and drop stretch"
        );
        let mut after2: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(view_addr2 as *const _, &mut after2, size_of_val(&after2)) } != 0
        {
            assert!(
                after2.State == MEM_FREE || after2.AllocationBase as usize != view_addr2,
                "reused-section stretch view must be unmapped after drop"
            );
        }

        crate::watch::title_memory_remapped(title_base, LEN);
        // SAFETY: section created in this test.
        unsafe { CloseHandle(section) };
    }
}
