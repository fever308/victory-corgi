//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory. For physical-page views,
//! guard notifications and content hashes detect changes.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UNORM;

use crate::host::{HostList, host};
use crate::object::This;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Monotonic tick stamped on every staging cache hit or insert for LRU eviction.
static USE_TICK: AtomicU64 = AtomicU64::new(1);

fn next_use_tick() -> u64 {
    USE_TICK.fetch_add(1, Ordering::Relaxed)
}

/// Host physical RAM reported by Windows, or `None` if the query fails.
fn total_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: `MEMORYSTATUSEX` is plain data and `dwLength` is initialized
    // before calling `GlobalMemoryStatusEx`.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: valid out pointer with `dwLength` matching `MEMORYSTATUSEX`.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0 && status.ullTotalPhys != 0).then_some(status.ullTotalPhys)
}

/// Upload-heap byte budget for [`STAGING`], scaled to host physical RAM.
fn staging_budget_for_ram(total_ram_bytes: Option<u64>) -> u64 {
    match total_ram_bytes {
        None => 512 * MIB,
        Some(ram) if ram <= 16 * GIB => 256 * MIB,
        Some(ram) if ram <= 32 * GIB => 512 * MIB,
        Some(_) => GIB,
    }
}

/// Cached staging upload-heap byte budget for this host.
fn staging_budget() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let ram = total_physical_ram();
        let budget = staging_budget_for_ram(ram);
        tracing::info!(
            total_ram_mib = ram.map(|b| b / MIB),
            budget_mib = budget / MIB,
            "texture staging cache budget"
        );
        budget
    })
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// Made by the pass over textures picked by index, for one upload
    /// ([`once`]): read when the list is submitted, and never looked at again.
    once: std::sync::atomic::AtomicBool,
    /// The staging buffer's bytes (counted in `STAGED` while it lives).
    bytes: u64,
    memory: *const u8,
    /// Whether unchanged view guards can defer hashing this texture.
    guard_view: bool,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: 5,411 subresource copies
    /// of 1,120 textures in one frame of run 18.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// How many bytes of the texture's memory were mapped when a list last
    /// recorded an upload of it ([`fill_unlocked`]; 0: not asked).
    expected: std::sync::atomic::AtomicU64,
    /// Monotonic resource creation sequence for epoch-based retirement.
    born_seq: u64,
    /// Monotonic [`USE_TICK`] stamped on every staging cache hit or insert.
    last_used: AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<Digest>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// Other addresses its pages show at ([`crate::inplace::aliases`]), as
    /// of the views' generation `aliases_at` (0: not looked for yet).
    aliases: Vec<(usize, usize)>,
    aliases_at: u64,
    /// More of them than are asked: the hash decides every time.
    aliases_more: bool,
    /// Hashes in a row that found the bytes as they were. A texture with
    /// other addresses is hashed at every check only while this is below
    /// [`alias_same`]; after that on its turn, as an unwritten one is.
    alias_same: u32,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// Bounded LRU cache of per-texture upload staging buffers.
///
/// Evicted entries with `Arc::strong_count == 1` are safe to drop directly
/// without going through [`crate::host::retire`]:
/// - `Staging.buffer` is only used by [`upload`], which immediately moves its
///   `Arc<Staging>` into `list.deferred.copies` via
///   `list.defer_texture(staging)` until `list.reset()` or `list.drop()`.
/// - `Host::execute` holds `&HostList` until the GPU fence completes, so any
///   staging buffer referenced by an open or in-flight command list has
///   `Arc::strong_count >= 2` and is skipped by eviction.
/// - No descriptor heap ever references `Staging.buffer`.
#[derive(Default)]
struct StagingCache {
    entries: HashMap<usize, Arc<Staging>>,
    total_bytes: u64,
}

impl StagingCache {
    fn insert(&mut self, key: usize, staging: Arc<Staging>) -> Option<Arc<Staging>> {
        self.total_bytes = self.total_bytes.saturating_add(staging.bytes);
        let previous = self.entries.insert(key, staging);
        if let Some(prev) = &previous {
            self.total_bytes = self.total_bytes.saturating_sub(prev.bytes);
        }
        previous
    }

    fn remove(&mut self, key: usize) -> Option<Arc<Staging>> {
        let removed = self.entries.remove(&key)?;
        self.total_bytes = self.total_bytes.saturating_sub(removed.bytes);
        Some(removed)
    }

    /// Evicts least-recently-used idle entries (`Arc::strong_count == 1`)
    /// other than `keep_key` until `self.total_bytes <= budget` or no idle
    /// candidates remain. Callers must drop the returned `Vec` after releasing
    /// the [`STAGING`] lock.
    fn evict_to_budget(&mut self, budget: u64, keep_key: usize) -> Vec<Arc<Staging>> {
        if self.total_bytes <= budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, usize)> = self
            .entries
            .iter()
            .filter_map(|(&key, staging)| {
                if key == keep_key || Arc::strong_count(staging) != 1 {
                    return None;
                }
                Some((staging.last_used.load(Ordering::Relaxed), key))
            })
            .collect();
        candidates.sort_unstable_by_key(|&(tick, _)| tick);
        let mut evicted = Vec::new();
        for (_, key) in candidates {
            if self.total_bytes <= budget {
                break;
            }
            if let Some(staging) = self.remove(key) {
                evicted.push(staging);
            }
        }
        evicted
    }
}

static STAGING: Mutex<Option<StagingCache>> = Mutex::new(None);

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    let first = list.first_use(placed as usize);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !first && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a guarded texture whose pages show at another address too is
/// hashed at every check (`VICTORY_TEXTURE_ALIASES=0`: no, the guard alone
/// decides, as before).
///
/// Hashing guards nothing, and loads (run 20261124-161527: no missed
/// change in 2,678 frames, about 1 ms a world frame of refreshes). Guarding
/// the other addresses instead stopped loading at the same point every
/// time (2.2 GB read, then no more reads: runs -155007 and -160049), and so
/// did holding the kernel's writes open (-160539); what writes there
/// without faulting is not yet known.
fn aliases_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_ALIASES").is_none_or(|v| v != "0"))
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. A staging buffer that holds nothing
/// yet is filled with what is there all the same ([`fill_unlocked`]): a list
/// has recorded the upload from it by then.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_some_and(|v| v == "1"))
}

/// Whether a refresh that does not wait ([`no_wait`]) and finds a page
/// missing fills a staging buffer that was never filled from the pages that
/// are there (`VICTORY_TEXTURE_FILL=0`: it was left as it was made).
///
/// Left for the next refresh, it was uploaded as it was: the upload is
/// recorded before the refresh, and a texture picked by index has no next
/// refresh (its staging buffer goes with the upload). A texture the title
/// holds only the smaller levels of is never all there, so whenever the
/// runtime was remapping at that moment, its levels became zeros on the
/// card, good ones too.
fn fill_unlocked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_FILL").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there; staging buffers
/// filled from the pages there ([`fill_unlocked`]), and of them those that
/// got less than was mapped (their slots are looked at again).
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Bytes of textures hashed since the last frame line, by why the hash was
/// taken: 0 no write watch and no guard over the memory (hashed at every
/// check), 1 the guard saw nothing and it was the texture's turn (every
/// [`hash_every`]th frame), 2 the guard saw nothing but the pages show at
/// another address too, 3 the guard saw a write.
pub(crate) static HASH_WHY: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Hashes in a row that must find a texture with other addresses unchanged
/// before it is hashed on its turn only (`VICTORY_TEXTURE_ALIAS_SAME`,
/// default 4; 0 hashes such a texture at every check, as before). Run
/// 20261125-002006 hashed 185 MB a frame of them, about all the hashing
/// there was, and found 0.5 MB changed. A texture written through its other
/// address (the boot video's ring, every eighth frame) keeps finding
/// changes, so it stays hashed at every check; one that stopped changing
/// shows a new write at most [`hash_every`] frames late.
pub(crate) fn alias_same() -> u32 {
    static SAME: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *SAME.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_ALIAS_SAME").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(4)
    })
}

/// How many frames apart a guarded, unwritten texture is hashed anyway
/// (`VICTORY_TEXTURE_HASH_EVERY`, default 8): the hash finds a write the
/// guard cannot see (one through another view of the same pages). Higher
/// is less reading; such a write shows that many frames late at most.
pub(crate) fn hash_every() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_HASH_EVERY").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&v| v > 0).unwrap_or(8)
    })
}

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    REFRESH_WAITS[slot].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left). Moves the buffer out of the
/// thread-local `RefCell` during `f` so VEH / `lazy_back` re-entrancy on the
/// same thread can never panic with `RefCell already borrowed`.
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    let mut read = READ.with(|slot| {
        slot.try_borrow_mut()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default()
    });
    if read.len() < size {
        read.resize(size, 0);
    }
    let out = f(&mut read[..size]);
    READ.with(|slot| {
        if let Ok(mut saved) = slot.try_borrow_mut()
            && read.capacity() > saved.capacity()
        {
            *saved = read;
        }
    });
    out
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        crate::texdebug::forgotten(self.memory as usize);
    }
}

/// The bytes of title memory the staged texture at `memory` is read from
/// (for the texture debugger's verifier).
pub(crate) fn memory_size(memory: usize) -> Option<usize> {
    let cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    cache
        .as_ref()?
        .entries
        .values()
        .find(|s| s.memory as usize == memory)
        .map(|s| s.layout.size as usize)
}

/// For the memory line and the texture debugger: staging buffers alive,
/// their bytes, and how many of them their texture still has (the others
/// wait for a list's reset).
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |cache| cache.entries.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    let (dropped, went, bytes) = {
        let mut cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let Some(cache) = cache.as_mut() else {
            return;
        };
        if cache.entries.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        // A list that recorded an upload or a check holds the staging until it
        // is reset: one only the table holds is named by no list, recorded or
        // running.
        let idle: Vec<usize> = cache
            .entries
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES)
            })
            .map(|(&k, _)| k)
            .collect();
        let mut dropped = Vec::with_capacity(idle.len());
        let (mut went, mut bytes) = (0u64, 0u64);
        for key in idle {
            if let Some(s) = cache.remove(key) {
                LET_GO[0].fetch_add(1, Relaxed);
                LET_GO[1].fetch_add(s.bytes, Relaxed);
                went += 1;
                bytes += s.bytes;
                dropped.push(s);
            }
        }
        (dropped, went, bytes)
    };
    drop(dropped);
    if went > 0 {
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// Title textures whose host copy a list has recorded an upload into
/// since it was made: the others hold nothing yet (zeros), and a slot made
/// from one before its upload shows the neutral texture instead
/// (`crate::descriptor::host`).
static FILLED: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

fn note_filled(resource: This) {
    let mut set = FILLED.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default).insert(resource as usize);
}

/// Whether a list has recorded an upload into `resource`'s host copy since
/// it was made.
pub(crate) fn filled(resource: This) -> bool {
    let set = FILLED.lock().unwrap_or_else(|e| e.into_inner());
    set.as_ref().is_some_and(|set| set.contains(&(resource as usize)))
}

/// `resource`'s host copy went (`crate::resource::let_go_idle`): the next
/// one is made empty.
pub(crate) fn unfilled(resource: usize) {
    let mut set = FILLED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    unfilled(resource);
    let staging = {
        let mut cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        cache.as_mut().and_then(|cache| cache.remove(resource))
    };
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
    drop(set);
    if let Some(staging) = staging {
        let born = staging.born_seq;
        crate::host::retire_born(staging, born);
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

thread_local! {
    /// The thread records the upload of a texture picked by index, of which
    /// that many bytes were found mapped just now (`u64::MAX`: not asked).
    pub(crate) static PICKING: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Whether a staging buffer made for one upload of a texture picked by index
/// is filled without asking the write watch and the guards
/// (`VICTORY_TEXTURE_ONCE=0`: with, as before). What they say is for the next
/// refresh, and there is none: the buffer goes with the upload. Asking guards
/// the texture's blocks (a change of protection, under the lock every other
/// refresh wants), and the title's next write there faults to open them.
fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_ONCE").is_none_or(|v| v != "0"))
}

/// A 128-bit digest of `bytes`, to tell a texture's memory changed: four
/// lanes of folded multiplications over 64 bytes at a time. (Not a
/// cryptographic one, which took 0.27 ms a megabyte, four times the detiling
/// it was to spare; nobody makes up textures to collide.)
pub(crate) fn digest(bytes: &[u8]) -> Digest {
    const K: [u64; 8] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
        0xeb44_acca_b455_d165,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
    ];
    let fold = |a: u64, b: u64| {
        let p = u128::from(a) * u128::from(b);
        p as u64 ^ (p >> 64) as u64
    };
    let word = |b: &[u8]| {
        let mut w = [0; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut step = |block: &[u8; 64]| {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let a = word(&block[16 * i..16 * i + 8]);
            let b = word(&block[16 * i + 8..16 * i + 16]);
            // (What went in is added again: a product with a factor of zero
            // forgets neither the lane nor the block.)
            *lane = fold(a ^ K[4 + i], b ^ *lane).wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41)) ^ b;
        }
    };
    let mut blocks = bytes.chunks_exact(64);
    for block in &mut blocks {
        step(block.try_into().unwrap_or(&[0; 64]));
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high = fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len) ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    Digest(u128::from(low) << 64 | u128::from(high))
}

/// See [`digest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Digest(u128);

/// Reads what is there of the `dst.len()` bytes of title memory at `src`,
/// zeros for the rest; returns how many bytes were there. Safe while the
/// runtime remaps title memory (the kernel reads). What the runtime's
/// record says is mapped is read in one go each, and by pieces where that
/// fails; memory it has no record of by pieces.
pub(crate) fn read_there(src: *const u8, dst: &mut [u8]) -> usize {
    if dst.is_empty() {
        return 0;
    }
    dst.fill(0);
    let (start, len) = (src as usize, dst.len());
    let mut got = 0;
    let mut runs = crate::inplace::shown_runs(start, len);
    if runs.is_empty() {
        runs.push((start, len));
    }
    for (at, n) in runs {
        let piece = &mut dst[at - start..at - start + n];
        if crate::host::read_into(at as *const u8, piece) {
            got += n;
            continue;
        }
        // (What failed may have written its first pages.)
        piece.fill(0);
        crate::host::for_pieces(at, n, 64 << 10, |from, n| {
            let piece = &mut dst[from - start..from - start + n];
            if crate::host::read_into(from as *const u8, piece) {
                got += n;
                return;
            }
            piece.fill(0);
            crate::host::for_pieces(from, n, 4 << 10, |from, n| {
                let piece = &mut dst[from - start..from - start + n];
                if crate::host::read_into(from as *const u8, piece) {
                    got += n;
                } else {
                    piece.fill(0);
                }
            });
        });
    }
    got
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    let picking = PICKING.get();
    if picking.is_none() {
        // A draw samples it through a table: kept and checked from here on.
        staging.once.store(false, Relaxed);
    }
    staging.used.store(crate::host::moment(), Relaxed);
    if no_wait() && fill_unlocked() {
        let there = match picking {
            Some(there) if there != u64::MAX => there,
            _ => mapped_bytes(staging.memory, staging.layout.size as usize),
        };
        staging.expected.store(there, Relaxed);
    }
    crate::texdebug::uploaded(staging.memory as usize);
    note_filled(resource);
    // Without levels on demand: once per submission however many of its
    // lists sample the texture; the list's prelude then records the GPU copy
    // only if `refresh` detiled new contents (or this is the first upload),
    // so unchanged textures skip the copy and changed ones upload on the
    // same submission. A texture with levels on demand (`crate::tiles`) is
    // uploaded by the list itself, below, which then gives its levels their
    // tiles in the order of the list's steps.
    if !staging.on_demand {
        list.defer_texture(staging.clone());
        list.defer_texture_upload(copy, staging);
        return;
    }
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left, and what it finds changed the next list that
    // samples the texture uploads (a frame late where every list of the
    // frame was recorded before that check).
    if staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        list.defer_texture(staging);
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexUploads, 1);
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.bytes,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // When the list is submitted, its copy takes what the staging buffer
    // holds by then: noted before the check below, which sets `dirty`
    // again if it detiles (the next list then uploads once more).
    {
        let staging = staging.clone();
        list.defer_host(move || {
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);
        });
    }
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level, so each one
    // given here is written whole by it (the rest are copies into no
    // memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    let Some(on_demand) = on_demand else {
        // Once per submission however many of its lists sample the texture.
        list.defer_texture(staging);
        return;
    };
    list.defer_texture(staging.clone());
    list.defer_host(move || {
        let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
        staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
        // A texture picked by index has no staging buffer after this
        // upload: its slots are made due again by the texture pass.
        if waiting {
            crate::descriptor::host::levels_waiting(staging.memory as u64);
        }
    });
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.layout.size as usize)
    }

    /// Marks this staging entry as most recently used.
    fn touch(&self) {
        self.last_used.store(next_use_tick(), Ordering::Relaxed);
    }

    /// Whether this staging buffer needs to be copied into its host texture
    /// now (first upload or detiled since the last upload), marking it
    /// uploaded. Called from `HostList::prelude` after `refresh` has run.
    pub(crate) fn take_upload(&self) -> bool {
        use std::sync::atomic::Ordering::AcqRel;
        let first = !self.uploaded.swap(true, AcqRel);
        let dirty = self.dirty.swap(false, AcqRel);
        first || dirty
    }

    /// Records the copy of every subresource from the staging buffer into
    /// `texture` (which must be in `D3D12_RESOURCE_STATE_COPY_DEST`).
    pub(crate) fn record_copy(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        crate::texdebug::refreshed(started.elapsed().as_micros() as u64);
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        REFRESH_WAITS[3].fetch_add(whole.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        read
    }

    fn refresh_timed(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        let alone = self.once.load(std::sync::atomic::Ordering::Relaxed);
        if alone && last.hash.is_some() {
            return 0;
        }
        if !alone && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false)) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                crate::texdebug::skipped(start, crate::texdebug::SKIP_WATCH);
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read, crate::texdebug::WATCH);
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken every eighth frame only, by the
        // texture's address, so the textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mut why = 0usize;
        if !alone
            && self.guard_view
            && *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, true))
        {
            let before = last.asked;
            let mut unwritten = before != 0 && written.iter().all(|&(_, w)| w <= before);
            why = if unwritten { 1 } else { 3 };
            last.asked = now;
            // The pages may show at another address too, and a write there
            // does not fault in this range's guards: the boot video's ring
            // was written through the pool view it was filled in and read
            // through the tiled heap the GPU mapped the same pages into,
            // and showed a new frame only on every eighth (run 20261124-
            // 154124, 1,176 changes the texture debugger saw missed). Such a
            // texture is hashed at every check instead: the other addresses
            // are not guarded (guarding them stopped loading, see
            // `aliases_on`), so nothing else changes.
            if unwritten && aliases_on() {
                let generation = crate::inplace::generation();
                if last.aliases_at != generation {
                    let (found, more) = crate::inplace::aliases(start, size);
                    crate::texdebug::aliased(start, found.len(), found.iter().map(|a| a.1).sum(), more);
                    last.aliases = found;
                    last.aliases_at = generation;
                    last.aliases_more = more;
                }
                let settled = alias_same() != 0 && last.alias_same >= alias_same();
                if (last.aliases_more || !last.aliases.is_empty()) && !settled {
                    unwritten = false;
                    why = 2;
                    crate::texdebug::alias_written(start);
                }
            }
            let turn = crate::frame::current().wrapping_add((start >> 16) as u64).is_multiple_of(hash_every());
            if unwritten && !turn {
                crate::texdebug::skipped(start, crate::texdebug::SKIP_GUARD);
                return 0;
            }
        }
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let _mapped = if no_wait() {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_now(start, size) else {
                return self.refresh_unlocked(last, size);
            };
            mapped
        } else {
            waited(2, || crate::watch::title_memory_reading(start, size))
        };
        if !committed(self.memory, size) {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // shortcut: read again at most every 30 frames (reading around
            // unmapped pages is slow, and nothing tells us when pages
            // arrive): a mip that streams in shows up to 30 frames late.
            let frame = crate::frame::current();
            if last.partial.is_some_and(|at| frame < at + 30) {
                crate::texdebug::skipped(start, crate::texdebug::SKIP_PARTIAL);
                return 0;
            }
            last.partial = Some(frame);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            HASH_WHY[why].fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                let hash = digest(read);
                last.alias_same = if last.hash == Some(hash) { last.alias_same.saturating_add(1) } else { 0 };
                if last.hash == Some(hash) {
                    return false;
                }
                last.hash = Some(hash);
                if read.iter().all(|&b| b == 0) {
                    return false;
                }
                self.detile(read, crate::texdebug::PARTIAL);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            let _ = changed;
            return size;
        }
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let hash = digest(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                HASH_WHY[why].fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);
                last.alias_same = if last.hash == Some(hash) { last.alias_same.saturating_add(1) } else { 0 };
                if last.hash == Some(hash) {
                    crate::texdebug::skipped(start, crate::texdebug::SKIP_SAME_HASH);
                    return size;
                }
                last.hash = Some(hash);
                self.detile(tiled, crate::texdebug::THROUGH);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = digest(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        HASH_WHY[why].fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);
        last.alias_same = if last.hash == Some(hash) { last.alias_same.saturating_add(1) } else { 0 };
        if last.hash == Some(hash) {
            crate::texdebug::skipped(start, crate::texdebug::SKIP_SAME_HASH);
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled, crate::texdebug::HASH);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then a staging buffer that was filled
    /// before keeps what it holds and the texture is looked at again at its
    /// next refresh (what was learnt about writes to it is forgotten, so
    /// that it is not passed over as unwritten); one never filled takes the
    /// pages that are there ([`Staging::fill`]). Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let hash = digest(read);
            last.alias_same = if last.hash == Some(hash) { last.alias_same.saturating_add(1) } else { 0 };
            if last.hash != Some(hash) {
                last.hash = Some(hash);
                self.detile(read, crate::texdebug::UNLOCKED);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else if last.hash.is_none() && fill_unlocked() {
            self.fill(last, size)
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            crate::texdebug::skipped(self.memory as usize, crate::texdebug::SKIP_UNREADABLE);
            last.asked = 0;
            0
        }
    }

    /// The first filling of the staging buffer while the runtime is
    /// remapping title memory, of a texture not all of which can be read:
    /// with the pages that are there, zeros for the others, as a partly
    /// mapped texture is filled under the lock. Pages that are away only
    /// for the moment of a change (the runtime unmaps the rest of a view to
    /// change one page of it) are read when that change is over. If less
    /// was read than was mapped when the upload was recorded, or is mapped
    /// now, the texture's slots are looked at again.
    fn fill(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let start = self.memory as u64;
        let mut there = 0;
        let got = with_read(size, |read| {
            let mut got = 0;
            for _ in 0..4 {
                got = read_there(self.memory, read) as u64;
                there = mapped_bytes(self.memory, size);
                if got >= there.max(self.expected.load(Relaxed)) {
                    break;
                }
                crate::watch::change_over(self.memory as usize, size);
            }
            if got != 0 {
                last.hash = Some(digest(read));
                self.detile(read, crate::texdebug::UNLOCKED);
            }
            got
        });
        last.partial = Some(crate::frame::current());
        UNLOCKED[2].fetch_add(1, Relaxed);
        if got < there.max(self.expected.load(Relaxed)) {
            UNLOCKED[3].fetch_add(1, Relaxed);
            last.asked = 0;
            crate::descriptor::host::read_short(start);
        } else {
            crate::descriptor::host::read_all(start);
        }
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer;
    /// `why`: how the refresh found it changed (`crate::texdebug`).
    fn detile(&self, tiled: &[u8], why: u8) {
        crate::texdebug::detiled(self.memory as usize, tiled, why);
        self.dirty.store(true, std::sync::atomic::Ordering::Release);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::Relaxed;
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            let held = self.held_levels(judged);
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let removed = {
        let mut cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        cache.as_mut().and_then(|cache| cache.remove(resource as usize))
    };
    drop(removed);
}

/// Where a staged title texture's memory is: address and bytes, and the
/// ranges of it (offsets from the address, bytes) that [`fingerprint`]
/// reads besides its three windows: the start of every mip of every slice
/// ([`victory_formats::texture::Layout::sample_ranges`]). None with
/// `VICTORY_HEAP_RECHECK_MIPS=0`: the three windows alone, as before.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64, Option<std::sync::Arc<[(u64, u64)]>>)> {
    let cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = cache.as_ref()?.entries.get(&(resource as usize))?;
    let (at, len) = staging.memory();
    let mips = recheck_mips().then(|| staging.layout.sample_ranges(MIP_WINDOW, MIP_RANGES).into());
    Some((at as u64, len as u64, mips))
}

/// Bytes [`fingerprint`] reads at the start of each mip, and the most mips
/// it reads of one texture (spread over its slices and mips when it has
/// more). A 4096-square texture with all its mips: 10 reads, 12 KiB; a cube
/// map with 10 mips: 40 reads, 52 KiB.
const MIP_WINDOW: u64 = 1 << 10;
const MIP_RANGES: usize = 64;

/// Whether [`fingerprint`] also reads the start of every mip of every
/// slice (`VICTORY_HEAP_RECHECK_MIPS=0`: no).
fn recheck_mips() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK_MIPS").is_none_or(|v| v != "0"))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end,
/// and of the ranges `mips` (offsets from `at`, bytes; see
/// [`staged_memory`]). A texture streams in whole mips, and the three
/// windows alone see only the smallest mips and mip 0 of a plain texture,
/// and only two slices of an array or cube map: a mip or slice that
/// arrived after the upload elsewhere was not seen. Read through the
/// kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64, mips: &[(u64, u64)]) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut print = 0u128;
    let mut read = READ.with(|slot| {
        slot.try_borrow_mut()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default()
    });
    let longest = mips.iter().map(|&(_, n)| n as usize).max().unwrap_or(0);
    read.resize((3 * WINDOW).max(longest), 0);
    let mut take = |from: usize, bytes: usize| {
        read[..bytes].fill(0);
        crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
        print = print.rotate_left(37) ^ digest(&read[..bytes]).0;
    };
    if len <= 3 * WINDOW {
        take(0, len);
    } else {
        take(0, WINDOW);
        take((len / 2) & !0xfff, WINDOW);
        take(len - WINDOW, WINDOW);
        for &(from, bytes) in mips {
            let from = from as usize;
            if from < len {
                take(from, (bytes as usize).min(len - from));
            }
        }
    }
    READ.with(|slot| {
        if let Ok(mut saved) = slot.try_borrow_mut()
            && read.capacity() > saved.capacity()
        {
            *saved = read;
        }
    });
    let first = (print as u64 ^ (print >> 64) as u64).to_le_bytes();
    u64::from_le_bytes(first) | 1
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = cache.as_ref()?.entries.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    cache
        .as_ref()
        .is_some_and(|cache| cache.entries.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
/// (Whether it is mapped is asked only if `ask`; else it counts as mapped.)
pub(crate) fn staged(resource: This, ask: bool) -> Option<(u64, bool)> {
    let s = {
        let cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        cache.as_ref()?.entries.get(&(resource as usize))?.clone()
    };
    let size = s.layout.size as usize;
    Some((size as u64, !ask || committed(s.memory, size)))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again).
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let end = start.saturating_add(len);
    // Streamed textures lie in views of physical pages, of which the runtime
    // knows which are there: one look, where the host was asked for every
    // 64 KiB of the texture (and once more for each that is not there).
    // `VICTORY_MAPPED_BY_VIEWS=0`: the host is asked, as before.
    static BY_VIEWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *BY_VIEWS.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_VIEWS").is_none_or(|v| v != "0"))
        && let Some(there) = crate::inplace::shown_bytes(start, len)
    {
        return there;
    }
    if !committed(at, 1) && !committed((end - 1) as *const u8, 1) {
        return 0;
    }
    if !by_query() {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if readable(lo, 1, true).0 == 1 {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// Allocates a fresh staging buffer for a title texture without inserting it
/// into [`STAGING`].
fn create_staging(
    resource: This,
    texture: &ID3D12Resource,
    on_demand: bool,
) -> Option<Arc<Staging>> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    // Software-decoded movie planes are linear R8 textures. The decoder can
    // write through another view of the same pages, bypassing the guards on
    // the sampled view. Check their content on every submission so cycling
    // through the movie's textures does not replay old frames.
    let guard_view = !(d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Format == DXGI_FORMAT_R8_UNORM
        && d.MipLevels == 1
        && d.DepthOrArraySize == 1
        && layout.desc.swizzle == victory_formats::texture::Swizzle::LINEAR);
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let s = Arc::new(Staging {
        buffer,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        once: std::sync::atomic::AtomicBool::new(once() && !on_demand && PICKING.get().is_some()),
        bytes: total,
        memory,
        guard_view,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        expected: std::sync::atomic::AtomicU64::new(0),
        born_seq: crate::host::birth_seq(),
        last_used: AtomicU64::new(next_use_tick()),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
    });
    crate::texdebug::created(memory as usize, total, || {
        format!(
            "{}x{}x{} f{} mips {} dim {}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0, d.MipLevels, d.Dimension.0
        )
    });
    Some(s)
}

/// The staging state of a title texture, made on first use (or after LRU
/// eviction).
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let key = resource as usize;
    let (s, replaced, evicted) = {
        let mut cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let cache = cache.get_or_insert_with(StagingCache::default);
        if let Some(s) = cache.entries.get(&key) {
            s.touch();
            return Some(s.clone());
        }
        let s = create_staging(resource, texture, on_demand)?;
        let replaced = cache.insert(key, s.clone());
        let evicted = cache.evict_to_budget(staging_budget(), key);
        (s, replaced, evicted)
    };
    // Drop replaced/evicted staging buffers after releasing `STAGING` so freeing
    // D3D12 upload resources does not stall concurrent cache lookups.
    drop((replaced, evicted));
    Some(s)
}

/// Live entry count, total bytes, and in-use (`strong_count > 1`) entry count
/// in [`STAGING`] for periodic memory diagnostics.
pub(crate) fn cache_stats() -> (usize, u64, usize) {
    let cache = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(cache) = cache.as_ref() else {
        return (0, 0, 0);
    };
    let pinned = cache
        .entries
        .values()
        .filter(|s| Arc::strong_count(s) > 1)
        .count();
    (cache.entries.len(), cache.total_bytes, pinned)
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    /// What is there of a range some of whose pages are not: those bytes,
    /// counted, and zeros for the rest, whatever the buffer held.
    #[test]
    fn what_is_there_is_read_and_the_rest_is_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const SIZE: usize = 0x8_0000;
        // SAFETY: a reservation of ours; only committed pages are written.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!at.is_null());
            // 0-0x1_1000 and 0x3_0000-0x4_0000 are there.
            for (from, len) in [(0, 0x1_1000), (0x3_0000, 0x1_0000)] {
                assert!(!VirtualAlloc(at.add(from).cast(), len, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.add(from).write_bytes(0x5a, len);
            }
            // From an address inside a page, as textures are packed.
            let mut out = vec![0x77u8; SIZE - 0x300];
            assert_eq!(super::read_there(at.add(0x300), &mut out), 0x1_1000 - 0x300 + 0x1_0000);
            for (i, &b) in out.iter().enumerate() {
                let from = i + 0x300;
                let there = from < 0x1_1000 || (0x3_0000..0x4_0000).contains(&from);
                assert_eq!(b, if there { 0x5a } else { 0 }, "byte {from:#x}");
            }
            // All there: one read.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x1000), &mut out), 0x1_0000);
            assert!(out.iter().all(|&b| b == 0x5a));
            // Nothing there.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x5_0000), &mut out), 0);
            assert!(out.iter().all(|&b| b == 0));
            assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The digest tells every length and every changed byte apart, zeros
    /// among them.
    #[test]
    fn a_digest_tells_bytes_apart() {
        use super::digest;
        let zeros = [0u8; 400];
        let mut seen: Vec<super::Digest> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| ((i * 37) >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable_by_key(|d| d.0);
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }

    /// What a look at how much of a texture is there costs: by questions to
    /// the host, and by the runtime's record of its views.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn look_costs() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const BLOCK: usize = 64 << 10;
        for (name, blocks, there) in [("350 KB, all there", 6, 6), ("22 MB, all there", 350, 350), ("22 MB, 1 MB there", 350, 16)] {
            let make = || {
                // SAFETY: a reservation of our own, of which the first part
                // is committed and touched.
                unsafe {
                    let at = VirtualAlloc(std::ptr::null(), blocks * BLOCK, MEM_RESERVE, PAGE_READWRITE);
                    VirtualAlloc(at, there * BLOCK, MEM_COMMIT, PAGE_READWRITE);
                    std::ptr::write_bytes(at.cast::<u8>(), 1, there * BLOCK);
                    at as usize
                }
            };
            let (asked, noted) = (make(), make());
            for block in 0..there {
                crate::title_memory_view(noted + block * BLOCK, BLOCK, 9, 0);
            }
            for (how, at) in [("asking the host", asked), ("by the views", noted)] {
                const N: u32 = 200;
                let began = std::time::Instant::now();
                for _ in 0..N {
                    assert_eq!(super::mapped_bytes(at as *const u8, blocks * BLOCK), (there * BLOCK) as u64);
                }
                println!("{name}, {how}: {:.1} us a look", began.elapsed().as_secs_f64() * 1e6 / f64::from(N));
            }
            crate::inplace::views_gone(noted, blocks * BLOCK);
        }
    }

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_texture() -> (This, &'static ID3D12Resource) {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::core::Interface;

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
        let host_tex = crate::resource::host_resource(tex).expect("host texture");
        (tex, host_tex)
    }

    #[test]
    fn forget_drops_staging_immediately_when_idle() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let (tex, host_tex) = test_texture();
        let weak = {
            let s = staging(tex, host_tex).expect("staging");
            Arc::downgrade(&s)
        };
        assert_eq!(weak.strong_count(), 1);
        forget(tex as usize);
        assert_eq!(weak.strong_count(), 0);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
        assert_eq!(release(tex), 0);
    }

    #[test]
    fn forget_retires_staging_until_active_epoch_finishes() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let (tex, host_tex) = test_texture();
        let weak = {
            let s = staging(tex, host_tex).expect("staging");
            Arc::downgrade(&s)
        };
        assert_eq!(weak.strong_count(), 1);

        let list = host.new_list().expect("host list");
        list.mark_work();

        forget(tex as usize);
        assert_eq!(
            weak.strong_count(),
            1,
            "forgotten texture staging must wait for active epochs that recorded after its birth"
        );

        list.close();
        list.reset();
        assert_eq!(weak.strong_count(), 0);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
        assert_eq!(release(tex), 0);
    }

    #[test]
    fn forget_drops_new_staging_despite_older_pinned_list() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let stale = host.new_list().expect("stale list");
        stale.mark_work();

        let (tex, host_tex) = test_texture();
        let weak = {
            let s = staging(tex, host_tex).expect("staging");
            Arc::downgrade(&s)
        };
        assert_eq!(weak.strong_count(), 1);

        forget(tex as usize);
        assert_eq!(
            weak.strong_count(),
            0,
            "staging created after stale list's last work must not leak behind it"
        );
        stale.close();
        stale.reset();
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
        assert_eq!(release(tex), 0);
    }

    #[test]
    fn staging_budget_scales_across_ram_tiers() {
        assert_eq!(staging_budget_for_ram(None), 512 * MIB);
        assert_eq!(staging_budget_for_ram(Some(8 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_for_ram(Some(16 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_for_ram(Some(16 * GIB + 1)), 512 * MIB);
        assert_eq!(staging_budget_for_ram(Some(32 * GIB)), 512 * MIB);
        assert_eq!(staging_budget_for_ram(Some(32 * GIB + 1)), GIB);
        assert_eq!(staging_budget_for_ram(Some(64 * GIB)), GIB);
    }

    #[test]
    fn staging_evicts_lru_idle_entries_when_over_budget_and_skips_in_use_entries() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };

        let (t1, h1) = test_texture();
        let (t2, h2) = test_texture();
        let (t3, h3) = test_texture();
        let (t4, h4) = test_texture();

        let (mem1, desc1) = crate::resource::texture_memory(t1).expect("texture memory");
        let size1 = crate::resource::texture_layout(&desc1)
            .expect("texture layout")
            .size as usize;
        // SAFETY: committed title memory for `t1` holds at least `size1` writable bytes.
        unsafe { std::slice::from_raw_parts_mut(mem1.cast_mut(), size1).fill(0x5a) };

        let read_rgba_pixels = |s: &Staging| -> Vec<u8> {
            let fp = s.footprints[0];
            let pitch = fp.Footprint.RowPitch as usize;
            let row_bytes = fp.Footprint.Width as usize * 4;
            let rows = s.rows[0] as usize;
            let mut out = Vec::with_capacity(row_bytes * rows);
            if rows == 0 {
                return out;
            }
            let span = (rows - 1) * pitch + row_bytes;
            assert!(fp.Offset as usize + span <= s.bytes as usize);
            // SAFETY: `s.data` is a mapped upload buffer covering `fp.Offset..fp.Offset + span`.
            let mapped =
                unsafe { std::slice::from_raw_parts(s.data.add(fp.Offset as usize), span) };
            for row in 0..rows {
                out.extend_from_slice(&mapped[row * pitch..row * pitch + row_bytes]);
            }
            out
        };

        let s1 = create_staging(t1, h1, false).expect("staging 1");
        s1.refresh();
        assert_eq!(read_rgba_pixels(&s1), vec![0x5a; 64]);

        let s2 = create_staging(t2, h2, false).expect("staging 2");
        let s3 = create_staging(t3, h3, false).expect("staging 3");
        let entry_bytes = s1.bytes;
        assert!(entry_bytes > 0);
        assert_eq!(s2.bytes, entry_bytes);
        assert_eq!(s3.bytes, entry_bytes);

        let mut cache = StagingCache::default();
        assert!(cache.insert(t1 as usize, s1.clone()).is_none());
        // Re-inserting the same key returns the replaced entry to the caller so it
        // can be dropped after releasing the `STAGING` lock.
        let s1_dup = s1.clone();
        assert!(cache.insert(t1 as usize, s1_dup).is_some());
        assert_eq!(cache.total_bytes, entry_bytes);
        assert!(cache.insert(t2 as usize, s2.clone()).is_none());
        assert!(cache.insert(t3 as usize, s3.clone()).is_none());
        assert_eq!(cache.total_bytes, 3 * entry_bytes);

        // Touch `t1` so LRU order is `t2` (oldest) < `t3` < `t1` (newest).
        s1.touch();

        // Hold `t2` in an open `HostList` so its strong count is 2 (in-use).
        let list = host.new_list().expect("host list");
        list.defer_texture(s2.clone());

        let (w1, w2, w3) = (
            Arc::downgrade(&s1),
            Arc::downgrade(&s2),
            Arc::downgrade(&s3),
        );
        drop((s1, s2, s3));
        assert_eq!(w1.strong_count(), 1);
        assert_eq!(w2.strong_count(), 2);
        assert_eq!(w3.strong_count(), 1);

        // Insert `t4`, pushing `total_bytes` to 4 * `entry_bytes`.
        let s4 = create_staging(t4, h4, false).expect("staging 4");
        assert_eq!(s4.bytes, entry_bytes);
        assert!(cache.insert(t4 as usize, s4).is_none());
        assert_eq!(cache.total_bytes, 4 * entry_bytes);

        // Budget of 3 entries must skip in-use `t2` and evict oldest idle `t3`,
        // leaving newer `t1` and kept `t4`.
        let evicted = cache.evict_to_budget(3 * entry_bytes, t4 as usize);
        assert_eq!(evicted.len(), 1);
        assert_eq!(cache.total_bytes, 3 * entry_bytes);
        assert!(!cache.entries.contains_key(&(t3 as usize)));
        assert!(cache.entries.contains_key(&(t2 as usize)));
        assert!(cache.entries.contains_key(&(t1 as usize)));
        assert!(cache.entries.contains_key(&(t4 as usize)));
        drop(evicted);
        assert_eq!(
            w3.strong_count(),
            0,
            "evicted idle staging must drop immediately"
        );
        assert_eq!(w2.strong_count(), 2);
        assert_eq!(w1.strong_count(), 1);

        // Tightening to 2 entries evicts `t1` while still protecting in-use `t2`.
        let evicted = cache.evict_to_budget(2 * entry_bytes, t4 as usize);
        assert_eq!(evicted.len(), 1);
        assert_eq!(cache.total_bytes, 2 * entry_bytes);
        assert!(!cache.entries.contains_key(&(t1 as usize)));
        assert!(cache.entries.contains_key(&(t2 as usize)));
        assert!(cache.entries.contains_key(&(t4 as usize)));
        drop(evicted);
        assert_eq!(
            w1.strong_count(),
            0,
            "evicted idle staging must drop immediately"
        );

        // Tightening to 1 entry while `t2` is still in-use evicts nothing:
        // neither in-use `t2` nor `keep_key` `t4` may be evicted.
        let evicted = cache.evict_to_budget(entry_bytes, t4 as usize);
        assert!(evicted.is_empty());
        assert_eq!(cache.total_bytes, 2 * entry_bytes);

        // Resetting `list` drops its reference to `t2`; now `t2` becomes idle
        // and is evicted on the next budget pass.
        list.close();
        list.reset();
        assert_eq!(w2.strong_count(), 1);
        let evicted = cache.evict_to_budget(entry_bytes, t4 as usize);
        assert_eq!(evicted.len(), 1);
        assert_eq!(cache.total_bytes, entry_bytes);
        assert!(!cache.entries.contains_key(&(t2 as usize)));
        assert!(cache.entries.contains_key(&(t4 as usize)));
        drop(evicted);
        assert_eq!(w2.strong_count(), 0);

        // Re-creating and refreshing evicted `t1` fully re-detiles its title
        // memory without zeroing or stale contents.
        let recreated = create_staging(t1, h1, false).expect("recreated staging");
        recreated.refresh();
        assert_eq!(read_rgba_pixels(&recreated), vec![0x5a; 64]);
        drop(recreated);

        // Also run a live upload + GPU execution for `t1` after eviction.
        let upload_list = host.new_list().expect("upload list");
        upload(&upload_list, t1, h1);
        upload_list.close();
        host.execute(&[&upload_list]);
        upload_list.reset();
        forget(t1 as usize);

        drop(cache);
        for tex in [t1, t2, t3, t4] {
            let release = crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10);
            assert_eq!(release(tex), 0);
        }
    }
}
