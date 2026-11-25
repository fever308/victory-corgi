//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

// (A lock for readers and writers: during a submission every refresh asks
// the journal, from six threads at once, and nearly every answer is read from
// it; under one mutex they stood in line.)
static JOURNAL: std::sync::RwLock<Journal> = std::sync::RwLock::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

/// Mirrors `Journal::kept` so `Batch::flush` / `ask_ahead` / `ask_ahead_done`
/// can skip range collection, `in_view` lock lookups, sorting, and exclusive
/// `JOURNAL.write()` locks while a submission-wide `ask_once` is active.
static ASK_KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline]
fn set_kept(j: &mut Journal, kept: bool) {
    j.kept = kept;
    ASK_KEPT.store(kept, std::sync::atomic::Ordering::Release);
}

/// Whether a submission-wide write-watch asking (`ask_once`) currently stands.
#[inline]
pub(crate) fn ask_kept() -> bool {
    ASK_KEPT.load(std::sync::atomic::Ordering::Acquire)
}

/// Experiment (`VICTORY_WATCH_STATS=1`): the kernel askings by where they
/// come from and how they end: ahead ok, ahead failed, `writes` ok,
/// `writes` failed (then the guards), the pages asked in each, and the
/// microseconds of each.
static ASKS: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
/// Of the askings that were answered: those that found no written page
/// (count, pages, microseconds), the pages found written in the others,
/// and askings spared by [`cold`] (count, pages).
static CLEAN: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn asks_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_STATS").is_some_and(|v| v == "1"))
}

fn asked_kernel(kind: usize, pages: usize, started: std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    ASKS[kind].fetch_add(1, Relaxed);
    ASKS[4 + kind].fetch_add(pages as u64, Relaxed);
    ASKS[8 + kind].fetch_add(started.elapsed().as_nanos() as u64 / 1000, Relaxed);
}

/// The counts since the last call, for the frame line's neighbour.
pub(crate) fn asks_say(frames: f64) {
    if !asks_on() || frames <= 0.0 {
        return;
    }
    let v: Vec<f64> = ASKS.iter().map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames).collect();
    let c: Vec<f64> = CLEAN.iter().map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "write-watch askings per frame, answered: {:.0} found nothing ({:.0} MB, {:.1} ms); the others found {:.0} pages written; spared as cold {:.0} ({:.0} MB)",
        c[0],
        c[1] * PAGE as f64 / 1048576.0,
        c[2] / 1000.0,
        c[3],
        c[4],
        c[5] * PAGE as f64 / 1048576.0
    );
    tracing::info!(
        "write-watch askings per frame (VICTORY_WATCH_STATS=1): ahead ok {:.0} ({:.0} MB, {:.1} ms), ahead failed {:.0} ({:.0} MB, {:.1} ms), single ok {:.0} ({:.0} MB, {:.1} ms), single failed {:.0} ({:.0} MB, {:.1} ms)",
        v[0], v[4] * PAGE as f64 / 1048576.0, v[8] / 1000.0,
        v[1], v[5] * PAGE as f64 / 1048576.0, v[9] / 1000.0,
        v[2], v[6] * PAGE as f64 / 1048576.0, v[10] / 1000.0,
        v[3], v[7] * PAGE as f64 / 1048576.0, v[11] / 1000.0,
    );
}

/// Whether pages `[first, last)` begin in a view of the title's physical
/// pages (`crate::inplace`'s record of what the runtime mapped). View
/// memory is not write-watched: the kernel refuses the asking, and the
/// guards answer instead. Knowing it beforehand saves the refused call:
/// in gameplay 25,000 of the 26,700 askings of a frame were refused, 62 ms
/// of the threads' time under the journal's lock (run 51).
/// `VICTORY_WATCH_VIEWS=0` asks the kernel all the same, as before.
fn in_view(first: usize, last: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_VIEWS").is_none_or(|v| v != "0")) {
        return false;
    }
    let _ = last;
    crate::inplace::cover(first * PAGE, 1) != crate::inplace::Cover::None
}

/// `VICTORY_WATCH_COLD=<n>` (experiment, 0: off): a run of pages whose
/// last `n` askings found nothing written is asked once a frame, not at
/// every submission; the submissions between take it as not written. Half
/// of a gameplay frame's askings found nothing (771 of 1,187, 4.1 GB,
/// 27 ms on the submitting thread, run 53): buffers the title filled once.
///
/// What it gives up: a page of such a run written between two submissions
/// of one frame reaches the card with the next frame's first submission.
/// The first asking that finds a write makes the run asked every time
/// again.
fn cold_after() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_WATCH_COLD").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// Per run of pages: askings in a row that found nothing, and the frame of
/// the last one.
static COLD: Mutex<Option<std::collections::HashMap<(usize, usize), (u32, u64)>>> = Mutex::new(None);
/// The pages the last answered asking found written.
static LAST_FOUND: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// [`ask_kernel`], unless the run is cold and was asked this frame
/// ([`cold_after`]): then nothing is asked and it counts as answered.
fn ask_or_spare(j: &mut Journal, first: usize, last: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let after = cold_after();
    if after == 0 {
        return ask_kernel(j, first, last);
    }
    let frame = crate::frame::current();
    let mut cold = COLD.lock().unwrap_or_else(|e| e.into_inner());
    let cold = cold.get_or_insert_with(Default::default);
    if let Some(&(streak, at)) = cold.get(&(first, last))
        && streak >= after
        && at == frame
    {
        CLEAN[4].fetch_add(1, Relaxed);
        CLEAN[5].fetch_add((last - first) as u64, Relaxed);
        return true;
    }
    let ok = ask_kernel(j, first, last);
    if ok {
        if cold.len() > 200_000 {
            cold.clear();
        }
        let e = cold.entry((first, last)).or_insert((0, 0));
        e.0 = if LAST_FOUND.load(Relaxed) == 0 { e.0.saturating_add(1) } else { 0 };
        e.1 = frame;
    }
    ok
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// [`AHEAD_GAP`], or `VICTORY_WATCH_GAP_MB` (experiment: an asking that
/// resets written pages makes every core running the process drop them
/// from its address cache, which costs by the asking and by the cores:
/// 365 such askings took 47 ms of a gameplay frame, run 54).
fn ahead_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("VICTORY_WATCH_GAP_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(AHEAD_GAP, |mb| (mb << 20) / PAGE)
    })
}

/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    if ask_kept() {
        return;
    }
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        // View memory is the guards' (`in_view`).
        .filter(|&(first, last)| !in_view(first, last))
        .collect();
    // View memory the guards already hold (read-only, noted absent, or known
    // viewed) is also left to `writes`, which answers it from the guards: the
    // kernel's write watch can only fail there. Asked here with the others, a
    // run holding such a range failed, split by halves down to single ranges,
    // failing each time, and then went to `writes` all the same
    // (`VICTORY_VIEW_ASK=1` asks as before).
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    if !view_ask_disabled() && !pieces.is_empty() {
        let g = GUARDS.read().unwrap_or_else(|e| e.into_inner());
        let vk = view_known();
        pieces.retain(|&(first, _)| {
            let block = first * PAGE / BLOCK;
            !(g.protected.contains(&block)
                || g.absent.contains(&block)
                || (vk && g.viewed.contains(&block)))
        });
    }
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + ahead_gap()
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    if ask_kept() {
        return;
    }
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    let submits = TITLE_SUBMITS.load(std::sync::atomic::Ordering::Acquire);
    if ask_shared() {
        use std::sync::atomic::Ordering::Relaxed;
        let stamp = CURRENT_STAMP.load(Relaxed);
        let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
        let j = &mut *j;
        // The title gave this submission before the asking that stands
        // was made: what it wrote before giving it is in that asking.
        if stamp != 0 && stamp <= ASK_STAMP.load(Relaxed) && !j.ahead.is_empty() {
            for &(start, end) in ranges {
                if start >= end {
                    continue;
                }
                if known_view(start) {
                    #[cfg(feature = "diagnostics")]
                    AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                let (first, last) = (start / PAGE, end.div_ceil(PAGE));
                if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && !in_view(first, last) && ask_or_spare(j, first, last) {
                    j.ahead.push((first, last));
                }
            }
            set_kept(j, true);
            SHARED[0].fetch_add(1, Relaxed);
            return;
        }
        SHARED[1].fetch_add(1, Relaxed);
        let n = SHARED[1].load(Relaxed);
        if n.is_power_of_two() && n >= 256 {
            tracing::info!(shared = SHARED[0].load(Relaxed), asked = n, "submissions that took the asking before them, and that asked anew (VICTORY_ASK_SHARED=1)");
        }
        set_kept(j, false);
        j.ahead.clear();
    }
    ask_once_end();
    ask_ahead(ranges);
    ASK_STAMP.store(submits, std::sync::atomic::Ordering::Relaxed);
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about: the asking can
        // only fail there, one failed kernel call for each such copy at
        // every submission, and `writes` answers it from the guards.
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && !in_view(first, last) && ask_or_spare(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    let kept = !j.ahead.is_empty();
    set_kept(j, kept);
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    // (With `ask_shared` the asking stands for the submissions the title
    // gave before it; `ask_once_void` ends it.)
    if ask_shared() {
        return;
    }
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        set_kept(&mut j, false);
        j.ahead.clear();
    }
}

/// Title memory was written in a way the journal was not told: no asking
/// stands, with [`ask_shared`] too.
pub(crate) fn ask_once_void() {
    ASK_STAMP.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    set_kept(&mut j, false);
    j.ahead.clear();
}

/// Submissions the title has given, counted when it gives them.
pub(crate) static TITLE_SUBMITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// [`TITLE_SUBMITS`] as it was when the submission being prepared was
/// given (0: not known).
pub(crate) static CURRENT_STAMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// [`TITLE_SUBMITS`] when the asking that stands was made.
static ASK_STAMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SHARED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// `VICTORY_ASK_SHARED=1` (experiment, with `VICTORY_ASYNC_SUBMIT`): one
/// asking of the kernel stands for every submission the title gave before
/// it was made. The title gives a frame's submissions in a burst and the
/// thread that runs the lines prepares them after it; each asked the
/// kernel for itself (35% of that thread's time, sampled), about what the
/// title wrote before giving it, which the asking made for the first of
/// the burst holds already. A submission names ranges the standing asking
/// lacks: those are asked and added.
/// shortcut: what we write into title memory ourselves between two such
/// submissions (the copy back of what shaders wrote) is not seen by the
/// second through another copy of that memory.
fn ask_shared() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_SHARED").is_some_and(|v| v == "1"))
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here. The page,
/// and the same physical page at its other title addresses (`others`,
/// page numbers), count as written at a new asking, which every host copy
/// that holds them takes at its next refresh. Returns the asking that had
/// last found `page` written (0: never) and the new one, for the copy
/// that wrote to note itself current.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    let before = j.written.insert(page, now).unwrap_or(0);
    for &other in others {
        if other != page {
            j.written.insert(other, now);
        }
    }
    (before, now)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_or_spare(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if asks_on() {
        asked_kernel(usize::from(failed), last - first, started);
    }
    if failed {
        return false;
    }
    LAST_FOUND.store(count, std::sync::atomic::Ordering::Relaxed);
    if asks_on() {
        use std::sync::atomic::Ordering::Relaxed;
        if count == 0 {
            CLEAN[0].fetch_add(1, Relaxed);
            CLEAN[1].fetch_add((last - first) as u64, Relaxed);
            CLEAN[2].fetch_add(started.elapsed().as_nanos() as u64 / 1000, Relaxed);
        } else {
            CLEAN[3].fetch_add(count as u64, Relaxed);
        }
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let page = j.found[k] / PAGE;
        j.written.insert(page, now);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.read().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Asked ahead with the submission's other ranges: read from the journal,
    // beside the other threads that read it.
    {
        let j = JOURNAL.read().unwrap_or_else(|e| e.into_inner());
        if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            return Some((j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()));
        }
    }
    // View memory is the guards' (`in_view` or `known_view`), which have a
    // lock of their own: asked without the journal's, which is taken to write
    // only if they found something. Every refresh of a copy of view memory
    // took the journal to write, and the six threads of a submission stood in
    // line behind it (45% of the submitting thread's samples, in a cutscene).
    if guarded && (in_view(first, last) || known_view(first * PAGE)) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let found = guard(first * PAGE, last * PAGE)?;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        if found.is_empty() {
            let j = JOURNAL.read().unwrap_or_else(|e| e.into_inner());
            return Some((j.asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()));
        }
        let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
        j.asked += 1;
        let now = j.asked;
        for page in found {
            j.written.insert(page, now);
        }
        return Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()));
    }
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    let started = std::time::Instant::now();
    // View memory: the kernel would refuse (`in_view`, and the views the
    // guards already hold).
    let known_view = guarded && known_view(first * PAGE);
    let view = known_view || in_view(first, last);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = view
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    if asks_on() && !view {
        asked_kernel(2 + usize::from(failed), last - first, started);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    #[cfg(feature = "diagnostics")]
    if failed && !view && guarded {
        failed_asking(first * PAGE, last * PAGE);
    }
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Diagnostics: one in 2,048 of the askings the kernel's write watch
/// refused for memory the guards did not know (each followed by a question
/// to Windows about the memory): what the memory is and what the guards
/// hold of its first block.
#[cfg(feature = "diagnostics")]
fn failed_asking(start: usize, end: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2048 != 0 {
        return;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    let (protected, clean, absent, viewed, aliased) = {
        let g = GUARDS.read().unwrap_or_else(|e| e.into_inner());
        let b = start / BLOCK;
        (
            g.protected.contains(&b),
            g.clean.contains(&b),
            g.absent.contains(&b),
            g.viewed.contains(&b),
            g.aliased.contains(&b),
        )
    };
    tracing::info!(
        "a failed write-watch asking (one in 2048): {start:#x}+{:#x}; Windows says (answered {n}): base {:#x} region {:#x} state {:#x} type {:#x} protect {:#x} allocation protect {:#x}; \
         the guards of its first block: protected {protected} clean {clean} absent {absent} viewed {viewed} aliased {aliased}; is a view: {}",
        end - start,
        info.BaseAddress as usize,
        info.RegionSize,
        info.State,
        info.Type,
        info.Protect,
        info.AllocationProtect,
        is_view(start, end)
    );
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether skipping kernel write-watch calls for known view blocks is disabled
/// (`VICTORY_VIEW_ASK=1` or `VICTORY_GUARD_ASK=1`).
fn view_ask_disabled() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    if view_ask_disabled() {
        return false;
    }
    let g = GUARDS.read().unwrap_or_else(|e| e.into_inner());
    let first = start / BLOCK;
    g.protected.contains(&first)
        || g.absent.contains(&first)
        || (view_known() && g.viewed.contains(&first))
}

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BTreeSet<usize>,
    /// Blocks the kernel is about to write for the runtime ([`kernel_writing`]),
    /// keyed by disjoint `[first, last)` block intervals with their expiry: a
    /// read may be asynchronous and land after the call that asked for it
    /// returned, and the kernel does not fault into our handler, it fails.
    /// Until then they are not made read-only, and every check reports them
    /// written.
    pins: BTreeMap<usize, (usize, std::time::Instant)>,
}

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1` or `alt`; off by default).
fn view_known() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_VIEW_KNOWN").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "view memory stays known while its guard is open from now (VICTORY_VIEW_KNOWN=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

// (Readers and writers: nearly every asking finds every block of its range
// clean, and six threads ask at once during a submission.)
static GUARDS: std::sync::RwLock<Guards> = std::sync::RwLock::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    aliased: BTreeSet::new(),
    viewed: BTreeSet::new(),
    pins: BTreeMap::new(),
});

/// How long blocks stay writable after [`kernel_writing`].
const PIN: std::time::Duration = std::time::Duration::from_secs(2);

impl Guards {
    /// Whether `block` is left writable for a kernel write (`O(log P)`).
    fn pinned(&self, block: usize) -> bool {
        if self.pins.is_empty() {
            return false;
        }
        self.pins
            .range(..=block)
            .next_back()
            .is_some_and(|(_, &(end, _))| block < end)
    }

    /// Marks `[first, last)` pinned until at least `until`, coalescing any
    /// overlapping or adjacent intervals so `pins` stays compact and disjoint.
    fn pin(&mut self, mut first: usize, mut last: usize, mut until: std::time::Instant) {
        if first >= last {
            return;
        }
        let scan_from = self
            .pins
            .range(..=first)
            .next_back()
            .filter(|&(_, &(p1, _))| p1 >= first)
            .map_or(first, |(&k, _)| k);
        let merged: Vec<(usize, usize, std::time::Instant)> = self
            .pins
            .range(scan_from..=last)
            .map(|(&p0, &(p1, u))| (p0, p1, u))
            .collect();
        for (p0, p1, u) in merged {
            self.pins.remove(&p0);
            first = first.min(p0);
            last = last.max(p1);
            if u > until {
                until = u;
            }
        }
        self.pins.insert(first, (last, until));
    }

    fn prune_pins(&mut self) {
        if !self.pins.is_empty() {
            let now = std::time::Instant::now();
            self.pins.retain(|_, &mut (_, until)| until > now);
        }
    }

    /// Removes `[first, last)` from active pins (the mapping there is gone or
    /// replaced, so a reused address must not inherit an old view's grace).
    fn unpin(&mut self, first: usize, last: usize) {
        if self.pins.is_empty() || first >= last {
            return;
        }
        let scan_from = self
            .pins
            .range(..first)
            .next_back()
            .filter(|&(_, &(p1, _))| p1 > first)
            .map_or(first, |(&k, _)| k);
        let hits: Vec<(usize, usize, std::time::Instant)> = self
            .pins
            .range(scan_from..last)
            .map(|(&p0, &(p1, u))| (p0, p1, u))
            .collect();
        for (p0, p1, until) in hits {
            self.pins.remove(&p0);
            if p0 < first {
                self.pins.insert(p0, (first, until));
            }
            if last < p1 {
                self.pins.insert(last, (p1, until));
            }
        }
    }
}

/// `Guards::clean` again as bits, one a block, for blocks under
/// [`CLEAN_BLOCKS`]: read without the lock by [`guard`]'s first look
/// (`VICTORY_GUARD_BITS=0`: under the lock, walking the set, as before).
/// Written only with the lock held alone, with the set.
const CLEAN_BLOCKS: usize = 1 << 26;

fn clean_bits() -> Option<&'static [std::sync::atomic::AtomicU64]> {
    static BITS: std::sync::OnceLock<Option<Box<[std::sync::atomic::AtomicU64]>>> = std::sync::OnceLock::new();
    BITS.get_or_init(|| {
        std::env::var_os("VICTORY_GUARD_BITS")
            .is_none_or(|v| v != "0")
            .then(|| (0..CLEAN_BLOCKS / 64).map(|_| std::sync::atomic::AtomicU64::new(0)).collect())
    })
    .as_deref()
}

fn clean_bit(block: usize, on: bool) {
    use std::sync::atomic::Ordering::SeqCst;
    if block < CLEAN_BLOCKS && let Some(bits) = clean_bits() {
        let (word, bit) = (&bits[block / 64], 1u64 << (block % 64));
        if on {
            word.fetch_or(bit, SeqCst);
        } else {
            word.fetch_and(!bit, SeqCst);
        }
    }
}

/// Whether every block of `first..last` is clean by the bits; `None` if
/// the bits do not say (off, or blocks past them).
fn clean_by_bits(first: usize, last: usize) -> Option<bool> {
    use std::sync::atomic::Ordering::SeqCst;
    let bits = clean_bits()?;
    if last > CLEAN_BLOCKS || last <= first {
        return None;
    }
    let (w0, w1) = (first / 64, (last - 1) / 64);
    for w in w0..=w1 {
        let mut mask = u64::MAX;
        if w == w0 {
            mask &= u64::MAX << (first % 64);
        }
        if w == w1 {
            mask &= u64::MAX >> (63 - (last - 1) % 64);
        }
        if bits[w].load(SeqCst) & mask != mask {
            return Some(false);
        }
    }
    Some(true)
}

fn guards() -> std::sync::RwLockWriteGuard<'static, Guards> {
    GUARDS.write().unwrap_or_else(|e| e.into_inner())
}

/// Whether the block `page` is read-only by our hand and unwritten since a
/// copy took it (diagnostics of `VICTORY_BACK_OWN_CHECK`).
pub(crate) fn clean(page: usize) -> bool {
    GUARDS.read().unwrap_or_else(|e| e.into_inner()).clean.contains(&page)
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // Every block of the range guarded and clean: nothing written, said
    // beside the other threads that ask (and without a list of the blocks).
    {
        let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
        match clean_by_bits(first, last) {
            Some(true) => return Some(Vec::new()),
            Some(false) => {}
            None => {
                let g = GUARDS.read().unwrap_or_else(|e| e.into_inner());
                if last > first && g.protected.contains(&first) && g.clean.range(first..last).count() == last - first {
                    return Some(Vec::new());
                }
            }
        }
    }
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        let g = GUARDS.read().unwrap_or_else(|e| e.into_inner());
        let first = start / BLOCK;
        g.protected.contains(&first)
            || g.absent.contains(&first)
            || (view_known() && g.viewed.contains(&first))
    };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // By the bits: the runs of blocks that are not clean, found without
    // the lock; only those are walked with it held (a range of 174 MB with
    // a few written pages held it for all its 42,000 blocks, and the
    // threads that refresh at once stood behind one another).
    if let Some(bits) = clean_bits()
        && last <= CLEAN_BLOCKS
    {
        use std::sync::atomic::Ordering::SeqCst;
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut block = first;
        while block < last {
            let word = bits[block / 64].load(SeqCst);
            if block % 64 == 0 && word == u64::MAX && block + 64 <= last {
                block += 64;
                continue;
            }
            if word & (1 << (block % 64)) == 0 {
                match runs.last_mut() {
                    Some(r) if r.1 == block => r.1 = block + 1,
                    _ => runs.push((block, block + 1)),
                }
            }
            block += 1;
        }
        let mut written = Vec::new();
        if runs.is_empty() {
            return Some(written);
        }
        let mut g = guards();
        g.prune_pins();
        for (from, to) in runs {
            guard_run(&mut g, from, to, &mut written);
        }
        return Some(written);
    }
    let mut g = guards();
    g.prune_pins();
    let mut written = Vec::new();
    guard_run(&mut g, first, last, &mut written);
    Some(written)
}

/// [`guard`] for the blocks `first..last`, with the lock held alone.
fn guard_run(g: &mut Guards, first: usize, last: usize, written: &mut Vec<usize>) {
    let mut run: Option<usize> = None;
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        return;
    }
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if open && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.pinned(block) {
                // The kernel may be writing it: left writable, reported
                // written at every check until the pin ends.
            } else if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
                clean_bit(block, true);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(g, from, block);
        }
    }
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    // One look at the views for the whole run, and no list made for a page
    // with one view (most of them): this runs under the guards' lock, which
    // every copy's check waits for.
    crate::inplace::for_each_alias_in(
        from.saturating_mul(BLOCK),
        to.saturating_mul(BLOCK),
        |block, shown| {
            if !g.protected.contains(&block) {
                return;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for &alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    protect_alias(g, alias);
                }
            }
        },
    );
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(&block) || g.pinned(block) {
        return;
    }
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        g.protected.insert(block);
        if view_known() {
            g.viewed.insert(block);
        }
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
        if view_known() {
            g.viewed.extend(from..to);
        }
        (from..to).for_each(|b| clean_bit(b, true));
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else {
            #[cfg(feature = "diagnostics")]
            let error = std::io::Error::last_os_error();
            if block < uncommitted_end(block) {
                g.absent.insert(block);
            } else {
                // Diagnostics: a committed block that will not be made
                // read-only is reported written at every asking and never
                // known to the guards (one in 1,024 said).
                #[cfg(feature = "diagnostics")]
                {
                    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 1024 == 0 {
                        tracing::info!(
                            "a block of title memory could not be made read-only (one in 1024): {:#x}: {error}",
                            block * BLOCK
                        );
                    }
                }
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again. Reports blocks whose cached
/// contents became dirty, including other views of the same pages.
fn open(g: &mut Guards, start: usize, end: usize, mut dirtied: impl FnMut(usize)) -> bool {
    let any = open_here(g, start, end, &mut dirtied);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK), &mut dirtied) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(&(alias / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize, mut dirtied: impl FnMut(usize)) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    if blocks.is_empty() {
        return false;
    }
    for &block in &blocks {
        g.protected.remove(&block);
        if g.clean.remove(&block) {
            dirtied(block * BLOCK);
        }
        clean_bit(block, false);
    }
    // Runs of blocks next to each other in one call (fewer kernel
    // transitions and TLB shootdowns). One call cannot span two views: the
    // title's page mappings are one view a page, and a run over several
    // failed whole, leaving blocks read-only that were no longer noted
    // guarded (a kernel read into them then failed and the title stopped).
    // So a run that fails is given back block by block.
    let give = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: blocks we made read-only; only their protection changes.
        let whole = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READWRITE,
                &mut old,
            )
        } != 0;
        if !whole && to - from > 1 {
            for block in from..to {
                // SAFETY: as above, one block.
                unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
            }
        }
    };
    let mut run_start = blocks[0];
    let mut run_end = run_start + 1;
    for &block in &blocks[1..] {
        if block == run_end {
            run_end = block + 1;
        } else {
            give(run_start, run_end);
            run_start = block;
            run_end = block + 1;
        }
    }
    give(run_start, run_end);
    true
}

/// Before our own synchronous user-mode write into title memory
/// `[at, at + len)` (GPU copy-back, `Step::Copy`): opens any guarded blocks
/// without pinning them writable for [`PIN`] (unlike async kernel reads in
/// [`kernel_writing`]), and uses a read-lock fast path so parallel relay
/// workers do not contend on the guards write lock when blocks are already open.
pub(crate) fn writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    let needs_open = {
        let g = GUARDS.read().unwrap_or_else(|e| e.into_inner());
        g.protected.range(first..last).next().is_some()
            || g.aliased.range(first..last).next().is_some()
    };
    if needs_open {
        let mut g = guards();
        open(&mut g, at, end, |_| {});
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// Before a write into title memory `[at, at + len)` through the kernel (which
/// does not fault into our handler): opens its guarded blocks and leaves them
/// writable for [`PIN`] across every view of their pages. The runtime calls it
/// (as `kernel_will_write`) before the title's kernel reads into its own memory.
pub fn kernel_writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    // The length is the title's (`fread`'s size times count, saturated).
    let end = at.saturating_add(len);
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    {
        let mut g = guards();
        g.prune_pins();
        let until = std::time::Instant::now() + PIN;
        g.pin(first, last, until);
        open(&mut g, at, end, |_| {});
        if view_guard() {
            // The same pages at their other views: the write lands in them too,
            // so their copies must take them again, now and when it has landed.
            crate::inplace::for_each_alias_in(
                first.saturating_mul(BLOCK),
                last.saturating_mul(BLOCK),
                |block, shown| {
                    for &alias in shown {
                        let alias_block = alias / BLOCK;
                        if alias_block != block {
                            g.pin(alias_block, alias_block + 1, until);
                            open_here(&mut g, alias, alias.saturating_add(BLOCK), |_| {});
                        }
                    }
                },
            );
        }
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end, |_| {});
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n), |_| {}) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Coordinates in-place CPU reads of title memory with mapping changes.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
///
/// Only used when `VICTORY_MAPPING_RANGES=0` ([`by_range`]); otherwise reads
/// and changes hold only the bytes they touch ([`SPANS`]).
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Whether a read in place and a change of mapping keep apart only when
/// their bytes meet ([`SPANS`]), not whatever memory each touches
/// ([`MAPPING`]); `VICTORY_MAPPING_RANGES=0` holds all of title memory for
/// each. On by default.
pub(crate) fn by_range() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MAPPING_RANGES").is_none_or(|v| v != "0"))
}

/// Reads of title memory in place under way, and changes of its mapping,
/// by the bytes they touch ([`by_range`]).
struct Spans {
    /// To tell them apart.
    next: u64,
    /// Reads under way: number, start and end.
    reads: Vec<(u64, usize, usize)>,
    /// Changes: number, start, end, and whether under way (else waiting for
    /// the reads over its bytes to end; no read begins over them meanwhile,
    /// so that one read after another cannot keep a change waiting).
    changes: Vec<(u64, usize, usize, bool)>,
    /// Threads waiting on [`SPANS_LEFT`].
    waiting: usize,
}

static SPANS: Mutex<Spans> = Mutex::new(Spans {
    next: 0,
    reads: Vec::new(),
    changes: Vec::new(),
    waiting: 0,
});

/// Told when a read or a change ends while a thread waits.
static SPANS_LEFT: std::sync::Condvar = std::sync::Condvar::new();

fn spans() -> std::sync::MutexGuard<'static, Spans> {
    SPANS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits until a read or a change ends.
fn spans_wait(
    mut s: std::sync::MutexGuard<'static, Spans>,
) -> std::sync::MutexGuard<'static, Spans> {
    s.waiting += 1;
    let mut s = SPANS_LEFT.wait(s).unwrap_or_else(|e| e.into_inner());
    s.waiting -= 1;
    s
}

/// Whether `[a, b)` and `[c, d)` share a byte.
fn meet(a: usize, b: usize, c: usize, d: usize) -> bool {
    a < b && c < d && a < d && c < b
}

impl Spans {
    fn read(&mut self, at: usize, end: usize) -> Span {
        self.next += 1;
        self.reads.push((self.next, at, end));
        Span(self.next, false)
    }
}

/// A read's or a change's place in [`SPANS`] (its number, and whether a
/// change), given up when dropped.
struct Span(u64, bool);

impl Drop for Span {
    fn drop(&mut self) {
        let mut s = spans();
        if self.1 {
            s.changes.retain(|c| c.0 != self.0);
        } else if let Some(i) = s.reads.iter().position(|r| r.0 == self.0) {
            s.reads.swap_remove(i);
        }
        if s.waiting > 0 {
            SPANS_LEFT.notify_all();
        }
    }
}

/// What a read or a change of mapping holds: its bytes ([`by_range`]) or
/// the lock over all of title memory.
enum Held<G> {
    All(#[allow(dead_code)] G),
    Bytes(#[allow(dead_code)] Span),
}

/// Since the last frame-rate line: changes of title memory's mapping, the
/// microseconds they waited for reads in place, and those they were under
/// way.
pub(crate) static REMAPS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping {
    _held: Held<std::sync::RwLockWriteGuard<'static, ()>>,
    since: std::time::Instant,
}

impl Drop for Remapping {
    fn drop(&mut self) {
        REMAPS[2].fetch_add(
            self.since.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory `[at, at + len)`: until what it returns is dropped, no read of
/// those bytes in place is under way and none begins
/// ([`title_memory_reading`]). Taken around the whole change, the unmapping
/// and the mapping again, over all the bytes whose mapping it changes.
pub fn title_memory_remapping(at: usize, len: usize) -> Remapping {
    use std::sync::atomic::Ordering::Relaxed;
    let started = std::time::Instant::now();
    let held = if by_range() {
        let end = at.saturating_add(len);
        let mut s = spans();
        s.next += 1;
        let id = s.next;
        s.changes.push((id, at, end, false));
        while s.reads.iter().any(|r| meet(r.1, r.2, at, end))
            || s.changes.iter().any(|c| c.3 && meet(c.1, c.2, at, end))
        {
            s = spans_wait(s);
        }
        if let Some(c) = s.changes.iter_mut().find(|c| c.0 == id) {
            c.3 = true;
        }
        Held::Bytes(Span(id, true))
    } else {
        Held::All(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
    };
    REMAPS[0].fetch_add(1, Relaxed);
    REMAPS[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    Remapping {
        _held: held,
        since: std::time::Instant::now(),
    }
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] Held<std::sync::RwLockReadGuard<'static, ()>>);

/// For reading title memory `[at, at + len)` in place: until what it
/// returns is dropped, the runtime changes no mapping of those bytes, so
/// memory found mapped stays mapped. Not to be taken twice by one thread,
/// nor held across anything that maps title memory.
pub fn title_memory_reading(at: usize, len: usize) -> Reading {
    if !by_range() {
        return Reading(Held::All(MAPPING.read().unwrap_or_else(|e| e.into_inner())));
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    while s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        s = spans_wait(s);
    }
    Reading(Held::Bytes(s.read(at, end)))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping of those bytes (or waits to), for a caller
/// that has another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    if !by_range() {
        return match MAPPING.try_read() {
            Ok(guard) => Some(Reading(Held::All(guard))),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(Held::All(e.into_inner()))),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    if s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        return None;
    }
    Some(Reading(Held::Bytes(s.read(at, end))))
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    g.unpin(range.start, range.end);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        clean_bit(block, false);
        // A view mapped a moment ago may already have been made
        // read-only as another view of a guarded page (between the
        // runtime's `title_memory_view` and this call): it is given
        // back, or the title's next write there would fault with no
        // guard to answer for it.
        let mut old = 0;
        // SAFETY: only the protection of title view memory changes; the
        // call fails harmlessly on memory that is no longer a view.
        unsafe {
            windows_sys::Win32::System::Memory::VirtualProtect(
                (block * BLOCK) as *const _,
                BLOCK,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
                &mut old,
            )
        };
    }
    let aliased: Vec<usize> = g.aliased.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    if view_known() {
        // The runtime calls this for mapping or unmapping section views and
        // placeholders: what stands there is view memory or a hole in it, so
        // the write watch's asking can only fail there.
        g.viewed.extend(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK));
        return;
    }
    // What the range shows is new: no longer known to be a view.
    let viewed: Vec<usize> = g
        .viewed
        .range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK))
        .copied()
        .collect();
    for block in viewed {
        g.viewed.remove(&block);
    }
}

/// The runtime left `[at, at + len)` mapped to the exact same physical pages
/// during a page-mapping batch (`CopyPageMappingsBatchX`), skipping redundant
/// unmap/remap syscalls. Placed textures and guarded buffer copies over the
/// range still treat its contents as potentially updated via another alias,
/// and any newly mapped alias of a guarded block becomes guarded too.
pub fn title_memory_alias_rebound(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    crate::descriptor::host::remapped(at, len);
    wrote(at, len);
    let mut g = guards();
    let end = at.saturating_add(len);
    let range = at / BLOCK..end.div_ceil(BLOCK);
    let clean: Vec<usize> = g.clean.range(range.clone()).copied().collect();
    for block in clean {
        g.clean.remove(&block);
        clean_bit(block, false);
    }
    if view_guard() && g.protected.range(range.clone()).next().is_some() {
        crate::inplace::for_each_alias_in(
            range.start.saturating_mul(BLOCK),
            range.end.saturating_mul(BLOCK),
            |block, shown| {
                if !g.protected.contains(&block) {
                    return;
                }
                for &alias in shown {
                    let alias = alias / BLOCK;
                    g.aliased.insert(alias);
                    if alias != block {
                        protect_alias(&mut g, alias);
                    }
                }
            },
        );
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks when `open` returns false: another thread may
        // have opened the block since this one faulted, and a copy may have
        // guarded it again since (then it is ours again, not read-only for
        // some other reason). Collect dirtied blocks and drop `GUARDS` before
        // calling `descriptor::host::remapped` so VEH faults cannot deadlock
        // with `REMAPPED`.
        let mut dirty_blocks = Vec::new();
        let handled = {
            let mut g = guards();
            if open(&mut g, at, at + 1, |block| dirty_blocks.push(block)) {
                Some(true)
            } else if writable(at) {
                Some(false)
            } else {
                None
            }
        };
        match handled {
            Some(true) => {
                for block in dirty_blocks {
                    // Indexed textures aren't checked each time a shader samples them.
                    if crate::descriptor::host::remapped(block, BLOCK) {
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::GuardTextureDirty, 1);
                    }
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            Some(false) => return EXCEPTION_CONTINUE_EXECUTION,
            None => {}
        }
        // Read-only, in a view we were told of, and not noted guarded: only we
        // make title views read-only, so it is ours to give back.
        if view_guard() && crate::inplace::has_view_at(at) {
            let mut old = 0;
            let block = at - at % BLOCK;
            // SAFETY: only the protection of title view memory changes.
            let opened = unsafe {
                windows_sys::Win32::System::Memory::VirtualProtect(
                    block as *const _,
                    BLOCK,
                    windows_sys::Win32::System::Memory::PAGE_READWRITE,
                    &mut old,
                )
            } != 0;
            if opened {
                tracing::warn!(
                    at = format_args!("{at:#x}"),
                    "a read-only view block that no guard knew was opened"
                );
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages:
    /// - synchronous user-mode [`writing`] opens the other views and immediately
    ///   re-guards on the next check without leaving a 2 s pin;
    /// - async [`kernel_writing`] keeps all views writable and reported written
    ///   while pinned, then re-guards cleanly once the pin expires.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Synchronous write (GPU copy-back): opens without pinning, so the next
        // check reports it once and re-guards immediately.
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Async kernel write: pinned writable for 2 s across all views.
        kernel_writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert!(writable(guarded + BLOCK));
        assert!(writable(other + BLOCK));
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        // Once the pin expires, the block re-guards cleanly.
        guards().pins.clear();
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        kernel_writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        guards().pins.clear();
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view block left read-only without being tracked in `Guards` (e.g.
    /// raced with a remap) is recovered by the vectored write handler when
    /// `inplace` knows it is a view.
    #[test]
    fn orphaned_readonly_view_block_is_opened_on_write_fault() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        let (section, one, two) = two_views(BLOCK);
        // Ensure the vectored handler is installed, then leave `one` read-only
        // without noting it in `Guards`.
        install();
        let mut old = 0;
        // SAFETY: our mapped view from `two_views`.
        assert_ne!(
            unsafe { VirtualProtect(one as *const _, BLOCK, PAGE_READONLY, &mut old) },
            0
        );
        // SAFETY: handler restores PAGE_READWRITE because `inplace` knows `one`.
        unsafe { ((one + 16) as *mut u8).write_volatile(0x5a) };
        assert!(writable(one));
        // SAFETY: readable byte we just wrote.
        assert_eq!(unsafe { ((one + 16) as *const u8).read_volatile() }, 0x5a);
        drop_views(section, &[one, two], BLOCK);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    #[test]
    fn alias_rebound_marks_guarded_block_unclean_and_guards_new_aliases() {
        let (section, a, b) = two_views(BLOCK);
        // Pin `b` initially so guarding `a` leaves `b` writable (simulating
        // an alias not yet guarded when an identical alias remap lands on `a`).
        let until = std::time::Instant::now() + PIN;
        guards().pin(b / BLOCK, b / BLOCK + 1, until);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]));
        assert!(writable(b));

        // Clear the pin, write through `b`, then notify an identical alias rebind on `a`.
        guards().pins.clear();
        // SAFETY: `b` is writable committed view memory.
        unsafe { ((b + 4) as *mut u8).write_volatile(0x42) };
        title_memory_alias_rebound(a, BLOCK);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]));

        // `b` must also be guarded now so subsequent CPU writes through `b` open `a`.
        assert!(!writable(b));
        // SAFETY: writing through guarded `b` faults into the VEH and opens `a` too.
        unsafe { ((b + 12) as *mut u8).write_volatile(0x99) };
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));

        drop_views(section, &[a, b], BLOCK);
    }

    /// A VEH write fault on a guarded alias view opens the clean view of the
    /// same physical page and calls `descriptor::host::remapped` after dropping
    /// `GUARDS`, refreshing indexed textures without risking lock inversion.
    #[test]
    fn alias_write_fault_triggers_remapped_without_deadlock() {
        let (section, a, b) = two_views(BLOCK);
        // Guard `a`: `a` is protected+clean, `b` is protected as an alias.
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(Vec::new()));

        // Direct `open` check: opening alias `b` reports clean view `a`.
        let mut dirty = Vec::new();
        assert!(open(&mut guards(), b + 8, b + 9, |block| dirty.push(block)));
        assert_eq!(dirty, [a]);

        // Re-guard `a` so a real CPU write through `b` enters the VEH handler.
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(Vec::new()));

        if let Some(heap) = crate::descriptor::host::HostHeap::new(false, 8) {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (a as u32, (a >> 32) as u32);
            let view = crate::descriptor::View::Placed(
                false,
                [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0],
                desc,
            );
            assert!(heap.note_buffer(1, Some(&view), false).is_empty());
            let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
            assert_eq!(queued, 1);
            heap.texture_slot_done(1, write, None);
            assert!(heap.texture_slots_due(u64::MAX, 8).0.is_empty());

            // SAFETY: writing to guarded alias `b` faults into `on_write`,
            // drops `GUARDS`, and calls `descriptor::host::remapped(a, BLOCK)`.
            unsafe { ((b + 16) as *mut u8).write_volatile(0x7c) };
            assert!(
                heap.texture_slots_remapped(50) >= 1,
                "alias write fault must queue texture slot refresh for `a`"
            );
            let due: Vec<u32> = heap.texture_slots_due(51, 8).0.into_iter().map(|d| d.0).collect();
            assert_eq!(due, [1]);
        } else {
            // SAFETY: even without a host GPU, VEH fault through `b` opens `a`.
            unsafe { ((b + 16) as *mut u8).write_volatile(0x7c) };
        }
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        drop_views(section, &[a, b], BLOCK);
    }

    /// Guarded view memory is left out of `ask_ahead` / `ask_once` (even when
    /// not registered in `inplace`, so `in_view` is false and only `known_view`
    /// identifies it), while non-view write-watched memory is still queried via
    /// `GetWriteWatch` and its writes are reported.
    #[test]
    fn ask_once_skips_known_guarded_views_and_watches_private_memory() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: fresh write-watched private allocation and pagefile view.
        let (priv_p, section, view_p) = unsafe {
            let priv_p = VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            assert_ne!(priv_p, 0);
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view_p = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE).Value as usize;
            assert_ne!(view_p, 0);
            (priv_p, section, view_p)
        };
        // Baseline: clear private watch bit and guard the view so `known_view` is true.
        assert!(writes(priv_p, priv_p + PAGE, false).is_some());
        assert!(guard(view_p, view_p + PAGE).is_some());
        assert!(known_view(view_p));
        assert!(!known_view(priv_p));

        // Write a byte to private memory, then call `ask_once` with both ranges.
        // SAFETY: committed private write-watched page.
        unsafe { ((priv_p + 16) as *mut u8).write_volatile(0x7a) };
        ask_once(&[(priv_p, priv_p + PAGE), (view_p, view_p + PAGE)]);
        let (now, priv_written) = asked(priv_p, priv_p + PAGE).expect("private range asked ahead");
        assert!(
            priv_written.iter().any(|&(p, w)| p == priv_p / PAGE && w == now),
            "private write must still be observed via GetWriteWatch"
        );
        assert!(
            asked(view_p, view_p + PAGE).is_none(),
            "known guarded view must be skipped by ask_once"
        );
        ask_once_end();

        // `wrote_own` records both the page and its aliases in the journal.
        let (before, own_now) = wrote_own(view_p / PAGE, &[view_p / PAGE + 1]);
        assert!(own_now > before);

        title_memory_remapped(view_p, PAGE);
        // SAFETY: clean up view, section, and private allocation.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_p as *mut _ });
            CloseHandle(section);
            VirtualFree(priv_p as *mut _, 0, MEM_RELEASE);
        }
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{
        PAGE, by_range, title_memory_reading, title_memory_reading_now, title_memory_remapping,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Addresses only these tests hold (nothing is read there).
    const HERE: usize = 0x7e00_0000_0000;

    /// A change of mapping waits for a read in place of its bytes that is
    /// under way, and such a read waits for the change.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading(HERE + PAGE / 2, PAGE);
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping(HERE + PAGE, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !changed.load(Ordering::SeqCst),
            "the mapping changed under a read in place"
        );
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping(HERE, 2 * PAGE);
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading(HERE + PAGE + 16, 16);
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !read.load(Ordering::SeqCst),
            "a read in place began during a change of mapping"
        );
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// Reads and changes whose bytes do not meet do not wait for each other,
    /// and a change waiting for a read keeps new reads of its bytes out (but
    /// not of others).
    #[test]
    fn reads_and_remappings_of_other_bytes_go_on() {
        if !by_range() {
            return; // VICTORY_MAPPING_RANGES=0: one lock over all of it
        }
        let at = HERE + 16 * PAGE;
        let reading = title_memory_reading(at, PAGE);
        // Another page changes while the read is under way, and another
        // read goes on while that change is.
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let changing = title_memory_remapping(at + PAGE, PAGE);
            let (read, was_read) = mpsc::channel();
            std::thread::spawn(move || {
                let _reading = title_memory_reading(at + 2 * PAGE, PAGE);
                read.send(()).unwrap();
            });
            let other = was_read.recv_timeout(Duration::from_secs(10)).is_ok();
            drop(changing);
            done.send(other).unwrap();
        });
        assert_eq!(
            finished.recv_timeout(Duration::from_secs(10)),
            Ok(true),
            "kept apart though their bytes do not meet"
        );
        // A change of the read's page waits, and keeps a new read of its
        // bytes out meanwhile; a read elsewhere is not kept out.
        let changed = Arc::new(AtomicBool::new(false));
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                let _changing = title_memory_remapping(at, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        while !super::spans().changes.iter().any(|c| c.1 == at) {
            std::thread::yield_now();
        }
        assert!(
            title_memory_reading_now(at + 8, 8).is_none(),
            "a read began over a change that waits"
        );
        assert!(title_memory_reading_now(at + 3 * PAGE, 8).is_some());
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !changed.load(Ordering::SeqCst),
            "the mapping changed under a read in place"
        );
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
    }

    /// Zero-length spans never overlap anything and ranges past `usize::MAX`
    /// saturate without overflowing.
    #[test]
    fn empty_and_saturating_ranges_are_safe() {
        assert!(!super::meet(HERE, HERE + PAGE, HERE + 8, HERE + 8));
        assert!(!super::meet(HERE + 8, HERE + 8, HERE, HERE + PAGE));
        assert!(super::meet(0, usize::MAX, HERE, HERE + PAGE));
        if !by_range() {
            return;
        }
        let at = HERE + 32 * PAGE;
        let changing = title_memory_remapping(at, PAGE);
        assert!(title_memory_reading_now(at + 8, 0).is_some());
        drop(changing);

        let empty = title_memory_remapping(at + 8, 0);
        assert!(title_memory_reading_now(at, PAGE).is_some());
        drop(empty);

        let top = title_memory_remapping(usize::MAX - 8, PAGE);
        assert!(title_memory_reading_now(usize::MAX - 4, 16).is_none());
        drop(top);
    }
}
