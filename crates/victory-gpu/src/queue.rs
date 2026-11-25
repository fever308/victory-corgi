//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given after a wait for a fence that had not been
    /// reached, in order; it runs when the fence gets there.
    #[cfg(windows)]
    held: std::sync::Mutex<Vec<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence (its address) to reach the value.
    Wait(usize, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), and the order they run in.
    Run(Vec<crate::list::Detached>, Vec<usize>),
    /// A signal of the queue's, due after what it holds before it.
    Signal(usize, u64),
    /// A present ([`async_submit`]): shown when the line gets there.
    Present(Box<dyn FnOnce() + Send>),
    /// How many submissions the title had given when it gave the run
    /// behind this (`crate::watch::TITLE_SUBMITS`).
    Stamp(u64),
    /// Placeholder at the head of a queue's line while a part taken from it
    /// is waiting in [`Joined`] or running without the line lock: keeps the
    /// line non-empty so concurrent submissions and signals stay behind it.
    Running,
}

/// Presents given to lines and not shown yet ([`async_submit`]): the title
/// is held at its present while there are two.
#[cfg(windows)]
static PRESENTS: (std::sync::Mutex<u32>, std::sync::Condvar) = (std::sync::Mutex::new(0), std::sync::Condvar::new());

/// Whether a queue other than `q` holds something.
#[cfg(windows)]
fn others_hold(q: usize) -> bool {
    lock(&WAITING).iter().any(|&w| {
        // SAFETY: a live queue: one released is taken off the list.
        w != q && !lock(unsafe { &(*(w as *const Queue)).held }).is_empty()
    })
}

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether queues wait for fences. `VICTORY_QUEUE_WAIT=0`: they do not,
/// and every submission runs when it is made (as before).
#[cfg(windows)]
fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE_WAIT").is_none_or(|v| v != "0"))
}

/// Whether a submission is held list by list: the lists before the first
/// one whose wait for a memory value is not met run at once, and each list
/// that waits is held with the lists after it, behind its own waits.
/// Holding a submission whole made the title's queues wait for each other:
/// a graphics submission was held for a value a compute list writes, and
/// that compute list for a value an earlier list of the same graphics
/// submission writes. Nothing moved until the present forced it, and the
/// title reached its present only after waiting about 2.5 s, every frame
/// (runs 27-30: 2.4-2.9 s from the last submission to the present call,
/// 13-32 ms before waits were honoured; the frame's main graphics
/// submissions, 88 lists, ran forced, the compute lists before them).
/// On by default; `VICTORY_SPLIT_WAITS=0` holds whole submissions, as
/// before.
#[cfg(windows)]
fn split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPLIT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a list is cut into host lists at its waits for memory values
/// (`crate::list::cut`), and a queue runs a submission part by part: it
/// stands at each wait until the value is in memory, with everything
/// recorded before the wait run and its bottom-of-pipe writes done, as on
/// the console. Held list by list ([`split`]) a wait still held its whole
/// list: the title's lists are chains of jobs, each behind a wait for a
/// value another queue's list writes in the middle of its own chain
/// (frame 858 of run 33: a compute list of 433 commands with 13 waits;
/// graphics lists of the form "341 dispatches, wait, draws"), so two lists
/// each waited for a write behind the other's wait and only the present
/// moved them, in the wrong order.
/// On by default where waits are honoured list by list;
/// `VICTORY_CUT_WAITS=0` holds whole lists, as before.
#[cfg(windows)]
pub(crate) fn cuts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    waits() && split() && *ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a present runs what queues still hold (on by default: a wait
/// is not kept across frames). `VICTORY_WAIT_FORCE=0` keeps it held: for
/// finding out whether the title's waits reach across its presents; a
/// wait nothing ever meets then stops the title for good.
#[cfg(windows)]
fn forces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_FORCE").is_none_or(|v| v != "0"))
}

/// Restores `crate::watch::CURRENT_STAMP` to `0` on scope exit, even on unwind.
#[cfg(windows)]
struct CurrentStampGuard;

#[cfg(windows)]
impl Drop for CurrentStampGuard {
    fn drop(&mut self) {
        crate::watch::CURRENT_STAMP.store(0, Ordering::Relaxed);
    }
}

/// Runs `f` with `crate::watch::CURRENT_STAMP` set to `stamp` and resets it to
/// `0` when `f` returns or unwinds.
#[cfg(windows)]
fn with_current_stamp<R>(stamp: u64, f: impl FnOnce() -> R) -> R {
    crate::watch::CURRENT_STAMP.store(stamp, Ordering::Relaxed);
    let _guard = CurrentStampGuard;
    f()
}

/// A submission to a queue whose lists are cut at their waits ([`cuts`]).
/// The lists before the first one that waits run at once if the queue
/// holds nothing; the rest goes behind what the queue holds, part by part
/// (`crate::list::detach_parts`): the waits of a part, then its host
/// lists. [`release_held`] runs the queue's line as far as the values are
/// there, now and whenever a submission has run.
#[cfg(windows)]
fn submit_cut(this: This, lists: &[This]) {
    if lists.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    let stamp = crate::watch::TITLE_SUBMITS.fetch_add(1, Ordering::AcqRel) + 1;
    // (With `async_submit` nothing runs in the title's call: all of it goes
    // to the queue's line.)
    let first = if !async_submit() && lock(&queue(this).held).is_empty() {
        lists
            .iter()
            .position(|&l| crate::list::cut_list(l))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {first} run at once, the rest part by part)",
            Some(queue(this).kind),
            first < lists.len()
        )
    });
    if first > 0 {
        with_current_stamp(stamp, || {
            if crate::host::late() {
                // Sent to the card and not waited for: their host sides are taken
                // from the title's lists (which it may reset before they have run)
                // and come back, with their tables' slots, when they are settled.
                let (taken, order) = crate::list::detach(&lists[..first]);
                host.execute_late(taken, order);
            } else {
                let now: Vec<_> = lists[..first]
                    .iter()
                    .filter_map(|&l| crate::list::host_list(l))
                    .collect();
                host.execute(&now);
                // They have run: their descriptor tables' slots are free again, as
                // on the path without cuts ([`execute`]). This path replaced that
                // one without the call, so the lists that run at once kept their
                // slots until the title's reset and the ring ran out as before
                // (run 45: 3,143 of 15,942 tables refused in one frame).
                crate::list::tables_done(&lists[..first]);
            }
        });
    }
    if first < lists.len() {
        let parts = crate::list::detach_parts(&lists[first..]);
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = first,
                parts = parts.len(),
                waits = parts.iter().map(|p| p.0.len()).sum::<usize>(),
                "a submission is given to its queue part by part (its lists wait for memory values)"
            );
        }
        {
            let mut held = lock(&queue(this).held);
            for (waits, taken, order) in parts {
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Stamp(stamp));
                held.push(Held::Run(taken, order));
            }
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just
    // given to the queue may be free to run already.
    release_held(false);
}

/// A submission to a queue that honours waits, list by list ([`split`]).
/// shortcut: a wait still holds its whole list, not only the commands
/// after it; and lists named twice among those held stay in one piece.
#[cfg(windows)]
fn submit_split(this: This, lists: &[This]) {
    if lists.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    let stamp = crate::watch::TITLE_SUBMITS.fetch_add(1, Ordering::AcqRel) + 1;
    let by = crate::list::waits_by_list(lists);
    // Behind what the queue already holds, nothing runs now.
    let cut = if lock(&queue(this).held).is_empty() {
        (0..lists.len())
            .find(|&i| by[i].iter().any(|w| !w.met()))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {cut} run at once)",
            Some(queue(this).kind),
            cut < lists.len()
        )
    });
    if cut > 0 {
        let now: Vec<_> = lists[..cut]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        with_current_stamp(stamp, || host.execute(&now));
    }
    if cut < lists.len() {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = cut,
                waits = ?by[cut],
                "lists of a submission are held for a memory value"
            );
        }
        let rest = &lists[cut..];
        // A list's host side can be taken once.
        let twice = rest.iter().enumerate().any(|(i, l)| rest[..i].contains(l));
        let mut pieces: Vec<Held> = Vec::new();
        let mut from = cut;
        while from < lists.len() {
            // Up to the next list that waits for anything (met or not: it
            // is looked at again when its turn comes).
            let to = if twice {
                lists.len()
            } else {
                (from + 1..lists.len())
                    .find(|&i| !by[i].is_empty())
                    .unwrap_or(lists.len())
            };
            let waits: Vec<crate::list::ValueWait> = by[from..to].concat();
            pieces.extend(waits.into_iter().map(Held::Value));
            let (taken, order) = crate::list::detach(&lists[from..to]);
            pieces.push(Held::Stamp(stamp));
            pieces.push(Held::Run(taken, order));
            from = to;
        }
        lock(&queue(this).held).extend(pieces);
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just held
    // may be free to run already.
    release_held(false);
}

/// Adds `what()` to what the queue holds, if it holds anything (it waits
/// for a fence): whether it was added.
#[cfg(windows)]
fn hold(this: This, what: impl FnOnce() -> Held) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        return false;
    }
    held.push(what());
    true
}

/// Adds `Held::Stamp(stamp)` followed by `Held::Run(taken, order)` to what
/// the queue holds, if it holds anything (under the same lock): whether it was added.
#[cfg(windows)]
fn hold_stamped(
    this: This,
    stamp: u64,
    what: impl FnOnce() -> (Vec<crate::list::Detached>, Vec<usize>),
) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        return false;
    }
    let (taken, order) = what();
    held.push(Held::Stamp(stamp));
    held.push(Held::Run(taken, order));
    true
}

/// Runs what waiting queues hold as far as their fences have been reached
/// (`all`: all of it), each queue's in the order it was given. Called
/// where a fence may have moved: after a signal and after a submission (a
/// list's bottom-of-pipe write may be into a fence), never from inside
/// one.
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    // With `async_submit` the line is run by a thread of ours: the caller
    // (a title thread, mostly) only wakes it.
    if !all && async_submit() && !ASYNC_WORKER.with(|w| w.get()) {
        async_poke();
        return;
    }
    release_held_now(all);
}

thread_local! {
    /// This thread runs the queues' lines ([`async_submit`]).
    static ASYNC_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `VICTORY_ASYNC_SUBMIT=1` (experiment): a submission is given to its
/// queue's line and the call returns; a thread of ours runs the lines
/// ([`release_held_now`]), preparing and sending each submission while the
/// title records the next. Preparing (asking what the title wrote, copying
/// it up) was done in the title's call, one submission after another: 27
/// to 48 ms of a cutscene frame of 42 to 105, during which the title's
/// thread did nothing else.
///
/// What the title is told (fence signals behind the submissions, the
/// values its lists wait for) is told when the line gets there, as when a
/// queue holds a submission behind a fence. Title memory is read when the
/// line prepares the submission, later than the call: as on the console,
/// where the card reads it when it runs.
///
/// A present runs what the lines still hold itself, as before.
#[cfg(windows)]
pub(crate) fn async_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::host::late() && std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1"))
}

/// Something was given to a line, or a fence moved: the thread that runs
/// the lines looks again.
#[cfg(windows)]
fn async_poke() {
    static POKED: (std::sync::Mutex<bool>, std::sync::Condvar) = (std::sync::Mutex::new(false), std::sync::Condvar::new());
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("victory-lines".into()).spawn(|| {
            crate::host::keep_on_fast_cores("victory-lines");
            ASYNC_WORKER.with(|w| w.set(true));
            loop {
                {
                    let mut poked = lock(&POKED.0);
                    while !*poked {
                        poked = POKED.1.wait(poked).unwrap_or_else(|e| e.into_inner());
                    }
                    *poked = false;
                }
                release_held_now(false);
            }
        });
        if let Err(e) = spawned {
            tracing::error!(error = %e, "VICTORY_ASYNC_SUBMIT: no thread to run the queues' lines");
        } else {
            tracing::info!("submissions are prepared and sent by a thread of ours (VICTORY_ASYNC_SUBMIT=1)");
        }
    });
    *lock(&POKED.0) = true;
    POKED.1.notify_one();
}

/// Whether parts that follow one another in the queues' lines go to the
/// host as one submission (`VICTORY_JOIN_PARTS=1`; `alt`: twenty seconds
/// with, twenty without, in turn, each change said in the log). Off by
/// default.
///
/// Why: the title's lists wait for memory values (19 to 20 waits a frame
/// in the first rooms, each met by the bottom-of-pipe write of the part
/// before it), and every wait cuts a submission into parts: 29 host
/// submissions a frame from about 15 of the title's. Each part is prepared
/// on its own (write-watch askings, refresh steps, heap sync, merge under
/// flights in the air, and on the card one sync list and one compare list).
///
/// With the switch, [`release_held_now`] takes consecutive ready parts off
/// the queues' lines in the order they would have run and gives them to the
/// host together as one submission: the same host lists in the same order,
/// prepared once. Between two parts a wait passes if memory meets it, if a
/// write of a flight in the air is known to meet it, or if a bottom-of-pipe
/// write of a part taken before it meets it (that part is ahead of it in
/// the joined submission, which is the order the wait asks for). A signal
/// met behind parts not sent yet is delivered right behind the joined
/// submission: never before the part it followed has run. Anything else (a
/// wait not known to be met, a present, the end of what is free to run)
/// sends what was taken first. At most [`join_most`] parts or
/// [`join_lists`] lists are joined, so that the card has work while the
/// next ones are prepared. Each queue with a part taken keeps the mark
/// [`Held::Running`] at the head of its line until the part is sent: what
/// the title gives the queue meanwhile, signals too, stays behind.
#[cfg(windows)]
fn join_parts() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_JOIN_PARTS").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        };
        if mode != 0 {
            tracing::info!(
                "parts that follow one another in the queues' lines go to the host as one submission, at most {} parts or {} lists (VICTORY_JOIN_PARTS=1; alt: in turn)",
                join_most(),
                join_lists()
            );
        }
        mode
    });
    match mode {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "parts joined into one host submission from now (VICTORY_JOIN_PARTS=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// The most parts joined into one host submission (`VICTORY_JOIN_MOST`, 4).
#[cfg(windows)]
fn join_most() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_JOIN_MOST")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(4)
    })
}

/// The most host lists joined into one host submission (`VICTORY_JOIN_LISTS`, 48).
#[cfg(windows)]
fn join_lists() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_JOIN_LISTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(48)
    })
}

/// Combines submission stamps for a joined batch (`crate::watch::CURRENT_STAMP`):
/// 0 (unknown) poisons the batch; otherwise the maximum stamp stands so a
/// shared write-watch asking is only reused if it was made after every part
/// in the batch was submitted.
#[cfg(windows)]
fn combine_stamp(current: u64, parts: usize, next: u64) -> u64 {
    if parts == 0 {
        next
    } else if current == 0 || next == 0 {
        0
    } else {
        current.max(next)
    }
}

/// With [`join_parts`]: parts taken off the queues' lines and not yet
/// given to the host, in the order they run.
#[cfg(windows)]
#[derive(Default)]
struct Joined {
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    parts: usize,
    /// Combined `TITLE_SUBMITS` stamp for the parts taken (0 if unknown).
    stamp: u64,
    /// Signals met behind those parts: delivered when the parts are sent.
    signals: Vec<(usize, u64)>,
    /// The queues with a part here: each has [`Held::Running`] at the
    /// head of its line.
    marked: Vec<usize>,
}

#[cfg(windows)]
impl Joined {
    fn is_empty(&self) -> bool {
        self.parts == 0
    }

    /// One more part behind those taken: its lists keep their order.
    fn add(&mut self, lists: Vec<crate::list::Detached>, order: Vec<usize>, stamp: u64) {
        self.stamp = combine_stamp(self.stamp, self.parts, stamp);
        let (lists, order) = join(
            std::mem::take(&mut self.lists),
            std::mem::take(&mut self.order),
            lists,
            order,
        );
        self.lists = lists;
        self.order = order;
        self.parts += 1;
    }

    fn full(&self) -> bool {
        self.parts >= join_most() || self.order.len() >= join_lists()
    }

    /// Whether a bottom-of-pipe write of a part taken meets `wait`, with no
    /// later joined list overwriting it or touching the address unsurely.
    fn meets(&self, wait: &crate::list::ValueWait) -> bool {
        let (address, bytes) = wait.place();
        let mut last_exact: Option<bool> = None;
        for &i in &self.order {
            match self.lists[i]
                .host()
                .will_write_state(address, bytes, |v| wait.holds_for(v))
            {
                crate::host::WriteMatch::None => {}
                crate::host::WriteMatch::Exact(met) => last_exact = Some(met),
                crate::host::WriteMatch::Unsure => return false,
            }
        }
        last_exact == Some(true)
    }
}

/// The things of one part behind those of another: `order` names places in
/// `lists`, so the second part's places move up by the first's count.
#[cfg(windows)]
fn join<T>(
    mut lists: Vec<T>,
    mut order: Vec<usize>,
    more: Vec<T>,
    more_order: Vec<usize>,
) -> (Vec<T>, Vec<usize>) {
    let base = lists.len();
    lists.extend(more);
    order.extend(more_order.into_iter().map(|i| i + base));
    (lists, order)
}

/// Parts joined, host submissions they went in, their lists, signals
/// delivered behind them, and waits passed for a write of a part taken.
#[cfg(windows)]
static JOINED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Gives the host what [`Joined`] holds, as one submission, delivers the
/// signals met behind it, and takes the marks off the lines. Called with
/// no line locked.
#[cfg(windows)]
fn send_joined(joined: &mut Joined) {
    if joined.parts > 0 {
        let (lists, order) = (
            std::mem::take(&mut joined.lists),
            std::mem::take(&mut joined.order),
        );
        let stamp = std::mem::take(&mut joined.stamp);
        JOINED[0].fetch_add(joined.parts as u64, Ordering::Relaxed);
        let sent = JOINED[1].fetch_add(1, Ordering::Relaxed) + 1;
        JOINED[2].fetch_add(order.len() as u64, Ordering::Relaxed);
        JOINED[3].fetch_add(joined.signals.len() as u64, Ordering::Relaxed);
        if sent % 2048 == 0 {
            let n: Vec<u64> = JOINED.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
            tracing::info!(
                "parts joined (VICTORY_JOIN_PARTS=1): {} parts went to the host in {} submissions ({:.1} parts, {:.1} lists each), {} signals delivered behind them, {} waits passed for a write of a part not sent yet",
                n[0],
                n[1],
                n[0] as f64 / n[1].max(1) as f64,
                n[2] as f64 / n[1].max(1) as f64,
                n[3],
                n[4]
            );
        }
        with_current_stamp(stamp, || match crate::host::host() {
            Some(host) => host.execute_late(lists, order),
            // (Not reached: parts are only taken with a host.)
            None => crate::list::recycle(lists, &order),
        });
        joined.parts = 0;
    }
    // Behind what was just sent, in the order they were met.
    for (fence, value) in joined.signals.drain(..) {
        if !crate::host::late_signal(fence, value) {
            crate::fence::set(fence as This, value);
        }
    }
    for q in joined.marked.drain(..) {
        // SAFETY: a live queue, as in the loop that marked it.
        let line = unsafe { &(*(q as *const Queue)).held };
        let mut held = lock(line);
        if matches!(held.first(), Some(Held::Running)) {
            held.remove(0);
        }
        if held.is_empty() {
            lock(&WAITING).retain(|&w| w != q);
        }
    }
}

/// [`release_held`], on the calling thread.
#[cfg(windows)]
pub(crate) fn release_held_now(all: bool) {
    // One caller at a time, so a queue's submissions keep their order.
    static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let _one = lock(&RELEASING);
    // Parts taken and not sent yet ([`join_parts`]): none when the lines
    // are forced, and none is left when this returns.
    let joining = !all && crate::host::late() && crate::host::host().is_some() && join_parts();
    let mut joined = Joined::default();
    let mut present_now = false;
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        let mut presents_wait = false;
        for q in queues {
            // SAFETY: a live queue: one released is taken off the list.
            let line = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(line);
            let mut part_stamp = 0u64;
            loop {
                // A queue with a part in `joined` has our mark at the head
                // of its line: what is looked at is behind the mark.
                let base = usize::from(joined.marked.contains(&q));
                match held.get(base) {
                    None | Some(Held::Running) => break,
                    Some(&Held::Wait(fence, value))
                        if !all && crate::fence::completed(fence as This) < value =>
                    {
                        break;
                    }
                    // With parts taken and not sent: only what is known
                    // without a wait for the card. A wait that stays is
                    // looked at again when the parts have been sent.
                    Some(Held::Value(wait)) if !all && !joined.is_empty() => {
                        if !wait.met() && !crate::host::late_met_known(wait) {
                            if !joined.meets(wait) {
                                break;
                            }
                            JOINED[4].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Some(Held::Value(wait)) if !all && !value_met(wait) => break,
                    // The frame is shown behind the parts taken: they are
                    // sent first.
                    Some(Held::Present(..)) if !joined.is_empty() => break,
                    // A present goes behind what every queue was given
                    // before it: other lines first, unless nothing moves.
                    Some(Held::Present(_)) if !all && !present_now && others_hold(q) => {
                        presents_wait = true;
                        break;
                    }
                    Some(_) => {}
                }
                let item = held.remove(base);
                match item {
                    Held::Wait(..) | Held::Running => {}
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if all {
                            drop(held);
                            if !value_met(&wait) {
                                static SEEN: AtomicU64 = AtomicU64::new(0);
                                if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                                    tracing::warn!(
                                        queue = format_args!("{q:#x}"),
                                        ?wait,
                                        "a wait for a memory value was not met at a present: the queue runs on"
                                    );
                                }
                            }
                            held = lock(line);
                        }
                    }
                    Held::Run(lists, order) => {
                        #[cfg(feature = "diagnostics")]
                        crate::capture::op(q, || {
                            format!(
                                "RUN held submission of {} lists (forced {all})",
                                order.len()
                            )
                        });
                        // Diagnostics: the part behind this one stands at
                        // waits that are met already, so the two could have
                        // been one host submission (one wait for the card,
                        // each range shaders write compared once).
                        #[cfg(feature = "diagnostics")]
                        {
                            let waits = held[base..]
                                .iter()
                                .take_while(|h| matches!(h, Held::Value(_)))
                                .count();
                            if waits > 0
                                && matches!(
                                    held.get(base + waits),
                                    Some(Held::Run(..) | Held::Stamp(..))
                                )
                                && held[base..base + waits]
                                    .iter()
                                    .all(|h| matches!(h, Held::Value(w) if w.met()))
                            {
                                crate::copystat::add(crate::copystat::C::PartsTogether, 1);
                            }
                        }
                        let stamp = std::mem::take(&mut part_stamp);
                        if joining {
                            // Taken, to go with the parts that follow it.
                            // The line stays open behind our mark.
                            if !joined.marked.contains(&q) {
                                held.insert(0, Held::Running);
                                joined.marked.push(q);
                            }
                            joined.add(lists, order, stamp);
                            if joined.full() {
                                drop(held);
                                send_joined(&mut joined);
                                held = lock(line);
                            }
                            moved = true;
                            continue;
                        }
                        // Run without the line's lock: the title gives the queue
                        // its next submission meanwhile (`async_submit`). Keep
                        // `Held::Running` at the head so concurrent signals stay
                        // behind this submission.
                        held.insert(0, Held::Running);
                        drop(held);
                        if crate::host::late()
                            && let Some(host) = crate::host::host()
                        {
                            // Not waited for: the host keeps the lists
                            // until they are settled.
                            host.execute_late(lists, order);
                        } else {
                            let ran = {
                                let runs: Vec<&crate::host::HostList> =
                                    order.iter().map(|&i| lists[i].host()).collect();
                                match crate::host::host() {
                                    Some(host) => {
                                        host.execute(&runs);
                                        true
                                    }
                                    None => false,
                                }
                            };
                            // They have run: their host lists are used again.
                            if ran {
                                crate::list::recycle(lists, &order);
                            }
                        }
                        if all {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::add(crate::copystat::C::Forced, 1);
                            static ONCE: AtomicBool = AtomicBool::new(false);
                            if !ONCE.swap(true, Ordering::Relaxed) {
                                tracing::warn!(
                                    "a queue still held a submission at a present: its fence was not reached"
                                );
                            }
                        }
                        crate::watch::CURRENT_STAMP.store(0, Ordering::Relaxed);
                        held = lock(line);
                        if matches!(held.first(), Some(Held::Running)) {
                            held.remove(0);
                        }
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if !joined.is_empty() {
                            // Behind parts not sent yet: with them.
                            joined.signals.push((fence, value));
                        } else {
                            drop(held);
                            if !crate::host::late_signal(fence, value) {
                                crate::fence::set(fence as This, value);
                            }
                            held = lock(line);
                        }
                    }
                    Held::Stamp(stamp) => {
                        if joining {
                            part_stamp = stamp;
                        } else {
                            crate::watch::CURRENT_STAMP.store(stamp, Ordering::Relaxed);
                        }
                    }
                    Held::Present(show) => {
                        held.insert(0, Held::Running);
                        drop(held);
                        show();
                        PRESENTS.0.lock().map(|mut n| *n = n.saturating_sub(1)).ok();
                        PRESENTS.1.notify_all();
                        held = lock(line);
                        if matches!(held.first(), Some(Held::Running)) {
                            held.remove(0);
                        }
                    }
                }
                moved = true;
            }
            if held.is_empty() {
                lock(&WAITING).retain(|&w| w != q);
            }
        }
        if !moved {
            if !joined.is_empty() {
                // Nothing more is free to run behind the parts taken: they go
                // to the host, and what stood at a wait is looked at again.
                send_joined(&mut joined);
                present_now = false;
                continue;
            }
            // Only presents left that wait for other lines, which stand at
            // waits: the presents go.
            if presents_wait && !present_now {
                present_now = true;
                continue;
            }
            break;
        }
        present_now = false;
    }
}

/// Whether a queue may pass `wait`: memory holds the value, or, with
/// submissions sent to the card without a wait (`crate::host::late`), it
/// will once those in the air have run.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    wait.met() || (crate::host::late() && crate::host::late_met(wait))
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    let left = object::release_boxed::<Queue>(this);
    #[cfg(windows)]
    if left == 0 {
        lock(&WAITING).retain(|&q| q != this as usize);
    }
    left
}

/// Runs the lists' host lists and waits for them (decision 0007). A queue
/// that waits for a fence keeps them until the fence is reached.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        if lists.is_empty() {
            return;
        }
        if cuts() && !this.is_null() {
            submit_cut(this, lists);
            return;
        }
        if waits() && split() && !this.is_null() {
            submit_split(this, lists);
            return;
        }
        let stamp = crate::watch::TITLE_SUBMITS.fetch_add(1, Ordering::AcqRel) + 1;
        // A list that waits for a memory value not there yet: the queue
        // holds the submission, as it does behind a fence.
        if waits() && !this.is_null() {
            let unmet = crate::list::unmet_waits(lists);
            if !unmet.is_empty() {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(kind = queue(this).kind, ?unmet, "a submission is held for a memory value");
                }
                lock(&queue(this).held).extend(unmet.into_iter().map(Held::Value));
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
        }
        let kept = hold_stamped(this, stamp, || crate::list::detach(lists));
        // Diagnostics: the order submissions are made in, next to the
        // commands of the frame's `ops` file (which are in recording order).
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || {
            let kind = (!this.is_null()).then(|| queue(this).kind);
            // The lists as the frame's other lines name them (their host sides).
            let named: Vec<String> = lists
                .iter()
                .map(|&l| match crate::list::host_list(l) {
                    Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                    None => format!("title {l:?}"),
                })
                .collect();
            format!("SUBMIT queue kind {kind:?} held {kept} lists {named:?}")
        });
        if kept {
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        with_current_stamp(stamp, || host.execute(&host_lists));
        crate::list::tables_done(lists);
        // What the lists wrote at the bottom of the pipe may be a fence
        // another queue waits for.
        release_held(false);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::fence::pace::signal_called(fence as usize, value);
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(fence as usize, value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late_signal(fence as usize, value) {
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`: how the console orders work between queues (the title's
/// compute queue waits for the depth the graphics queue draws). Our
/// submissions run when they are made, so a wait for a value already
/// reached is nothing; otherwise what the queue is given next is held
/// ([`Held`]) until a signal or a list's bottom-of-pipe write gets the
/// fence there.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if !this.is_null() {
        let now = crate::fence::completed(fence);
        // Whether the title uses this at all (the first calls).
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(kind = queue(this).kind, now, value, "a queue is told to wait for a fence");
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || format!("WAIT queue kind {} for fence {fence:?} value {value} (now {now})", queue(this).kind));
        let behind = {
            let mut held = lock(&queue(this).held);
            let behind = !held.is_empty() || now < value;
            if behind && waits() {
                held.push(Held::Wait(fence as usize, value));
            }
            behind
        };
        if behind {
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    now,
                    value,
                    held = waits(),
                    "a queue waits for a fence not reached yet"
                );
            }
        }
        if behind && waits() {
            {
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
            // The fence may have got there meanwhile.
            release_held(false);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// Frames presented so far (also in release builds).
#[cfg(windows)]
pub(crate) fn frame_count() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

#[cfg(any(windows, feature = "diagnostics"))]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    // Submissions sent to the card and not waited for are settled first,
    // and what that frees runs in its own order, before anything is forced.
    // With `async_submit` (and the present sent ahead): the frame is shown
    // by the thread that runs the lines, behind the submissions before it,
    // and the title goes on to its next frame. It is held here while two
    // presents wait to be shown.
    #[cfg(windows)]
    let in_line = async_submit() && crate::host::late_free_present() && !this.is_null();
    #[cfg(windows)]
    if in_line {
        let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
        let shown: Option<usize> = (planes > 0)
            // SAFETY: the title passes `planes` plane descriptions.
            .then(|| unsafe { plane_parameters.as_ref() })
            .flatten()
            .filter(|plane| plane.count > 0 && !plane.resources.is_null())
            // SAFETY: the plane lists `count` resources.
            .map(|plane| unsafe { *plane.resources } as usize);
        {
            let mut waiting = PRESENTS.0.lock().unwrap_or_else(|e| e.into_inner());
            while *waiting >= 2 {
                waiting = PRESENTS.1.wait(waiting).unwrap_or_else(|e| e.into_inner());
            }
            *waiting += 1;
        }
        lock(&queue(this).held).push(Held::Present(Box::new(move || {
            // The diagnostics a present drives (target dumps, the debugger's keys, commands
            // and recordings), here as the frame is shown: the title's present returned
            // before them, and F9 did nothing with `VICTORY_ASYNC_SUBMIT=1`.
            #[cfg(feature = "diagnostics")]
            {
                crate::capture::frame_drew(frame, crate::draw::drawn());
                if crate::capture::listed(frame) {
                    crate::capture::dump_targets(frame);
                }
                crate::debugger::on_present(frame);
            }
            let Some(resource) = shown.map(|r| r as This) else {
                return;
            };
            if let (Some(texture), Some(host)) = (crate::resource::host_resource(resource), crate::host::host()) {
                const COPY_SOURCE: u32 = 0x800;
                let state = crate::resource::host_transition(resource, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before);
                let _ = host.present(texture, state, frame);
                crate::resource::host_transition(resource, state);
            }
        })));
        {
            let mut waiting = lock(&WAITING);
            if !waiting.contains(&(this as usize)) {
                waiting.push(this as usize);
            }
        }
        async_poke();
        #[cfg(feature = "diagnostics")]
        crate::fence::pace::frame();
        return S_OK;
    }
    #[cfg(windows)]
    if crate::host::late() && !crate::host::late_free_present() {
        crate::host::settle_for_present();
        release_held_now(false);
        crate::host::settle_for_present();
    } else if crate::host::late() {
        // The present is sent ahead too: what waits for a value a flight
        // will write runs behind that flight, and nothing is settled here.
        release_held_now(false);
    }
    // shortcut: what a queue still holds at a present runs now, whatever
    // its fence: a wait is not kept across frames.
    #[cfg(windows)]
    release_held_now(forces());
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::frame_drew(frame, crate::draw::drawn());
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
        crate::debugger::on_present(frame);
    }
    #[cfg(windows)]
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        let shown = host.present(texture, state, frame);
        crate::resource::host_transition(resource, state);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::fence::pace::frame();
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    use crate::resource::{
        PageMapOp,
        PageMapping::{Consecutive, Repeated},
    };

    let batch_map = crate::resource::page_batch_mapper();
    let single_map = crate::resource::page_mapper();
    if batch_map.is_none() && single_map.is_none() {
        return E_INVALIDARG;
    }
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    let mut ops = if batch_map.is_some() {
        Vec::new()
    } else {
        Vec::with_capacity(0)
    };
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        if batch_map.is_some() {
            ops.reserve(ranges.len());
        }
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            let op = match r.kind {
                0 if inside(r.count) => Some(PageMapOp {
                    to: at,
                    from: source,
                    len,
                    mapping: Consecutive,
                }),
                1 if inside(1) => Some(PageMapOp {
                    to: at,
                    from: source,
                    len,
                    mapping: Repeated,
                }),
                2 => Some(PageMapOp {
                    to: at,
                    from: 0,
                    len,
                    mapping: Consecutive,
                }),
                3 => None,
                _ => {
                    ok = false;
                    None
                }
            };
            if let Some(op) = op {
                if batch_map.is_some() {
                    if op.len > 0 {
                        ops.push(op);
                    }
                } else if let Some(map) = single_map {
                    ok &= map(op.to, op.from, op.len, op.mapping);
                }
            }
            at += len;
        }
    }
    if let Some(batch_map) = batch_map
        && !ops.is_empty()
    {
        ok &= batch_map(&ops);
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn joined_parts_keep_their_lists_and_their_order() {
        // a part of three lists run as 2, 0, 1; then one of two run as 1, 0; then one of one
        let (lists, order) = join(vec!["a0", "a1", "a2"], vec![2, 0, 1], vec!["b0", "b1"], vec![1, 0]);
        let (lists, order) = join(lists, order, vec!["c0"], vec![0]);
        assert_eq!(lists, ["a0", "a1", "a2", "b0", "b1", "c0"]);
        let run: Vec<&str> = order.iter().map(|&i| lists[i]).collect();
        assert_eq!(
            run,
            ["a2", "a0", "a1", "b1", "b0", "c0"],
            "each part's lists in its own order, the parts one after the other"
        );
        // a part with a list named twice, and an empty one
        let (lists, order) = join(vec!["a0"], vec![0, 0], Vec::new(), Vec::new());
        let (lists, order) = join(lists, order, vec!["b0"], vec![0, 0]);
        assert_eq!(
            order.iter().map(|&i| lists[i]).collect::<Vec<_>>(),
            ["a0", "a0", "b0", "b0"]
        );
    }

    #[cfg(windows)]
    #[test]
    fn joined_parts_combine_submission_stamps_conservatively() {
        assert_eq!(combine_stamp(0, 0, 5), 5);
        assert_eq!(combine_stamp(5, 1, 9), 9);
        assert_eq!(combine_stamp(9, 2, 7), 9);
        // Unknown stamp poisons the batch so shared write-watch askings re-query.
        assert_eq!(combine_stamp(9, 2, 0), 0);
        assert_eq!(combine_stamp(0, 1, 9), 0);
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }

    #[cfg(windows)]
    #[test]
    fn with_current_stamp_restores_zero_on_return_and_unwind() {
        struct ResetStamp;
        impl Drop for ResetStamp {
            fn drop(&mut self) {
                crate::watch::CURRENT_STAMP.store(0, Ordering::Relaxed);
            }
        }
        let _reset = ResetStamp;
        crate::watch::CURRENT_STAMP.store(0, Ordering::Relaxed);

        let out = with_current_stamp(42, || {
            assert_eq!(crate::watch::CURRENT_STAMP.load(Ordering::Relaxed), 42);
            7u32
        });
        assert_eq!(out, 7);
        assert_eq!(crate::watch::CURRENT_STAMP.load(Ordering::Relaxed), 0);

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_current_stamp(99, || {
                assert_eq!(crate::watch::CURRENT_STAMP.load(Ordering::Relaxed), 99);
                panic!("simulated host execution panic");
            });
        }));
        assert!(panicked.is_err());
        assert_eq!(
            crate::watch::CURRENT_STAMP.load(Ordering::Relaxed),
            0,
            "CurrentStampGuard must reset CURRENT_STAMP on unwind"
        );
    }

    #[cfg(windows)]
    #[test]
    fn submit_cut_assigns_single_stamp_to_prefix_and_held_parts() {
        if crate::host::host().is_none() || !cuts() || async_submit() {
            return;
        }
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);

        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let mut allocator = null;
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );

        let mut lists = [null; 3];
        let wait_slot = if crate::device::newer() { 0x248 } else { 0x238 };
        for l in &mut lists {
            assert_eq!(
                crate::list::create_list(null, 0, 0, allocator, null, &list_iid, l),
                S_OK
            );
        }

        let gate_mem: u64 = 0;
        let gate_addr = std::ptr::from_ref(&gate_mem) as u64;
        // lists[0] has no wait (runs in prefix `first = 1`);
        // lists[1] and lists[2] wait on unmet values so `detach_parts` produces two held parts.
        for (i, reference) in [(1usize, 1u64), (2usize, 2u64)] {
            let wait_value = slot::<extern "system" fn(This, u64, u32, u64)>(lists[i], wait_slot);
            wait_value(lists[i], gate_addr, 3, reference);
            assert!(crate::list::cut_list(lists[i]));
        }
        for &l in &lists {
            assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(l, 0x48)(l), S_OK);
        }

        // Empty submission must not mint a stamp.
        let before = crate::watch::TITLE_SUBMITS.load(Ordering::Acquire);
        submit_cut(q, &[]);
        assert_eq!(crate::watch::TITLE_SUBMITS.load(Ordering::Acquire), before);
        assert_eq!(crate::watch::CURRENT_STAMP.load(Ordering::Relaxed), 0);

        submit_cut(q, &lists);
        let after = crate::watch::TITLE_SUBMITS.load(Ordering::Acquire);
        assert_eq!(
            after,
            before + 1,
            "submit_cut must mint one stamp across prefix and all held parts"
        );
        assert_eq!(crate::watch::CURRENT_STAMP.load(Ordering::Relaxed), 0);

        {
            let held = lock(&queue(q).held);
            let mut stamps = Vec::new();
            let mut prev_was_stamp = false;
            for item in held.iter() {
                match item {
                    Held::Stamp(s) => {
                        stamps.push(*s);
                        prev_was_stamp = true;
                    }
                    Held::Run(..) => {
                        assert!(prev_was_stamp, "Held::Stamp must immediately precede Held::Run");
                        prev_was_stamp = false;
                    }
                    _ => prev_was_stamp = false,
                }
            }
            assert_eq!(stamps, vec![after, after]);
        }

        // Drain the held parts cleanly and verify CURRENT_STAMP returns to 0.
        release_held(true);
        assert!(lock(&queue(q).held).is_empty());
        assert_eq!(crate::watch::CURRENT_STAMP.load(Ordering::Relaxed), 0);

        for l in lists {
            slot::<extern "system" fn(This) -> u32>(l, 0x10)(l);
        }
        slot::<extern "system" fn(This) -> u32>(allocator, 0x10)(allocator);
        slot::<extern "system" fn(This) -> u32>(q, 0x10)(q);
    }
}
