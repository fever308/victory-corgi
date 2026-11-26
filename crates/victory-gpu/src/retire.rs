//! Epoch-based deferred retirement for host D3D12 resources and wrappers.
//!
//! Host objects the title let go of while an open list or a queued/running
//! submission may still use them (a texture its descriptors name, a descriptor
//! slot's old resource, a buffer shadow, or texture staging) are held until
//! every active epoch whose recorded `work_seq` could observe them has
//! finished.
//!
//! Lock invariant: `RETIRE_STATE` is a strict leaf lock — never lock anything
//! else while holding it, and ALWAYS drop retired objects AFTER releasing it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Warn once when pinned epochs hold back this many retired host objects
/// (for example, an unreset command list or a submission stalled behind an
/// unsignalled fence).
const RETIRED_WARN_THRESHOLD: usize = 4096;

/// Monotonic sequence bumped whenever a tracked host resource (`HostTexture`,
/// `Shadow`, `Staging`) is created. Starts at 1 so `0` means "unbounded /
/// existed before any sequence" (used by legacy [`retire`]).
pub(crate) static RESOURCE_SEQ: AtomicU64 = AtomicU64::new(1);

/// Returns a strictly increasing birth sequence for a newly allocated host
/// resource. Active command lists whose last recorded command occurred before
/// this sequence cannot reference the resource and do not block its release.
pub(crate) fn birth_seq() -> u64 {
    RESOURCE_SEQ.fetch_add(1, Ordering::SeqCst) + 1
}

/// Sentinel work sequence shared by active [`crate::host::Host::execute`]
/// submissions so any object retired while the GPU is executing stays alive
/// until execution finishes.
pub(crate) fn exec_work_seq() -> Arc<AtomicU64> {
    static SEQ: OnceLock<Arc<AtomicU64>> = OnceLock::new();
    SEQ.get_or_init(|| Arc::new(AtomicU64::new(u64::MAX)))
        .clone()
}

struct ActiveEpoch {
    /// Highest [`RESOURCE_SEQ`] observed when this epoch's owner recorded work
    /// (or `u64::MAX` for an active execution submission).
    work_seq: Arc<AtomicU64>,
    /// Whether this epoch was acquired with [`exec_work_seq`] (`u64::MAX`).
    is_exec: bool,
    /// Slab indices in [`RetireState::retired`] waiting on this epoch.
    waiting: Vec<usize>,
}

struct RetiredEntry {
    remaining: usize,
    object: Option<Retired>,
}

pub(crate) struct RetireState {
    next_epoch: u64,
    active_epochs: BTreeMap<u64, ActiveEpoch>,
    /// Subset of `active_epochs` keys whose `work_seq == u64::MAX` (active GPU
    /// execution epochs). Allows `retire_executing` (`born_seq == u64::MAX`) to
    /// inspect only executing submissions instead of scanning hundreds of open
    /// command-list epochs with `SeqCst` loads.
    exec_epochs: Vec<u64>,
    retired: Vec<Option<RetiredEntry>>,
    free_slots: Vec<usize>,
    pending_count: usize,
    warned: bool,
}

impl RetireState {
    pub(crate) const fn new() -> Self {
        Self {
            next_epoch: 1,
            active_epochs: BTreeMap::new(),
            exec_epochs: Vec::new(),
            retired: Vec::new(),
            free_slots: Vec::new(),
            pending_count: 0,
            warned: false,
        }
    }
}

pub(crate) static RETIRE_STATE: Mutex<RetireState> = Mutex::new(RetireState::new());

/// Objects [`retire_born`] dropped at once (no blocking epoch was active) and
/// kept until their blocking epochs finished, since the start.
pub(crate) static RETIREMENTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Live retirement telemetry: `(pending_objects, active_epochs, oldest_epoch)`.
pub(crate) fn retire_stats() -> (usize, usize, u64) {
    let state = RETIRE_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let oldest = state.active_epochs.keys().next().copied().unwrap_or(0);
    (state.pending_count, state.active_epochs.len(), oldest)
}

/// RAII guard pinning an active GPU execution epoch until the submission
/// finishes (or lands, for `late` flights).
pub(crate) struct ExecEpochGuard {
    #[cfg(test)]
    pub(crate) scope: Option<Arc<Mutex<RetireState>>>,
    pub(crate) epoch: u64,
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
    static RETIRE_SCOPE: std::cell::RefCell<Option<Arc<Mutex<RetireState>>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct RetireTestScope {
    prev: Option<Arc<Mutex<RetireState>>>,
}

#[cfg(test)]
impl Drop for RetireTestScope {
    fn drop(&mut self) {
        RETIRE_SCOPE.with(|cell| *cell.borrow_mut() = self.prev.take());
    }
}

#[cfg(test)]
pub(crate) fn current_retire_scope() -> Option<Arc<Mutex<RetireState>>> {
    RETIRE_SCOPE.with(|cell| cell.borrow().clone())
}

/// Isolates retirement epochs and queued drops on the current test thread (and
/// any `HostList` created or acquired on it) from concurrent GPU tests.
#[cfg(test)]
pub(crate) fn isolated_retire_scope() -> RetireTestScope {
    let fresh = Arc::new(Mutex::new(RetireState::new()));
    let prev = RETIRE_SCOPE.with(|cell| cell.borrow_mut().replace(fresh));
    RetireTestScope { prev }
}

/// Acquires an execution epoch guard in the active test scope (if any) or
/// global `RETIRE_STATE`.
#[cfg(test)]
pub(crate) fn acquire_exec_guard() -> ExecEpochGuard {
    let scope = current_retire_scope();
    let epoch = acquire_epoch_in(scope.as_deref().unwrap_or(&RETIRE_STATE), exec_work_seq());
    ExecEpochGuard { scope, epoch }
}

/// A retired object, dropped on whatever thread finishes its last blocking
/// epoch.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Acquires a monotonically increasing epoch (never 0) that keeps retired
/// objects alive until [`release_epoch_in`] is called with it.
pub(crate) fn acquire_epoch_in(
    retire_state: &Mutex<RetireState>,
    work_seq: Arc<AtomicU64>,
) -> u64 {
    let is_exec = work_seq.load(Ordering::Relaxed) == u64::MAX;
    let mut state = retire_state.lock().unwrap_or_else(|e| e.into_inner());
    let e = state.next_epoch;
    state.next_epoch += 1;
    state.active_epochs.insert(
        e,
        ActiveEpoch {
            work_seq,
            is_exec,
            waiting: Vec::new(),
        },
    );
    if is_exec {
        state.exec_epochs.push(e);
    }
    e
}

/// Releases `epoch` and drops every retired object whose last blocking epoch
/// was `epoch` (outside `retire_state`).
pub(crate) fn release_epoch_in(retire_state: &Mutex<RetireState>, epoch: u64) {
    let done: Vec<Retired> = {
        let mut state = retire_state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(active) = state.active_epochs.remove(&epoch) else {
            return;
        };
        if active.is_exec
            && let Some(pos) = state.exec_epochs.iter().position(|&e| e == epoch)
        {
            state.exec_epochs.swap_remove(pos);
        }
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

pub(crate) fn retire_in<T: 'static>(retire_state: &Mutex<RetireState>, object: T, born_seq: u64) {
    if born_seq != u64::MAX {
        // Pair with `work_seq` `SeqCst` stores/RMWs in `HostList::mark_work`
        // and `RESOURCE_SEQ` `SeqCst` increments in `birth_seq`.
        std::sync::atomic::fence(Ordering::SeqCst);
    }
    let warn = {
        let mut state = retire_state.lock().unwrap_or_else(|e| e.into_inner());
        let slot = state.free_slots.pop().unwrap_or(state.retired.len());
        let mut blockers = 0usize;
        if born_seq == u64::MAX {
            // Fast path for `retire_executing`: only active GPU execution
            // epochs have `work_seq == u64::MAX`, tracked in `exec_epochs`
            // under this same mutex.
            if !state.exec_epochs.is_empty() {
                let RetireState {
                    exec_epochs,
                    active_epochs,
                    ..
                } = &mut *state;
                for &epoch in exec_epochs.iter() {
                    if let Some(entry) = active_epochs.get_mut(&epoch) {
                        entry.waiting.push(slot);
                        blockers += 1;
                    }
                }
            }
        } else {
            for entry in state.active_epochs.values_mut() {
                if born_seq == 0 || entry.work_seq.load(Ordering::SeqCst) >= born_seq {
                    entry.waiting.push(slot);
                    blockers += 1;
                }
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

/// Computes the conservative batch `born_seq` across a set of item `born_seq`s.
///
/// An epoch blocks iff `born_seq == 0 || entry.work_seq >= born_seq`, so a
/// smaller `born_seq` is strictly more conservative (blocks a superset of
/// epochs), and `0` dominates every other sequence. Returns `u64::MAX` for an
/// empty iterator.
#[allow(dead_code)]
pub(crate) fn safe_batch_born_seq(seqs: impl IntoIterator<Item = u64>) -> u64 {
    seqs.into_iter().min().unwrap_or(u64::MAX)
}

/// Retires a non-empty batch of objects under a single `RETIRE_STATE` lock and
/// single boxed entry at `born_seq` (use [`safe_batch_born_seq`] when items
/// have distinct birth sequences).
#[allow(dead_code)]
pub(crate) fn retire_batch_born<T: 'static>(objects: Vec<T>, born_seq: u64) {
    if !objects.is_empty() {
        retire_born(objects, born_seq);
    }
}

/// Drops `object` once every currently executing [`crate::host::Host::execute`]
/// submission (`work_seq == u64::MAX`) has finished: at once if none is
/// executing. Avoids bumping global [`RESOURCE_SEQ`] when retiring overwritten
/// or released descriptor-heap references that newly recorded lists cannot
/// observe.
pub(crate) fn retire_executing<T: 'static>(object: T) {
    retire_born(object, u64::MAX);
}

/// Retires a non-empty batch of overwritten or released objects once every
/// currently executing submission has finished, acquiring `RETIRE_STATE` once.
pub(crate) fn retire_executing_batch<T: 'static>(objects: Vec<T>) {
    if !objects.is_empty() {
        retire_executing(objects);
    }
}

/// Drops `object` once every epoch active now has finished: at once if none is.
#[cfg(test)]
pub(crate) fn retire<T: 'static>(object: T) {
    retire_born(object, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Counter(Arc<AtomicUsize>);
    impl Drop for Counter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn safe_batch_born_seq_picks_most_conservative_minimum() {
        assert_eq!(safe_batch_born_seq([]), u64::MAX);
        assert_eq!(safe_batch_born_seq([u64::MAX, u64::MAX]), u64::MAX);
        assert_eq!(safe_batch_born_seq([50, 10, 30]), 10);
        assert_eq!(safe_batch_born_seq([u64::MAX, 42, 100]), 42);
        // 0 ("unbounded") must dominate every finite and executing sequence.
        assert_eq!(safe_batch_born_seq([10, 0, u64::MAX]), 0);
    }

    #[test]
    fn retire_batch_born_respects_min_born_seq_across_epochs() {
        let _scope = isolated_retire_scope();
        let state = current_retire_scope().unwrap();

        let seq1 = birth_seq();
        let e1 = acquire_epoch_in(&state, Arc::new(AtomicU64::new(seq1)));
        let seq2 = birth_seq();
        let e2 = acquire_epoch_in(&state, Arc::new(AtomicU64::new(seq2)));

        // Batch containing items born at `seq1` and `seq2` must use `min(seq1, seq2) = seq1`,
        // so BOTH `e1` (work_seq == seq1) and `e2` (work_seq == seq2) block the batch.
        let drops = Arc::new(AtomicUsize::new(0));
        let batch = vec![Counter(drops.clone()), Counter(drops.clone())];
        retire_batch_born(batch, safe_batch_born_seq([seq2, seq1]));
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        // Releasing `e2` alone must NOT drop the batch because `e1` still pins `seq1`.
        release_epoch_in(&state, e2);
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        // Releasing `e1` drops the entire batch.
        release_epoch_in(&state, e1);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn retire_batch_born_zero_dominates_and_pins_all_epochs() {
        let _scope = isolated_retire_scope();
        let state = current_retire_scope().unwrap();

        let early = acquire_epoch_in(&state, Arc::new(AtomicU64::new(1)));
        let late_seq = birth_seq();

        let drops = Arc::new(AtomicUsize::new(0));
        let batch = vec![Counter(drops.clone()), Counter(drops.clone())];
        retire_batch_born(batch, safe_batch_born_seq([late_seq, 0]));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "born_seq 0 in batch must pin even older epochs"
        );
        release_epoch_in(&state, early);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn retire_executing_batch_fast_path_only_waits_for_exec_epochs() {
        let _scope = isolated_retire_scope();
        let state = current_retire_scope().unwrap();

        // Pin many non-executing list epochs.
        let list_epochs: Vec<u64> = (0..64)
            .map(|_| acquire_epoch_in(&state, Arc::new(AtomicU64::new(birth_seq()))))
            .collect();

        // With no active exec epoch, `retire_executing_batch` drops immediately.
        let drops1 = Arc::new(AtomicUsize::new(0));
        retire_executing_batch(vec![Counter(drops1.clone()), Counter(drops1.clone())]);
        assert_eq!(drops1.load(Ordering::SeqCst), 2);

        // With an active exec epoch, batch is pinned ONLY by the exec epoch.
        let exec = acquire_exec_guard();
        let drops2 = Arc::new(AtomicUsize::new(0));
        retire_executing_batch(vec![
            Counter(drops2.clone()),
            Counter(drops2.clone()),
            Counter(drops2.clone()),
        ]);
        assert_eq!(drops2.load(Ordering::SeqCst), 0);

        // Dropping the exec guard releases the whole batch even while all 64 list epochs remain active.
        drop(exec);
        assert_eq!(drops2.load(Ordering::SeqCst), 3);

        for e in list_epochs {
            release_epoch_in(&state, e);
        }
    }
}
