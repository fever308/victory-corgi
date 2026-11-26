//! Submission-time parallel preparation helpers for [`super::Batch`]:
//! - [`wide_fills`] (`VICTORY_WIDE_FILLS`, `=0` disables): batches `Step::Fill`
//!   copies of title memory into upload memory across worker threads, with a
//!   single batched [`crate::lazy_back::materialize_ranges`] pass when
//!   `VICTORY_LAZY_BACK=1` is active so workers never spam single-page DMAs or
//!   deadlock.
//! - [`ahead_of_writes`] (`VICTORY_REFRESH_AHEAD`, `=0` disables): refreshes
//!   shadows targeted by `Step::GpuWrite` ahead of a submission's deferred
//!   steps in one parallel [`crate::shadow::refresh_batch`] pass.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use super::{
    Batch, HostList, PREP_FILL_BYTES, PREP_FILL_COUNT, PREP_FILL_US, PREP_WATCH_US, PREPARING,
    PREPARING_COPIES, Step, copy_mapped, copy_mapped_raw_checked, wide_refresh,
};

/// Whether a submission's fills of upload memory are batched and run across
/// several threads (`VICTORY_WIDE_FILLS=0`: one by one, where the lists name
/// them).
pub(super) fn wide_fills() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_WIDE_FILLS").is_none_or(|v| v != "0") && wide_refresh()
    })
}

/// Whether copies that a submission's shaders write are refreshed together on
/// several threads ahead of its deferred steps (`VICTORY_REFRESH_AHEAD=0`:
/// each one right before its `Step::GpuWrite`).
pub(super) fn ahead_of_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_REFRESH_AHEAD").is_none_or(|v| v != "0") && wide_refresh()
    })
}

/// What a deferred step means to pre-write refreshes ([`written_spans`]).
#[derive(Clone, Copy)]
enum Touch {
    /// A refresh of a copy (by address), from, to.
    Refresh(usize, usize, usize),
    /// The copy is given to shaders (`Step::GpuWrite`).
    Write(usize),
    /// Title memory is written (`Step::Copy` or `Step::Run`).
    Title,
    Nothing,
}

/// Of `steps`, up to the first that writes title memory: the copies given to
/// shaders, each with the union `[from, to)` of its refreshes before its first
/// `Step::GpuWrite`.
fn written_spans(steps: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize)> {
    // `(from, to, given_to_shaders)` keyed by shadow pointer.
    let mut spans: HashMap<usize, (usize, usize, bool)> = HashMap::new();
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                let span = spans.entry(copy).or_insert((from, to, false));
                if !span.2 {
                    (span.0, span.1) = (span.0.min(from), span.1.max(to));
                }
            }
            // Not refreshed before: nothing of it is due ahead of the write.
            Touch::Write(copy) => {
                if let Some(span) = spans.get_mut(&copy) {
                    span.2 = true;
                }
            }
            Touch::Title => break,
            Touch::Nothing => {}
        }
    }
    spans
        .into_iter()
        .filter(|(_, s)| s.2)
        .map(|(copy, (from, to, _))| (copy, from, to))
        .collect()
}

impl Batch {
    /// Executes all pending deferred `Step::Fill` copies in `self.fills`.
    /// When `lazy_back` is active, materializes all source ranges in one
    /// batched DMA before dispatching parallel `copy_mapped_raw_checked` work.
    pub(super) fn fill(&mut self) {
        // Below this many fills, thread dispatch costs more than it saves.
        const FEW: usize = 96;
        const PART: usize = 64;
        if self.fills.is_empty() {
            return;
        }
        let started = std::time::Instant::now();
        let fills = std::mem::take(&mut self.fills);
        let count = fills.len() as u64;
        let bytes: u64 = fills.iter().map(|&(_, _, len)| len as u64).sum();
        let lazy = crate::lazy_back::active();
        if lazy {
            let ranges: Vec<(usize, usize)> = fills
                .iter()
                .filter(|&&(_, _, len)| len > 0)
                .map(|&(_, src, len)| (src, len))
                .collect();
            if !ranges.is_empty() {
                crate::lazy_back::materialize_ranges(&ranges);
            }
        }
        if fills.len() < FEW {
            for &(dst, src, len) in &fills {
                if lazy {
                    if !copy_mapped_raw_checked(dst as *mut u8, src as *const u8, len, false) {
                        copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    }
                } else {
                    crate::shadow::copy_mapped_wide(dst as *mut u8, src as *const u8, len);
                }
            }
        } else {
            let fills = Arc::new(fills);
            let all_ok = Arc::new(AtomicBool::new(true));
            let chunks = fills.len().div_ceil(PART);
            let fills_w = fills.clone();
            let ok_w = all_ok.clone();
            crate::shadow::wide_each(chunks, move |i| {
                for &(dst, src, len) in fills_w.iter().skip(i * PART).take(PART) {
                    if !copy_mapped_raw_checked(dst as *mut u8, src as *const u8, len, false) {
                        ok_w.store(false, Ordering::Release);
                    }
                }
            });
            if !all_ok.load(Ordering::Acquire) {
                // A concurrent relay landing re-deferred a lazy block between
                // pre-materialization and copy: re-materialize and retry safely.
                let ranges: Vec<(usize, usize)> = fills
                    .iter()
                    .filter(|&&(_, _, len)| len > 0)
                    .map(|&(_, src, len)| (src, len))
                    .collect();
                if !ranges.is_empty() {
                    crate::lazy_back::materialize_ranges(&ranges);
                }
                for &(dst, src, len) in fills.iter() {
                    if !copy_mapped_raw_checked(dst as *mut u8, src as *const u8, len, false) {
                        copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    }
                }
            }
        }
        let us = started.elapsed().as_micros() as u64;
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(bytes, Ordering::Relaxed);
        PREPARING_COPIES[PREP_FILL_US].fetch_add(us, Ordering::Relaxed);
        PREPARING_COPIES[PREP_FILL_COUNT].fetch_add(count, Ordering::Relaxed);
        PREPARING_COPIES[PREP_FILL_BYTES].fetch_add(bytes, Ordering::Relaxed);
    }

    /// Ahead of `runs`' deferred steps ([`ahead_of_writes`]): refreshes in
    /// parallel every shadow that will be written by shaders before the first
    /// CPU write to title memory, recording the covered spans in `self.ahead`.
    pub(super) fn refresh_ahead(&mut self, runs: &[&HostList]) {
        let mut copies: HashMap<usize, Arc<crate::shadow::Shadow>> = HashMap::new();
        let mut touches = Vec::new();
        'steps: for list in runs {
            for step in &list.deferred().copies {
                let touch = match step {
                    Step::Refresh(shadow, at, len) => {
                        let key = Arc::as_ptr(shadow) as usize;
                        copies.entry(key).or_insert_with(|| shadow.clone());
                        Touch::Refresh(key, *at, *at + *len)
                    }
                    Step::GpuWrite(shadow, ..) => Touch::Write(Arc::as_ptr(shadow) as usize),
                    Step::Copy(..) | Step::Run(_) => Touch::Title,
                    Step::Fill(..) | Step::Host(_) | Step::Texture(_) => Touch::Nothing,
                };
                touches.push(touch);
                if matches!(touch, Touch::Title) {
                    break 'steps;
                }
            }
        }
        let due: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = written_spans(touches.into_iter())
            .into_iter()
            .filter_map(|(key, from, to)| Some((copies.get(&key)?.clone(), from, to)))
            .collect();
        if due.is_empty() {
            return;
        }
        for (shadow, from, to) in &due {
            self.ahead.insert(Arc::as_ptr(shadow) as usize, (*from, *to));
        }
        let ahead_watch = due.len() > 1 && !crate::watch::ask_kept();
        if ahead_watch {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = due
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
        let copied = crate::shadow::refresh_batch(due, wide_refresh());
        let us = started.elapsed().as_micros() as u64;
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        if ahead_watch {
            crate::watch::ask_ahead_done();
        }
    }
}

#[cfg(test)]
mod written_spans_tests {
    use super::{Touch, written_spans};

    #[test]
    fn what_is_written_is_refreshed_ahead_up_to_its_write_and_to_title_writes() {
        use Touch::{Nothing, Refresh, Title, Write};
        let mut found = written_spans(
            [
                Refresh(1, 10, 20),
                Refresh(2, 0, 5),
                Nothing,
                Refresh(1, 30, 40),
                Write(1),
                Refresh(1, 0, 100),
                Write(3),
                Refresh(3, 1, 2),
                Refresh(4, 7, 9),
                Write(4),
                Title,
                Write(2),
            ]
            .into_iter(),
        );
        found.sort_unstable();
        assert_eq!(found, [(1, 10, 40), (4, 7, 9)]);
    }
}
