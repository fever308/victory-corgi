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

    #[test]
    fn range_settle_is_default_on_and_respects_disable() {
        use super::range_settle_enabled;
        assert!(range_settle_enabled(None));
        assert!(range_settle_enabled(Some("1")));
        assert!(!range_settle_enabled(Some("0")));
        assert!(range_settle_enabled(Some("auto")));
    }

    #[test]
    fn line_overlaps_stale_distinguishes_disjoint_and_overlapping_flights() {
        use super::{ExecEpochGuard, Flight, Late, Line, line_overlaps_stale};
        use crate::shadow::Shadow;
        use std::sync::Arc;

        let mut side = vec![0u8; 4 * crate::watch::PAGE];
        let Some(shadow_a) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x1000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let Some(shadow_b) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x1000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let ptr_a = Arc::as_ptr(&shadow_a) as usize;
        let ptr_b = Arc::as_ptr(&shadow_b) as usize;

        let make_flight = |number: u64, back: Vec<super::Declared>| Flight {
            number,
            fence: number,
            lists: Vec::new(),
            order: Vec::new(),
            card: None,
            back,
            texture_back_ranges: Vec::new(),
            marks: None,
            _exec_epoch: ExecEpochGuard {
                scope: None,
                epoch: 0,
            },
        };

        let mut line = Line {
            line: vec![
                Late::Run(make_flight(
                    1,
                    vec![(
                        shadow_a.clone(),
                        vec![(0x1000_0000, 0x1001_0000)],
                        Vec::new(),
                    )],
                )),
                Late::Run(make_flight(
                    2,
                    vec![(
                        shadow_b.clone(),
                        vec![(0x1002_0000, 0x1003_0000)],
                        Vec::new(),
                    )],
                )),
            ],
            written: Vec::new(),
            lists: 2,
            target: 2,
            settling: false,
            poke: false,
            landed: Vec::new(),
            in_hand: Vec::new(),
            in_hand_back: Vec::new(),
        };

        // Disjoint ranges on the same or other shadow must not match.
        assert!(!line_overlaps_stale(&line, &[(ptr_a, 0x1001_0000, 0x1002_0000)]));
        assert!(!line_overlaps_stale(&line, &[(ptr_b, 0x1000_0000, 0x1001_0000)]));
        assert!(!line_overlaps_stale(&line, &[]));

        // Overlapping range on shadow_a matches flight 1.
        assert!(line_overlaps_stale(&line, &[(ptr_a, 0x1000_8000, 0x1001_8000)]));
        // Overlapping range on shadow_b matches flight 2.
        assert!(line_overlaps_stale(&line, &[(ptr_b, 0x1002_0000, 0x1002_1000)]));

        // When flight 1 moves into `in_hand_back` (`settling == true`), it still overlaps
        // until landing completes and clears `in_hand_back`.
        line.line.remove(0);
        line.settling = true;
        line.in_hand_back.push((ptr_a, 0x1000_0000, 0x1001_0000));
        assert!(line_overlaps_stale(&line, &[(ptr_a, 0x1000_0000, 0x1000_1000)]));
        assert!(!line_overlaps_stale(&line, &[(ptr_a, 0x1001_0000, 0x1002_0000)]));

        line.in_hand_back.clear();
        line.settling = false;
        assert!(!line_overlaps_stale(&line, &[(ptr_a, 0x1000_0000, 0x1000_1000)]));
        assert!(line_overlaps_stale(&line, &[(ptr_b, 0x1002_0000, 0x1003_0000)]));
    }

    #[test]
    fn range_scoped_wait_unblocks_without_draining_unrelated_flights() {
        use super::{
            ExecEpochGuard, Flight, Late, Line, WHY_STALE, line_overlaps_stale, wait_on_line,
        };
        use crate::shadow::Shadow;
        use std::sync::{Arc, Condvar, Mutex};

        let mut side = vec![0u8; 4 * crate::watch::PAGE];
        let Some(shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            0x2000_0000,
            (4 * crate::watch::PAGE) as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return;
        };
        let s_ptr = Arc::as_ptr(&shadow) as usize;
        let stale_hit = [(s_ptr, 0x2000_0000usize, 0x2001_0000usize)];
        let stale_disjoint = [(s_ptr, 0x2003_0000usize, 0x2004_0000usize)];

        let make_flight = |number: u64, range: (usize, usize)| Flight {
            number,
            fence: number,
            lists: Vec::new(),
            order: Vec::new(),
            card: None,
            back: vec![(shadow.clone(), vec![range], Vec::new())],
            texture_back_ranges: Vec::new(),
            marks: None,
            _exec_epoch: ExecEpochGuard {
                scope: None,
                epoch: 0,
            },
        };

        let line = Arc::new(Mutex::new(Line {
            line: vec![
                Late::Run(make_flight(1, (0x2000_0000, 0x2001_0000))),
                Late::Run(make_flight(2, (0x2002_0000, 0x2003_0000))),
            ],
            written: vec![(0x2000_0000, 0x2001_0000), (0x2002_0000, 0x2003_0000)],
            lists: 2,
            target: 2,
            settling: false,
            poke: false,
            landed: Vec::new(),
            in_hand: Vec::new(),
            in_hand_back: Vec::new(),
        }));
        let cv = Arc::new(Condvar::new());

        // Disjoint stale range does not block at all even with both flights in flight.
        assert_eq!(
            wait_on_line(&line, &cv, WHY_STALE, false, |l| !line_overlaps_stale(l, &stale_disjoint)),
            None
        );

        // Overlapping stale range blocks until flight 1 lands, even while flight 2 stays in line.
        let stage = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let line_w = line.clone();
        let cv_w = cv.clone();
        let stage_w = stage.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            {
                let mut l = line_w.lock().unwrap();
                l.line.remove(0);
                l.settling = true;
                l.in_hand_back.push((s_ptr, 0x2000_0000, 0x2001_0000));
                stage_w.store(1, std::sync::atomic::Ordering::Release);
            }
            cv_w.notify_all();
            std::thread::sleep(std::time::Duration::from_millis(20));
            {
                let mut l = line_w.lock().unwrap();
                l.in_hand_back.clear();
                l.settling = false;
                l.written = vec![(0x2002_0000, 0x2003_0000)];
                stage_w.store(2, std::sync::atomic::Ordering::Release);
            }
            cv_w.notify_all();
        });

        let waited = wait_on_line(&line, &cv, WHY_STALE, false, |l| {
            !line_overlaps_stale(l, &stale_hit)
        });
        assert!(waited.is_some());
        assert_eq!(stage.load(std::sync::atomic::Ordering::Acquire), 2);
        assert_eq!(line.lock().unwrap().line.len(), 1, "unrelated flight 2 remains in LINE");
        handle.join().unwrap();
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
    fn retire_executing_pins_only_active_execute_epochs_without_bumping_resource_seq() {
        let _retire = isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let open = host.new_list().expect("open list");
        open.mark_work();

        let seq_before = RESOURCE_SEQ.load(Ordering::SeqCst);
        let (c_open, d_open) = DropCounter::new();
        retire_executing(c_open);
        assert_eq!(
            RESOURCE_SEQ.load(Ordering::SeqCst),
            seq_before,
            "retire_executing must not bump global RESOURCE_SEQ"
        );
        assert_eq!(
            d_open.load(Ordering::SeqCst),
            1,
            "newly opened list with finite work_seq must not pin retire_executing objects"
        );

        let scope = current_retire_scope();
        let epoch = acquire_epoch_in(scope.as_deref().unwrap_or(&RETIRE_STATE), exec_work_seq());
        let exec_guard = ExecEpochGuard { scope, epoch };
        let (c_exec, d_exec) = DropCounter::new();
        retire_executing(c_exec);
        assert_eq!(RESOURCE_SEQ.load(Ordering::SeqCst), seq_before);
        assert_eq!(
            d_exec.load(Ordering::SeqCst),
            0,
            "active Host::execute epoch (work_seq == u64::MAX) must pin retire_executing objects"
        );
        drop(exec_guard);
        assert_eq!(
            d_exec.load(Ordering::SeqCst),
            1,
            "finishing Host::execute epoch must drop pinned object"
        );

        drop(open);
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
        parse_rebar_mb_override, rebar_hash_pipeline, rebar_loose_policy, rebar_policy,
        resolve_rebar_budgets, spans_to_piece_runs, transition,
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
        assert_eq!(rebar_loose_policy(false, false, Some(32 * GIB), true), 0);
        assert_eq!(rebar_loose_policy(true, true, Some(64 * GIB), true), 0);
        assert_eq!(rebar_loose_policy(true, false, Some(32 * GIB), false), 0);
        assert_eq!(rebar_loose_policy(true, false, None, true), 0);

        // Low-VRAM discrete GPUs (<= 8 GiB) stay on system-RAM shadows.
        assert_eq!(rebar_policy(true, false, Some(4 * GIB), true), (false, 0));
        assert_eq!(rebar_policy(true, false, Some(8 * GIB), true), (false, 0));
        assert_eq!(rebar_loose_policy(true, false, Some(8 * GIB), true), 0);

        // 12–14 GiB tier: conservative 512 MiB buffer budget, 0 loose budget.
        assert_eq!(
            rebar_policy(true, false, Some(12 * GIB), true),
            (true, 512 * MIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(14 * GIB), true),
            (true, 512 * MIB)
        );
        assert_eq!(rebar_loose_policy(true, false, Some(12 * GIB), true), 0);
        assert_eq!(rebar_loose_policy(true, false, Some(14 * GIB), true), 0);

        // 16 GiB tier (< 20 GiB): 1 GiB buffer budget, 2 GiB loose budget.
        assert_eq!(rebar_policy(true, false, Some(16 * GIB), true), (true, GIB));
        assert_eq!(rebar_loose_policy(true, false, Some(16 * GIB), true), 2 * GIB);

        // 20–24 GiB tier (< 28 GiB): 2 GiB buffer budget, 4 GiB loose budget.
        assert_eq!(
            rebar_policy(true, false, Some(20 * GIB), true),
            (true, 2 * GIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(24 * GIB), true),
            (true, 2 * GIB)
        );
        assert_eq!(rebar_loose_policy(true, false, Some(20 * GIB), true), 4 * GIB);
        assert_eq!(rebar_loose_policy(true, false, Some(24 * GIB), true), 4 * GIB);

        // 28–32+ GiB tier (>= 28 GiB, e.g. RTX 5090): 3 GiB buffer, 6 GiB loose.
        assert_eq!(
            rebar_policy(true, false, Some(28 * GIB), true),
            (true, 3 * GIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(32 * GIB), true),
            (true, 3 * GIB)
        );
        assert_eq!(
            rebar_policy(true, false, Some(48 * GIB), true),
            (true, 3 * GIB)
        );
        assert_eq!(rebar_loose_policy(true, false, Some(28 * GIB), true), 6 * GIB);
        assert_eq!(rebar_loose_policy(true, false, Some(32 * GIB), true), 6 * GIB);
        assert_eq!(rebar_loose_policy(true, false, Some(48 * GIB), true), 6 * GIB);

        // Pure env override parsing & policy composition (`VICTORY_REBAR_MB` / `VICTORY_REBAR_LOOSE_MB`).
        assert_eq!(parse_rebar_mb_override(None), None);
        assert_eq!(parse_rebar_mb_override(Some("  ")), None);
        assert_eq!(parse_rebar_mb_override(Some("abc")), None);
        assert_eq!(parse_rebar_mb_override(Some("0")), Some(0));
        assert_eq!(parse_rebar_mb_override(Some(" 1536 ")), Some(1536 * MIB));

        assert_eq!(
            resolve_rebar_budgets(true, false, Some(32 * GIB), true, Some("2048"), Some("0")),
            (true, 2 * GIB, 0)
        );
        assert_eq!(
            resolve_rebar_budgets(true, false, Some(16 * GIB), true, Some("0"), Some("1024")),
            (true, 0, GIB)
        );
        // `VICTORY_REBAR=0`, UMA, or unsupported GPU upload disables all ReBAR despite overrides.
        assert_eq!(
            resolve_rebar_budgets(true, false, Some(32 * GIB), false, Some("2048"), Some("4096")),
            (false, 0, 0)
        );
        assert_eq!(
            resolve_rebar_budgets(true, true, Some(64 * GIB), true, Some("2048"), Some("4096")),
            (false, 0, 0)
        );
        assert_eq!(
            resolve_rebar_budgets(false, false, Some(32 * GIB), true, Some("2048"), Some("4096")),
            (false, 0, 0)
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

    #[test]
    fn is_fast_copy_rip_matches_only_try_copy_bounds() {
        let (avx_s, avx_e, avx_fail, sse_s, sse_e, sse_fail) = (
            &raw const super::victory_copy_try_avx2_start as u64,
            &raw const super::victory_copy_try_avx2_end as u64,
            &raw const super::victory_copy_try_avx2_fail as u64,
            &raw const super::victory_copy_try_sse2_start as u64,
            &raw const super::victory_copy_try_sse2_end as u64,
            &raw const super::victory_copy_try_sse2_fail as u64,
        );
        assert!(avx_e > avx_s);
        assert!(sse_e > sse_s);
        assert!(super::is_fast_copy_rip(avx_s));
        assert!(super::is_fast_copy_rip(avx_e - 1));
        assert!(!super::is_fast_copy_rip(avx_e));
        assert!(!super::is_fast_copy_rip(avx_fail));
        assert!(super::is_fast_copy_rip(sse_s));
        assert!(super::is_fast_copy_rip(sse_e - 1));
        assert!(!super::is_fast_copy_rip(sse_e));
        assert!(!super::is_fast_copy_rip(sse_fail));
        assert!(!super::is_fast_copy_rip(0));
        assert!(!super::is_fast_copy_rip(
            is_fast_copy_rip_matches_only_try_copy_bounds as *const () as usize as u64
        ));
    }

    #[test]
    fn sync_copy_queue_orders_copy_behind_direct_queue_signal() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::Interface;

        let Some(h) = host() else {
            return;
        };
        let errors_before = h.d3d12_errors();

        let (src_up, src_ptr) = h.upload_buffer(256).expect("upload buffer");
        let dst_default = h
            .default_buffer_in_state(256, D3D12_RESOURCE_STATE_COMMON)
            .expect("default buffer");
        let (rb_buf, rb_ptr) = h.readback_buffer(256).expect("readback buffer");
        // SAFETY: mapped 256-byte upload buffer.
        unsafe { std::ptr::write_bytes(src_ptr, 0x6D, 256) };

        // Record and execute a copy on the DIRECT queue via `Host::execute` (which signals `last_signalled`).
        let direct_list = h.new_list().expect("direct list");
        // SAFETY: live buffers on open direct command list.
        unsafe {
            direct_list.list.ResourceBarrier(&[transition(
                &dst_default,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            direct_list
                .list
                .CopyBufferRegion(&dst_default, 0, &src_up, 0, 256);
            direct_list.list.ResourceBarrier(&[transition(
                &dst_default,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        direct_list.close();
        h.execute(&[&direct_list]);
        direct_list.reset();
        assert!(h.last_signalled.lock().unwrap().0 > 0);

        // Create a private COPY queue, call `h.sync_copy_queue`, and read back `dst_default`.
        // SAFETY: plain D3D12 COPY queue, allocator, list, fence, and Win32 event.
        unsafe {
            let copy_queue: ID3D12CommandQueue = h
                .device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })
                .expect("copy queue");
            let copy_alloc: ID3D12CommandAllocator = h
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)
                .expect("copy allocator");
            let copy_list: ID3D12GraphicsCommandList = h
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &copy_alloc, None)
                .expect("copy list");
            copy_list.CopyBufferRegion(&rb_buf, 0, &dst_default, 0, 256);
            copy_list.Close().expect("close copy list");

            h.sync_copy_queue(&copy_queue).expect("sync_copy_queue");
            let cl: ID3D12CommandList = copy_list.cast().expect("cast list");
            copy_queue.ExecuteCommandLists(&[Some(cl)]);
            let copy_fence: ID3D12Fence = h
                .device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("copy fence");
            let event = CreateEventW(None, false, false, None).expect("event");
            copy_queue.Signal(&copy_fence, 1).expect("signal copy fence");
            copy_fence
                .SetEventOnCompletion(1, event)
                .expect("set event");
            WaitForSingleObject(event, INFINITE);
            let _ = windows::Win32::Foundation::CloseHandle(event);

            let got = std::slice::from_raw_parts(rb_ptr, 256);
            assert_eq!(got, &[0x6Du8; 256]);
        }

        h.log_messages();
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during sync_copy_queue test"
        );
    }

    #[test]
    fn ask_once_void_forces_fresh_get_write_watch_after_settle_stale() {
        use std::sync::atomic::Ordering;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        struct CleanupGuard {
            base: usize,
        }
        impl Drop for CleanupGuard {
            fn drop(&mut self) {
                crate::watch::CURRENT_STAMP.store(0, Ordering::Relaxed);
                crate::watch::ask_once_void();
                // SAFETY: `self.base` is a private allocation owned by this test.
                unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
            }
        }

        // SAFETY: private write-watched page simulating title RAM.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        let _cleanup = CleanupGuard { base };

        // Baseline reset so kernel write-watch bit starts clean.
        assert!(crate::watch::writes(base, base + crate::watch::PAGE, false).is_some());

        // Mint a title submission stamp and set `CURRENT_STAMP` to it so after
        // the first `ask_once`, `CURRENT_STAMP <= ASK_STAMP` holds.
        let stamp = crate::watch::TITLE_SUBMITS.fetch_add(1, Ordering::AcqRel) + 1;
        crate::watch::CURRENT_STAMP.store(stamp, Ordering::Relaxed);

        let ranges = [(base, base + crate::watch::PAGE)];
        crate::watch::ask_once(&ranges);
        let (epoch1, _) = crate::watch::asked(base, base + crate::watch::PAGE)
            .expect("first ask_once covers private range");

        // Simulate `settle_stale` landing GPU writes directly into private
        // write-watched title memory without updating `DIR`.
        // SAFETY: committed private `PAGE_READWRITE` memory owned by this test.
        unsafe { (base as *mut u8).write_volatile(0x5A) };

        // Voiding the standing asking clears `ASK_STAMP` and `AHEAD`.
        crate::watch::ask_once_void();
        assert!(
            crate::watch::asked(base, base + crate::watch::PAGE).is_none(),
            "ask_once_void must clear standing ask_once state"
        );

        // Even with `CURRENT_STAMP` still at `stamp` (`<=` the prior `ASK_STAMP`),
        // re-asking must perform a fresh kernel `GetWriteWatch` query and observe
        // the newly dirtied page.
        crate::watch::ask_once(&ranges);
        let (epoch2, written) = crate::watch::asked(base, base + crate::watch::PAGE)
            .expect("second ask_once re-queries private range");
        assert!(epoch2 > epoch1);
        assert!(
            written
                .iter()
                .any(|&(p, w)| p == base / crate::watch::PAGE && w == epoch2),
            "re-ask after ask_once_void must observe newly dirtied write-watch page"
        );
    }
}

