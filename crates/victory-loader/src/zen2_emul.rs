//! Clean-room emulator for AMD Zen 2 console-era x86-64 instructions that
//! fault with `STATUS_ILLEGAL_INSTRUCTION` on host CPUs lacking those ISA
//! extensions.
//!
//! Designed independently from the AMD64 Architecture Programmer's Manual
//! (Vol. 3 & Vol. 4), the Intel 64 and IA-32 Architectures Software
//! Developer's Manual (Vol. 2), and the crate's black-box behavioral
//! specification.
//!
//! # Supported Instructions
//! - `UD2` (`0F 0B`) — explicitly recognized as a trap, never emulated.
//! - `CLWB` (`66 0F AE /6` mem) — patched to `NOP` in place.
//! - `MONITORX` (`0F 01 FA`) — architectural no-op.
//! - `MWAITX` (`0F 01 FB`) — CPU spin-loop pause hint.
//! - `CLZERO` (`0F 01 FC`) — 64-byte cache-line zeroing, with 12-byte and
//!   15-byte bulk loop accelerators.
//! - `RDPRU` (`0F 01 FD`) — host `QueryPerformanceCounter` for `ECX` 0/1,
//!   zero otherwise.
//! - `RDPID` (`F3 0F C7 /7` reg) — host logical processor ID via
//!   `GetCurrentProcessorNumber`.
//! - SSE4a `EXTRQ` (`66 0F 78 /0 ib ib`, `66 0F 79 /r`) and `INSERTQ`
//!   (`F2 0F 78 /r ib ib`, `F2 0F 79 /r`).

use std::cell::RefCell;
use std::mem::size_of;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, M128A};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS,
    PAGE_PROTECTION_FLAGS, PAGE_READWRITE, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Performance::QueryPerformanceCounter;
use windows_sys::Win32::System::Threading::GetCurrentProcessorNumber;

/// Return value telling Windows VEH to resume execution at the updated `Rip`.
const CONTINUE_EXECUTION: i32 = -1;
/// Return value telling Windows VEH this handler did not handle the fault.
const CONTINUE_SEARCH: i32 = 0;

/// Architectural x86-64 maximum single-instruction encoding length
/// (AMD64 APM Vol. 3 §1.1; Intel SDM Vol. 2A §2.3.11).
const MAX_INSN_BYTES: usize = 15;

/// Code-fetch window size (16 bytes covers any 15-byte instruction plus
/// adjacent loop tail lookahead).
const FETCH_WINDOW: usize = 16;

/// Upper bound on bulk zeroing range for `CLZERO` loop accelerators (2 GiB).
const MAX_BULK_ZERO_BYTES: usize = 0x8000_0000;

/// Cache line size on AMD Zen 2 x86-64 (bytes).
const CACHE_LINE_BYTES: usize = 64;

/// Number of entries in the per-thread code-readability MRU cache.
const CODE_CACHE_SLOTS: usize = 8;

/// Number of entries in the per-thread writable-region MRU cache.
const WRITE_CACHE_SLOTS: usize = 4;

/// Number of writable-cache hits before evicting cached regions so memory
/// protection changes cannot stay stale indefinitely.
const WRITE_CACHE_EPOCH: u32 = 4096;

/// `CLZERO` 15-byte loop lookbehind distance (`mov rax, ptr; add ptr, 0x40`).
const LOOP15_HEAD_BYTES: usize = 7;

// ---------------------------------------------------------------------------
// Configuration & telemetry
// ---------------------------------------------------------------------------

/// Returns `true` when `VICTORY_CLZERO_ASK=1` disables per-thread
/// `VirtualQuery` caching.
fn bypass_mem_cache() -> bool {
    static BYPASS: OnceLock<bool> = OnceLock::new();
    *BYPASS.get_or_init(|| std::env::var_os("VICTORY_CLZERO_ASK").is_some_and(|v| v == "1"))
}

/// Returns `true` when the 15-byte `CLZERO` loop accelerator is enabled
/// (default on; disabled only by `VICTORY_CLZERO_LOOP15=0`).
fn allow_loop15() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_CLZERO_LOOP15").is_none_or(|v| v != "0"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum EmulKind {
    Clwb = 0,
    Monitorx = 1,
    Mwaitx = 2,
    Clzero = 3,
    ClzeroLoop12 = 4,
    ClzeroLoop15 = 5,
    Rdpru = 6,
    Rdpid = 7,
    Extrq = 8,
    Insertq = 9,
}

impl EmulKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Clwb => "CLWB",
            Self::Monitorx => "MONITORX",
            Self::Mwaitx => "MWAITX",
            Self::Clzero => "CLZERO",
            Self::ClzeroLoop12 => "CLZERO (12-byte loop)",
            Self::ClzeroLoop15 => "CLZERO (15-byte loop)",
            Self::Rdpru => "RDPRU",
            Self::Rdpid => "RDPID",
            Self::Extrq => "EXTRQ",
            Self::Insertq => "INSERTQ",
        }
    }
}

fn record_emulation(kind: EmulKind, rip: usize) {
    static SEEN_KINDS: AtomicU32 = AtomicU32::new(0);
    static TOTAL_EMULATED: AtomicU64 = AtomicU64::new(0);

    let bit = 1u32 << (kind as u32);
    if (SEEN_KINDS.fetch_or(bit, Ordering::Relaxed) & bit) == 0 {
        tracing::warn!(
            insn = kind.label(),
            rip = format_args!("{rip:#018x}"),
            "emulating Zen 2 console instruction"
        );
    }

    let count = TOTAL_EMULATED.fetch_add(1, Ordering::Relaxed) + 1;
    if count >= 65_536 && count.is_power_of_two() {
        tracing::info!(
            count,
            last_insn = kind.label(),
            "console instruction emulation count milestone"
        );
    }
}

// ---------------------------------------------------------------------------
// Windows memory validation & per-thread MRU caches
// ---------------------------------------------------------------------------

struct MemCaches {
    code: [(usize, u8); CODE_CACHE_SLOTS],
    writable: [(usize, usize); WRITE_CACHE_SLOTS],
    write_hits: u32,
}

impl MemCaches {
    const fn new() -> Self {
        Self {
            code: [(0, 0); CODE_CACHE_SLOTS],
            writable: [(0, 0); WRITE_CACHE_SLOTS],
            write_hits: 0,
        }
    }
}

thread_local! {
    static MEM_CACHES: RefCell<MemCaches> = const { RefCell::new(MemCaches::new()) };
}

fn query_region(addr: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let mut mbi = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: `VirtualQuery` reads no memory at `addr` and only writes into `mbi`.
    let ret = unsafe {
        VirtualQuery(
            addr as *const core::ffi::c_void,
            &mut mbi,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if ret == 0 { None } else { Some(mbi) }
}

fn region_bounds(mbi: &MEMORY_BASIC_INFORMATION) -> (usize, usize) {
    let start = mbi.BaseAddress as usize;
    let end = start.saturating_add(mbi.RegionSize);
    (start, end)
}

fn is_readable_region(mbi: &MEMORY_BASIC_INFORMATION) -> bool {
    mbi.State == MEM_COMMIT && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0
}

/// Note: `PAGE_GUARD` is intentionally NOT masked out here so that writing to
/// a `PAGE_READWRITE | PAGE_GUARD` page triggers Windows's normal guard-page
/// notification instead of turning the fault into an unhandled `#UD`.
fn is_writable_region(mbi: &MEMORY_BASIC_INFORMATION) -> bool {
    mbi.State == MEM_COMMIT && (mbi.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE)) != 0
}

/// Determines how many bytes (in `0..=FETCH_WINDOW`) are readable starting at
/// `rip` within the single committed memory region containing `rip`.
fn readable_code_len(rip: usize) -> usize {
    if !bypass_mem_cache() {
        let cached = MEM_CACHES.with(|c| {
            let caches = c.borrow();
            caches
                .code
                .iter()
                .find_map(|&(addr, len)| (addr == rip && len > 0).then_some(len as usize))
        });
        if let Some(len) = cached {
            return len;
        }
    }

    let Some(mbi) = query_region(rip) else {
        return 0;
    };
    if !is_readable_region(&mbi) {
        return 0;
    }
    let (start, end) = region_bounds(&mbi);
    if rip < start || rip >= end {
        return 0;
    }

    let len = (end - rip).min(FETCH_WINDOW);
    if !bypass_mem_cache() {
        MEM_CACHES.with(|c| {
            let mut caches = c.borrow_mut();
            caches.code.rotate_right(1);
            caches.code[0] = (rip, len as u8);
        });
    }
    len
}

/// Copies up to `FETCH_WINDOW` readable code bytes starting at `rip` into `buf`
/// and returns the initialized prefix slice.
fn fetch_code(rip: usize, buf: &mut [u8; FETCH_WINDOW]) -> &[u8] {
    let n = readable_code_len(rip);
    if n == 0 {
        return &[];
    }
    // SAFETY: `readable_code_len` verified `[rip, rip + n)` is committed and
    // readable (neither `PAGE_NOACCESS` nor `PAGE_GUARD`).
    unsafe {
        ptr::copy_nonoverlapping(rip as *const u8, buf.as_mut_ptr(), n);
    }
    &buf[..n]
}

/// Checks whether `[start, start + len)` is readable within a single committed
/// memory region (used when lookbehind crosses backward across a 4 KiB page).
fn span_readable(start: usize, len: usize) -> bool {
    let Some(end) = start.checked_add(len) else {
        return false;
    };
    let Some(mbi) = query_region(start) else {
        return false;
    };
    if !is_readable_region(&mbi) {
        return false;
    }
    let (r_start, r_end) = region_bounds(&mbi);
    r_start <= start && end <= r_end
}

/// Returns the exclusive upper bound (clamped to `end`) of the committed,
/// writable region containing `va`, or `None` if `va` is not writable.
fn writable_until(va: usize, end: usize) -> Option<usize> {
    if va >= end {
        return None;
    }

    let use_cache = !bypass_mem_cache();
    if use_cache {
        let hit = MEM_CACHES.with(|c| {
            let mut caches = c.borrow_mut();
            let found = caches
                .writable
                .iter()
                .find_map(|&(r_start, r_end)| (r_start <= va && va < r_end).then_some(r_end));
            if let Some(r_end) = found {
                if caches.write_hits >= WRITE_CACHE_EPOCH {
                    caches.writable = [(0, 0); WRITE_CACHE_SLOTS];
                    caches.write_hits = 0;
                } else {
                    caches.write_hits += 1;
                }
                Some(r_end.min(end))
            } else {
                None
            }
        });
        if hit.is_some() {
            return hit;
        }
    }

    let mbi = query_region(va)?;
    if !is_writable_region(&mbi) {
        return None;
    }
    let (r_start, r_end) = region_bounds(&mbi);
    if va < r_start || va >= r_end {
        return None;
    }

    if use_cache {
        MEM_CACHES.with(|c| {
            let mut caches = c.borrow_mut();
            caches.writable.rotate_right(1);
            caches.writable[0] = (r_start, r_end);
        });
    }
    Some(r_end.min(end))
}

/// Zeroes contiguous writable memory starting at `start` up to `end`, stopping
/// at the first unwritable region boundary. Returns the first unwritten address
/// (equal to `end` on full completion, or `start` if the initial address is
/// unwritable).
fn zero_writable_range(start: usize, end: usize) -> usize {
    let mut cursor = start;
    while cursor < end {
        let Some(stop) = writable_until(cursor, end) else {
            break;
        };
        if stop <= cursor {
            break;
        }
        // SAFETY: `writable_until` verified `[cursor, stop)` is committed and
        // writable (`PAGE_READWRITE` or `PAGE_EXECUTE_READWRITE`).
        unsafe {
            ptr::write_bytes(cursor as *mut u8, 0, stop - cursor);
        }
        cursor = stop;
    }
    cursor
}

/// Zeroes the single 64-byte cache line containing `addr`. Returns `false` if
/// the target cache line is not writable (leaving memory untouched).
fn zero_single_cache_line(addr: usize) -> bool {
    let line = addr & !(CACHE_LINE_BYTES - 1);
    let Some(end) = line.checked_add(CACHE_LINE_BYTES) else {
        return false;
    };
    if writable_until(line, end) != Some(end) {
        return false;
    }
    // SAFETY: `writable_until` verified `[line, line + 64)` is committed and
    // writable.
    unsafe {
        ptr::write_bytes(line as *mut u8, 0, CACHE_LINE_BYTES);
    }
    true
}

/// Replaces `[rip, rip + len)` with x86 `NOP` (`0x90`) bytes if the page
/// protection can temporarily be elevated to `PAGE_EXECUTE_READWRITE`.
fn patch_with_nops(rip: usize, len: usize) {
    let mut old_protect: PAGE_PROTECTION_FLAGS = 0;
    // SAFETY: `rip` points to the decoded instruction of `len` readable bytes.
    // We only write after `VirtualProtect` grants write permission, and
    // immediately restore the original protection flags.
    unsafe {
        let ok = VirtualProtect(
            rip as *const core::ffi::c_void,
            len,
            PAGE_EXECUTE_READWRITE,
            &mut old_protect,
        );
        if ok != 0 {
            ptr::write_bytes(rip as *mut u8, 0x90, len);
            let mut unused: PAGE_PROTECTION_FLAGS = 0;
            let _ = VirtualProtect(
                rip as *const core::ffi::c_void,
                len,
                old_protect,
                &mut unused,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// CONTEXT register accessors (AMD64 standard 4-bit encoding order)
// ---------------------------------------------------------------------------

fn read_gpr(ctx: &CONTEXT, reg: u8) -> u64 {
    match reg & 0x0F {
        0 => ctx.Rax,
        1 => ctx.Rcx,
        2 => ctx.Rdx,
        3 => ctx.Rbx,
        4 => ctx.Rsp,
        5 => ctx.Rbp,
        6 => ctx.Rsi,
        7 => ctx.Rdi,
        8 => ctx.R8,
        9 => ctx.R9,
        10 => ctx.R10,
        11 => ctx.R11,
        12 => ctx.R12,
        13 => ctx.R13,
        14 => ctx.R14,
        _ => ctx.R15,
    }
}

fn write_gpr(ctx: &mut CONTEXT, reg: u8, value: u64) {
    match reg & 0x0F {
        0 => ctx.Rax = value,
        1 => ctx.Rcx = value,
        2 => ctx.Rdx = value,
        3 => ctx.Rbx = value,
        4 => ctx.Rsp = value,
        5 => ctx.Rbp = value,
        6 => ctx.Rsi = value,
        7 => ctx.Rdi = value,
        8 => ctx.R8 = value,
        9 => ctx.R9 = value,
        10 => ctx.R10 = value,
        11 => ctx.R11 = value,
        12 => ctx.R12 = value,
        13 => ctx.R13 = value,
        14 => ctx.R14 = value,
        _ => ctx.R15 = value,
    }
}

fn read_xmm(ctx: &CONTEXT, reg: u8) -> M128A {
    // SAFETY: On x86-64 Windows `CONTEXT`, `Anonymous.FltSave` is the active
    // legacy/XMM save view containing `XmmRegisters[0..16]`.
    unsafe { ctx.Anonymous.FltSave.XmmRegisters[(reg & 0x0F) as usize] }
}

fn write_xmm(ctx: &mut CONTEXT, reg: u8, low: u64, high: i64) {
    // SAFETY: Same active `FltSave` union view as `read_xmm`.
    unsafe {
        ctx.Anonymous.FltSave.XmmRegisters[(reg & 0x0F) as usize] = M128A {
            Low: low,
            High: high,
        };
    }
}

// ---------------------------------------------------------------------------
// Targeted x86-64 prefix & instruction decoder
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Prefixes {
    count: usize,
    rex: u8,
    saw_66: bool,
    saw_67: bool,
    last_rep: u8, // 0, 0xF2, or 0xF3
}

/// Scans legacy prefixes (excluding `0xF0` LOCK) and REX prefixes according to
/// AMD64 APM Vol. 3 §1.2:
/// - Any legacy prefix following a REX byte nullifies the earlier REX byte.
/// - Consecutive REX bytes immediately before the opcode use the last REX.
/// - `last_rep` tracks the last Group-1 repeat prefix (`0xF2` or `0xF3`).
fn scan_prefixes(bytes: &[u8]) -> Prefixes {
    let mut p = Prefixes {
        count: 0,
        rex: 0,
        saw_66: false,
        saw_67: false,
        last_rep: 0,
    };

    for &b in bytes {
        match b {
            0x66 => {
                p.saw_66 = true;
                p.rex = 0;
            }
            0x67 => {
                p.saw_67 = true;
                p.rex = 0;
            }
            0xF2 | 0xF3 => {
                p.last_rep = b;
                p.rex = 0;
            }
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 => {
                p.rex = 0;
            }
            0x40..=0x4F => {
                p.rex = b;
            }
            _ => break,
        }
        p.count += 1;
    }

    p
}

/// Computes the total byte length of a ModRM memory operand (ModRM + optional
/// SIB + optional disp8/disp32) per AMD64 APM Vol. 3 §1.4 / Intel SDM Vol. 2A
/// Table 2-2. Returns `None` if `mod == 3` (register operand) or if `tail` is
/// truncated.
fn decode_modrm_mem_len(tail: &[u8]) -> Option<usize> {
    let modrm = *tail.first()?;
    let mode = modrm >> 6;
    let rm = modrm & 7;

    let need = match mode {
        3 => return None,
        0 => match rm {
            4 => {
                let sib = *tail.get(1)?;
                if (sib & 7) == 5 { 6 } else { 2 }
            }
            5 => 5,
            _ => 1,
        },
        1 => {
            if rm == 4 {
                3
            } else {
                2
            }
        }
        _ => {
            if rm == 4 {
                6
            } else {
                5
            }
        }
    };

    (tail.len() >= need).then_some(need)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodedInsn {
    Ud2,
    Clwb {
        len: usize,
    },
    Monitorx {
        len: usize,
    },
    Mwaitx {
        len: usize,
    },
    Clzero {
        len: usize,
        prefix_len: usize,
        addr32: bool,
    },
    Rdpru {
        len: usize,
    },
    Rdpid {
        len: usize,
        dst: u8,
    },
    ExtrqImm {
        len: usize,
        dst: u8,
        raw_len: u8,
        raw_idx: u8,
    },
    ExtrqReg {
        len: usize,
        dst: u8,
        src: u8,
    },
    InsertqImm {
        len: usize,
        dst: u8,
        src: u8,
        raw_len: u8,
        raw_idx: u8,
    },
    InsertqReg {
        len: usize,
        dst: u8,
        src: u8,
    },
}

/// Decodes one supported instruction from `code`.
///
/// Strictly enforces:
/// - Architectural 15-byte maximum x86-64 instruction length
///   (AMD64 APM Vol. 3 §1.1; Intel SDM Vol. 2A §2.3.11).
/// - Mandatory prefix precedence (`F2`/`F3` overrides `66`, last of `F2`/`F3`
///   wins) for `CLWB`, `RDPID`, `EXTRQ`, and `INSERTQ`
///   (Intel SDM Vol. 2A §2.1.1 & Table A-6; AMD64 APM Vol. 3 §1.2.1).
fn decode_insn(code: &[u8]) -> Option<DecodedInsn> {
    let p = scan_prefixes(code);
    let i = p.count;

    // Every supported instruction begins with the two-byte escape 0x0F.
    if *code.get(i)? != 0x0F {
        return None;
    }
    let op2 = *code.get(i + 1)?;
    let rex_r = (p.rex & 0x04) << 1; // REX.R -> bit 3
    let rex_b = (p.rex & 0x01) << 3; // REX.B -> bit 3

    let check_len = |len: usize| -> Option<usize> {
        (len <= MAX_INSN_BYTES && len <= code.len()).then_some(len)
    };

    match op2 {
        // UD2: [prefixes] 0F 0B
        0x0B => {
            check_len(i + 2)?;
            Some(DecodedInsn::Ud2)
        }

        // Three-byte 0F 01 xx opcodes: MONITORX, MWAITX, CLZERO, RDPRU
        0x01 => {
            let op3 = *code.get(i + 2)?;
            let len = check_len(i + 3)?;
            match op3 {
                0xFA => Some(DecodedInsn::Monitorx { len }),
                0xFB => Some(DecodedInsn::Mwaitx { len }),
                0xFC => Some(DecodedInsn::Clzero {
                    len,
                    prefix_len: i,
                    addr32: p.saw_67,
                }),
                0xFD => Some(DecodedInsn::Rdpru { len }),
                _ => None,
            }
        }

        // CLWB: 66 [REX] 0F AE /6 (memory operand only; F2/F3 overrides 66 -> #UD)
        0xAE => {
            if !p.saw_66 || p.last_rep != 0 {
                return None;
            }
            let tail = code.get(i + 2..)?;
            let modrm = *tail.first()?;
            if ((modrm >> 3) & 7) != 6 {
                return None;
            }
            let mem_len = decode_modrm_mem_len(tail)?;
            let len = check_len(i + 2 + mem_len)?;
            Some(DecodedInsn::Clwb { len })
        }

        // RDPID: F3 [REX] 0F C7 /7 (register operand only; last repeat prefix must be F3)
        0xC7 => {
            if p.last_rep != 0xF3 {
                return None;
            }
            let modrm = *code.get(i + 2)?;
            if (modrm >> 6) != 3 || ((modrm >> 3) & 7) != 7 {
                return None;
            }
            let len = check_len(i + 3)?;
            let dst = (modrm & 7) | rex_b;
            Some(DecodedInsn::Rdpid { len, dst })
        }

        // SSE4a immediate forms:
        // - INSERTQ: F2 [REX] 0F 78 /r imm8_len imm8_idx (last_rep == 0xF2)
        // - EXTRQ:   66 [REX] 0F 78 /0 imm8_len imm8_idx (last_rep == 0 && saw_66)
        0x78 => {
            let modrm = *code.get(i + 2)?;
            if (modrm >> 6) != 3 {
                return None;
            }
            let raw_len = *code.get(i + 3)?;
            let raw_idx = *code.get(i + 4)?;
            let len = check_len(i + 5)?;

            if p.last_rep == 0xF2 {
                let dst = ((modrm >> 3) & 7) | rex_r;
                let src = (modrm & 7) | rex_b;
                Some(DecodedInsn::InsertqImm {
                    len,
                    dst,
                    src,
                    raw_len,
                    raw_idx,
                })
            } else if p.last_rep == 0 && p.saw_66 {
                if ((modrm >> 3) & 7) != 0 {
                    return None;
                }
                let dst = (modrm & 7) | rex_b;
                Some(DecodedInsn::ExtrqImm {
                    len,
                    dst,
                    raw_len,
                    raw_idx,
                })
            } else {
                None
            }
        }

        // SSE4a register forms:
        // - INSERTQ: F2 [REX] 0F 79 /r (last_rep == 0xF2)
        // - EXTRQ:   66 [REX] 0F 79 /r (last_rep == 0 && saw_66)
        0x79 => {
            let modrm = *code.get(i + 2)?;
            if (modrm >> 6) != 3 {
                return None;
            }
            let len = check_len(i + 3)?;
            let dst = ((modrm >> 3) & 7) | rex_r;
            let src = (modrm & 7) | rex_b;

            if p.last_rep == 0xF2 {
                Some(DecodedInsn::InsertqReg { len, dst, src })
            } else if p.last_rep == 0 && p.saw_66 {
                Some(DecodedInsn::ExtrqReg { len, dst, src })
            } else {
                None
            }
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CLZERO bulk loop accelerators
// ---------------------------------------------------------------------------

enum LoopOutcome {
    /// Code/register preconditions did not match this loop shape; fall back.
    NotMatched,
    /// Matched and emulated; return `CONTINUE_EXECUTION`.
    Handled,
    /// Matched, but initial target address is unwritable; return `CONTINUE_SEARCH`.
    Unwritable,
}

/// Recognizes and accelerates the 12-byte `CLZERO` loop starting at `ctx.Rip`:
/// ```text
/// top:
///     clzero                 ; 0F 01 FC
///     add  rax, 0x40         ; 48 83 C0 40
///     cmp  rax, <end_reg>    ; [48|4C] 39 <modrm: mod=11, rm=000>
///     jb   top               ; 72 F4  (rel8 = -12)
/// ```
///
/// Architectural safeguards (falling back to single-line `CLZERO` whenever any
/// invariant fails):
/// - Requires zero prefix bytes before `0F` (so `jb -12` genuinely targets `rip`).
/// - Validates `ModRM.mod == 3 && ModRM.rm == 0` so `cmp` truly compares `RAX`.
/// - Requires 64-byte aligned `RAX`, `end > start`, and total clear length
///   `< 2 GiB`, rounding `end` up to the next 64-byte line boundary to match
///   hardware cache-line zeroing and post-loop `RAX`.
fn try_clzero_loop12(ctx: &mut CONTEXT, rip: usize, code: &[u8]) -> LoopOutcome {
    if code.len() < 12 {
        return LoopOutcome::NotMatched;
    }
    if code[0..3] != [0x0F, 0x01, 0xFC]
        || code[3..7] != [0x48, 0x83, 0xC0, 0x40]
        || (code[7] != 0x48 && code[7] != 0x4C)
        || code[8] != 0x39
        || code[10..12] != [0x72, 0xF4]
    {
        return LoopOutcome::NotMatched;
    }

    let cmp_modrm = code[9];
    // Require register-register `cmp rax, <end_reg>` (mod == 3, rm == 0).
    if (cmp_modrm & 0xC7) != 0xC0 {
        return LoopOutcome::NotMatched;
    }

    let end_reg = ((cmp_modrm >> 3) & 7) | if code[7] == 0x4C { 8 } else { 0 };
    let start = ctx.Rax as usize;
    let end = read_gpr(ctx, end_reg) as usize;

    // Require 64-byte alignment, positive range, and bounded size (< 2 GiB).
    // Any edge case (unaligned `start`, `end <= start`, or `>= 2 GiB`) falls
    // back to single-line `CLZERO` so hardware `add; cmp; jb` executes cleanly.
    if (start & (CACHE_LINE_BYTES - 1)) != 0 || end <= start {
        return LoopOutcome::NotMatched;
    }
    let lines = (end - start).div_ceil(CACHE_LINE_BYTES);
    let Some(total_bytes) = lines.checked_mul(CACHE_LINE_BYTES) else {
        return LoopOutcome::NotMatched;
    };
    if total_bytes >= MAX_BULK_ZERO_BYTES {
        return LoopOutcome::NotMatched;
    }
    let Some(last) = start.checked_add(total_bytes) else {
        return LoopOutcome::NotMatched;
    };

    let reached = zero_writable_range(start, last);
    if reached == start {
        return LoopOutcome::Unwritable;
    }

    record_emulation(EmulKind::ClzeroLoop12, rip);
    ctx.Rax = reached as u64;
    if reached == last {
        ctx.Rip = (rip as u64).wrapping_add(12);
    }
    // On partial writability (`start < reached < last`), `Rip` stays at `clzero`
    // so the next execution retries at the unwritable boundary `reached`.
    LoopOutcome::Handled
}

/// Recognizes and accelerates the 15-byte `CLZERO` loop with `ctx.Rip` on
/// `clzero` (byte offset `+7`):
/// ```text
/// top:                          ; rip - 7
///     mov  rax, <ptr_reg>       ; [48|4C] 89 [C0 | (pl << 3)]
///     add  <ptr_reg>, 0x40      ; [48|49] 83 [C0 | pl] 40
///     clzero                    ; 0F 01 FC   <-- ctx.Rip
///     cmp  <ptr_reg>, <end_reg> ; [48|49|4C|4D] 39 <modrm>
///     jb   top                  ; 72 F1  (rel8 = -15)
/// ```
fn try_clzero_loop15(ctx: &mut CONTEXT, rip: usize, code: &[u8]) -> LoopOutcome {
    if !allow_loop15() || code.len() < 8 {
        return LoopOutcome::NotMatched;
    }

    let cmp_rex = code[3];
    let cmp_modrm = code[5];
    if code[0..3] != [0x0F, 0x01, 0xFC]
        || (cmp_rex & 0xFA) != 0x48
        || code[4] != 0x39
        || (cmp_modrm >> 6) != 3
        || code[6..8] != [0x72, 0xF1]
    {
        return LoopOutcome::NotMatched;
    }

    let end_reg = ((cmp_modrm >> 3) & 7) | ((cmp_rex & 0x04) << 1);
    let ptr_reg = (cmp_modrm & 7) | ((cmp_rex & 0x01) << 3);

    // `ptr_reg` cannot be RAX or RSP; `end_reg` cannot be RAX or `ptr_reg`.
    if ptr_reg == 0 || ptr_reg == 4 || end_reg == 0 || end_reg == ptr_reg {
        return LoopOutcome::NotMatched;
    }

    // Read the 7-byte loop prologue at `[rip - 7 .. rip)`.
    let Some(head_addr) = rip.checked_sub(LOOP15_HEAD_BYTES) else {
        return LoopOutcome::NotMatched;
    };
    if (rip & 0xFFF) < LOOP15_HEAD_BYTES && !span_readable(head_addr, LOOP15_HEAD_BYTES) {
        return LoopOutcome::NotMatched;
    }

    let mut head = [0u8; LOOP15_HEAD_BYTES];
    // SAFETY: Either `[head_addr, rip)` lies in the same committed readable
    // 4 KiB page as `rip`, or `span_readable` verified it via `VirtualQuery`.
    unsafe {
        ptr::copy_nonoverlapping(head_addr as *const u8, head.as_mut_ptr(), LOOP15_HEAD_BYTES);
    }

    let ph = (ptr_reg >> 3) & 1;
    let pl = ptr_reg & 7;
    let expected_head = [
        0x48 | (ph << 2),
        0x89,
        0xC0 | (pl << 3),
        0x48 | ph,
        0x83,
        0xC0 | pl,
        0x40,
    ];
    if head != expected_head {
        return LoopOutcome::NotMatched;
    }

    let rax = ctx.Rax as usize;
    let ptr = read_gpr(ctx, ptr_reg) as usize;
    let end = read_gpr(ctx, end_reg) as usize;

    if (rax & (CACHE_LINE_BYTES - 1)) != 0 || ptr < rax || ptr != rax.wrapping_add(CACHE_LINE_BYTES)
    {
        return LoopOutcome::NotMatched;
    }

    let more_lines = if ptr < end {
        (end - ptr).div_ceil(CACHE_LINE_BYTES)
    } else {
        0
    };
    let Some(more_bytes) = more_lines.checked_mul(CACHE_LINE_BYTES) else {
        return LoopOutcome::NotMatched;
    };
    let Some(last) = ptr.checked_add(more_bytes) else {
        return LoopOutcome::NotMatched;
    };
    if last - rax >= MAX_BULK_ZERO_BYTES {
        return LoopOutcome::NotMatched;
    }

    let reached = zero_writable_range(rax, last);
    if reached == rax {
        return LoopOutcome::Unwritable;
    }

    record_emulation(EmulKind::ClzeroLoop15, rip);
    if reached == last {
        ctx.Rax = (last - CACHE_LINE_BYTES) as u64;
        write_gpr(ctx, ptr_reg, last as u64);
        // Land on the loop's native `cmp; jb` so architectural `EFlags` are set.
        ctx.Rip = (rip as u64).wrapping_add(3);
    } else {
        ctx.Rax = reached as u64;
        write_gpr(ctx, ptr_reg, (reached + CACHE_LINE_BYTES) as u64);
        // Stay at `clzero` so execution retries on the unwritable boundary.
    }
    LoopOutcome::Handled
}

// ---------------------------------------------------------------------------
// SSE4a bit-field helpers
// ---------------------------------------------------------------------------

fn compute_extrq(dst_lo: u64, raw_len: u8, raw_idx: u8) -> u64 {
    let width = u32::from(raw_len & 63);
    let shift = u32::from(raw_idx & 63);
    let shifted = dst_lo >> shift;
    if width == 0 {
        shifted
    } else {
        shifted & ((1u64 << width) - 1)
    }
}

fn compute_insertq(dst_lo: u64, src_lo: u64, raw_len: u8, raw_idx: u8) -> u64 {
    let width = u32::from(raw_len & 63);
    let shift = u32::from(raw_idx & 63);
    let mask = if width == 0 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let positioned_mask = mask << shift;
    let positioned_bits = (src_lo & mask) << shift;
    (dst_lo & !positioned_mask) | positioned_bits
}

// ---------------------------------------------------------------------------
// Crate-visible entry point
// ---------------------------------------------------------------------------

/// Attempts to emulate the Zen 2 console instruction faulting at `ctx.Rip`.
///
/// Returns `-1` (`EXCEPTION_CONTINUE_EXECUTION`) if the instruction was
/// recognized and emulated (with `ctx` updated accordingly), or `0`
/// (`EXCEPTION_CONTINUE_SEARCH`) if unhandled (leaving `ctx` and memory
/// untouched).
///
/// # Safety
/// `ctx` must reference a valid x86-64 Windows thread `CONTEXT` structure.
pub(crate) unsafe fn emulate(ctx: &mut CONTEXT) -> i32 {
    let rip = ctx.Rip as usize;
    let mut code_buf = [0u8; FETCH_WINDOW];
    let code = fetch_code(rip, &mut code_buf);
    if code.is_empty() {
        return CONTINUE_SEARCH;
    }

    let Some(insn) = decode_insn(code) else {
        return CONTINUE_SEARCH;
    };

    match insn {
        DecodedInsn::Ud2 => CONTINUE_SEARCH,

        DecodedInsn::Clwb { len } => {
            patch_with_nops(rip, len);
            record_emulation(EmulKind::Clwb, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::Monitorx { len } => {
            record_emulation(EmulKind::Monitorx, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::Mwaitx { len } => {
            std::hint::spin_loop();
            record_emulation(EmulKind::Mwaitx, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::Clzero {
            len,
            prefix_len,
            addr32,
        } => {
            if prefix_len == 0 && !addr32 {
                match try_clzero_loop12(ctx, rip, code) {
                    LoopOutcome::Handled => return CONTINUE_EXECUTION,
                    LoopOutcome::Unwritable => return CONTINUE_SEARCH,
                    LoopOutcome::NotMatched => {}
                }
                match try_clzero_loop15(ctx, rip, code) {
                    LoopOutcome::Handled => return CONTINUE_EXECUTION,
                    LoopOutcome::Unwritable => return CONTINUE_SEARCH,
                    LoopOutcome::NotMatched => {}
                }
            }

            let addr = if addr32 {
                (ctx.Rax & 0xFFFF_FFFF) as usize
            } else {
                ctx.Rax as usize
            };
            if !zero_single_cache_line(addr) {
                return CONTINUE_SEARCH;
            }
            record_emulation(EmulKind::Clzero, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::Rdpru { len } => {
            let mut qpc: i64 = 0;
            // SAFETY: `QueryPerformanceCounter` writes a 64-bit counter into `qpc`.
            unsafe {
                QueryPerformanceCounter(&mut qpc);
            }
            let value = if (ctx.Rcx as u32) <= 1 { qpc as u64 } else { 0 };
            ctx.Rax = value & 0xFFFF_FFFF;
            ctx.Rdx = value >> 32;
            record_emulation(EmulKind::Rdpru, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::Rdpid { len, dst } => {
            // SAFETY: `GetCurrentProcessorNumber` has no preconditions.
            let cpu = unsafe { GetCurrentProcessorNumber() } as u64;
            write_gpr(ctx, dst, cpu);
            record_emulation(EmulKind::Rdpid, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::ExtrqImm {
            len,
            dst,
            raw_len,
            raw_idx,
        } => {
            let dst_lo = read_xmm(ctx, dst).Low;
            let result = compute_extrq(dst_lo, raw_len, raw_idx);
            write_xmm(ctx, dst, result, 0);
            record_emulation(EmulKind::Extrq, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::ExtrqReg { len, dst, src } => {
            let dst_lo = read_xmm(ctx, dst).Low;
            let ctrl = read_xmm(ctx, src).Low;
            let result = compute_extrq(dst_lo, ctrl as u8, (ctrl >> 8) as u8);
            write_xmm(ctx, dst, result, 0);
            record_emulation(EmulKind::Extrq, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::InsertqImm {
            len,
            dst,
            src,
            raw_len,
            raw_idx,
        } => {
            let dst_lo = read_xmm(ctx, dst).Low;
            let src_lo = read_xmm(ctx, src).Low;
            let result = compute_insertq(dst_lo, src_lo, raw_len, raw_idx);
            write_xmm(ctx, dst, result, 0);
            record_emulation(EmulKind::Insertq, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }

        DecodedInsn::InsertqReg { len, dst, src } => {
            let dst_lo = read_xmm(ctx, dst).Low;
            let src_val = read_xmm(ctx, src);
            let ctrl = src_val.High as u64;
            let result = compute_insertq(dst_lo, src_val.Low, ctrl as u8, (ctrl >> 8) as u8);
            write_xmm(ctx, dst, result, 0);
            record_emulation(EmulKind::Insertq, rip);
            ctx.Rip = (rip as u64).wrapping_add(len as u64);
            CONTINUE_EXECUTION
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: standalone unit tests + 136-case black-box corpus conformance
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use windows_sys::Win32::System::Memory::{
        MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, VirtualAlloc, VirtualFree,
    };

    const PAGE_SIZE: usize = 0x1000;
    const CORPUS_TEXT: &str = include_str!("../tests/fixtures/console_insn_corpus.txt");

    /// Guard that releases a `VirtualAlloc` region on drop.
    struct VmRegion {
        base: *mut u8,
        size: usize,
    }

    impl VmRegion {
        fn alloc(size: usize, alloc_type: u32, protect: PAGE_PROTECTION_FLAGS) -> Self {
            // SAFETY: Standard Win32 virtual memory allocation.
            let ptr = unsafe { VirtualAlloc(ptr::null(), size, alloc_type, protect) } as *mut u8;
            assert!(!ptr.is_null(), "VirtualAlloc failed for size {size:#x}");
            Self { base: ptr, size }
        }

        fn alloc_low32(size: usize) -> Self {
            let mut candidate: usize = 0x2000_0000;
            while candidate + size <= 0x7FFF_0000 {
                // SAFETY: Trying explicit low-32-bit hint address.
                let ptr = unsafe {
                    VirtualAlloc(
                        candidate as *const core::ffi::c_void,
                        size,
                        MEM_COMMIT | MEM_RESERVE,
                        PAGE_READWRITE,
                    )
                } as *mut u8;
                if !ptr.is_null() {
                    let addr = ptr as usize;
                    if addr + size <= 0x1_0000_0000 {
                        return Self { base: ptr, size };
                    }
                    // SAFETY: Releasing address that landed above 4 GiB.
                    unsafe {
                        VirtualFree(ptr as *mut core::ffi::c_void, 0, MEM_RELEASE);
                    }
                }
                candidate += 0x0001_0000;
            }
            panic!("failed to allocate low-32-bit memory region");
        }

        fn protect_subrange(&self, offset: usize, len: usize, protect: PAGE_PROTECTION_FLAGS) {
            assert!(offset + len <= self.size);
            let mut old = 0;
            // SAFETY: Subrange is within our owned `VirtualAlloc` allocation.
            let ok = unsafe {
                VirtualProtect(
                    self.base.add(offset) as *const core::ffi::c_void,
                    len,
                    protect,
                    &mut old,
                )
            };
            assert_ne!(ok, 0, "VirtualProtect failed");
        }

        fn fill(&self, offset: usize, len: usize, byte: u8) {
            assert!(offset + len <= self.size);
            // SAFETY: Subrange is committed and writable at call time.
            unsafe {
                ptr::write_bytes(self.base.add(offset), byte, len);
            }
        }

        fn write_at(&self, offset: usize, bytes: &[u8]) {
            assert!(offset + bytes.len() <= self.size);
            // SAFETY: Subrange is committed and writable at call time.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(offset), bytes.len());
            }
        }

        fn read_slice(&self, offset: usize, len: usize) -> &[u8] {
            assert!(offset + len <= self.size);
            // SAFETY: Caller only reads readable subranges of our allocation.
            unsafe { std::slice::from_raw_parts(self.base.add(offset), len) }
        }

        fn addr(&self) -> usize {
            self.base as usize
        }
    }

    impl Drop for VmRegion {
        fn drop(&mut self) {
            // SAFETY: Releasing our own `VirtualAlloc` region.
            unsafe {
                VirtualFree(self.base as *mut core::ffi::c_void, 0, MEM_RELEASE);
            }
        }
    }

    fn zeroed_context() -> CONTEXT {
        // SAFETY: Windows `CONTEXT` is a plain-old-data C struct.
        unsafe { std::mem::zeroed() }
    }

    fn gpr_index(name: &str) -> u8 {
        match name {
            "rax" => 0,
            "rcx" => 1,
            "rdx" => 2,
            "rbx" => 3,
            "rsp" => 4,
            "rbp" => 5,
            "rsi" => 6,
            "rdi" => 7,
            "r8" => 8,
            "r9" => 9,
            "r10" => 10,
            "r11" => 11,
            "r12" => 12,
            "r13" => 13,
            "r14" => 14,
            "r15" => 15,
            other => panic!("unknown GPR name: {other}"),
        }
    }

    fn xmm_index(name: &str) -> u8 {
        let rest = name
            .strip_prefix("xmm")
            .unwrap_or_else(|| panic!("unknown XMM name: {name}"));
        let idx: u8 = rest.parse().expect("valid XMM index");
        assert!(idx < 16);
        idx
    }

    fn parse_hex_bytes(s: &str) -> Vec<u8> {
        s.split_whitespace()
            .map(|tok| u8::from_str_radix(tok, 16).expect("hex byte"))
            .collect()
    }

    fn parse_u64_hex(s: &str) -> u64 {
        let hex = s
            .strip_prefix("0x")
            .unwrap_or_else(|| panic!("expected 0x prefix in {s}"));
        u64::from_str_radix(hex, 16).expect("valid hex u64")
    }

    #[derive(Clone, Debug)]
    enum RegVal {
        Literal(u64),
        MemPlus(u64),
        MemHi32Plus(u64),
    }

    impl RegVal {
        fn parse(s: &str) -> Self {
            if let Some(off) = s.strip_prefix("MEM+") {
                Self::MemPlus(parse_u64_hex(off))
            } else if let Some(off) = s.strip_prefix("MEM_HI32+") {
                Self::MemHi32Plus(parse_u64_hex(off))
            } else {
                Self::Literal(parse_u64_hex(s))
            }
        }

        fn resolve(&self, mem_base: usize) -> u64 {
            match *self {
                Self::Literal(v) => v,
                Self::MemPlus(off) => (mem_base as u64).wrapping_add(off),
                Self::MemHi32Plus(off) => {
                    0xDEAD_BEEF_0000_0000 | (((mem_base as u64) & 0xFFFF_FFFF) + off)
                }
            }
        }
    }

    #[derive(Clone, Debug)]
    struct CorpusCase {
        id: String,
        env: String,
        code_layout: String,
        before: Vec<u8>,
        code: Vec<u8>,
        mem_layout: String,
        in_eflags: u32,
        in_gprs: Vec<(u8, RegVal)>,
        in_xmms: Vec<(u8, u64, u64)>,
        expect_ret: i32,
        expect_rip_delta: i64,
        expect_eflags: u32,
        expect_gprs: Vec<(u8, RegVal)>,
        expect_xmms: Vec<(u8, u64, u64)>,
        expect_rdpru_qpc: bool,
        expect_rdpid_cpu: Option<u8>,
        expect_code_patched: String,
        expect_zero_range: Option<(usize, usize)>,
        expect_other_regs_unchanged: bool,
    }

    fn parse_corpus() -> Vec<CorpusCase> {
        let mut cases = Vec::new();
        let mut cur: Option<CorpusCase> = None;

        for raw_line in CORPUS_TEXT.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("=== CASE ") {
                if let Some(prev) = cur.take() {
                    cases.push(prev);
                }
                let id = rest.strip_suffix(" ===").expect("case header end").trim();
                cur = Some(CorpusCase {
                    id: id.to_string(),
                    env: String::new(),
                    code_layout: String::new(),
                    before: Vec::new(),
                    code: Vec::new(),
                    mem_layout: String::new(),
                    in_eflags: 0,
                    in_gprs: Vec::new(),
                    in_xmms: Vec::new(),
                    expect_ret: 0,
                    expect_rip_delta: 0,
                    expect_eflags: 0,
                    expect_gprs: Vec::new(),
                    expect_xmms: Vec::new(),
                    expect_rdpru_qpc: false,
                    expect_rdpid_cpu: None,
                    expect_code_patched: String::new(),
                    expect_zero_range: None,
                    expect_other_regs_unchanged: false,
                });
                continue;
            }

            let c = cur.as_mut().expect("key before case header");
            let (key, val) = line.split_once(':').expect("key: value line");
            let key = key.trim();
            let val = val.trim();

            if let Some(reg) = key.strip_prefix("in_gpr ") {
                c.in_gprs.push((gpr_index(reg.trim()), RegVal::parse(val)));
            } else if let Some(reg) = key.strip_prefix("in_xmm ") {
                let mut parts = val.split_whitespace();
                let lo = parse_u64_hex(parts.next().expect("xmm low"));
                let hi = parse_u64_hex(parts.next().expect("xmm high"));
                c.in_xmms.push((xmm_index(reg.trim()), lo, hi));
            } else if let Some(reg) = key.strip_prefix("expect_gpr ") {
                c.expect_gprs
                    .push((gpr_index(reg.trim()), RegVal::parse(val)));
            } else if let Some(reg) = key.strip_prefix("expect_xmm ") {
                let mut parts = val.split_whitespace();
                let lo = parse_u64_hex(parts.next().expect("xmm low"));
                let hi = parse_u64_hex(parts.next().expect("xmm high"));
                c.expect_xmms.push((xmm_index(reg.trim()), lo, hi));
            } else {
                match key {
                    "env" => c.env = val.to_string(),
                    "code_layout" => c.code_layout = val.to_string(),
                    "before" => c.before = parse_hex_bytes(val),
                    "code" => c.code = parse_hex_bytes(val),
                    "mem_layout" => c.mem_layout = val.to_string(),
                    "in_eflags" => c.in_eflags = parse_u64_hex(val) as u32,
                    "expect_ret" => c.expect_ret = val.parse().expect("i32"),
                    "expect_rip_delta" => c.expect_rip_delta = val.parse().expect("i64"),
                    "expect_eflags" => c.expect_eflags = parse_u64_hex(val) as u32,
                    "expect_rdpru_qpc" => c.expect_rdpru_qpc = val == "true",
                    "expect_rdpid_cpu" => c.expect_rdpid_cpu = Some(gpr_index(val)),
                    "expect_code_patched" => c.expect_code_patched = val.to_string(),
                    "expect_zero_range" => {
                        if val == "none" {
                            c.expect_zero_range = None;
                        } else {
                            let (s, e) = val.split_once("..").expect("start..end");
                            c.expect_zero_range =
                                Some((parse_u64_hex(s) as usize, parse_u64_hex(e) as usize));
                        }
                    }
                    "expect_other_regs_unchanged" => {
                        c.expect_other_regs_unchanged = val == "true";
                    }
                    other => panic!("unexpected corpus key: {other}"),
                }
            }
        }
        if let Some(last) = cur.take() {
            cases.push(last);
        }
        cases
    }

    struct CaseEnv {
        _code_region: Option<VmRegion>,
        mem_region: Option<VmRegion>,
        rip: usize,
        code_readable: bool,
        mem_readable: bool,
        ctx: CONTEXT,
    }

    fn setup_case(c: &CorpusCase) -> CaseEnv {
        let mut code_readable = true;
        let (code_region, rip) = if c.code_layout == "normal" {
            let r = VmRegion::alloc(PAGE_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
            r.fill(0, PAGE_SIZE, 0xCC);
            let rip_off = 0x100;
            r.write_at(rip_off - c.before.len(), &c.before);
            r.write_at(rip_off, &c.code);
            let rip = r.addr() + rip_off;
            (Some(r), rip)
        } else if c.code_layout == "page_end" {
            let r = VmRegion::alloc(
                2 * PAGE_SIZE,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            );
            r.fill(0, PAGE_SIZE, 0xCC);
            let rip_off = PAGE_SIZE - c.code.len();
            r.write_at(rip_off - c.before.len(), &c.before);
            r.write_at(rip_off, &c.code);
            r.protect_subrange(PAGE_SIZE, PAGE_SIZE, PAGE_NOACCESS);
            let rip = r.addr() + rip_off;
            (Some(r), rip)
        } else if let Some(off_s) = c.code_layout.strip_prefix("page_start_prev_noaccess ") {
            let rip_offset: usize = off_s.trim().parse().expect("rip_offset");
            let r = VmRegion::alloc(
                2 * PAGE_SIZE,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            );
            r.fill(PAGE_SIZE, PAGE_SIZE, 0xCC);
            r.write_at(PAGE_SIZE + rip_offset, &c.code);
            r.protect_subrange(0, PAGE_SIZE, PAGE_NOACCESS);
            let rip = r.addr() + PAGE_SIZE + rip_offset;
            (Some(r), rip)
        } else if let Some(off_s) = c.code_layout.strip_prefix("page_start_prev_rw ") {
            let rip_offset: usize = off_s.trim().parse().expect("rip_offset");
            let r = VmRegion::alloc(
                2 * PAGE_SIZE,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            );
            r.fill(0, 2 * PAGE_SIZE, 0xCC);
            let rip_off = PAGE_SIZE + rip_offset;
            r.write_at(rip_off - c.before.len(), &c.before);
            r.write_at(rip_off, &c.code);
            let rip = r.addr() + rip_off;
            (Some(r), rip)
        } else if let Some(split_s) = c.code_layout.strip_prefix("cross_region ") {
            let split_at: usize = split_s.trim().parse().expect("split_at");
            let r = VmRegion::alloc(
                2 * PAGE_SIZE,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            );
            r.fill(0, 2 * PAGE_SIZE, 0xCC);
            let rip_off = PAGE_SIZE - split_at;
            r.write_at(rip_off, &c.code);
            r.protect_subrange(PAGE_SIZE, PAGE_SIZE, PAGE_READONLY);
            code_readable = false;
            let rip = r.addr() + rip_off;
            (Some(r), rip)
        } else if let Some(kind) = c.code_layout.strip_prefix("unreadable ") {
            code_readable = false;
            match kind.trim() {
                "noaccess" => {
                    let r = VmRegion::alloc(PAGE_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS);
                    let rip = r.addr();
                    (Some(r), rip)
                }
                "guard" => {
                    let r = VmRegion::alloc(
                        PAGE_SIZE,
                        MEM_COMMIT | MEM_RESERVE,
                        PAGE_READWRITE | PAGE_GUARD,
                    );
                    let rip = r.addr();
                    (Some(r), rip)
                }
                "reserve" => {
                    let r = VmRegion::alloc(PAGE_SIZE, MEM_RESERVE, PAGE_NOACCESS);
                    let rip = r.addr();
                    (Some(r), rip)
                }
                "null" => (None, 0),
                other => panic!("unknown unreadable layout: {other}"),
            }
        } else {
            panic!("unknown code_layout: {}", c.code_layout);
        };

        let mut mem_readable = true;
        let parts: Vec<&str> = c.mem_layout.split_whitespace().collect();
        let mem_region = match parts.as_slice() {
            ["none"] => {
                mem_readable = false;
                None
            }
            ["rw", pages] => {
                let sz = pages.parse::<usize>().unwrap() * PAGE_SIZE;
                let r = VmRegion::alloc(sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                r.fill(0, sz, 0xAA);
                Some(r)
            }
            ["rwx", pages] => {
                let sz = pages.parse::<usize>().unwrap() * PAGE_SIZE;
                let r = VmRegion::alloc(sz, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
                r.fill(0, sz, 0xAA);
                Some(r)
            }
            ["ro", pages] => {
                let sz = pages.parse::<usize>().unwrap() * PAGE_SIZE;
                let r = VmRegion::alloc(sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                r.fill(0, sz, 0xAA);
                r.protect_subrange(0, sz, PAGE_READONLY);
                Some(r)
            }
            ["noaccess", pages] => {
                let sz = pages.parse::<usize>().unwrap() * PAGE_SIZE;
                mem_readable = false;
                Some(VmRegion::alloc(sz, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS))
            }
            ["reserve", pages] => {
                let sz = pages.parse::<usize>().unwrap() * PAGE_SIZE;
                mem_readable = false;
                Some(VmRegion::alloc(sz, MEM_RESERVE, PAGE_NOACCESS))
            }
            ["rw_then_ro", rw_p, ro_p] => {
                let rw_sz = rw_p.parse::<usize>().unwrap() * PAGE_SIZE;
                let ro_sz = ro_p.parse::<usize>().unwrap() * PAGE_SIZE;
                let r = VmRegion::alloc(rw_sz + ro_sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                r.fill(0, rw_sz + ro_sz, 0xAA);
                r.protect_subrange(rw_sz, ro_sz, PAGE_READONLY);
                Some(r)
            }
            ["rw_then_rwx", rw_p, rwx_p] => {
                let rw_sz = rw_p.parse::<usize>().unwrap() * PAGE_SIZE;
                let rwx_sz = rwx_p.parse::<usize>().unwrap() * PAGE_SIZE;
                let r = VmRegion::alloc(rw_sz + rwx_sz, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                r.fill(0, rw_sz + rwx_sz, 0xAA);
                r.protect_subrange(rw_sz, rwx_sz, PAGE_EXECUTE_READWRITE);
                Some(r)
            }
            ["rw_low32", pages] => {
                let sz = pages.parse::<usize>().unwrap() * PAGE_SIZE;
                let r = VmRegion::alloc_low32(sz);
                r.fill(0, sz, 0xAA);
                Some(r)
            }
            other => panic!("unknown mem_layout: {other:?}"),
        };

        let mem_base = mem_region.as_ref().map_or(0, VmRegion::addr);
        let mut ctx = zeroed_context();
        ctx.Rip = rip as u64;
        ctx.EFlags = c.in_eflags;
        for (reg, val) in &c.in_gprs {
            write_gpr(&mut ctx, *reg, val.resolve(mem_base));
        }
        for &(reg, lo, hi) in &c.in_xmms {
            write_xmm(&mut ctx, reg, lo, hi as i64);
        }

        CaseEnv {
            _code_region: code_region,
            mem_region,
            rip,
            code_readable,
            mem_readable,
            ctx,
        }
    }

    /// Expected outcomes for the 8 corpus cases where the legacy implementation
    /// had documented architectural bugs per AMD64 APM / Intel SDM:
    ///
    /// 1. `prefix_13_bytes_total_16_accepted`:
    ///    AMD64 APM Vol. 3 §1.1 & Intel SDM Vol. 2A §2.3.11 strictly cap x86-64
    ///    instruction length at 15 bytes (`#GP` if >15). Clean-room rejects 16B.
    /// 2. `clwb_with_seg_a32_f3_quirk` (`65 67 66 f3 41 0f ae 30`):
    ///    Intel SDM Vol. 2A Table A-6 (Group 15 `0F AE /6`) & AMD64 APM Vol. 3:
    ///    `F3` overrides `66` as mandatory prefix; `F3 0F AE /6` (mem) is `#UD`.
    /// 3. `rdpid_f3_then_f2_quirk` (`f3 f2 0f c7 f9`):
    ///    Intel SDM Vol. 2B (`RDPID`) & AMD64 APM Vol. 3 (`RDPID`): `F2` after
    ///    `F3` makes `F2` the active Group-1 prefix; `F2 0F C7 /7` is `#UD`.
    /// 4. `clzero_loop12_unaligned_quirk`:
    ///    AMD64 APM Vol. 3 (`CLZERO`) always zeroes whole 64-byte aligned lines
    ///    (`addr & !63`). Legacy zeroed unaligned `[0x45..0xa3)` and set
    ///    `RAX=0xa3`. Clean-room falls back to single-line `CLZERO` (`[0x40..0x80)`,
    ///    `rip_delta=3`) so native `add; cmp; jb` runs accurately.
    /// 5. `clzero_loop12_end_equal_start_quirk`:
    ///    In `clzero; add rax, 0x40; cmp; jb`, `clzero` always executes at least
    ///    once before `cmp`/`jb`. Legacy skipped zeroing. Clean-room falls back
    ///    to single-line `CLZERO` (`[0x40..0x80)`, `rip_delta=3`).
    /// 6. `clzero_loop12_end_below_start_quirk`:
    ///    Same do-while semantics: legacy skipped zeroing and clobbered `RAX=end`.
    ///    Clean-room falls back to single-line `CLZERO` (`[0x100..0x140)`, `rip_delta=3`).
    /// 7. `clzero_loop12_over_2gib_quirk`:
    ///    Legacy silently skipped a >=2 GiB clear and jumped past the loop.
    ///    Clean-room falls back to single-line `CLZERO` (`[0x00..0x40)`, `rip_delta=3`),
    ///    matching `clzero_loop15_over_2gib_fallback`.
    /// 8. `clzero_loop12_with_harmless_prefix_quirk`:
    ///    `jb -12` (`72 F4`) jumps to `rip+1` (`0F`), not `rip` (`40`).
    ///    Clean-room requires `prefix_len == 0` for the 12-byte loop and falls
    ///    back to single-line `CLZERO` (`[0x40..0x80)`, `rip_delta=4`).
    fn documented_manual_fix(case_id: &str) -> Option<(i32, i64, Option<(usize, usize)>)> {
        match case_id {
            "prefix_13_bytes_total_16_accepted" => Some((0, 0, None)),
            "clwb_with_seg_a32_f3_quirk" => Some((0, 0, None)),
            "rdpid_f3_then_f2_quirk" => Some((0, 0, None)),
            "clzero_loop12_unaligned_quirk" => Some((-1, 3, Some((0x0040, 0x0080)))),
            "clzero_loop12_end_equal_start_quirk" => Some((-1, 3, Some((0x0040, 0x0080)))),
            "clzero_loop12_end_below_start_quirk" => Some((-1, 3, Some((0x0100, 0x0140)))),
            "clzero_loop12_over_2gib_quirk" => Some((-1, 3, Some((0x0000, 0x0040)))),
            "clzero_loop12_with_harmless_prefix_quirk" => Some((-1, 4, Some((0x0040, 0x0080)))),
            _ => None,
        }
    }

    fn verify_case_against_spec(c: &CorpusCase, env: &mut CaseEnv, ret: i32) {
        let mem_base = env.mem_region.as_ref().map_or(0, VmRegion::addr);
        let ctx_before = {
            let mut tmp = zeroed_context();
            tmp.Rip = env.rip as u64;
            tmp.EFlags = c.in_eflags;
            for (reg, val) in &c.in_gprs {
                write_gpr(&mut tmp, *reg, val.resolve(mem_base));
            }
            for &(reg, lo, hi) in &c.in_xmms {
                write_xmm(&mut tmp, reg, lo, hi as i64);
            }
            tmp
        };

        let (
            exp_ret,
            exp_rip_delta,
            exp_zero_range,
            exp_gprs,
            exp_code_patched,
            exp_rdpru_qpc,
            exp_rdpid_cpu,
        ) = if let Some((r, d, zr)) = documented_manual_fix(&c.id) {
            (r, d, zr, Vec::new(), "none".to_string(), false, None)
        } else {
            (
                c.expect_ret,
                c.expect_rip_delta,
                c.expect_zero_range,
                c.expect_gprs.clone(),
                c.expect_code_patched.clone(),
                c.expect_rdpru_qpc,
                c.expect_rdpid_cpu,
            )
        };

        assert_eq!(ret, exp_ret, "[{}] return value mismatch", c.id);
        let rip_delta = env.ctx.Rip.wrapping_sub(env.rip as u64) as i64;
        assert_eq!(rip_delta, exp_rip_delta, "[{}] rip_delta mismatch", c.id);
        assert_eq!(
            env.ctx.EFlags, c.expect_eflags,
            "[{}] eflags mismatch",
            c.id
        );

        let mut modified_gprs = [false; 16];
        let mut modified_xmms = [false; 16];

        for (reg, val) in &exp_gprs {
            modified_gprs[*reg as usize] = true;
            let got = read_gpr(&env.ctx, *reg);
            let want = val.resolve(mem_base);
            assert_eq!(got, want, "[{}] gpr {reg} mismatch", c.id);
        }

        for &(reg, want_lo, want_hi) in &c.expect_xmms {
            modified_xmms[reg as usize] = true;
            let got = read_xmm(&env.ctx, reg);
            assert_eq!(got.Low, want_lo, "[{}] xmm{reg}.Low mismatch", c.id);
            assert_eq!(
                got.High as u64, want_hi,
                "[{}] xmm{reg}.High mismatch",
                c.id
            );
        }

        if exp_rdpru_qpc {
            modified_gprs[0] = true;
            modified_gprs[2] = true;
            assert_eq!(env.ctx.Rax >> 32, 0, "[{}] rdpru rax hi32 nonzero", c.id);
            assert_eq!(env.ctx.Rdx >> 32, 0, "[{}] rdpru rdx hi32 nonzero", c.id);
            let qpc = (env.ctx.Rdx << 32) | env.ctx.Rax;
            assert!(qpc > 0, "[{}] rdpru qpc not positive", c.id);
        }

        if let Some(cpu_reg) = exp_rdpid_cpu {
            modified_gprs[cpu_reg as usize] = true;
            let val = read_gpr(&env.ctx, cpu_reg);
            assert_eq!(val >> 32, 0, "[{}] rdpid hi32 nonzero", c.id);
            assert!(val < 256, "[{}] rdpid cpu id out of range: {val}", c.id);
        }

        if c.expect_other_regs_unchanged {
            for r in 0..16u8 {
                if !modified_gprs[r as usize] {
                    assert_eq!(
                        read_gpr(&env.ctx, r),
                        read_gpr(&ctx_before, r),
                        "[{}] unexpected change to gpr {r}",
                        c.id
                    );
                }
                if !modified_xmms[r as usize] {
                    let a = read_xmm(&env.ctx, r);
                    let b = read_xmm(&ctx_before, r);
                    assert_eq!(
                        (a.Low, a.High),
                        (b.Low, b.High),
                        "[{}] unexpected change to xmm{r}",
                        c.id
                    );
                }
            }
        }

        if env.code_readable && !c.code.is_empty() {
            // SAFETY: `env.rip` has `c.code.len()` readable bytes.
            let after_code =
                unsafe { std::slice::from_raw_parts(env.rip as *const u8, c.code.len()) };
            if exp_code_patched == "none" {
                assert_eq!(
                    after_code,
                    c.code.as_slice(),
                    "[{}] code unexpectedly patched",
                    c.id
                );
            } else if let Some(cnt_s) = exp_code_patched.strip_prefix("nops ") {
                let cnt: usize = cnt_s.trim().parse().expect("nop count");
                assert!(
                    after_code[..cnt].iter().all(|&b| b == 0x90),
                    "[{}] expected {cnt} NOPs",
                    c.id
                );
                assert_eq!(
                    &after_code[cnt..],
                    &c.code[cnt..],
                    "[{}] trailing code bytes modified",
                    c.id
                );
            } else {
                panic!("unknown expect_code_patched: {exp_code_patched}");
            }
        }

        if env.mem_readable {
            let mem = env.mem_region.as_ref().unwrap();
            let bytes = mem.read_slice(0, mem.size);
            for (idx, &b) in bytes.iter().enumerate() {
                let want = match exp_zero_range {
                    Some((s, e)) if idx >= s && idx < e => 0x00,
                    _ => 0xAA,
                };
                assert_eq!(
                    b, want,
                    "[{}] mem byte at offset {idx:#x} mismatch (got {b:#x}, want {want:#x})",
                    c.id
                );
            }
        }
    }

    fn run_corpus_for_env(env_filter: &str) -> usize {
        let cases = parse_corpus();
        assert_eq!(cases.len(), 136, "expected 136 corpus cases");

        // Keep allocated regions alive across the whole run so cached virtual
        // addresses are never reused with different protections.
        let mut live_envs: Vec<CaseEnv> = Vec::new();
        let mut total = 0usize;

        for c in &cases {
            if c.env != env_filter {
                continue;
            }
            total += 1;

            let mut env = setup_case(c);
            // SAFETY: `env.ctx` is a valid synthetic `CONTEXT`.
            let ret = unsafe { emulate(&mut env.ctx) };
            verify_case_against_spec(c, &mut env, ret);

            live_envs.push(env);
        }

        total
    }

    fn run_child_test(test_name: &str, env_key: &str, env_val: &str) {
        let exe = std::env::current_exe().expect("current test exe");
        let out = Command::new(exe)
            .args(["--exact", test_name, "--ignored", "--nocapture"])
            .env(env_key, env_val)
            .output()
            .expect("spawn child test");
        assert!(
            out.status.success(),
            "child test {test_name} ({env_key}={env_val}) failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    #[test]
    fn corpus_conformance_all_136_cases() {
        let default_count = run_corpus_for_env("default");
        assert_eq!(default_count, 133);
        run_child_test(
            "zen2_emul::tests::corpus_child_loop15_disabled",
            "VICTORY_CLZERO_LOOP15",
            "0",
        );
        run_child_test(
            "zen2_emul::tests::corpus_child_clzero_ask",
            "VICTORY_CLZERO_ASK",
            "1",
        );
    }

    #[test]
    #[ignore]
    fn corpus_child_loop15_disabled() {
        let count = run_corpus_for_env("VICTORY_CLZERO_LOOP15=0");
        assert_eq!(count, 1);
    }

    #[test]
    #[ignore]
    fn corpus_child_clzero_ask() {
        let count = run_corpus_for_env("VICTORY_CLZERO_ASK=1");
        assert_eq!(count, 2);
    }

    // -----------------------------------------------------------------------
    // Standalone unit tests (decoder + edge cases beyond corpus)
    // -----------------------------------------------------------------------

    #[test]
    fn decoder_enforces_15_byte_architectural_limit() {
        // 12 legacy prefixes + 3-byte MONITORX = 15 bytes (valid)
        let mut insn15 = [0x66u8; 15];
        insn15[12] = 0x0F;
        insn15[13] = 0x01;
        insn15[14] = 0xFA;
        assert_eq!(
            decode_insn(&insn15),
            Some(DecodedInsn::Monitorx { len: 15 })
        );

        // 13 legacy prefixes + 3-byte MONITORX = 16 bytes (exceeds 15-byte limit -> None)
        let mut insn16 = [0x66u8; 16];
        insn16[13] = 0x0F;
        insn16[14] = 0x01;
        insn16[15] = 0xFA;
        assert_eq!(decode_insn(&insn16), None);
    }

    #[test]
    fn decoder_rejects_lock_prefix_and_invalid_mandatory_prefixes() {
        assert_eq!(decode_insn(&[0xF0, 0x0F, 0x01, 0xFC]), None);
        // CLWB with F3 or F2 overriding 66 is #UD
        assert_eq!(decode_insn(&[0x66, 0xF3, 0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode_insn(&[0x66, 0xF2, 0x0F, 0xAE, 0x30]), None);
        // CLWB with GS + addr32 + 66 + REX.B (no F2/F3) is valid
        assert_eq!(
            decode_insn(&[0x65, 0x67, 0x66, 0x41, 0x0F, 0xAE, 0x30]),
            Some(DecodedInsn::Clwb { len: 7 })
        );
        // RDPID with F3 then F2 is #UD; F2 then F3 is valid
        assert_eq!(decode_insn(&[0xF3, 0xF2, 0x0F, 0xC7, 0xF8]), None);
        assert_eq!(
            decode_insn(&[0xF2, 0xF3, 0x0F, 0xC7, 0xF8]),
            Some(DecodedInsn::Rdpid { len: 5, dst: 0 })
        );
    }

    #[test]
    fn clzero_loop12_rejects_non_rax_cmp_modrm() {
        let code_page =
            VmRegion::alloc(PAGE_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        // `clzero; add rax, 0x40; cmp rcx, rdx (48 39 D1); jb -12`
        // ModRM 0xD1 compares RCX (not RAX) against RDX -> must NOT trigger loop12 fast path!
        let seq = [
            0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD1, 0x72, 0xF4,
        ];
        code_page.write_at(0x100, &seq);

        let mem = VmRegion::alloc(PAGE_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        mem.fill(0, PAGE_SIZE, 0xAA);

        let mut ctx = zeroed_context();
        ctx.Rip = (code_page.addr() + 0x100) as u64;
        ctx.Rax = (mem.addr() + 0x40) as u64;
        ctx.Rdx = (mem.addr() + 0x200) as u64;

        // SAFETY: Valid synthetic CONTEXT.
        let ret = unsafe { emulate(&mut ctx) };
        assert_eq!(ret, CONTINUE_EXECUTION);
        // Must have fallen back to single-line CLZERO (rip + 3, only 64B cleared, RAX untouched).
        assert_eq!(ctx.Rip, (code_page.addr() + 0x103) as u64);
        assert_eq!(ctx.Rax, (mem.addr() + 0x40) as u64);
        assert!(mem.read_slice(0x40, 0x40).iter().all(|&b| b == 0));
        assert!(mem.read_slice(0x80, 0x40).iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn clzero_loop12_aligned_start_unaligned_end_rounds_up_lines() {
        let code_page =
            VmRegion::alloc(PAGE_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        let seq = [
            0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4,
        ];
        code_page.write_at(0x100, &seq);

        let mem = VmRegion::alloc(PAGE_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        mem.fill(0, PAGE_SIZE, 0xAA);

        let mut ctx = zeroed_context();
        ctx.Rip = (code_page.addr() + 0x100) as u64;
        ctx.Rax = (mem.addr() + 0x40) as u64;
        ctx.Rdx = (mem.addr() + 0xA5) as u64; // not 64B-aligned; hardware exits at 0xC0

        // SAFETY: Valid synthetic CONTEXT.
        let ret = unsafe { emulate(&mut ctx) };
        assert_eq!(ret, CONTINUE_EXECUTION);
        assert_eq!(ctx.Rip, (code_page.addr() + 0x10C) as u64);
        assert_eq!(ctx.Rax, (mem.addr() + 0xC0) as u64);
        assert!(mem.read_slice(0x00, 0x40).iter().all(|&b| b == 0xAA));
        assert!(mem.read_slice(0x40, 0x80).iter().all(|&b| b == 0x00));
        assert!(mem.read_slice(0xC0, 0x40).iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn sse4a_bitfield_unit_checks() {
        // EXTRQ: extract 8 bits starting at bit 16
        assert_eq!(compute_extrq(0x1234_5678_9ABC_DEF0, 8, 16), 0xBC);
        // EXTRQ: length 0 means 64 bits
        assert_eq!(
            compute_extrq(0xDEAD_BEEF_CAFE_BABE, 0, 0),
            0xDEAD_BEEF_CAFE_BABE
        );
        // EXTRQ: upper bits of len/idx masked & 63
        assert_eq!(compute_extrq(0x0000_0000_0000_0010, 0xC1, 0xC4), 1);
        // INSERTQ: insert 16 bits at bit 16
        assert_eq!(
            compute_insertq(0x1111_2222_3333_4444, 0xFFFF_ABCD, 16, 16),
            0x1111_2222_ABCD_4444
        );
        // INSERTQ: len=0, idx=32 replaces upper 32 bits with lower 32 of src
        assert_eq!(
            compute_insertq(0x1122_3344_5566_7788, 0xAAAA_BBBB_CCCC_DDDD, 0, 32),
            0xCCCC_DDDD_5566_7788
        );
    }

    #[test]
    fn all_gprs_and_xmms_roundtrip() {
        let mut ctx = zeroed_context();
        for i in 0..16u8 {
            let val = 0x1000_0000_0000_0000 + u64::from(i) * 0x0101_0101;
            write_gpr(&mut ctx, i, val);
            assert_eq!(read_gpr(&ctx, i), val);

            let lo = 0xAAAA_0000 + u64::from(i);
            let hi = (0xBBBB_0000 + i64::from(i)) << 32;
            write_xmm(&mut ctx, i, lo, hi);
            let got = read_xmm(&ctx, i);
            assert_eq!((got.Low, got.High), (lo, hi));
        }
    }
}
