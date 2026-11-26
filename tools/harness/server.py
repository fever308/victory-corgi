"""victory-harness: MCP server for building, launching, watching and screenshotting VIctory.

Runs on the Windows machine that hosts the VIctory source tree. Every log and
capture is written under this server's own folder (outside the repository), as
VIctory's asset policy requires.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import io
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prologue  # noqa: E402  (next to this file: the launch-to-world pilot)

# --------------------------------------------------------------------------- config

parser = argparse.ArgumentParser(description="victory-harness MCP server")
parser.add_argument("--repo", default=r"C:\Users\Corgan\Desktop\victory-corgi", help="VIctory source tree")
parser.add_argument(
    "--game-exe",
    default=r"C:\path\to\title\title.exe",
    help="title executable passed to 'victory run'",
)
parser.add_argument("--work", default=str(Path(__file__).resolve().parent), help="folder for run logs and captures")
parser.add_argument("--debug", action="store_true", help="verbose server logging to stderr")
ARGS, _unknown = parser.parse_known_args()

REPO = Path(ARGS.repo)
GAME_EXE = Path(ARGS.game_exe)
WORK = Path(ARGS.work)
RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("VICTORY_DATA_DIR", Path(os.environ["LOCALAPPDATA"]) / "victory" / "data"))
CARGO_BIN = Path(os.environ["USERPROFILE"]) / ".cargo" / "bin"
RENDERDOC_CMD = WORK / "renderdoc" / "RenderDoc_1.45_64" / "renderdoccmd.exe"
# VIctory polls this file at every present (VICTORY_CAPTURE_TRIGGER): when it exists it is
# removed and the next frame is saved, its render targets dumped and (inside RenderDoc) captured.
TRIGGER_FILE = WORK / "trigger.txt"
# Keys pressed from outside (VICTORY_INPUT_FILE, diag builds): `press()` and the prologue pilot append to it.
INPUT_FILE = WORK / "input.txt"

logging.basicConfig(
    stream=sys.stderr,  # stdout carries the MCP protocol
    level=logging.DEBUG if ARGS.debug else logging.INFO,
    format="%(asctime)s [victory-harness] %(levelname)s %(message)s",
)
log = logging.getLogger("victory-harness")

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MAX_RETURN_CHARS = 60_000
CREATE_NO_WINDOW = 0x08000000

mcp = FastMCP("victory-harness")

# --------------------------------------------------------------------------- run state


@dataclass
class Run:
    proc: subprocess.Popen
    profile: str
    env: dict
    log_path: Path
    capture_dir: Optional[Path]
    started: float = field(default_factory=time.time)
    handle: Optional[io.BufferedWriter] = None
    watcher: Optional["Watcher"] = None


_run: Optional[Run] = None
_lock = threading.Lock()

# Lines worth surfacing and screenshotting by default. fps lines are kept (progress), throttled for shots.
# Case-sensitive: a log level is upper case, so "ERROR" never matches a WARN line's "error=".
DEFAULT_TRIGGERS = [
    r"\sERROR\s",
    r"device removed|DEVICE_HUNG|TDR|host GPU failure",
    r"panicked|access violation|unhandled exception|crashes\.log|stack overflow",
    r"\bfps\b",
    r"(?i)loading|world|mission|title card|intro|video|save|profile|start screen|legal|language",
]
URGENT = re.compile(r"\sERROR\s|device removed|DEVICE_HUNG|host GPU failure|panicked")
# What ends a capture: the device or process failing (a logged ERROR alone does not).
FATAL = re.compile(r"device removed|DEVICE_HUNG|host GPU failure|panicked|E_OUTOFMEMORY|0x887A000[56]")
NOISE = re.compile(r"shader does not translate|PipelineLibrary|a shader reads a texture|copied from 0x")


@dataclass
class Event:
    seq: int
    t: float
    line: str
    trigger: str


@dataclass
class Shot:
    seq: int
    t: float
    path: Path
    reason: str
    black: bool
    alt: Optional[Path] = None  # screen grab taken when the window capture came back black


class Watcher(threading.Thread):
    """Follows the run log as it is written. Keeps every line (bounded), records trigger hits as events,
    and screenshots the game window on triggers (throttled) and on a fixed period."""

    def __init__(self, run: "Run"):
        super().__init__(daemon=True, name="victory-watcher")
        self.run_ref = run
        self.lines: list[tuple[int, float, str]] = []  # (seq, time, text)
        self.events: list[Event] = []
        self.shots: list[Shot] = []
        self.seq = 0
        self.cond = threading.Condition()
        self.stop_flag = threading.Event()
        self.triggers = [re.compile(p) for p in DEFAULT_TRIGGERS]
        self.shot_gap = 8.0  # min seconds between trigger screenshots
        self.periodic = 20.0  # screenshot at least this often while running (0 = off)
        self.urgent = URGENT
        self.last_shot = 0.0
        self.shot_dir = run.log_path.with_suffix("")  # runs\<stamp>-<profile>\
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        self.max_lines = 50_000

    def configure(self, triggers: Optional[list[str]], shot_gap: Optional[float], periodic: Optional[float]) -> None:
        with self.cond:
            if triggers is not None:
                self.triggers = [re.compile(p) for p in triggers]
            if shot_gap is not None:
                self.shot_gap = max(1.0, shot_gap)
            if periodic is not None:
                self.periodic = max(0.0, periodic)

    def _take(self, reason: str) -> None:
        run = self.run_ref
        if run.proc.poll() is not None:
            return
        wins = _windows_of(run.proc.pid)
        if not wins:
            return
        try:
            img = _print_window(wins[0][0])
            black = _is_black(img)
            path = self.shot_dir / f"{time.strftime('%H%M%S')}-{self.seq:07d}.png"
            img.save(path)
            alt = None
            if black:
                alt = _screen_grab(wins[0][2], path.with_name(path.stem + "-screen.png"))
            with self.cond:
                self.shots.append(Shot(self.seq, time.time(), path, reason[:300], black, alt))
                self.cond.notify_all()
            self.last_shot = time.time()
        except Exception as exc:  # window closing, minimized, etc.
            log.debug("watcher screenshot failed: %s", exc)

    def run(self) -> None:
        path = self.run_ref.log_path
        pos = 0
        partial = ""
        while not self.stop_flag.is_set():
            new = ""
            if path.exists():
                with path.open("rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read(4 * 1024 * 1024)
                    pos += len(chunk)
                new = chunk.decode("utf-8", errors="replace")
            if new:
                text = partial + new
                parts = text.split("\n")
                partial = parts.pop()
                urgent_reason = None
                trigger_reason = None
                with self.cond:
                    for raw in parts:
                        line = _strip(raw.rstrip("\r"))
                        if not line:
                            continue
                        self.seq += 1
                        now = time.time()
                        self.lines.append((self.seq, now, line))
                        if NOISE.search(line):
                            continue
                        for rx in self.triggers:
                            if rx.search(line):
                                self.events.append(Event(self.seq, now, line[:1500], rx.pattern))
                                if self.urgent.search(line):
                                    urgent_reason = line
                                elif trigger_reason is None and not re.search(r"\bfps\b", line):
                                    trigger_reason = line
                                break
                    if len(self.lines) > self.max_lines:
                        del self.lines[: len(self.lines) - self.max_lines]
                    if len(self.events) > 5000:
                        del self.events[: len(self.events) - 5000]
                    self.cond.notify_all()
                if urgent_reason:
                    self._take(f"urgent: {urgent_reason}")
                elif trigger_reason and time.time() - self.last_shot >= self.shot_gap:
                    self._take(f"trigger: {trigger_reason}")
            if self.periodic and time.time() - self.last_shot >= self.periodic:
                self._take("periodic")
            if self.run_ref.proc.poll() is not None and not new:
                with self.cond:
                    self.seq += 1
                    msg = f"[harness] process exited with code {self.run_ref.proc.returncode}"
                    self.lines.append((self.seq, time.time(), msg))
                    self.events.append(Event(self.seq, time.time(), msg, "exit"))
                    self.cond.notify_all()
                break
            time.sleep(0.25)


def _strip(text: str) -> str:
    return ANSI.sub("", text)


def _clip(text: str) -> str:
    if len(text) <= MAX_RETURN_CHARS:
        return text
    return f"[... clipped {len(text) - MAX_RETURN_CHARS} chars ...]\n" + text[-MAX_RETURN_CHARS:]


def _env_with_cargo() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{CARGO_BIN};{env.get('PATH', '')}"
    return env


def _exe_for(profile: str) -> Path:
    if profile not in ("release", "diag"):
        raise ValueError("profile must be 'release' or 'diag'")
    return REPO / "target" / profile / "victory.exe"


def _read_tail(path: Path, lines: int, max_bytes: int = 8 * 1024 * 1024) -> list[str]:
    """Last `lines` lines of a possibly huge file, reading at most `max_bytes` from its end."""
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - max_bytes))
        data = fh.read().decode("utf-8", errors="replace")
    out = _strip(data).splitlines()
    return out[-lines:]


def _alive() -> bool:
    return _run is not None and _run.proc.poll() is None


def _latest_crash_log() -> Optional[Path]:
    if not DATA_DIR.exists():
        return None
    logs = list(DATA_DIR.rglob("crashes.log"))
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


# --------------------------------------------------------------------------- win32 window capture

user32 = ctypes.WinDLL("user32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")
EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
user32.EnumWindows.argtypes = [EnumWindowsProc, wt.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.IsHungAppWindow.argtypes = [wt.HWND]
try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2: real pixel coordinates
except Exception:  # older Windows
    pass
DWMWA_EXTENDED_FRAME_BOUNDS = 9


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wt.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _process_tree(pid: int) -> set[int]:
    """`pid` and its descendants (a run started through RenderDoc is renderdoccmd's child)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    parents: dict[int, int] = {}
    if snap and snap != wt.HANDLE(-1).value:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(_ProcessEntry)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
        k32.CloseHandle(snap)
    tree = {pid}
    grew = True
    while grew:
        grew = False
        for child, parent in parents.items():
            if parent in tree and child not in tree:
                tree.add(child)
                grew = True
    return tree


def _windows_of(pid: int) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    found: list[tuple[int, str, tuple[int, int, int, int]]] = []
    tree = _process_tree(pid)

    def cb(hwnd, _lp):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in tree and user32.IsWindowVisible(hwnd):
            rect = wt.RECT()
            if dwmapi.DwmGetWindowAttribute(
                wt.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
            ) != 0:
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w, h = rect.right - rect.left, rect.bottom - rect.top
            if w > 50 and h > 50:
                buf = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd, buf, 512)
                found.append((hwnd, buf.value, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    found.sort(key=lambda t: (t[2][2] - t[2][0]) * (t[2][3] - t[2][1]), reverse=True)
    return found


# --------------------------------------------------------------------------- tools: build


@mcp.tool()
def build(profile: str = "diag", locked: bool = True) -> str:
    """Build VIctory. profile: 'release' (target\\release) or 'diag' (diagnostics feature, target\\diag).
    Returns exit status, duration and the tail of cargo's output (errors and warnings in full)."""
    if _alive():
        return "victory.exe is running; call kill first (the linker cannot replace a running exe)."
    if profile == "release":
        cmd = ["cargo", "build", "--release", "-p", "victory"]
    elif profile == "diag":
        cmd = ["cargo", "build", "--profile", "diag", "-p", "victory", "--features", "diagnostics"]
    else:
        return "profile must be 'release' or 'diag'"
    if locked:
        cmd.append("--locked")
    build_log = WORK / f"build-{profile}.log"
    t0 = time.time()
    log.info("build: %s", " ".join(cmd))
    try:
        res = subprocess.run(
            cmd, cwd=REPO, env=_env_with_cargo(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800, creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError:
        return f"cargo not found (looked in {CARGO_BIN} and PATH)"
    except subprocess.TimeoutExpired:
        return "build timed out after 30 minutes"
    out = _strip(res.stdout + res.stderr)
    build_log.write_text(out, encoding="utf-8")
    lines = out.splitlines()
    important = [l for l in lines if re.match(r"\s*(error|warning)(\[|:)", l)]
    exe = _exe_for(profile)
    exe_info = f"{exe} {exe.stat().st_size} bytes, {time.ctime(exe.stat().st_mtime)}" if exe.exists() else "exe missing"
    return _clip(
        f"exit={res.returncode} in {time.time() - t0:.1f}s\n{exe_info}\nfull log: {build_log}\n"
        f"errors/warnings: {len(important)}\n--- tail ---\n" + "\n".join(lines[-40:])
    )


# --------------------------------------------------------------------------- tools: run control


_console: subprocess.Popen | None = None


def _open_console(log_path: Path, pid: int) -> None:
    """A console window the user can watch: the run's log as it is written (the
    per-range copy lines left out). The last run's window is closed when a new run
    starts; after a run ends its window stays open to read. VICTORY_MCP_NO_CONSOLE=1
    turns it off."""
    global _console
    if os.environ.get("VICTORY_MCP_NO_CONSOLE") == "1":
        return
    if _console is not None and _console.poll() is None:
        try:
            _console.kill()
        except OSError:
            pass
    title = f"VIctory log - pid {pid} - {log_path.name}"
    script = (
        f"$host.UI.RawUI.WindowTitle = '{title}'; "
        f"Get-Content -LiteralPath '{log_path}' -Wait -Tail 80 | "
        "Where-Object { $_ -notmatch 'copystat:   copied from' }"
    )
    try:
        _console = subprocess.Popen(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", script],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    except OSError as e:
        log.warning("no log console: %s", e)


@mcp.tool()
def launch(
    profile: str = "diag",
    warp: bool = False,
    d3d12_debug: bool = False,
    dred: bool = False,
    gbv: bool = False,
    log_filter: str = "info",
    shader_loop_limit: int = 0,
    skip_pipelines: str = "",
    capture_frames: str = "",
    console: str = "series-x",
    extra_env: dict[str, str] | None = None,
    renderdoc_frames: str = "",
    renderdoc: bool = False,
    dump_targets: bool = True,
    preset: str = "",
) -> str:
    """Start the title under VIctory. Output (stdout+stderr) goes to a fresh run log.
    renderdoc: run the title inside RenderDoc (renderdoccmd starts victory.exe and hooks it), so
    capture_frame() also writes a RenderDoc .rdc of the frame (runs/<stamp>-rdc). Needs GPU memory
    to spare: VIctory's own use is near a 12 GB card's budget in the world.
    renderdoc_frames: e.g. '300,600' also captures those presented frames automatically (implies renderdoc).
    dump_targets: capture_frame() saves every render target the frame writes (diag builds).
    profile 'diag' is needed for warp/d3d12_debug/dred/gbv/capture/loop-limit/skip-pipelines (debugging aids).
    capture_frames: e.g. '1,30,60,120' saves those presented frames as images (read with list_captures/get_capture).
    skip_pipelines: comma-separated 16-digit shader hashes whose draws are omitted.
    preset: a known set of settings ('lod': dkdknight's levels-on-demand settings, own cache folder;
    'plain'); extra_env is applied on top. Keys can be pressed with press() in diag builds."""
    global _run
    with _lock:
        if _alive():
            return f"already running (pid {_run.proc.pid}); call kill first"
        exe = _exe_for(profile)
        if not exe.exists():
            return f"{exe} not built; call build(profile='{profile}')"
        if not GAME_EXE.exists():
            return f"title not found: {GAME_EXE}"
        if console not in ("series-x", "series-s"):
            return "console must be 'series-x' or 'series-s'"

        stamp = time.strftime("%Y%m%d-%H%M%S")
        env = _env_with_cargo()
        set_vars: dict[str, str] = {"VICTORY_LOG": log_filter}
        if warp:
            set_vars["VICTORY_D3D12_WARP"] = "1"
        if d3d12_debug:
            set_vars["VICTORY_D3D12_DEBUG"] = "1"
        if gbv:
            set_vars["VICTORY_D3D12_DEBUG"] = "1"
            set_vars["VICTORY_D3D12_GBV"] = "1"
        if dred:
            set_vars["VICTORY_D3D12_DRED"] = "1"
        if shader_loop_limit > 0:
            set_vars["VICTORY_SHADER_LOOP_LIMIT"] = str(shader_loop_limit)
        if skip_pipelines.strip():
            set_vars["VICTORY_SKIP_PIPELINES"] = skip_pipelines.strip()
        capture_dir = None
        if capture_frames.strip():
            capture_dir = RUNS / f"{stamp}-capture"
            capture_dir.mkdir(parents=True, exist_ok=True)
            set_vars["VICTORY_CAPTURE"] = str(capture_dir)
            set_vars["VICTORY_CAPTURE_FRAMES"] = capture_frames.strip()
        # capture_frame() works in every diag run: the trigger file, and with dump_targets a folder.
        TRIGGER_FILE.unlink(missing_ok=True)
        set_vars["VICTORY_CAPTURE_TRIGGER"] = str(TRIGGER_FILE)
        if dump_targets:
            if capture_dir is None:
                capture_dir = RUNS / f"{stamp}-capture"
                capture_dir.mkdir(parents=True, exist_ok=True)
                set_vars["VICTORY_CAPTURE"] = str(capture_dir)
            set_vars["VICTORY_DUMP_TARGETS"] = "1"
        if preset:
            try:
                set_vars.update(prologue.preset_env(preset))
            except ValueError as exc:
                return str(exc)
        INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        INPUT_FILE.write_text("", encoding="ascii")
        set_vars["VICTORY_INPUT_FILE"] = str(INPUT_FILE)
        if extra_env:
            set_vars.update({str(k): str(v) for k, v in extra_env.items()})
        for k in [k for k in env if k.startswith("VICTORY_")]:
            del env[k]  # no leftovers from the server's own environment
        env.update(set_vars)

        if profile == "release" and any(
            k in set_vars for k in ("VICTORY_D3D12_WARP", "VICTORY_D3D12_DEBUG", "VICTORY_D3D12_DRED",
                                    "VICTORY_CAPTURE", "VICTORY_SHADER_LOOP_LIMIT", "VICTORY_SKIP_PIPELINES")
        ):
            note = "WARNING: release builds ignore debugging variables; use profile='diag'.\n"
        else:
            note = ""

        log_path = RUNS / f"{stamp}-{profile}.log"
        cmd = [str(exe), "run", str(GAME_EXE), "--execute", "--console", console]
        if renderdoc_frames.strip() or renderdoc:
            if not RENDERDOC_CMD.exists():
                return f"RenderDoc not found: {RENDERDOC_CMD}"
            rdc_dir = RUNS / f"{stamp}-rdc"
            rdc_dir.mkdir(parents=True, exist_ok=True)
            # VIctory writes its own log (RenderDoc's child may not pass output on);
            # renderdoccmd's output goes to a log of its own.
            if renderdoc_frames.strip():
                env["VICTORY_RENDERDOC_FRAMES"] = renderdoc_frames.strip()
                set_vars["VICTORY_RENDERDOC_FRAMES"] = renderdoc_frames.strip()
            env["VICTORY_LOG_FILE"] = str(log_path)
            set_vars["VICTORY_RENDERDOC"] = "1"
            cmd = [str(RENDERDOC_CMD), "capture", "-w", "-d", str(REPO), "-c", str(rdc_dir / "victory")] + cmd
            log_path.touch()
            handle = open(RUNS / f"{stamp}-renderdoccmd.log", "wb")
        else:
            handle = open(log_path, "wb")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        _open_console(log_path, proc.pid)
        _run = Run(proc=proc, profile=profile, env=set_vars, log_path=log_path, capture_dir=capture_dir, handle=handle)
        _run.watcher = Watcher(_run)
        _run.watcher.start()
        log.info("launched pid %s: %s", proc.pid, cmd)
        return (
            f"{note}launched pid {proc.pid} ({profile})\nenv {set_vars}\nlog {log_path}\n"
            f"live watcher on: screenshots to {_run.watcher.shot_dir} (triggers every >= {_run.watcher.shot_gap}s, "
            f"periodic {_run.watcher.periodic}s). Use watch() / live() to follow it."
        )


@mcp.tool()
def kill() -> str:
    """Stop the running title (and any victory.exe left from earlier runs)."""
    global _run
    msgs = []
    with _lock:
        if _run and _run.watcher:
            _run.watcher.stop_flag.set()
        if _alive():
            _run.proc.kill()
            try:
                _run.proc.wait(10)
            except subprocess.TimeoutExpired:
                pass
            msgs.append(f"killed pid {_run.proc.pid}")
        if _run and _run.handle:
            _run.handle.close()
    for image in ("victory.exe", "renderdoccmd.exe"):
        stray = subprocess.run(
            ["taskkill", "/F", "/IM", image], capture_output=True, text=True, creationflags=CREATE_NO_WINDOW
        )
        if stray.returncode == 0:
            msgs.append(stray.stdout.strip())
    return "\n".join(msgs) or "nothing running"


@mcp.tool()
def status() -> str:
    """PID, uptime, exit code, memory, window title/responsiveness, log size and newest crash entry time."""
    if _run is None:
        return "no run started by this server"
    p = _run.proc
    code = p.poll()
    parts = [
        f"pid {p.pid} ({_run.profile}) {'running' if code is None else f'exited code {code} (0x{code & 0xFFFFFFFF:08X})'}",
        f"uptime {time.time() - _run.started:.0f}s",
        f"env {_run.env}",
        f"log {_run.log_path} ({_run.log_path.stat().st_size if _run.log_path.exists() else 0} bytes)",
    ]
    if code is None:
        try:
            q = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Process -Id {p.pid}).WorkingSet64/1MB; (Get-Process -Id {p.pid}).CPU"],
                capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW,
            ).stdout.split()
            if len(q) >= 2:
                parts.append(f"memory {float(q[0]):.0f} MB, cpu {float(q[1]):.0f}s")
        except Exception as exc:
            parts.append(f"memory query failed: {exc}")
        wins = _windows_of(p.pid)
        for hwnd, title, rect in wins[:3]:
            hung = bool(user32.IsHungAppWindow(hwnd))
            parts.append(f"window '{title}' {rect[2]-rect[0]}x{rect[3]-rect[1]} {'NOT RESPONDING' if hung else 'responding'}")
        if not wins:
            parts.append("no visible window yet")
    crash = _latest_crash_log()
    if crash:
        parts.append(f"newest crashes.log {crash} modified {time.ctime(crash.stat().st_mtime)}")
    return "\n".join(parts)


@mcp.tool()
def wait(seconds: int = 30, until_regex: str = "") -> str:
    """Block up to `seconds` (max 100) until the run exits or a log line matches until_regex.
    Returns what happened plus the matching or last lines."""
    if _run is None:
        return "no run"
    seconds = max(1, min(seconds, 100))
    pat = re.compile(until_regex) if until_regex else None
    start_size = _run.log_path.stat().st_size if _run.log_path.exists() else 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _run.proc.poll() is not None:
            return f"exited code {_run.proc.returncode}\n" + "\n".join(_read_tail(_run.log_path, 30))
        if pat and _run.log_path.exists():
            with _run.log_path.open("rb") as fh:
                fh.seek(start_size)
                new = _strip(fh.read().decode("utf-8", errors="replace"))
            hits = [l for l in new.splitlines() if pat.search(l)]
            if hits:
                return "matched:\n" + _clip("\n".join(hits[-20:]))
        time.sleep(1)
    return f"still running after {seconds}s\n" + "\n".join(_read_tail(_run.log_path, 15))


# --------------------------------------------------------------------------- tools: logs


@mcp.tool()
def log_tail(lines: int = 80, run_log: str = "") -> str:
    """Last N lines (ANSI stripped) of the current run's log, or of run_log (a file name in the runs folder)."""
    path = RUNS / run_log if run_log else (_run.log_path if _run else None)
    if path is None or not path.exists():
        return "no log"
    return _clip("\n".join(_read_tail(path, max(1, min(lines, 2000)))))


@mcp.tool()
def log_grep(pattern: str, max_hits: int = 60, run_log: str = "", ignore_case: bool = True) -> str:
    """Regex search through the whole run log (streamed, any size). Returns the last max_hits matches with line numbers."""
    path = RUNS / run_log if run_log else (_run.log_path if _run else None)
    if path is None or not path.exists():
        return "no log"
    rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    hits: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = _strip(line.rstrip("\n"))
            if rx.search(line):
                total += 1
                hits.append(f"{n}: {line[:2000]}")
                if len(hits) > max_hits:
                    hits.pop(0)
    return _clip(f"{total} matching lines (showing last {len(hits)})\n" + "\n".join(hits))


@mcp.tool()
def crashes(entries: int = 1) -> str:
    """The newest crashes.log entries VIctory wrote for the title (found under the data folder)."""
    path = _latest_crash_log()
    if not path:
        return f"no crashes.log under {DATA_DIR}"
    text = _strip(path.read_text(encoding="utf-8", errors="replace"))
    blocks = re.split(r"(?=^unix time \d+, process \d+)", text, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip()]
    return _clip(f"{path}\n\n" + "\n\n".join(blocks[-max(1, entries):]))


@mcp.tool()
def list_runs() -> str:
    """Run logs and capture folders kept in the runs folder, newest first."""
    items = sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return "\n".join(
        f"{p.name}  {'dir' if p.is_dir() else str(p.stat().st_size) + ' bytes'}  {time.ctime(p.stat().st_mtime)}"
        for p in items[:40]
    ) or "none"


# --------------------------------------------------------------------------- tools: screenshots


def _encode(img, max_width: int) -> Image:
    from PIL import Image as PILImage

    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetDC.argtypes = [wt.HWND]
user32.GetDC.restype = wt.HDC
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.PrintWindow.argtypes = [wt.HWND, wt.HDC, wt.UINT]
user32.IsIconic.argtypes = [wt.HWND]
gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
gdi32.CreateCompatibleDC.restype = wt.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wt.HBITMAP
gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
gdi32.SelectObject.restype = wt.HGDIOBJ
gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wt.HDC]
gdi32.GetDIBits.argtypes = [wt.HDC, wt.HBITMAP, wt.UINT, wt.UINT, ctypes.c_void_p, ctypes.c_void_p, wt.UINT]
PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2  # Windows 8.1+: captures DirectX/flip-model content, even when covered


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
        ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
        ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
        ("biClrImportant", wt.DWORD),
    ]


def _print_window(hwnd: int):
    """The window's own client area via PrintWindow(PW_RENDERFULLCONTENT): independent of what covers it."""
    from PIL import Image as PILImage

    if user32.IsIconic(hwnd):
        raise RuntimeError("game window is minimized; restore it (covered is fine, minimized is not)")
    rc = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    w, h = rc.right - rc.left, rc.bottom - rc.top
    if w <= 0 or h <= 0:
        raise RuntimeError("game window has an empty client area")
    wdc = user32.GetDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(wdc)
    bmp = gdi32.CreateCompatibleBitmap(wdc, w, h)
    old = gdi32.SelectObject(mdc, bmp)
    try:
        if not user32.PrintWindow(hwnd, mdc, PW_CLIENTONLY | PW_RENDERFULLCONTENT):
            raise RuntimeError(f"PrintWindow failed (error {ctypes.get_last_error()})")
        bih = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(w * h * 4)
        if gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bih), 0) != h:
            raise RuntimeError("GetDIBits failed")
        img = PILImage.frombuffer("RGB", (w, h), buf.raw, "raw", "BGRX", 0, 1)
    finally:
        gdi32.SelectObject(mdc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(hwnd, wdc)
    return img


def _is_black(img) -> bool:
    from PIL import ImageStat

    return max(ImageStat.Stat(img.convert("RGB")).extrema[i][1] for i in range(3)) < 8


def _screen_grab(rect: tuple[int, int, int, int], path: Path) -> Optional[Path]:
    """The game window's area of the desktop. Fallback for a black window capture: tells a really black
    frame from a capture that missed the swapchain. Shows whatever covers the window."""
    from PIL import ImageGrab

    try:
        img = ImageGrab.grab(bbox=rect, all_screens=True)
    except Exception as exc:
        log.debug("screen grab failed: %s", exc)
        return None
    if _is_black(img):
        return None
    img.save(path)
    return path


@mcp.tool()
def screenshot(max_width: int = 1280, save: bool = True) -> Image:
    """Screenshot of the game's own window contents (client area), captured from the window itself,
    so it works while other windows cover it. The window must not be minimized.
    If that capture is all black, the screen area of the window is returned instead when it is not black."""
    if not _alive():
        raise RuntimeError("game not running")
    wins = _windows_of(_run.proc.pid)
    if not wins:
        raise RuntimeError("game has no visible window yet")
    hwnd, title, rect = wins[0]
    img = _print_window(hwnd)
    if _is_black(img):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        alt = _screen_grab(rect, RUNS / f"{stamp}-shot-screen.png")
        if alt is not None:
            from PIL import Image as PILImage

            log.info("window capture black for '%s'; returning the screen area instead", title)
            return _encode(PILImage.open(alt), max(320, min(max_width, 2560)))
        log.info("window capture and screen area both black for '%s': the frame is black", title)
    if save:
        img.save(RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-shot.png")
    log.info("screenshot of '%s' %sx%s", title, img.width, img.height)
    return _encode(img, max(320, min(max_width, 2560)))


@mcp.tool()
def capture_frame(wait_seconds: int = 90) -> str:
    """Capture the next frame of the running game (diag build): VIctory saves the presented frame and
    dumps every render target it writes (launch(dump_targets=True), the default), and when the run is
    inside RenderDoc (launch(renderdoc=True)) writes a RenderDoc .rdc of it. Waits up to wait_seconds
    (max 100) for the files and returns their paths, or the error if the game fails meanwhile.
    Open .rdc files with renderdoc/RenderDoc_1.45_64/qrenderdoc.exe."""
    if not _alive():
        return "no running game; launch first"
    log_path = _run.log_path
    start = log_path.stat().st_size if log_path.exists() else 0
    TRIGGER_FILE.write_text("capture\n")
    wait_seconds = max(5, min(wait_seconds, 100))
    deadline = time.time() + wait_seconds
    frame = None
    saved: list[str] = []
    rdc: list[str] = []
    in_renderdoc = "VICTORY_RENDERDOC" in _run.env or "VICTORY_RENDERDOC_FRAMES" in _run.env
    while time.time() < deadline:
        new = ""
        if log_path.exists():
            with log_path.open("rb") as fh:
                fh.seek(start)
                new = _strip(fh.read().decode("utf-8", errors="replace"))
        for line in new.splitlines():
            m = re.search(r"capture trigger: the next frame is saved and dumped frame=(\d+)", line)
            if m:
                frame = int(m.group(1))
            m = re.search(r"frame saved path=(\S+)", line)
            if m and frame is not None and re.search(rf"frame-{frame}(\.|-)", m.group(1)) and m.group(1) not in saved:
                saved.append(m.group(1))
            m = re.search(r"RenderDoc capture written index=\d+ path=(\S+)", line)
            if m and m.group(1) not in rdc:
                rdc.append(m.group(1))
            if FATAL.search(line):
                return (
                    f"the game failed while capturing frame {frame}:\n{_short(line, 1500)}\n"
                    f"saved so far: {len(saved)} files; RenderDoc: {rdc or 'none'}"
                )
        if _run.proc.poll() is not None:
            return f"the game exited (code {_run.proc.returncode}) before the capture finished; frame {frame}"
        rdc_done = bool(rdc) or not in_renderdoc
        saved_done = bool(saved) or _run.capture_dir is None
        # Target dumps arrive in a burst at the frame's present; give them a moment to finish.
        if frame is not None and saved_done and rdc_done:
            time.sleep(3)
            break
        time.sleep(1)
    else:
        if frame is None:
            return f"no frame picked the trigger up within {wait_seconds}s (is this a diag build? is the game presenting?)"
    folder = _run.capture_dir
    lines = [f"frame {frame}: {len(saved)} images saved"]
    if folder:
        files = sorted([*folder.glob(f"frame-{frame}.png"), *folder.glob(f"frame-{frame}-*.png")])
        lines.append(f"folder {folder} ({len(files)} files for this frame)")
        lines += [f"  {p.name}" for p in files[:12]]
        if len(files) > 12:
            lines.append(f"  ... and {len(files) - 12} more")
    if in_renderdoc:
        lines.append("RenderDoc: " + (", ".join(rdc) if rdc else "no .rdc written yet (it can take minutes for a world frame)"))
    return "\n".join(lines)


@mcp.tool()
def list_captures() -> str:
    """Frames VIctory saved for the current run (launch with capture_frames, diag build)."""
    if not _run or not _run.capture_dir:
        return "current run has no capture folder; launch with capture_frames"
    files = sorted(p for p in _run.capture_dir.rglob("*") if p.is_file())
    return "\n".join(f"{p.relative_to(_run.capture_dir)}  {p.stat().st_size} bytes" for p in files) or "no frames yet"


@mcp.tool()
def get_capture(name: str, max_width: int = 1280) -> Image:
    """Return one saved frame (name as shown by list_captures) as an image."""
    from PIL import Image as PILImage

    if not _run or not _run.capture_dir:
        raise RuntimeError("no capture folder")
    path = (_run.capture_dir / name).resolve()
    if _run.capture_dir.resolve() not in path.parents:
        raise ValueError("name must be inside the capture folder")
    return _encode(PILImage.open(path), max(320, min(max_width, 2560)))


# --------------------------------------------------------------------------- tools: live follow


def _watcher() -> "Watcher":
    if _run is None or _run.watcher is None:
        raise RuntimeError("no run launched by this harness; call launch first")
    return _run.watcher


def _fmt_t(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


_NUM = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-f]{16}\b|\d+(?:\.\d+)?")


def _short(line: str, limit: int = 300) -> str:
    """A log line without its long tails: pipeline sets, breadcrumb arrays and tables are summarized."""
    m = re.search(r" (pipelines|breadcrumbs[^=\[]*)=?\s*[\{\[]", line)
    if m:
        head, tail = line[: m.start()], line[m.start():]
        n = len(re.findall(r"\[(?:compute|mesh|vertex|pixel) ", tail)) or tail.count(",") + 1
        line = f"{head} {m.group(1).strip()}=<{n} entries>"
    return line if len(line) <= limit else line[:limit] + " ..."


def _collapse(events: list, max_groups: int = 40) -> list[str]:
    """Group repeated events (same message once numbers and hashes are removed) into one counted line,
    in order of first appearance, so a flood of one warning cannot hide the rare ones."""
    groups: dict[str, list] = {}
    for e in events:
        msg = re.sub(r"^\S+Z\s+", "", e.line)
        key = _NUM.sub("#", msg)[:140]
        g = groups.get(key)
        if g is None:
            groups[key] = [e, e, 1]
        else:
            g[1] = e
            g[2] += 1
    out = []
    for first, last, n in list(groups.values())[-max_groups:]:
        if n == 1:
            out.append(f"{first.seq} {_fmt_t(first.t)} {_short(first.line, 1200)}")
        else:
            out.append(
                f"x{n} {_fmt_t(first.t)}..{_fmt_t(last.t)} (seq {first.seq}..{last.seq}) last: {_short(last.line, 600)}"
            )
    if len(groups) > max_groups:
        out.insert(0, f"({len(groups) - max_groups} older event kinds not shown; use live() or log_grep)")
    return out


@mcp.tool()
def live(since: int = 0, max_lines: int = 150, regex: str = "", include_noise: bool = False) -> str:
    """Log lines written since sequence number `since` (0 = the last max_lines), as the game writes them.
    Returns the lines plus `next`, the cursor to pass next time so nothing is missed or repeated.
    regex filters lines; noise (untranslated-shader and pipeline-cache warnings) is hidden unless include_noise."""
    w = _watcher()
    rx = re.compile(regex, re.IGNORECASE) if regex else None
    with w.cond:
        lines = [l for l in w.lines if l[0] > since] if since else list(w.lines)
        last = w.seq
        nshots = len(w.shots)
    hidden = 0
    out = []
    for seq, t, text in lines:
        if not include_noise and NOISE.search(text):
            hidden += 1
            continue
        if rx and not rx.search(text):
            continue
        out.append(f"{seq} {_fmt_t(t)} {text[:1500]}")
    out = out[-max(1, min(max_lines, 2000)):]
    state = "running" if _alive() else f"exited {_run.proc.returncode}"
    return _clip(
        f"next={last}  {state}  uptime {time.time() - _run.started:.0f}s  shots={nshots}  "
        f"noise hidden={hidden}\n" + "\n".join(out)
    )


@mcp.tool()
def watch(seconds: int = 60, max_images: int = 4, since: int = 0, stop_on_urgent: bool = True) -> list:
    """Follow the game live for up to `seconds` (max 100): returns every trigger event (errors, GPU removal,
    fps/progress, stage changes) and the screenshots the watcher took meanwhile, newest last.
    Returns early on an urgent event (error / device removal / exit) when stop_on_urgent.
    `since`: event cursor from a previous watch (its `next`); 0 = start from now."""
    w = _watcher()
    seconds = max(1, min(seconds, 100))
    with w.cond:
        start_seq = since or w.seq
        start_t = time.time()
        deadline = start_t + seconds
        while time.time() < deadline:
            urgent = any(
                e.seq > start_seq and (w.urgent.search(e.line) or e.trigger == "exit") for e in w.events
            )
            if stop_on_urgent and urgent:
                break
            w.cond.wait(timeout=min(2.0, max(0.05, deadline - time.time())))
        if stop_on_urgent:
            time.sleep(0)  # let the watcher finish its urgent screenshot
        events = [e for e in w.events if e.seq > start_seq]
        shots = [s for s in w.shots if s.t >= start_t or s.seq > start_seq]
        end_seq = w.seq
    if stop_on_urgent and any(w.urgent.search(e.line) for e in events):
        time.sleep(1.5)  # urgent screenshot is taken right after the line; give it a moment
        with w.cond:
            shots = [s for s in w.shots if s.t >= start_t or s.seq > start_seq]
    fps = [e for e in events if re.search(r"\bfps\b", e.line)]
    other = [e for e in events if e not in fps]
    lines = [
        f"next={end_seq}  {'running' if _alive() else f'exited {_run.proc.returncode}'}  "
        f"watched {time.time() - start_t:.0f}s  uptime {time.time() - _run.started:.0f}s  "
        f"events={len(events)} shots={len(shots)}"
    ]
    if fps:
        lines.append(f"fps (latest of {len(fps)}): {_fmt_t(fps[-1].t)} {fps[-1].line[:400]}")
    lines.extend(_collapse(other))
    picked = shots[-max(0, min(max_images, 8)):] if max_images else []
    result: list = ["\n".join(lines)]
    from PIL import Image as PILImage

    for s in picked:
        if s.black and s.alt is not None:
            note = " [window capture black; showing the screen area, which may include covering windows]"
            show = s.alt
        elif s.black:
            note = " [all black: window capture and screen area both black, so the frame really is black]"
            show = s.path
        else:
            note, show = "", s.path
        result.append(f"screenshot {_fmt_t(s.t)} at log seq {s.seq} ({_short(s.reason)}){note}")
        try:
            result.append(_encode(PILImage.open(show), 1024))
        except Exception as exc:
            result.append(f"(could not open {show}: {exc})")
    return result


@mcp.tool()
def full_cycle(
    profile: str = "diag",
    rebuild: bool = True,
    seconds: int = 90,
    d3d12_debug: bool = True,
    dred: bool = True,
    warp: bool = False,
    shader_loop_limit: int = 0,
    skip_pipelines: str = "",
    max_images: int = 4,
) -> list:
    """cs2-harness style one-shot: build -> kill any old run -> launch -> follow the live log for `seconds`
    (max 100, stops early on error / GPU removal / exit) -> report: build result, every trigger event,
    newest crash entry if one was written, and the watcher's screenshots."""
    report: list[str] = []
    if rebuild:
        kill()
        built = build(profile)
        report.append("--- build ---\n" + built)
        if not built.startswith("exit=0"):
            return ["\n\n".join(report)]
    else:
        kill()
    report.append(
        "--- launch ---\n"
        + launch(profile=profile, warp=warp, d3d12_debug=d3d12_debug, dred=dred,
                 shader_loop_limit=shader_loop_limit, skip_pipelines=skip_pipelines)
    )
    if not _alive():
        return ["\n\n".join(report) + "\n\nprocess did not start"]
    watched = watch(seconds=seconds, max_images=max_images)
    report.append("--- live log (trigger events) ---\n" + watched[0])
    crash = _latest_crash_log()
    if crash and crash.stat().st_mtime >= _run.started:
        report.append("--- new crash entry ---\n" + crashes(1)[:6000])
    return ["\n\n".join(report)] + watched[1:]


@mcp.tool()
def watch_config(triggers: list[str] | None = None, shot_gap: float | None = None, periodic: float | None = None) -> str:
    """Change what the live watcher reacts to. triggers: regex list (replaces the set; omit to keep);
    shot_gap: minimum seconds between trigger screenshots; periodic: screenshot every N seconds (0 = off)."""
    w = _watcher()
    w.configure(triggers, shot_gap, periodic)
    return f"triggers={[r.pattern for r in w.triggers]} shot_gap={w.shot_gap}s periodic={w.periodic}s"


# --------------------------------------------------------------------------- tools: input and the prologue

_pilot: Optional[prologue.Pilot] = None


@mcp.tool()
def press(keys: str = "Enter", hold_ms: int = 150, repeat: int = 1, gap_ms: int = 600) -> str:
    """Press keys in the running title (diag builds, with or without focus; also the first pad's buttons).
    keys: comma-separated presses, '+' joins keys held together, e.g. 'Down,Down,Enter' or 'Enter+Down'.
    Keys: letters, digits, Enter (A), Back (B), Esc (Menu), Tab (View), Up/Down/Left/Right (d-pad),
    X, Y, Q/E (shoulders), F/H (stick clicks), W/A/S/D (left stick, without a real pad), Z/C (triggers).
    repeat: the whole list that many times; gap_ms between presses. Takes at most 60 s."""
    if not _alive():
        return "no run"
    try:
        chords = prologue.check_keys(keys)
    except ValueError as exc:
        return str(exc)
    repeat = max(1, repeat)
    total = len(chords) * repeat
    gap = max(0.05, gap_ms / 1000.0)
    if total * gap > 60:
        return f"{total} presses {gap_ms} ms apart take over 60 s; send fewer"
    for i in range(total):
        prologue.press(INPUT_FILE, chords[i % len(chords)], hold_ms)
        if i + 1 < total:
            time.sleep(gap)
    w = prologue.window_of(_run.proc.pid)
    return f"pressed {total}: {', '.join(chords)} x{repeat}; window '{w.title if w else '?'}'"


def _pilot_launch(profile: str, preset: str, extra_env: dict | None, console: str):
    def go() -> tuple[int, Path]:
        said = launch(profile=profile, preset=preset, extra_env=extra_env, console=console, dump_targets=False)
        if not _alive():
            raise RuntimeError(said)
        return _run.proc.pid, _run.log_path
    return go


@mcp.tool()
def to_prologue(
    preset: str = "lod",
    profile: str = "diag",
    relaunch: bool = False,
    keys: str = "",
    press_every: float = 0.0,
    first_press_after: float = 20.0,
    world_draws: int = 1500,
    settle: float = 20.0,
    stall: float = 240.0,
    attempt_timeout: float = 1800.0,
    restarts: int = 3,
    extra_env: dict[str, str] | None = None,
    console: str = "series-x",
    route: str = "first-run",
) -> str:
    """Get a run to the world (the prologue) with no one at the keyboard; returns at once, the pilot runs in
    the background (prologue_status / prologue_stop). Follows the running title unless relaunch (or nothing
    runs), else launches with `preset`. `route` (prologue.ROUTES) says which keys walk the menus and how the
    opening cutscene is skipped: 'first-run' taps Enter and Down in turn through the setup, legal and terms
    screens, holds Enter once the frame is the cutscene's, and counts the world from the loading screen
    after it; 'enter' only taps Enter. `keys` (comma list, cycled; Enter = A) and press_every, when given,
    replace the route's; presses start after first_press_after and go on until the window title shows
    >= world_draws draws/frame for `settle` seconds. A run that exits, stops moving for `stall` seconds (no log output and the same title; half that
    with the window not responding) or takes over attempt_timeout is killed and launched again, up to
    `restarts` times. The first launch of a new cache folder translates every shader (many minutes)."""
    global _pilot
    if _pilot is not None and _pilot.is_alive():
        return "a pilot is running; prologue_status() or prologue_stop()"
    attach = None
    if _alive() and not relaunch:
        attach = (_run.proc.pid, _run.log_path)
        if "VICTORY_INPUT_FILE" not in _run.env:
            return ("the running title was not launched with VICTORY_INPUT_FILE (keys would not reach it); "
                    "use relaunch=True")
    try:
        settings = prologue.Settings.of_route(
            route, keys=keys.strip() or None, press_every=press_every or None,
            first_press_after=first_press_after, world_draws=world_draws,
            settle=settle, stall=stall, attempt_timeout=attempt_timeout, restarts=restarts,
        )
    except ValueError as exc:
        return str(exc)

    def shot() -> None:
        try:
            screenshot(save=True)
        except Exception as exc:  # a minimized window
            log.info("prologue screenshot: %s", exc)

    try:
        _pilot = prologue.Pilot(
            settings, INPUT_FILE, _pilot_launch(profile, preset, extra_env, console), kill, _alive,
            attach=attach, on_world=shot,
        )
    except ValueError as exc:
        return str(exc)
    _pilot.start()
    time.sleep(2)
    return "pilot started\n" + _pilot.report.text()


@mcp.tool()
def prologue_status(with_screenshot: bool = False) -> list:
    """Where the prologue pilot is: state (launching, waiting, in-world, failed, stopped), attempt, presses,
    the window's fps and draws/frame, and its events; with_screenshot adds the game window."""
    if _pilot is None:
        return ["no pilot started"]
    out: list = [_pilot.report.text()]
    if with_screenshot and _alive():
        try:
            out.append(screenshot(save=False))
        except Exception as exc:
            out.append(f"screenshot failed: {exc}")
    return out


@mcp.tool()
def prologue_stop() -> str:
    """Stop the prologue pilot (the run keeps going)."""
    if _pilot is None or not _pilot.is_alive():
        return "no pilot running"
    _pilot.stop()
    _pilot.join(5)
    return _pilot.report.text()


if __name__ == "__main__":
    log.info("repo=%s game=%s work=%s data=%s", REPO, GAME_EXE, WORK, DATA_DIR)
    mcp.run()
