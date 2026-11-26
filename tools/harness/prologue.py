"""Drive a VIctory run of the title from launch to the first frames of the world (the prologue).

Used two ways:

* by the victory-harness MCP server (`to_prologue`, `prologue_status`, `prologue_stop`, `press`), which
  launches through its own `launch()` so its tools keep following the run;
* on its own, from a console:

      python prologue.py --repo C:\\Users\\Corgan\\Desktop\\victory-corgi ^
          --game-exe "D:\\...\\title.exe" --work C:\\Users\\Corgan\\victory-mcp --preset lod

    python prologue.py --game-exe "<exe>" --attach     (follow the run the harness launched)

How it works. VIctory (diagnostics builds) reads keys from `VICTORY_INPUT_FILE`: each line appended to it is
pressed when seen, with or without focus, on the keyboard and on the first pad. The pilot presses the
configured keys (Enter = A by default) every few seconds until the window title reports a frame of the world
(`<n> draws/frame` at or above `world_draws`) held for `settle` seconds. A run that stops moving (no log
output, the same title, or a window that does not respond, for `stall` seconds), exits, or does not reach the
world within `attempt_timeout` is killed and launched again, up to `restarts` times: a run stuck on a loading
screen is restarted, not debugged.

A route (ROUTES, `--route`, default "first-run") says which keys walk the menus and how the opening
cutscene is skipped: once the frame is the cutscene's, Enter is held instead of tapped, and the world
counts from the loading screen that follows.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("victory-prologue")

# Settings known to run well (dkdknight's play-lod.ps1, with this tree's card limits as defaults).
PRESETS: dict[str, dict[str, str]] = {
    "plain": {},
    "lod": {
        "VICTORY_LEVELS_ON_DEMAND": "1",
        "VICTORY_DISPLAY_MODE": "1920x1080",
        "VICTORY_RELAY": "2",
        "VICTORY_LATE_WAIT": "3",
        "VICTORY_WIDE_BACK": "1",
        "VICTORY_TABLES_MEMO": "2",
        "VICTORY_ASYNC_SUBMIT": "1",
        "VICTORY_FAST_COPY": "1",
        "VICTORY_ASK_SHARED": "1",
        "VICTORY_LIST_POOL_MOST": "1024",
        "VICTORY_LIST_POOL_MB": "2048",
        "VICTORY_INDIRECT_AMPLIFY": "1",
        "VICTORY_AMPLIFY_WAIT": "auto",
        "VICTORY_CLAMP_WAIT": "auto",
        "VICTORY_CARD_REHASH_RUNS": "1",
        "VICTORY_JOIN_COPIES": "1",
    },
}

# The merged build (levels on demand, depth bounds pipelines) keeps its own caches: a pipeline library of
# older builds holds pipelines under the same names, and runs that read it lost textures and lighting.
PRESET_CACHE = {"lod": "cache-lod"}

# Routes from the title's start screen to the world, by what the window title reports (draws/frame):
# the menus draw a few dozen times a frame, the opening cutscene hundreds, the world thousands.
#
#   keys             pressed in turn (short taps) while the menus are up
#   press_every      seconds between those taps
#   cutscene_draws   draws/frame from which the frame is the cutscene's (0: no cutscene step)
#   cutscene_hold_ms how long Enter (A) is held to skip it: the title wants the button held, and a
#                    hold has to span a few frames of a run that is still translating shaders
#   cutscene_wait    seconds of frames at or above cutscene_draws with no loading screen after a hold
#                    before they are taken for the world (a launch that has no cutscene to skip)
#
# "first-run" (recorded 2026-11-25, tested build): start screen (A) -> opening video (A skips) ->
# Subtitles (A) -> Brightness (A) -> Finish Setup (A) -> Legal Notices (each notice read for a few
# seconds: Down, Down, Down, A on Next) -> Legal Terms & Policies (A on each box, Down, A on Confirm;
# the user said to accept them) -> loading -> cutscene (hold A) -> loading -> the world.
# Enter and Down in turn walk all of those menus: Enter confirms where a screen has one button,
# Down moves to the next row where it has several, and no screen on the way loses anything to a Down.
ROUTES: dict[str, dict] = {
    "enter": {"keys": "Enter", "press_every": 4.0, "cutscene_draws": 0},
    "first-run": {
        "keys": "Enter,Down",
        "press_every": 2.0,
        "cutscene_draws": 150,
        "cutscene_hold_ms": 8000,
        "cutscene_wait": 60.0,
    },
}

TITLE = re.compile(r"(?P<fps>[\d.]+)\s*fps.*?(?P<draws>\d+)\s*draws/frame", re.I)

KEY_NAME = re.compile(r"^[A-Za-z0-9]$|^(Enter|Esc|Tab|Back|Up|Down|Left|Right|Menu|Start|View)$")


def preset_env(name: str) -> dict[str, str]:
    """The environment of a preset, with its own cache folder where it has one."""
    if name not in PRESETS:
        raise ValueError(f"no preset {name!r}; presets: {', '.join(PRESETS)}")
    env = dict(PRESETS[name])
    cache = PRESET_CACHE.get(name)
    if cache and "LOCALAPPDATA" in os.environ:
        folder = Path(os.environ["LOCALAPPDATA"]) / "victory" / cache
        (folder / "pipelines").mkdir(parents=True, exist_ok=True)
        env["VICTORY_CACHE_DIR"] = str(folder)
    return env


def check_keys(keys: str) -> list[str]:
    """`Enter+Down,Enter` -> ['Enter+Down', 'Enter']; raises on a key VIctory does not know."""
    out = []
    for chord in [c.strip() for c in keys.split(",") if c.strip()]:
        for key in chord.split("+"):
            if not KEY_NAME.match(key.strip()):
                raise ValueError(f"unknown key {key!r} (letters, digits, Enter, Esc, Tab, Back, Up, Down, Left, Right)")
        out.append(chord)
    if not out:
        raise ValueError("no keys")
    return out


def press(input_file: Path, chord: str, hold_ms: int = 150) -> None:
    """Append one press to the input file (VIctory presses it within 50 ms)."""
    input_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("a", encoding="ascii") as fh:
        fh.write(f"{chord} {int(hold_ms)}\n")


# --------------------------------------------------------------------------- windows

if sys.platform == "win32":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _EnumProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    _user32.EnumWindows.argtypes = [_EnumProc, wt.LPARAM]
    _user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    _user32.IsWindowVisible.argtypes = [wt.HWND]
    _user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    _user32.IsHungAppWindow.argtypes = [wt.HWND]


def _children(pid: int) -> set[int]:
    """`pid` and its descendants, by a process snapshot."""
    tree = {pid}
    if sys.platform != "win32":
        return tree

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
            ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    parents: dict[int, int] = {}
    if snap and snap != wt.HANDLE(-1).value:
        e = Entry()
        e.dwSize = ctypes.sizeof(Entry)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            parents[e.th32ProcessID] = e.th32ParentProcessID
            ok = k32.Process32NextW(snap, ctypes.byref(e))
        k32.CloseHandle(snap)
    grew = True
    while grew:
        grew = False
        for child, parent in parents.items():
            if parent in tree and child not in tree:
                tree.add(child)
                grew = True
    return tree


@dataclass
class Window:
    title: str = ""
    hung: bool = False
    fps: Optional[float] = None
    draws: Optional[int] = None


def window_of(pid: int) -> Optional[Window]:
    """The run's window with a frame counter in its title (VIctory's), else its first visible one."""
    if sys.platform != "win32":
        return None
    tree = _children(pid)
    found: list[tuple[int, str]] = []

    def cb(hwnd, _lp):
        owner = wt.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in tree and _user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(512)
            _user32.GetWindowTextW(hwnd, buf, 512)
            found.append((hwnd, buf.value))
        return True

    _user32.EnumWindows(_EnumProc(cb), 0)
    if not found:
        return None
    found.sort(key=lambda t: 0 if TITLE.search(t[1]) else 1)
    hwnd, title = found[0]
    w = Window(title=title, hung=bool(_user32.IsHungAppWindow(hwnd)))
    m = TITLE.search(title)
    if m:
        w.fps = float(m.group("fps"))
        w.draws = int(m.group("draws"))
    return w


def running_pid() -> Optional[int]:
    """The process id of a running victory.exe, if there is one."""
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq victory.exe", "/FO", "CSV", "/NH"], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[0].lower() == "victory.exe" and parts[1].isdigit():
            return int(parts[1])
    return None


def pid_alive(pid: int) -> bool:
    """Whether process `pid` is still running."""
    if sys.platform != "win32":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wt.HANDLE
    k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = wt.DWORD()
    ok = k32.GetExitCodeProcess(wt.HANDLE(handle), ctypes.byref(code))
    k32.CloseHandle(wt.HANDLE(handle))
    return bool(ok) and code.value == 259  # STILL_ACTIVE


# --------------------------------------------------------------------------- the pilot


@dataclass
class Settings:
    keys: str = "Enter"
    press_every: float = 4.0
    hold_ms: int = 150
    first_press_after: float = 20.0
    world_draws: int = 1500
    settle: float = 20.0
    stall: float = 240.0
    attempt_timeout: float = 1800.0
    restarts: int = 3
    stop_pressing_in_world: bool = True
    # The cutscene step of a route (see ROUTES); 0 draws: none.
    cutscene_draws: int = 0
    cutscene_hold_ms: int = 8000
    cutscene_wait: float = 60.0

    @classmethod
    def of_route(cls, route: str, **over) -> "Settings":
        """The settings of a route, with `over` (values that are not None) on top."""
        if route not in ROUTES:
            raise ValueError(f"no route {route!r}; routes: {', '.join(ROUTES)}")
        values = dict(ROUTES[route])
        values.update({k: v for k, v in over.items() if v is not None})
        return cls(**values)


@dataclass
class Report:
    state: str = "idle"  # idle, launching, waiting, in-world, failed, stopped
    phase: str = ""  # of a route with a cutscene step: menus, cutscene, loading
    attempt: int = 0
    started: float = field(default_factory=time.time)
    attempt_started: float = 0.0
    presses: int = 0
    pid: Optional[int] = None
    log: str = ""
    title: str = ""
    fps: Optional[float] = None
    draws: Optional[int] = None
    best_draws: int = 0
    world_since: Optional[float] = None
    reached_after: Optional[float] = None
    events: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"{stamp} {text}")
        del self.events[:-60]
        log.info(text)

    def text(self) -> str:
        d = asdict(self)
        events = d.pop("events")
        now = time.time()
        d["elapsed_s"] = round(now - self.started)
        d["attempt_elapsed_s"] = round(now - self.attempt_started) if self.attempt_started else 0
        return json.dumps(d, indent=1, default=str) + "\nevents:\n" + "\n".join(events[-25:])


class Pilot(threading.Thread):
    """Launches (through `launch`), presses, watches and relaunches until the world is on screen."""

    def __init__(
        self,
        settings: Settings,
        input_file: Path,
        launch: Callable[[], tuple[int, Path]],
        kill: Callable[[], None],
        alive: Callable[[], bool],
        attach: Optional[tuple[int, Path]] = None,
        on_world: Optional[Callable[[], None]] = None,
    ):
        super().__init__(daemon=True, name="victory-prologue")
        self.s = settings
        self.chords = check_keys(settings.keys)
        self.input_file = input_file
        self.launch_fn = launch
        self.kill_fn = kill
        self.alive_fn = alive
        self.attach = attach
        self.on_world = on_world
        self.report = Report()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _start_attempt(self) -> bool:
        r = self.report
        r.attempt += 1
        r.state = "launching"
        r.attempt_started = time.time()
        r.world_since = None
        r.best_draws = 0
        if self.attach and r.attempt == 1:
            r.pid, path = self.attach
            r.log = str(path)
            r.note(f"attempt {r.attempt}: following the running title (pid {r.pid})")
        else:
            self.kill_fn()
            try:
                r.pid, path = self.launch_fn()
            except Exception as exc:  # a failed build or a missing title: nothing to retry
                r.state = "failed"
                r.note(f"launch failed: {exc}")
                return False
            r.log = str(path)
            r.note(f"attempt {r.attempt}: launched pid {r.pid}, log {path}")
        r.state = "waiting"
        return True

    def run(self) -> None:
        r = self.report
        if not self._start_attempt():
            return
        last_press = time.time()
        chord_at = 0
        last_size = -1
        last_title = ""
        moved = time.time()
        cut = self.s.cutscene_draws
        # The cutscene step: frames at or above `cut` draws since, and when the step began.
        r.phase = "menus" if cut else ""
        cut_since: Optional[float] = None
        phase_since = time.time()
        while not self._stop.is_set():
            time.sleep(1.0)
            now = time.time()
            elapsed = now - r.attempt_started
            reason = ""
            if not self.alive_fn():
                reason = "the run exited"
            else:
                w = window_of(r.pid) if r.pid else None
                try:
                    size = Path(r.log).stat().st_size
                except OSError:
                    size = 0
                title = w.title if w else ""
                if size != last_size or title != last_title:
                    moved = now
                last_size, last_title = size, title
                if w:
                    r.title, r.fps, r.draws = w.title, w.fps, w.draws
                    if w.draws is not None:
                        r.best_draws = max(r.best_draws, w.draws)
                draws = w.draws if w else None
                if cut and draws is not None:
                    if r.phase == "menus":
                        if draws >= cut:
                            cut_since = cut_since or now
                            if now - cut_since >= 3.0:
                                r.phase, phase_since = "cutscene", now
                                r.note(f"cutscene frames ({draws} draws/frame): holding Enter")
                        else:
                            cut_since = None
                    elif r.phase == "cutscene" and draws < cut:
                        r.phase, phase_since = "loading", now
                        r.note("the cutscene gave way to a loading screen")
                    if r.phase == "cutscene" and now - last_press >= self.s.cutscene_hold_ms / 1000.0 + 1.5:
                        press(self.input_file, "Enter", self.s.cutscene_hold_ms)
                        r.presses += 1
                        last_press = now
                in_world = bool(draws is not None and draws >= self.s.world_draws)
                # With a cutscene step, frames that busy are the world's once the cutscene was
                # skipped, or after cutscene_wait of them with no loading screen (nothing to skip).
                if in_world and cut and r.phase != "loading":
                    in_world = r.phase == "cutscene" and now - phase_since >= self.s.cutscene_wait
                if in_world:
                    if r.world_since is None:
                        r.world_since = now
                        r.note(f"world frames: {w.draws} draws/frame at {w.fps} fps")
                    if now - r.world_since >= self.s.settle:
                        r.state = "in-world"
                        r.reached_after = round(now - r.started)
                        r.note(f"in the world after {r.reached_after}s ({r.attempt} attempt(s), {r.presses} presses)")
                        if self.on_world:
                            try:
                                self.on_world()
                            except Exception as exc:
                                r.note(f"on_world: {exc}")
                        return
                elif r.world_since is not None:
                    r.note(f"left world frames ({w.draws if w else '?'} draws/frame)")
                    r.world_since = None
                pressing = not (in_world and self.s.stop_pressing_in_world) and r.phase in ("", "menus")
                if pressing and elapsed >= self.s.first_press_after and now - last_press >= self.s.press_every:
                    chord = self.chords[chord_at % len(self.chords)]
                    chord_at += 1
                    press(self.input_file, chord, self.s.hold_ms)
                    r.presses += 1
                    last_press = now
                if w and w.hung and now - moved >= self.s.stall / 2:
                    reason = f"window not responding for {now - moved:.0f}s"
                elif now - moved >= self.s.stall:
                    reason = f"no log output and the same title for {now - moved:.0f}s"
                elif elapsed >= self.s.attempt_timeout:
                    reason = f"no world after {elapsed:.0f}s"
            if reason:
                r.note(f"attempt {r.attempt}: {reason}")
                if r.attempt > self.s.restarts:
                    r.state = "failed"
                    r.note("out of restarts")
                    return
                if not self._start_attempt():
                    return
                last_press, chord_at, last_size, last_title, moved = time.time(), 0, -1, "", time.time()
                r.phase, cut_since, phase_since = ("menus" if cut else ""), None, time.time()
        r.state = "stopped"
        r.note("stopped")


# --------------------------------------------------------------------------- standalone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=r"C:\Users\Corgan\Desktop\victory-corgi")
    ap.add_argument("--game-exe", required=True)
    ap.add_argument("--work", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--profile", default="diag", choices=["diag", "release"])
    ap.add_argument("--preset", default="lod", choices=list(PRESETS))
    ap.add_argument("--console", default="series-x", choices=["series-x", "series-s"])
    ap.add_argument("--route", default="first-run", choices=list(ROUTES))
    ap.add_argument("--keys", default=None, help="instead of the route's")
    ap.add_argument("--press-every", type=float, default=None, help="instead of the route's")
    ap.add_argument("--attach", action="store_true",
                    help="follow the victory.exe that is running (launched with VICTORY_INPUT_FILE=<work>\\input.txt) "
                         "instead of launching; a restart launches as usual")
    ap.add_argument("--world-draws", type=int, default=1500)
    ap.add_argument("--settle", type=float, default=20.0)
    ap.add_argument("--stall", type=float, default=240.0)
    ap.add_argument("--attempt-timeout", type=float, default=1800.0)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--watch-console", default="", help="a .cmd that follows the newest run log (opened per launch)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [prologue] %(message)s")

    repo, work = Path(a.repo), Path(a.work)
    runs = work / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    input_file = work / "input.txt"
    exe = repo / "target" / ("diag" if a.profile == "diag" else "release") / "victory.exe"
    state: dict = {"proc": None}

    def launch() -> tuple[int, Path]:
        if not exe.exists():
            raise FileNotFoundError(exe)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = runs / f"{stamp}-{a.profile}.log"
        env = {k: v for k, v in os.environ.items() if not k.startswith("VICTORY_")}
        env.update(preset_env(a.preset))
        env["VICTORY_LOG"] = "info"
        env["VICTORY_INPUT_FILE"] = str(input_file)
        input_file.write_text("", encoding="ascii")
        handle = open(path, "wb")
        state["proc"] = subprocess.Popen(
            [str(exe), "run", a.game_exe, "--execute", "--console", a.console],
            cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT,
        )
        if a.watch_console:
            time.sleep(1.0)
            subprocess.Popen(["cmd.exe", "/c", "start", "", "cmd.exe", "/k", a.watch_console])
        return state["proc"].pid, path

    def kill() -> None:
        p = state["proc"]
        if p is not None and p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        subprocess.run(["taskkill", "/F", "/IM", "victory.exe"], capture_output=True)

    attach = None
    if a.attach:
        pid = running_pid()
        logs = sorted(runs.glob("*.log"), key=lambda f: f.stat().st_mtime)
        if pid is None or not logs:
            print("nothing to follow: victory.exe is not running (or has no run log); launching")
        else:
            attach = (pid, logs[-1])
            state["attached"] = pid

    def alive() -> bool:
        p = state["proc"]
        if p is not None:
            return p.poll() is None
        return state.get("attached") is not None and pid_alive(state["attached"])

    s = Settings.of_route(a.route, keys=a.keys, press_every=a.press_every, world_draws=a.world_draws,
                          settle=a.settle, stall=a.stall, attempt_timeout=a.attempt_timeout, restarts=a.restarts)
    pilot = Pilot(s, input_file, launch, kill, alive, attach=attach)
    pilot.start()
    try:
        while pilot.is_alive():
            pilot.join(30)
            r = pilot.report
            log.info("state %s attempt %s draws %s fps %s presses %s", r.state, r.attempt, r.draws, r.fps, r.presses)
    except KeyboardInterrupt:
        pilot.stop()
        pilot.join(5)
    (work / "prologue-report.txt").write_text(pilot.report.text(), encoding="utf-8")
    print(pilot.report.text())
    return 0 if pilot.report.state == "in-world" else 1


if __name__ == "__main__":
    sys.exit(main())
