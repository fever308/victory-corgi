# the title frame rate under VIctory: findings so far (2026-11-25)

Notes from dkdknight's test series, branch `feature/watch-batch` of
`dkdknight/victory-fork`. Written with the help of an AI assistant (Claude
Code); every number is from our own runs. No game data is in this
repository: shader hashes below are only the names the logs print.

Setup: the title (Series X title build), RTX 5080 (16 GB), Windows 11,
display mode 1920x1080, `VICTORY_RELAY=2 VICTORY_LATE_WAIT=3
VICTORY_WIDE_BACK=1 VICTORY_CARD_BUFFERS_MIN_MB=1`. Reference scene: one
passage of the opening cutscene with about 4,950 draws a frame.

## Where it stands

| | Before this series | Best settings now |
|---|---|---|
| Full picture, `--console series-x` | ~3.5 fps | ~6.5 fps |
| Indirect mesh draws on the card | 137 ms/frame | ~22-25 ms/frame |
| All title commands on the card | ~200 ms/frame | ~85-100 ms/frame |
| `--console series-s` | lost the device while loading | ~8.6 fps, black screen |

The title itself is not the limit: with every draw, dispatch, refresh,
copy-back and upload dropped (`VICTORY_BARE=127`) it runs at 30.0 fps in
the open world. 30 fps with the picture needs under 33 ms of card work a
frame; a PIX capture of one frame shows 117-136 ms.

Best settings: `tools/perf/play-best.ps1` (paths are local ones; adapt).

## What was found

1. **The global UAV barrier before every indirect draw was the largest
   single cost.** About 5,000 pipeline drains a frame. With one-record
   indirect mesh draws going through an amplification stage
   (`VICTORY_INDIRECT_AMPLIFY=1`: the stage reads and bounds the record on
   the card, then `DispatchMesh`) and the barrier recorded only after a
   shader that may have written (`VICTORY_AMPLIFY_WAIT=auto`, about 1% of
   the draws), lists in flight on the card went from 86 ms to ~25 ms a
   frame. `VICTORY_CLAMP_WAIT=auto` does the same for the clamp pass of the
   other indirect commands (no measurable gain by itself).
2. **Measurement trap.** The indirect arguments are written by the title's
   compute shaders (GPU culling). With compute dropped the mesh draws ask
   for 0-127 groups a frame instead of 15,000-21,000, so a run without
   compute says nothing about geometry cost.
3. **There is no per-command floor.** The PIX capture gives medians of
   1.2 µs for `DispatchMesh`, 0.2 µs for `Dispatch`, 3.5 µs for
   `ExecuteIndirect`. Our own timestamp timing (`VICTORY_GPU_TIMING`) had
   suggested 5-7 µs each; that was an artefact. About 80 commands of
   18,000 hold more than 80 ms.
4. **Some compute shaders are ~150 times slower per thread than others**,
   and at least one of them is trivial code:

   | compute shader | threads | time | per thread |
   |---|---|---|---|
   | `269b802f878c24e1` | 47,360 | 6.25 ms | ~130 ns |
   | `570f1473f4641b30` | 16,448 | 2.34 ms | ~140 ns |
   | `969c1cef8b783b39` | 3,328 | 0.63 ms | ~190 ns |
   | `2c3592e9bbad543f` | 921,600 | 0.73 ms | ~0.8 ns |
   | `a7eeed0646be1ab7` | 1,497,600 | 0.60 ms | ~0.4 ns |

   `570f1473f4641b30` translated is 179 lines of IR: zero stores into a few
   heap buffers, no loop. So the cost is where its buffers live, not its
   code. `shadow.rs` already says host copies in the PC's memory are read
   and written over the bus; in the reference scene 3.5 GB of "windows
   outside buffers" and every title buffer under 1 MiB are such copies
   (`card_limits` clamps the minimum to 1 MiB), and 5.6 GB of shared memory
   is in use. **Not yet shown directly**: which buffers these shaders touch.
5. **The title renders at 2560x1440 internally** although the display mode
   is 1920x1080. Shadow maps are 2048x8192 and take about 1,985 draws a
   frame.
6. **`--console series-s` loads once R9G9B9E5 render targets get a stand-in**
   (`VICTORY_RGB9E5_TARGET=1`, R11G11B10_FLOAT). It then runs ~8.6 fps with
   the card 71% busy, but the screen stays black: the title uses FSR4 there
   (`amd_fidelityfx_upscaler_xbox.dll`) and its 30 passes fail in
   `DeserializeComputePipelineStateX` (`E_NOTIMPL`), with no fallback.
7. **CPU side.** Lock convoys on global mutexes explained the idle CPU:
   `JOURNAL`, `GUARDS`, descriptor and in-place `VIEWS`, `BUFFERS`, `LOOSE`
   became `RwLock`s with read fast paths; `kept` and `RETIRED` are sharded;
   `HostHeap::mirror` has a lock-free check; clean guard blocks are also
   kept as bits read without the lock, and the guard lock is held only for
   the blocks that are not clean. `GetWriteWatch` on view memory is skipped
   (it was refused 25,000 times a frame). Submissions run on a thread of
   their own (`VICTORY_ASYNC_SUBMIT=1`). Preparing a frame is still
   ~75-90 ms, about half of it copying ~130 MB of title memory.

## Ruled out (measured)

- In-place title memory on a discrete card: card time 77 ms → 538-1,170 ms.
- Reading indirect arguments on the CPU: flicker (they are shader-written).
- Mesh shader maxima: all are already 128/128/128.
- `VICTORY_INDIRECT_CLAMP=0`: the title's own fatal path while the story
  loads (reported on fxceless/victory-fork PR #169).
- `VICTORY_INDIRECT_WINDOW=1`: no gain (and 50,000-96,000 groups asked
  instead of ~21,000).
- Runaway translated loops: `VICTORY_SHADER_LOOP_LIMIT=64` changes nothing.
- Group-sync barriers of wave64 run as two wave32: `VICTORY_GROUP_SYNC=0`
  gives no gain (slower).
- Removing the submitting thread's wait (`VICTORY_LATE_FILL_AHEAD=1` with
  the merge switches): the wait goes, the card goes to 95% busy, same fps.
- Copy-back alone: without it (`VICTORY_BARE=8`) 7.8-8.6 fps, card 98%.
- Windows in the card's memory (`VICTORY_CARD_WINDOWS_MB=4096`,
  `VICTORY_CARD_BUFFERS_MB=2048`): no gain; with 4.4 GB of card buffers
  the frame got worse (card wait 70 → 155 ms) because the card was at 10.9
  of an 11.5 GB budget. `VICTORY_TEXTURE_BUDGET_MB=3500` did not bring the
  host textures under 6.4 GB. So this one is untested rather than refuted.

## Card memory and levels on demand (branch `feature/levels-on-demand`)

Branch `feature/levels-on-demand` is `feature/watch-batch` with
fxceless/victory-fork `main` merged in (levels on demand, one host copy per
placed texture, the depth bounds test, card room kept for buffer copies)
and that repository's PR #795. Launch: `tools/perf/play-lod.ps1`.

Why the card's memory was full, from the code:

- Every texture is one committed resource of its whole mip chain, even when
  the title has only streamed in its small levels; placed textures got one
  host copy per address and description; nothing is aliased.
- Textures may fill the card up to 90% of the budget, card copies of
  buffers and windows only while it is under 80%: once textures filled it,
  buffers under 1 MiB, ~3.5 GB of windows and the indirect clamp's budget
  counter stayed in the PC's memory, read over the bus.
- At idle, other programs (OBS, Wallpaper Engine, browsers, Discord,
  launchers) held 4.3 GB of the 16 GB card: the budget was ~11.4 GB.

Measured, same opening-cutscene passage (~5,000 draws), RTX 5080:

| Build and settings | fps | card per frame | card busy | card memory in use |
|---|---|---|---|---|
| `feature/watch-batch` (old best) | 7.3 | ~58 ms | ~75% | - |
| merged, cache shared with older builds, levels on demand off | ~4.3 | ~135 ms | ~81% | 10.45 / 11.37 GB |
| merged, cache shared with older builds, levels on demand on | ~4.8 | ~105 ms | ~72% | 8.98 / 11.46 GB |
| merged, own cache, levels on demand on | 5.4-6.4 | 65-84 ms | 42-72% | 9.3 / 12.4 GB |

- Levels on demand leaves ~3 GB of the card unallocated (levels the title
  does not hold) and is worth ~10% by itself.
- **The merged build must not share its cache directory with older builds**
  (`victory-lod.toml`, `cache_dir` of its own): with the old pipeline
  library, runs after the first lost textures and lighting and took ~30 ms
  more of the card. With its own cache, textures and lighting are right
  (checked by eye) after the first launch translates everything again.
- The old best build is faster but draws less (no depth bounds test,
  missing textures): the numbers are not of the same picture.
- With the card under 75% busy the frame now follows the submitting
  thread: ~52% of its time waits for flights to land (`relay_wait`, the
  "title memory read" settles, ~57 ms a frame), ~17% refreshes host copies
  (`Shadow::update_free`, partly in contention on the guard lock).

## Open questions

- Which buffers the slow compute shaders read and write, and whether
  putting exactly those in the card's memory removes the cost.
- Removing the submitting thread's wait now that the card has room
  (`VICTORY_LATE_FILL_AHEAD=1` with the merge switches: no gain when the
  card was the limit, to be measured again).
- FSR4's precompiled pipelines for the Series S mode.

## Tools (`tools/perf/`)

- `sampler.py <image> <pdb dir> [secs] [ms] [filter|0] [chain names...]`:
  stack sampler by suspend + `StackWalk64`, no admin rights. Filter
  `Host::submit` for the submitting thread, `relay_worker` for the landing
  thread; chain names such as `lock_contended` print who waits on what.
- `joinlog.py in out`, `auto5.sh <log> <exe>`: wait for the reference
  passage and print frame lines, card occupation and settles.
- **PIX capture that completes**: `VICTORY_PIX=1`,
  `VICTORY_PIX_TRIGGER=<file>`, `VICTORY_PARK_NULL_WRITE=1` (the title's
  watchdog fires its fatal null write ~36 s into a capture; this parks that
  thread), and in a diagnostics build `VICTORY_PIX_MARKERS=1` to name every
  command. Pipelines are remade under the capturer for about six minutes.
  `pixwait4.sh` triggers when the scene is whole. Analysis needs Windows
  developer mode:
  `MSYS_NO_PATHCONV=1 pixtool open-capture X.wpix save-event-list e.csv "--counters=*Duration*" "--counters=*Invocations"`,
  then `pixagg.py e.csv 45`.
- `VICTORY_SHADER_SAVE=<dir>` saves each used shader as given and as
  translated (also on cache hits); `dxc -dumpbin` reads the `.dxil`. Keep
  these local: they are game data.
- `VICTORY_BARE=<bits>` (1 mesh draws, 2 compute, 4 refreshes, 8 copy-back,
  16 other draws, 32 texture uploads, 64 fills and copies) and
  `VICTORY_MESH_TEST=skip|scissor|noexec` for isolation runs.
- `VICTORY_SHADER_CACHE_KEY=<name>` shares one shader cache between builds
  that leave the translator alone (`amp-2026-11-25` for builds with the
  amplification stage).
