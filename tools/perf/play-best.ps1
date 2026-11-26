# Launch the title with the best settings found so far (2026-11-25, branch
# feature/watch-batch of dkdknight/victory-fork): about 6.5 fps with the full
# picture in the opening cutscene on an RTX 5080, against about 3.5 before.
#   powershell -ExecutionPolicy Bypass -File "<work>\tools\play-best.ps1"
# The log goes to <work>\profiles\best-run.log.

$out = '<work>\profiles'
New-Item -ItemType Directory -Force $out | Out-Null
Set-Location '<repo>'

# As before this series.
$env:VICTORY_CONFIG = '<repo>\victory.toml'
$env:VICTORY_DISPLAY_MODE = '1920x1080'
$env:VICTORY_RELAY = '2'
$env:VICTORY_LATE_WAIT = '3'
$env:VICTORY_WIDE_BACK = '1'
$env:VICTORY_CARD_BUFFERS_MIN_MB = '1'
$env:VICTORY_CARD_BUFFERS_MB = '1024'

# One shader cache for every build with this translator (the amplification
# stage changes the mesh shaders: not the same cache as base-2026-11-25).
$env:VICTORY_SHADER_CACHE_KEY = 'amp-2026-11-25'

# CPU side: memoised tables, submissions on a thread of their own, title
# memory read in-process, one write-watch asking shared by the submissions
# given before it, more host lists kept for reuse.
$env:VICTORY_TABLES_MEMO = '2'
$env:VICTORY_ASYNC_SUBMIT = '1'
$env:VICTORY_FAST_COPY = '1'
$env:VICTORY_ASK_SHARED = '1'
$env:VICTORY_LIST_POOL_MOST = '1024'
$env:VICTORY_LIST_POOL_MB = '2048'

# Card side: one-record indirect mesh draws through an amplification stage,
# the wait for earlier shader writes only when a shader may have written,
# our hash passes and copies joined.
$env:VICTORY_INDIRECT_AMPLIFY = '1'
$env:VICTORY_AMPLIFY_WAIT = 'auto'
$env:VICTORY_CLAMP_WAIT = 'auto'
$env:VICTORY_CARD_REHASH_RUNS = '1'
$env:VICTORY_JOIN_COPIES = '1'

& '.\target\release\victory-best.exe' run '<title dir>\title.exe' --execute --console series-x *> "$out\best-run.log"
