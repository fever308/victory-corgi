# Launch the title on branch feature/levels-on-demand (feature/watch-batch with
# origin/main merged in, plus fxceless/victory-fork PR #795): the same settings
# as play-best.ps1, plus levels on demand (only the mip levels the title has
# loaded take card memory).
#   powershell -ExecutionPolicy Bypass -File "<work>\tools\play-lod.ps1"
#   ... -NoLod       same build, levels on demand off (A/B against it)
#   ... -Diag        diagnostics build: memory lines every few seconds
# Log: <work>\profiles\lod-run.log (lod-off-run.log with -NoLod).
#
# First launch: the shader cache is now keyed on the translator itself
# (upstream), not on VICTORY_SHADER_CACHE_KEY: every shader is translated
# again once (several minutes before the world is whole). Measure on the
# second launch.
# Close OBS, Wallpaper Engine, browsers, Discord and launchers first: at idle
# they held 4.3 GB of the card's 16 GB.
param([switch]$NoLod, [switch]$Diag)

$out = '<work>\profiles'
New-Item -ItemType Directory -Force $out | Out-Null
Set-Location '<repo>'

# Own cache directory (victory-lod-data): the pipeline library of the older
# builds held pipelines under the same names without the depth bounds test,
# and runs that read it lost textures and lighting.
$env:VICTORY_CONFIG = '<repo>\victory-lod.toml'
$env:VICTORY_DISPLAY_MODE = '1920x1080'
$env:VICTORY_RELAY = '2'
$env:VICTORY_LATE_WAIT = '3'
$env:VICTORY_WIDE_BACK = '1'
$env:VICTORY_CARD_BUFFERS_MIN_MB = '1'
$env:VICTORY_CARD_BUFFERS_MB = '1024'

# CPU side (as play-best.ps1).
$env:VICTORY_TABLES_MEMO = '2'
$env:VICTORY_ASYNC_SUBMIT = '1'
$env:VICTORY_FAST_COPY = '1'
$env:VICTORY_ASK_SHARED = '1'
$env:VICTORY_LIST_POOL_MOST = '1024'
$env:VICTORY_LIST_POOL_MB = '2048'

# Card side (as play-best.ps1).
$env:VICTORY_INDIRECT_AMPLIFY = '1'
$env:VICTORY_AMPLIFY_WAIT = 'auto'
$env:VICTORY_CLAMP_WAIT = 'auto'
$env:VICTORY_CARD_REHASH_RUNS = '1'
$env:VICTORY_JOIN_COPIES = '1'

# New: card memory for the mip levels the title holds only. (Upstream's
# VICTORY_CARD_ROOM_FOR_BUFFERS is on by default in this build: the texture
# queue keeps room for the card copies of title buffers.)
if ($NoLod) {
    Remove-Item Env:VICTORY_LEVELS_ON_DEMAND -ErrorAction SilentlyContinue
    $log = "$out\lod-off-run.log"
} else {
    $env:VICTORY_LEVELS_ON_DEMAND = '1'
    $log = "$out\lod-run.log"
}

if ($Diag) {
    $exe = '<repo>\target\diag\release\victory-lod-diag.exe'
} else {
    $exe = '<repo>\target\release\victory-lod.exe'
}

& $exe run '<title dir>\title.exe' --execute --console series-x *> $log
