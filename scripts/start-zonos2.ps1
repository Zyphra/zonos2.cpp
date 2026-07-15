# start-zonos2.ps1 — download the models if missing, launch zonos2-server, open the browser.
# Launched by start-zonos2.bat (double-click) with -ExecutionPolicy Bypass; mirrors
# scripts/start-zonos2.sh — keep the two in sync.
param(
    [string]$Quant,
    [switch]$Cpu,
    [switch]$Gpu,
    [string]$BindHost,
    [int]$Port,
    [switch]$Yes,
    [switch]$NoBrowser,
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'

$GPU_DEFAULT = 'cpu'   # CI stamps this to 'gpu' in the vulkan release archive
$ScriptDir = $PSScriptRoot

if (-not $Quant)    { $Quant    = if ($env:ZONOS2_QUANT)     { $env:ZONOS2_QUANT }     else { 'q6_k' } }
$ModelDir           = if ($env:ZONOS2_MODEL_DIR)             { $env:ZONOS2_MODEL_DIR } else { Join-Path $ScriptDir 'models' }
$BaseUrl            = if ($env:ZONOS2_BASE_URL)              { $env:ZONOS2_BASE_URL }  else { 'https://huggingface.co/Zyphra/ZONOS2-GGUF/resolve/main' }
if (-not $BindHost) { $BindHost = if ($env:ZONOS2_HOST)      { $env:ZONOS2_HOST }      else { '127.0.0.1' } }
if (-not $Port)     { $Port     = if ($env:ZONOS2_PORT)      { [int]$env:ZONOS2_PORT } else { 1919 } }
$ServerBin          = if ($env:ZONOS2_SERVER_BIN)            { $env:ZONOS2_SERVER_BIN } else { Join-Path $ScriptDir 'zonos2-server.exe' }
if ($env:ZONOS2_ASSUME_YES) { $Yes = $true }
if ($env:ZONOS2_NO_BROWSER) { $NoBrowser = $true }
$FfmpegUrl = $env:ZONOS2_FFMPEG_URL   # override with a single-binary .exe URL (used by tests); default is BtbN LGPL
$FfmpegDir = Join-Path $ModelDir 'bin'
$FfmpegBin = Join-Path $FfmpegDir 'ffmpeg.exe'
$GpuMode = $GPU_DEFAULT
if ($Cpu) { $GpuMode = 'cpu' }
if ($Gpu) { $GpuMode = 'gpu' }

if ($Help) {
    Write-Host @"
usage: start-zonos2.bat [options] [extra zonos2-server args]
  -Quant Q     model quant to fetch/serve (q4_k 4.9GB, q5_k 5.8GB, q6_k 6.8GB,
               q8_0 8.5GB, f16 15.3GB; default q6_k)
  -Cpu | -Gpu  backend (archive default: $GPU_DEFAULT)
  -BindHost H  bind address (default 127.0.0.1)
  -Port P      port (default 1919)
  -Yes         don't ask before downloading
  -NoBrowser   don't open the web UI
env overrides: ZONOS2_QUANT ZONOS2_MODEL_DIR ZONOS2_BASE_URL ZONOS2_HOST ZONOS2_PORT
               ZONOS2_SERVER_BIN ZONOS2_ASSUME_YES ZONOS2_NO_BROWSER ZONOS2_FFMPEG_URL
ffmpeg (voice cloning only) is fetched into <ModelDir>\bin if not already there or on PATH.
"@
    exit 0
}

# not Write-Error: under ErrorActionPreference=Stop it throws before reaching exit
function Fail([string]$msg) { Write-Host "start-zonos2: error: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $ServerBin)) {
    Fail "zonos2-server.exe not found at $ServerBin (run this script from the extracted release archive, or set ZONOS2_SERVER_BIN)"
}

$Sizes = @{
    'zonos2-f16.gguf' = '15.3'; 'zonos2-q8_0.gguf' = '8.5'; 'zonos2-q6_k.gguf' = '6.8'
    'zonos2-q5_k.gguf' = '5.8'; 'zonos2-q4_k.gguf' = '4.9'; 'dac.gguf' = '0.25'; 'spk-encoder.gguf' = '0.02'
}
function Get-SizeGB([string]$name) { if ($Sizes.ContainsKey($name)) { $Sizes[$name] } else { '?' } }

# Segmented parallel download: one HEAD resolves the exact size + final CDN URL (HF's resolve/
# endpoint 302s to a Xet CDN that honors Range), then N concurrent range requests are stitched
# back together — the trick hf_transfer uses to beat single-stream HF (~2x+). Falls back to a
# single stream on any hiccup. Tunable via ZONOS2_DL_CONNECTIONS (default 8, max 16).
function Download-Segmented([string]$name, [string]$url, [string]$dst) {
    $conns = 8
    if ($env:ZONOS2_DL_CONNECTIONS) {
        $parsed = 0
        if ([int]::TryParse($env:ZONOS2_DL_CONNECTIONS, [ref]$parsed) -and $parsed -gt 0) { $conns = $parsed }
    }
    if ($conns -lt 1) { $conns = 1 }
    if ($conns -gt 16) { $conns = 16 }
    if ($conns -le 1) { return $false }
    # resolve final URL + exact size from a single redirect-following HEAD (-match is case-insensitive)
    $head = & curl.exe -sIL $url
    if ($LASTEXITCODE -ne 0) { return $false }
    $size = [int64]0; $eff = $url
    foreach ($line in $head) {
        $l = $line.TrimEnd("`r")
        if ($l -match '^content-length:\s*(\d+)') { $v = [int64]$Matches[1]; if ($v -gt $size) { $size = $v } }
        elseif ($l -match '^location:\s*(\S+)')    { $eff = $Matches[1] }
    }
    if ($size -lt 33554432) { return $false }   # < 32 MB: not worth segmenting
    $seg = [int64]([math]::Floor($size / $conns))
    Get-ChildItem "$dst.part*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "downloading $name ($(Get-SizeGB $name) GB, $conns connections) ..."
    $procs = @()
    for ($k = 0; $k -lt $conns; $k++) {
        $s = [int64]$k * $seg
        if ($k -eq ($conns - 1)) { $e = $size - 1 } else { $e = [int64]($k + 1) * $seg - 1 }
        $procs += Start-Process -FilePath curl.exe -PassThru -NoNewWindow -ArgumentList `
            @('-sfL', '--retry', '3', '--range', "$s-$e", '-o', "$dst.part$k", $eff)
    }
    $ok = $true
    foreach ($p in $procs) { $p.WaitForExit(); if ($p.ExitCode -ne 0) { $ok = $false } }
    if (-not $ok) {
        Get-ChildItem "$dst.part*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        return $false
    }
    $out = [System.IO.File]::Create("$dst.part")   # concatenate segments in order, freeing each
    try {
        for ($k = 0; $k -lt $conns; $k++) {
            $part = "$dst.part$k"
            $in = [System.IO.File]::OpenRead($part)
            try { $in.CopyTo($out) } finally { $in.Close() }
            Remove-Item -Force $part -ErrorAction SilentlyContinue
        }
    } finally { $out.Close() }
    if ((Get-Item "$dst.part").Length -ne $size) {
        Remove-Item -Force "$dst.part" -ErrorAction SilentlyContinue; return $false
    }
    Move-Item -Force "$dst.part" $dst
    return $true
}

# curl -C - --fail exits 33 when the .part is already complete (HTTP 416): restart clean once.
function Download-One([string]$name) {
    $url = "$BaseUrl/$name"
    $dst = Join-Path $ModelDir $name
    if (Download-Segmented $name $url $dst) { return $true }   # fast path; falls through on failure
    Write-Host "downloading $name ($(Get-SizeGB $name) GB) ..."
    & curl.exe -L --fail --retry 3 -C - --progress-bar -o "$dst.part" $url
    if ($LASTEXITCODE -eq 33) {
        Remove-Item -Force "$dst.part" -ErrorAction SilentlyContinue
        & curl.exe -L --fail --retry 3 --progress-bar -o "$dst.part" $url
    }
    if ($LASTEXITCODE -ne 0) { return $false }
    Move-Item -Force "$dst.part" $dst
    return $true
}

# Fetch a static ffmpeg into $FfmpegDir — voice cloning only; basic TTS never uses it.
# Windows uses the BtbN LGPL build (a zip nesting the exe at ffmpeg-*/bin/ffmpeg.exe);
# override with ZONOS2_FFMPEG_URL (single-binary .exe URL).
function Download-Ffmpeg {
    New-Item -ItemType Directory -Force -Path $FfmpegDir | Out-Null
    if ($FfmpegUrl) {
        Write-Host "downloading ffmpeg (for voice cloning) ..."
        & curl.exe -L --fail --retry 3 --progress-bar -o "$FfmpegBin.part" $FfmpegUrl
        if ($LASTEXITCODE -ne 0) { return $false }
        Move-Item -Force "$FfmpegBin.part" $FfmpegBin
        return $true
    }
    $url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-lgpl-7.1.zip'
    $zip = Join-Path $FfmpegDir 'ffmpeg-dl.zip'
    $ex  = Join-Path $FfmpegDir '.extract'
    Write-Host "downloading ffmpeg (~90 MB, for voice cloning) ..."
    & curl.exe -L --fail --retry 3 --progress-bar -o $zip $url
    if ($LASTEXITCODE -ne 0) { return $false }
    if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
    Expand-Archive -Path $zip -DestinationPath $ex -Force
    $found = Get-ChildItem -Path $ex -Recurse -Filter ffmpeg.exe |
             Where-Object { $_.FullName -match '[\\/]bin[\\/]' } | Select-Object -First 1
    if (-not $found) { Remove-Item -Recurse -Force $zip, $ex -ErrorAction SilentlyContinue; return $false }
    Move-Item -Force $found.FullName $FfmpegBin
    Remove-Item -Recurse -Force $zip, $ex -ErrorAction SilentlyContinue
    return (Test-Path $FfmpegBin)
}

$Backbone = "zonos2-$Quant.gguf"
$Missing = @()
foreach ($f in @($Backbone, 'dac.gguf', 'spk-encoder.gguf')) {
    if (-not (Test-Path (Join-Path $ModelDir $f))) { $Missing += $f }
}

# ffmpeg is only needed for voice cloning: prefer our own copy, then a system ffmpeg
# on PATH, else fetch it alongside the models.
$needFfmpeg = $false
if (Test-Path $FfmpegBin) {
    $env:ZONOS2_FFMPEG = $FfmpegBin
} elseif ($env:ZONOS2_SKIP_PATH_FFMPEG -ne '1' -and (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue)) {
    # a system ffmpeg is on PATH; the server resolves it there
} else {
    $needFfmpeg = $true
}

if ($Missing.Count -gt 0 -or $needFfmpeg) {
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
        Fail "curl.exe is required to download models (it ships with Windows 10 1803+) — or place the .gguf files in $ModelDir yourself (see README)"
    }
    Write-Host "The following will be downloaded into ${ModelDir}:"
    foreach ($f in $Missing) { Write-Host "  $f  ($(Get-SizeGB $f) GB)  from $BaseUrl" }
    if ($needFfmpeg) { Write-Host "  ffmpeg  (~90 MB, voice cloning only)" }
    if (-not $Yes) {
        $ans = Read-Host 'Download now? [Y/n]'
        if ($ans -and $ans -notmatch '^[yY]') { Write-Host 'aborted.'; exit 1 }
    }
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    foreach ($f in $Missing) {
        if (-not (Download-One $f)) {
            if ($f -eq 'spk-encoder.gguf') {
                Write-Warning "failed to download $f — continuing without voice-clone upload support"
            } else {
                Fail "failed to download $f from $BaseUrl/$f"
            }
        }
    }
    if ($needFfmpeg) {
        if (Download-Ffmpeg) { $env:ZONOS2_FFMPEG = $FfmpegBin }
        else { Write-Warning "ffmpeg download failed — voice cloning disabled (basic TTS still works)" }
    }
}

# The UI, default voices, and emotion directions ship next to this script in release
# archives; in a source checkout the script lives in scripts\ with them one level up.
# Pass them explicitly because the server defaults are cwd-relative and a
# double-clicked .bat may not run from the archive dir.
function Get-Sibling([string]$name) {
    $p = Join-Path $ScriptDir $name
    if (Test-Path $p) { $p } else { Join-Path $ScriptDir "..\$name" }
}

$srvArgs = @((Join-Path $ModelDir $Backbone),
             '--dac', (Join-Path $ModelDir 'dac.gguf'),
             '--host', $BindHost, '--port', $Port,
             '--ui', (Get-Sibling 'web\tts_ui.html'))
if (Test-Path (Get-Sibling 'default_voices')) {
    $srvArgs += @('--tts-default-voices-dir', (Get-Sibling 'default_voices'))
}
if (Test-Path (Get-Sibling 'emotion_directions')) {
    $srvArgs += @('--tts-emotion-directions-dir', (Get-Sibling 'emotion_directions'))
}
if (Test-Path (Join-Path $ModelDir 'spk-encoder.gguf')) {
    $srvArgs += @('--spk', (Join-Path $ModelDir 'spk-encoder.gguf'))
}
if ($GpuMode -eq 'gpu') { $srvArgs += '--gpu' }
if ($ExtraArgs) { $srvArgs += $ExtraArgs }

# PS 5.1 joins -ArgumentList elements with spaces without quoting; quote them ourselves
$quotedArgs = $srvArgs | ForEach-Object { if ("$_" -match '\s') { '"{0}"' -f $_ } else { "$_" } }
$proc = Start-Process -FilePath $ServerBin -ArgumentList $quotedArgs -NoNewWindow -PassThru
try {
    $pollHost = if ($BindHost -eq '0.0.0.0' -or $BindHost -eq '::') { '127.0.0.1' } else { $BindHost }
    $url = "http://${pollHost}:$Port/"
    $ready = $false
    for ($i = 0; $i -lt 240; $i++) {
        if ($proc.HasExited) { Fail 'server exited during startup' }
        # -s without -S: stderr output under redirection throws NativeCommandError on PS 5.1
        & curl.exe -fs "${url}health" > $null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { Fail "server did not become ready at $url" }

    Write-Host "zonos2: ready -> $url"
    if (-not $NoBrowser) { Start-Process $url }

    Write-Host 'Press Ctrl-C (or close this window) to stop the server.'
    Wait-Process -Id $proc.Id
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}
