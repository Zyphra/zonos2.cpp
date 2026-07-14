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
               ZONOS2_SERVER_BIN ZONOS2_ASSUME_YES ZONOS2_NO_BROWSER
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

# curl -C - --fail exits 33 when the .part is already complete (HTTP 416): restart clean once.
function Download-One([string]$name) {
    $url = "$BaseUrl/$name"
    $dst = Join-Path $ModelDir $name
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

$Backbone = "zonos2-$Quant.gguf"
$Missing = @()
foreach ($f in @($Backbone, 'dac.gguf', 'spk-encoder.gguf')) {
    if (-not (Test-Path (Join-Path $ModelDir $f))) { $Missing += $f }
}

if ($Missing.Count -gt 0) {
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
        Fail "curl.exe is required to download models (it ships with Windows 10 1803+) — or place the .gguf files in $ModelDir yourself (see README)"
    }
    Write-Host "The following models are missing from $ModelDir and will be downloaded:"
    foreach ($f in $Missing) { Write-Host "  $f  ($(Get-SizeGB $f) GB)" }
    Write-Host "  from $BaseUrl"
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
