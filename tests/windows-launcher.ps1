# Run after the workflow's Prepare Windows launchers step, under both
# powershell.exe (Windows PowerShell 5.1) and pwsh (PowerShell 7).
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$powerShell = (Get-Process -Id $PID).Path
$cases = @(
    @{ Path = 'scripts/start-zonos2.ps1'; GpuDefault = 'cpu' },
    @{ Path = 'dist/start-zonos2.ps1'; GpuDefault = 'gpu' }
)

foreach ($case in $cases) {
    $path = Join-Path $repoRoot $case.Path
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -lt 3 -or $bytes[0] -ne 0xEF -or $bytes[1] -ne 0xBB -or $bytes[2] -ne 0xBF) {
        throw "$($case.Path) must be UTF-8 with BOM for Windows PowerShell 5.1"
    }

    # ParseFile must read the on-disk encoding, as the batch launcher does.
    # Decoding with Get-Content -Encoding UTF8 first would hide this regression.
    $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$null, [ref]$parseErrors) | Out-Null
    if ($parseErrors) {
        $parseErrors | Out-String | Write-Host
        throw "$($case.Path) has parse errors under PowerShell $($PSVersionTable.PSVersion)"
    }

    # Use a child of the current interpreter: -Help exits the launcher and must
    # succeed without a server binary, models, downloads, or a browser.
    $helpOutput = & $powerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $path -Help 2>&1
    if ($LASTEXITCODE -ne 0) {
        $helpOutput | Out-String | Write-Host
        throw "$($case.Path) -Help failed with exit code $LASTEXITCODE"
    }
    $helpText = $helpOutput -join "`n"
    if ($helpText -notmatch '(?m)^usage: start-zonos2\.bat') {
        throw "$($case.Path) -Help did not print usage"
    }
    $backendHelp = "backend (archive default: $($case.GpuDefault))"
    if ($helpText -notmatch [regex]::Escape($backendHelp)) {
        throw "$($case.Path) -Help did not report the expected backend: $backendHelp"
    }
    if ($helpText -notmatch 'ZONOS2_DL_CONNECTIONS \(download connections, default 1, max 16\)') {
        throw "$($case.Path) -Help did not document the single-connection default"
    }
    Write-Host "PASS: $($case.Path) BOM, parse, and -Help (PowerShell $($PSVersionTable.PSVersion))"
}

# Exercise real curl transfers under the same interpreter against a local server.
& python (Join-Path $PSScriptRoot 'windows-download.py') --powershell $powerShell
if ($LASTEXITCODE -ne 0) { throw 'Windows download regression tests failed' }
