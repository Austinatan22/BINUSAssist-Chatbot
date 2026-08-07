# Runs scripts/eval.py in its own window so the progress is watchable, and writes the output to a
# log file so it survives closing the window.
#
# The eval takes 9-10 minutes (96 questions, 2.5s pacing, plus ~60s of model loading), which is too
# long to sit in front of a blocked prompt and too long to run blind. The window stays open when
# the run finishes so the summary is still there to read.
#
#   .\scripts\run_eval.ps1              # new window, log under logs/
#   .\scripts\run_eval.ps1 -Here        # this window instead (for CI or a redirect)
#   .\scripts\run_eval.ps1 -LogPath x   # somewhere else
[CmdletBinding()]
param(
    [switch]$Here,
    [string]$LogPath
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "No virtualenv at $python. Create it and install requirements.txt first."
}

if (-not $LogPath) {
    $logDir = Join-Path $repo 'logs'
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
    $LogPath = Join-Path $logDir ("eval_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
}

if (-not $Here) {
    # Re-invoke this same script with -Here in a new window. Passing the script path beats
    # embedding the run in a -Command here-string: no nested quoting to get wrong, and the logic
    # below is the only copy of it. -NoExit keeps the window open on the summary.
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList '-NoExit', '-NoProfile', '-File', $PSCommandPath, '-Here', '-LogPath', "`"$LogPath`"" `
        -WorkingDirectory $repo | Out-Null
    Write-Host "Eval running in a new window. Log: $LogPath"
    return
}

Set-Location $repo
$env:PYTHONUNBUFFERED = '1'
# Without this, a single non-ASCII character in a printed question or answer raises
# UnicodeEncodeError against the console's legacy codepage and kills a 10-minute run outright.
$env:PYTHONIOENCODING = 'utf-8'
Write-Host "Running scripts/eval.py -- logging to $LogPath" -ForegroundColor Cyan
Write-Host ''

# Not Tee-Object: in Windows PowerShell 5.1 it writes UTF-16LE and has no -Encoding parameter, so
# the log came out as UTF-16 and every downstream reader (grep, sed, Python's default open) saw
# NUL-separated bytes instead of text. Confirmed on the 2026-08-08 run: the file opened with a
# BOM of ff fe and `sed -n '/--- Summary ---/,$p' ` matched nothing at all in a 250-line log that
# plainly contained it. A StreamWriter with UTF8Encoding($false) gives UTF-8 with no BOM, and
# AutoFlush keeps the file current so the log is as live as the window.
$writer = New-Object System.IO.StreamWriter(
    $LogPath, $false, (New-Object System.Text.UTF8Encoding($false))
)
$writer.AutoFlush = $true
try {
    & $python -u (Join-Path $repo 'scripts\eval.py') 2>&1 | ForEach-Object {
        Write-Host $_
        $writer.WriteLine([string]$_)
    }
} finally {
    $writer.Dispose()
}

Write-Host ''
Write-Host "Done. Log: $LogPath" -ForegroundColor Cyan
Write-Host 'Next: python scripts/grade_eval.py' -ForegroundColor Cyan
