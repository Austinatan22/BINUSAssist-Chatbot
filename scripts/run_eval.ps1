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
# MUST be Continue, not Stop, around the native call below. Windows PowerShell 5.1 wraps every
# line a native exe writes to stderr in a NativeCommandError ErrorRecord, and under 'Stop' the
# first one aborts the run. That killed a run after 3 lines: bm25s logs "Building index from IDs
# objects" to stderr at DEBUG, which is not an error at all. The earlier version of this script
# survived only by accident, because it passed a -Command string to a fresh process where the
# preference was still the default.
$ErrorActionPreference = 'Continue'
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
# Streams are merged by cmd, NOT by PowerShell's `2>&1`. Doing it in the PowerShell pipeline is
# what produces the ErrorRecord wrapping described above, so each stderr line arrives as a red
# NativeCommandError block with a call-stack trailer instead of the one line it actually is,
# which makes the progress window unreadable. cmd merges before PowerShell sees anything, so
# everything arrives as a plain string.
$evalPy = Join-Path $repo 'scripts\eval.py'
try {
    & cmd /c "`"$python`" -u `"$evalPy`" 2>&1" | ForEach-Object {
        Write-Host $_
        $writer.WriteLine([string]$_)
    }
} finally {
    $writer.Dispose()
}

Write-Host ''
Write-Host "Done. Log: $LogPath" -ForegroundColor Cyan
Write-Host 'Next: python scripts/grade_eval.py' -ForegroundColor Cyan
