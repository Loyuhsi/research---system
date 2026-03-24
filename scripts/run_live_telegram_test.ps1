# run_live_telegram_test.ps1 — Start bot, run live test observer, cleanup.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_live_telegram_test.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=== Telegram Live Test Wrapper (v3.18) ===" -ForegroundColor Cyan

# Ensure output directory exists
$outputDir = Join-Path $RepoRoot "output"
if (-not (Test-Path $outputDir)) { New-Item -ItemType Directory -Path $outputDir | Out-Null }

# Clear stale session log
$logPath = Join-Path $outputDir "telegram_session_log.jsonl"
if (Test-Path $logPath) { Remove-Item $logPath -Force }

# Run the live test script (which internally manages bot lifecycle)
Write-Host "`nStarting live test observer..." -ForegroundColor Yellow
python (Join-Path $RepoRoot "scripts" "live_telegram_test.py")
$exitCode = $LASTEXITCODE

# Display results
$resultPath = Join-Path $outputDir "telegram_live_test_v318.json"
if (Test-Path $resultPath) {
    Write-Host "`n=== Test Results ===" -ForegroundColor Cyan
    $results = Get-Content $resultPath -Raw | ConvertFrom-Json
    Write-Host "  Total: $($results.test_count)"
    Write-Host "  LIVE_PASS: $($results.live_pass)" -ForegroundColor Green
    Write-Host "  PARTIAL: $($results.partial)" -ForegroundColor Yellow
    Write-Host "  FAIL: $($results.fail)" -ForegroundColor Red
}

# Ensure no orphan bot processes
$botProcs = Get-Process -Name "python*" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*telegram_bot*" }
foreach ($p in $botProcs) {
    Write-Host "Cleaning up orphan bot process (PID $($p.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "`nDone." -ForegroundColor Cyan
exit $exitCode
