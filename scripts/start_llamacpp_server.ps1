# start_llamacpp_server.ps1 — Start llama-server with GPU offload.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/start_llamacpp_server.ps1 -ModelPath "C:\models\qwen2.5-7b-q4_k_m.gguf"
#   powershell -File scripts/start_llamacpp_server.ps1 -ModelPath "model.gguf" -GpuLayers 99 -ContextSize 8192 -Port 8080

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [int]$GpuLayers = 99,
    [int]$ContextSize = 8192,
    [int]$Port = 8080,
    [int]$BatchSize = 512,
    [int]$Threads = 0
)

$ErrorActionPreference = "Stop"

Write-Host "=== llama-server Startup (v3.18) ===" -ForegroundColor Cyan

# Validate model file
if (-not (Test-Path $ModelPath)) {
    Write-Host "ERROR: Model file not found: $ModelPath" -ForegroundColor Red
    exit 1
}
$ModelSize = (Get-Item $ModelPath).Length / 1MB
Write-Host "  Model: $ModelPath ($([math]::Round($ModelSize, 1)) MB)"

# Find llama-server binary
$llamaServer = $null
$searchPaths = @(
    "llama-server",
    "llama-server.exe",
    "$env:USERPROFILE\llama.cpp\build\bin\llama-server.exe",
    "C:\llama.cpp\build\bin\llama-server.exe",
    "C:\tools\llama-server.exe"
)

foreach ($path in $searchPaths) {
    $found = Get-Command $path -ErrorAction SilentlyContinue
    if ($found) {
        $llamaServer = $found.Source
        break
    }
    if (Test-Path $path) {
        $llamaServer = $path
        break
    }
}

if (-not $llamaServer) {
    Write-Host "ERROR: llama-server not found. Install from https://github.com/ggml-org/llama.cpp/releases" -ForegroundColor Red
    exit 1
}
Write-Host "  Binary: $llamaServer"

# Check VRAM availability
Write-Host "`n[VRAM Check]" -ForegroundColor Yellow
try {
    $vramInfo = nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>&1
    Write-Host "  GPU Memory: $vramInfo"
} catch {
    Write-Host "  WARNING: nvidia-smi not available" -ForegroundColor Yellow
}

# Build command arguments
$serverArgs = @(
    "-m", $ModelPath,
    "-ngl", $GpuLayers,
    "-c", $ContextSize,
    "--port", $Port,
    "-b", $BatchSize
)
if ($Threads -gt 0) {
    $serverArgs += @("-t", $Threads)
}

Write-Host "`n[Starting Server]" -ForegroundColor Yellow
Write-Host "  Port: $Port"
Write-Host "  GPU Layers: $GpuLayers"
Write-Host "  Context Size: $ContextSize"
Write-Host "  Batch Size: $BatchSize"
Write-Host "  Command: $llamaServer $($serverArgs -join ' ')"

# Start server
$process = Start-Process -FilePath $llamaServer -ArgumentList $serverArgs -PassThru -NoNewWindow

# Wait for health endpoint
Write-Host "`n[Waiting for server health...]" -ForegroundColor Yellow
$maxWait = 60
$waited = 0
$healthy = $false

while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 2
    $waited += 2
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 -ErrorAction Stop
        if ($response.status -eq "ok") {
            $healthy = $true
            break
        }
    } catch {
        # Server not ready yet
    }
    Write-Host "  Waiting... ($waited/$maxWait s)" -ForegroundColor Gray
}

if ($healthy) {
    Write-Host "`n✅ llama-server is healthy at http://127.0.0.1:$Port" -ForegroundColor Green
    Write-Host "  PID: $($process.Id)"

    # Query loaded model
    try {
        $models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 5
        $modelId = $models.data[0].id
        Write-Host "  Loaded model: $modelId"
    } catch {
        Write-Host "  Could not query loaded model" -ForegroundColor Yellow
    }

    # Save PID for cleanup
    $pidFile = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".." ".runtime" "llamacpp.pid"
    $pidDir = Split-Path -Parent $pidFile
    if (-not (Test-Path $pidDir)) { New-Item -ItemType Directory -Path $pidDir -Force | Out-Null }
    $process.Id | Out-File -FilePath $pidFile -Encoding ascii
    Write-Host "  PID file: $pidFile"
} else {
    Write-Host "`n❌ Server did not become healthy within $maxWait seconds" -ForegroundColor Red
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    exit 1
}

Write-Host "`nPress Ctrl+C to stop the server." -ForegroundColor Gray
try {
    Wait-Process -Id $process.Id
} catch {
    # Ctrl+C
}
