# build_llamacpp.ps1 — Build llama.cpp from source with CUDA support.
#
# Only use this if pre-built binaries are unavailable.
# Requires: git, cmake, CUDA toolkit, Visual Studio Build Tools.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build_llamacpp.ps1

$ErrorActionPreference = "Stop"
$BuildDir = "$env:TEMP\llama-cpp-build"
$OutputDir = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "tools"

Write-Host "=== llama.cpp Source Build (v3.18 fallback) ===" -ForegroundColor Cyan

# Check prerequisites
$prereqs = @("git", "cmake")
foreach ($cmd in $prereqs) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: $cmd not found. Install it first." -ForegroundColor Red
        exit 1
    }
}

# Check CUDA
$nvcc = Get-Command "nvcc" -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Write-Host "WARNING: nvcc not found. Build may not have CUDA support." -ForegroundColor Yellow
    Write-Host "Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
}

# Clone
if (Test-Path $BuildDir) {
    Write-Host "Removing existing build directory..."
    Remove-Item -Recurse -Force $BuildDir
}

Write-Host "`n[1/4] Cloning llama.cpp..."
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $BuildDir
if ($LASTEXITCODE -ne 0) { Write-Host "Clone failed" -ForegroundColor Red; exit 1 }

# Configure
Write-Host "`n[2/4] Configuring with CMake (CUDA enabled)..."
Push-Location $BuildDir
cmake -B build -DGGML_CUDA=ON 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure failed. Trying without CUDA..." -ForegroundColor Yellow
    cmake -B build 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configure failed completely" -ForegroundColor Red
        Pop-Location; exit 1
    }
}

# Build
Write-Host "`n[3/4] Building..."
cmake --build build --config Release -j $env:NUMBER_OF_PROCESSORS 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    Pop-Location; exit 1
}

# Copy binary
Write-Host "`n[4/4] Copying llama-server to $OutputDir..."
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$serverBin = Get-ChildItem -Path "$BuildDir\build" -Recurse -Filter "llama-server*" |
    Where-Object { -not $_.PSIsContainer } | Select-Object -First 1

if ($serverBin) {
    Copy-Item $serverBin.FullName -Destination $OutputDir
    Write-Host "Copied: $($serverBin.Name) -> $OutputDir" -ForegroundColor Green

    # Verify
    $outputBin = Join-Path $OutputDir $serverBin.Name
    & $outputBin --version 2>&1
    Write-Host "`n✅ Build complete. Binary at: $outputBin" -ForegroundColor Green
} else {
    Write-Host "❌ llama-server binary not found in build output" -ForegroundColor Red
    Pop-Location; exit 1
}

Pop-Location

# Cleanup
Write-Host "`nCleaning up build directory..."
Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue

Write-Host "Done." -ForegroundColor Cyan
