param(
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        return [pscustomobject]@{
            Name = $Name
            Found = $false
            Path = ""
        }
    }

    return [pscustomobject]@{
        Name = $Name
        Found = $true
        Path = $cmd.Source
    }
}

function Get-PortOwner {
    param([int]$Port)

    $conn = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq $Port } |
        Select-Object -First 1

    if ($null -eq $conn) {
        return [pscustomobject]@{
            Port = $Port
            InUse = $false
            LocalAddress = ""
            Process = ""
        }
    }

    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        Port = $Port
        InUse = $true
        LocalAddress = $conn.LocalAddress
        Process = if ($proc) { $proc.ProcessName } else { "PID $($conn.OwningProcess)" }
    }
}

function Test-Http {
    param([string]$Url)

    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return "OK $($resp.StatusCode)"
    }
    catch {
        return "FAIL"
    }
}

$vaultRoot = "C:\Users\User\Documents\AutoResearchVault"

Write-Host "== Commands ==" -ForegroundColor Cyan
$commands = @(
    "ollama",
    "lms",
    "wsl",
    "node",
    "npm",
    "python",
    "obsidian"
) | ForEach-Object { Test-Command $_ }
$commands | Format-Table -AutoSize

Write-Host ""
Write-Host "== Ports ==" -ForegroundColor Cyan
$ports = 11434, 1234, 8010 | ForEach-Object { Get-PortOwner $_ }
$ports | Format-Table -AutoSize

Write-Host ""
Write-Host "== HTTP ==" -ForegroundColor Cyan
[pscustomobject]@{
    Ollama = Test-Http "http://localhost:11434/api/tags"
    LMStudio = Test-Http "http://localhost:1234/v1/models"
    ScraplingMCP = Test-Http "http://localhost:8010"
} | Format-List

Write-Host ""
Write-Host "== Vault ==" -ForegroundColor Cyan
[pscustomobject]@{
    Path = $vaultRoot
    Exists = Test-Path $vaultRoot
    ResearchPathExists = Test-Path (Join-Path $vaultRoot "10_Research\AutoResearch")
} | Format-List

Write-Host ""
Write-Host "== WSL ==" -ForegroundColor Cyan
$wslListing = ""
try {
    $wslListing = cmd /c "wsl -l -v" 2>$null | Out-String
    Write-Host $wslListing
}
catch {
    Write-Host "WSL unavailable"
}

Write-Host "== Ollama Models ==" -ForegroundColor Cyan
try {
    ollama list
}
catch {
    Write-Host "Unable to query ollama"
}

$issues = @()
if (-not (Test-Path $vaultRoot)) { $issues += "Vault path missing: $vaultRoot" }
if (($ports | Where-Object { $_.Port -eq 8010 -and $_.InUse }).Count -gt 0) { $issues += "Port 8010 already in use" }
if (($commands | Where-Object { $_.Name -eq "ollama" -and -not $_.Found }).Count -gt 0) { $issues += "ollama command missing" }
if (($commands | Where-Object { $_.Name -eq "wsl" -and -not $_.Found }).Count -gt 0) { $issues += "wsl command missing" }
if ($commands | Where-Object { $_.Name -eq "wsl" -and $_.Found }) {
    if ($wslListing -notmatch "Ubuntu") {
        $issues += "Ubuntu WSL distro not found"
    }
}

if ($issues.Count -gt 0) {
    Write-Host ""
    Write-Host "== Issues ==" -ForegroundColor Yellow
    $issues | ForEach-Object { Write-Host "- $_" }
    if ($Strict) {
        exit 1
    }
}
