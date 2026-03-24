param(
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-Command {
    param([string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        Name = $Name
        Found = $null -ne $cmd
        Path = if ($null -ne $cmd) { $cmd.Source } else { "" }
    }
}

function Test-HttpStatus {
    param([string]$Url)

    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return [pscustomobject]@{
            Url = $Url
            Ok = $true
            Detail = "HTTP $($resp.StatusCode)"
        }
    }
    catch {
        return [pscustomobject]@{
            Url = $Url
            Ok = $false
            Detail = $_.Exception.Message
        }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$vaultRoot = if ($env:VAULT_ROOT) { $env:VAULT_ROOT } else { "C:\Users\User\Documents\AutoResearchVault" }
$commands = @("python", "wsl", "docker") | ForEach-Object { Test-Command $_ }
$httpChecks = @(
    Test-HttpStatus "http://127.0.0.1:11434/api/tags"
    Test-HttpStatus "http://127.0.0.1:1234/v1/models"
)

Write-Host "== Fast Preflight ==" -ForegroundColor Cyan
[pscustomobject]@{
    RepoRoot = $repoRoot
    VaultConfigured = -not [string]::IsNullOrWhiteSpace($vaultRoot)
    VaultExists = Test-Path $vaultRoot
    DockerInstalled = ($commands | Where-Object { $_.Name -eq "docker" }).Found
    WslInstalled = ($commands | Where-Object { $_.Name -eq "wsl" }).Found
    PythonInstalled = ($commands | Where-Object { $_.Name -eq "python" }).Found
} | Format-List

Write-Host ""
Write-Host "== HTTP ==" -ForegroundColor Cyan
$httpChecks | Format-Table -AutoSize

$issues = @()
foreach ($command in $commands) {
    if (-not $command.Found) {
        $issues += "Missing command: $($command.Name)"
    }
}
if (-not (Test-Path (Join-Path $repoRoot "config\runtime-modes.json"))) {
    $issues += "Missing config/runtime-modes.json"
}
if (-not (Test-Path (Join-Path $repoRoot "config\zones.json"))) {
    $issues += "Missing config/zones.json"
}
if (-not (Test-Path (Join-Path $repoRoot "config\tool-allowlist.json"))) {
    $issues += "Missing config/tool-allowlist.json"
}

if ($issues.Count -gt 0) {
    Write-Host ""
    Write-Host "== Issues ==" -ForegroundColor Yellow
    $issues | ForEach-Object { Write-Host "- $_" }
    if ($Strict) {
        exit 1
    }
}
