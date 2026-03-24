param(
    [switch]$IncludePreflight
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Convert-ToWslPath {
    param([string]$WindowsPath)

    $resolved = (Resolve-Path $WindowsPath).Path
    if ($resolved -match "^([A-Za-z]):\\") {
        $drive = $Matches[1].ToLowerInvariant()
        $suffix = $resolved.Substring(3).Replace("\", "/")
        return "/mnt/$drive/$suffix"
    }

    throw "Unsupported path for WSL conversion: $resolved"
}

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "== $Title ==" -ForegroundColor Cyan
    & $Action
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WslRepoRoot = Convert-ToWslPath -WindowsPath $RepoRoot

Push-Location $RepoRoot
try {
    if ($IncludePreflight) {
        Invoke-Step -Title "PowerShell preflight" -Action {
            powershell -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
        }
    }

    Invoke-Step -Title "Python syntax" -Action {
        python -m compileall .\src\ .\scripts\ .\tests\ .\docker\ -q
    }

    Invoke-Step -Title "Python tests" -Action {
        python -m unittest discover -s tests -v
    }

    Invoke-Step -Title "WSL shell syntax" -Action {
        wsl -d Ubuntu-24.04 --cd $WslRepoRoot bash -lc "bash -n scripts/*.sh"
    }

    Invoke-Step -Title "Shellcheck" -Action {
        wsl -d Ubuntu-24.04 --cd $WslRepoRoot bash -lc "if command -v shellcheck >/dev/null 2>&1; then shellcheck scripts/*.sh; else echo '[warn] shellcheck is not installed in Ubuntu-24.04'; fi"
    }

    Write-Host ""
    Write-Host "[ok] Validation completed" -ForegroundColor Green
}
finally {
    Pop-Location
}
