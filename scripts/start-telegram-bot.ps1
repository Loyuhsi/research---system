param(
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path $Path)) {
        return ""
    }

    foreach ($rawLine in Get-Content -Path $Path -Encoding utf8) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }

        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }

        if (-not $line.Contains("=")) {
            continue
        }

        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($key -ne $Name) {
            continue
        }

        if ($value.Length -ge 2) {
            $quote = $value[0]
            if (($quote -eq '"' -or $quote -eq "'") -and $value[-1] -eq $quote) {
                return $value.Substring(1, $value.Length - 2)
            }
        }

        return $value
    }

    return ""
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $RepoRoot ".runtime"
$EnvExamplePath = Join-Path $RepoRoot ".env.example"
$EnvPath = Join-Path $RepoRoot ".env"
$RequirementsPath = Join-Path $RepoRoot "requirements-telegram.txt"
$VenvPath = Join-Path $RuntimeDir "telegram-bot-venv"
$VenvPython = Join-Path $VenvPath "Scripts\\python.exe"
$RequirementsStamp = Join-Path $RuntimeDir "telegram-bot.requirements.sha256"
$BotScript = Join-Path $RepoRoot "scripts\\telegram_bot.py"

if (-not (Test-Path $EnvPath)) {
    if (-not (Test-Path $EnvExamplePath)) {
        throw "Missing .env.example"
    }
    Copy-Item -Path $EnvExamplePath -Destination $EnvPath
    Write-Host "[info] Created .env from .env.example"
}

$token = if ($env:TELEGRAM_BOT_TOKEN) { $env:TELEGRAM_BOT_TOKEN } else { Get-EnvValue -Path $EnvPath -Name "TELEGRAM_BOT_TOKEN" }
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "Missing TELEGRAM_BOT_TOKEN. Fill it in $EnvPath and rerun this script."
}

if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[info] Creating Telegram bot virtual environment..."
    python -m venv $VenvPath
}

if (-not (Test-Path $VenvPython)) {
    throw "Unable to create Telegram bot virtual environment at $VenvPath"
}

$requirementsHash = (Get-FileHash -Path $RequirementsPath -Algorithm SHA256).Hash
$installedHash = if (Test-Path $RequirementsStamp) { (Get-Content -Path $RequirementsStamp -Raw -Encoding utf8).Trim() } else { "" }

if ($installedHash -ne $requirementsHash) {
    Write-Host "[info] Installing Telegram bot dependencies..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r $RequirementsPath
    Set-Content -Path $RequirementsStamp -Value $requirementsHash -Encoding utf8
}

$env:PYTHONIOENCODING = "utf-8"

Write-Host "[info] Validating Telegram bot configuration..."
& $VenvPython $BotScript --validate-only
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($ValidateOnly) {
    exit 0
}

Write-Host "[info] Starting Telegram bot..."
& $VenvPython $BotScript
exit $LASTEXITCODE
