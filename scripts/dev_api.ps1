param(
  [int]$Port = 8000,
  [switch]$NoDeps,
  [switch]$SkipInstall,
  [string]$PyVersion = "",
  [switch]$RecreateVenv,
  [switch]$ProdLike,
  [string]$DbUrl = "",
  [switch]$AllowRemoteDb,
  [string]$RequirementsFile = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "== Capy Teacher API (dev) ==" -ForegroundColor Cyan

if (-not $RequirementsFile) {
  # Match VPS deploy image (Dockerfile.deploy) by default.
  $RequirementsFile = "requirements-api.txt"
}

if ($ProdLike) {
  Write-Host "Running in ProdLike mode (no auto-reload, api deps only)." -ForegroundColor Cyan
}

if ($DbUrl) {
  $env:DATABASE_URL = $DbUrl
}

# Guardrail: avoid accidentally using a production DATABASE_URL from .env when running locally.
if (-not $AllowRemoteDb) {
  $repoEnvPath = Join-Path $repoRoot ".env"
  $hasDbOverride = -not [string]::IsNullOrWhiteSpace($env:DATABASE_URL)
  if (-not $hasDbOverride -and (Test-Path $repoEnvPath)) {
    $dbLine = (Select-String -Path $repoEnvPath -Pattern '^\s*DATABASE_URL\s*=\s*' -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($dbLine) {
      $dbValue = ($dbLine.Line -replace '^\s*DATABASE_URL\s*=\s*', '').Trim().Trim('"').Trim("'")
      if ($dbValue -and ($dbValue -notmatch '@(localhost|127\.0\.0\.1|postgres)[:/]' )) {
        Write-Host "Refusing to run: .env contains a non-local DATABASE_URL." -ForegroundColor Red
        Write-Host "Set a local DB first, e.g.:" -ForegroundColor Yellow
        Write-Host "  `$env:DATABASE_URL=\"postgresql://capy:capy@localhost:5432/capy_teacher\"" -ForegroundColor Yellow
        Write-Host "Or pass -DbUrl <url> or -AllowRemoteDb if you really intend to use remote DB." -ForegroundColor Yellow
        exit 2
      }
    }
  }
}

if (-not $NoDeps) {
  Write-Host "Starting local dependencies (postgres/redis) via docker compose..." -ForegroundColor Cyan
  docker compose --profile local-deps up -d --remove-orphans postgres redis | Out-Host
}

# Pick a Python to create the venv.
# - Prefer `py -3.11` (or provided version) because PyVi wheels are commonly available on 3.10-3.12.
# - Fall back to `python`.
$pythonExe = "python"
$pythonPrefixArgs = @()

$hasPyLauncher = $null -ne (Get-Command py -ErrorAction SilentlyContinue)
if ($hasPyLauncher) {
  if ($PyVersion) {
    $pythonExe = "py"
    $pythonPrefixArgs = @("-$PyVersion")
  } else {
    $pythonExe = "py"
    $pythonPrefixArgs = @("-3.11")
  }
}

function Invoke-BasePython {
  param([Parameter(ValueFromRemainingArguments = $true)]$Args)
  & $pythonExe @pythonPrefixArgs @Args
}

# Recreate venv on demand (useful when you previously created .venv with Python 3.14,
# which cannot install PyVi reliably).
if ($RecreateVenv -and (Test-Path (Join-Path $repoRoot ".venv"))) {
  Write-Host "Removing existing virtualenv: .venv" -ForegroundColor Yellow
  Remove-Item -Recurse -Force (Join-Path $repoRoot ".venv")
}

# Create venv if missing
if (-not (Test-Path (Join-Path $repoRoot ".venv"))) {
  Write-Host "Creating virtualenv: .venv" -ForegroundColor Cyan
  Invoke-BasePython -m venv .venv
}

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"

Write-Host "Python in venv:" -ForegroundColor Cyan
& $py -c "import sys; print(sys.executable); print(sys.version)" | Out-Host

if (-not $SkipInstall) {
  Write-Host "Installing Python deps ($RequirementsFile)..." -ForegroundColor Cyan
  & $py -m pip install -U pip | Out-Host
  & $py -m pip install -r $RequirementsFile | Out-Host
}

Write-Host "PyVi check:" -ForegroundColor Cyan
& $py -c "
try:
    import pyvi
    from pyvi import ViTokenizer
    print('pyvi: OK')
except Exception as e:
    print('pyvi: NOT AVAILABLE ->', e)
" | Out-Host

Write-Host "Starting FastAPI with auto-reload on http://localhost:$Port ..." -ForegroundColor Green
Write-Host "Swagger UI: http://localhost:$Port/docs" -ForegroundColor Green

# api.py defines `app`
if ($ProdLike) {
  & $py -m uvicorn api:app --host 0.0.0.0 --port $Port
} else {
  & $py -m uvicorn api:app --reload --host 0.0.0.0 --port $Port
}
