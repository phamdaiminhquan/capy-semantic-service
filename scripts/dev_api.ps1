param(
  [int]$Port = 8000,
  [switch]$NoDeps,
  [switch]$SkipInstall,
  [string]$PyVersion = "",
  [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "== Capy Teacher API (dev) ==" -ForegroundColor Cyan

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
  Write-Host "Installing Python deps (requirements.txt)..." -ForegroundColor Cyan
  & $py -m pip install -U pip | Out-Host
  & $py -m pip install -r requirements.txt | Out-Host
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
& $py -m uvicorn api:app --reload --host 0.0.0.0 --port $Port
