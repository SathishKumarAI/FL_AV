<#
.SYNOPSIS
  One command, clean checkout to results. Windows.

.DESCRIPTION
  Runs the whole thing: environment probe, dataset, shards, holdout, fleet,
  validation, federated run, holdout evaluation, pass criteria, report. Then prints
  where the results are.

  Every step is the same command a person would type, in the same order, so this
  script is a runbook that happens to execute. Nothing here is a second code path.

.PARAMETER Profile
  demo (300 images per vehicle, minutes) or full (6308, hours). Default demo.

.PARAMETER Baseline
  Also train the centralised ceiling. Costs about as much as the federated run
  itself, and is what gives the federated number a scale.

.EXAMPLE
  .\scripts\run_pipeline.ps1
  .\scripts\run_pipeline.ps1 -Profile full -Rounds 6 -Epochs 4 -PerVehicle 1400 -Baseline
#>
param(
  [ValidateSet('demo', 'full')][string]$Profile = 'demo',
  [int]$Vehicles = 6,
  [int]$Rounds = 2,
  [int]$Epochs = 1,
  [int]$PerVehicle = 0,
  [ValidateSet('condition', 'random', 'mixed', 'dirichlet')][string]$Partition = 'condition',
  [double]$Alpha = 0.5,
  [string]$Strategy = 'fedavg',
  [int]$HoldoutSize = 1000,
  [switch]$Baseline,
  [string]$Python = "$env:USERPROFILE\venvs\fl_yolov8\Scripts\python.exe"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Test-Path $Python)) {
  Write-Host "No interpreter at $Python" -ForegroundColor Red
  Write-Host "Build one on python.org 3.12 (NOT conda -- Smart App Control blocks" -ForegroundColor Yellow
  Write-Host "conda-forge's _bz2.pyd; see docs/ENV_WINDOWS.md):" -ForegroundColor Yellow
  Write-Host "  py -3.12 -m venv `$env:USERPROFILE\venvs\fl_yolov8"
  Write-Host "  & `$env:USERPROFILE\venvs\fl_yolov8\Scripts\pip install -r my-project/requirements.txt -r pipeline/requirements.txt"
  exit 1
}

# Without this flwr builds its own runtime env, installs the CPU-only torch wheel,
# and every client trains on CPU at ~5.5x the wall clock with no error anywhere.
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = '1'

function Step([string]$Name, [scriptblock]$Body) {
  Write-Host ""
  Write-Host "=== $Name ===" -ForegroundColor Cyan
  & $Body
  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED at: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "Nothing downstream is run: a stage that fails halts the chain, because" -ForegroundColor Yellow
    Write-Host "continuing past a failure is how this project used to ship silent no-ops." -ForegroundColor Yellow
    exit $LASTEXITCODE
  }
}

Step "Tests (fast, catches a broken checkout before the GPU is touched)" {
  & $Python -m pytest pipeline/tests my-project/tests -q
}

Step "Shared holdout ($HoldoutSize images no vehicle may see)" {
  & $Python -m pipeline.holdout --build --size $HoldoutSize --seed 0
}

$runArgs = @(
  '-m', 'pipeline.runner', '--all',
  '--profile', $Profile, '--vehicles', $Vehicles, '--rounds', $Rounds,
  '--epochs', $Epochs, '--partition', $Partition, '--alpha', $Alpha,
  '--strategy', $Strategy, '--yes'
)
if ($PerVehicle -gt 0) { $runArgs += @('--per-vehicle', $PerVehicle) }
# --all includes the gated baseline stage, so without this the ceiling would be
# trained twice: once in the chain and once by the step below.
if (-not $Baseline) { $runArgs += @('--skip', 'baseline') }

Step "Full chain: shards, fleet, validate, federate, evaluate, verify$(if ($Baseline) {', baseline'})" {
  & $Python @runArgs
}

Step "Comparison against previous runs" {
  & $Python -m pipeline.compare --last 5
}

$report = Get-ChildItem pipeline/reports -Directory | Sort-Object Name | Select-Object -Last 1
Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  report      : $($report.FullName)\report.html"
Write-Host "  holdout curve: pipeline\.state\holdout_metrics.json"
Write-Host "  dashboards  : $Python -m pipeline.server   ->  http://127.0.0.1:8800"
