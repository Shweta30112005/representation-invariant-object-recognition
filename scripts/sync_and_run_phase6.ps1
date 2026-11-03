# =================================================================
# sync_and_run_phase6.ps1
# =================================================================
# Automated Sync & Execution Runner for Phase 6 Pipeline:
#   1. Attempts SSH sync with GPU server (172.25.0.179) if online
#   2. Executes Script 12: Multi-Criteria Top-10 Selection
#   3. Executes Script 13: Test-Time Augmentation (TTA) Calibration
#   4. Executes Script 14: Iterative Feedback Loop & Failure Diagnosis
#   5. Executes Script 15: Phase 6 Report & Interactive Dashboard
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\sync_and_run_phase6.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\sync_and_run_phase6.ps1 -SkipSync
# =================================================================

param(
    [switch]$SkipSync,
    [string]$ServerHost = "172.25.0.179",
    [string]$ServerUser = "akanksha"
)

$ErrorActionPreference = "Continue"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR
Set-Location $PROJECT_DIR

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  PHASE 6: MASTER SYNC & PIPELINE EXECUTION RUNNER" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Project Directory: $PROJECT_DIR"
Write-Host "  GPU Server Target: ${ServerUser}@${ServerHost}"
Write-Host "=================================================================" -ForegroundColor Cyan

# ─────────────────────────────────────────────────────────────
# Step 1: Optional Server Sync
# ─────────────────────────────────────────────────────────────
if (-not $SkipSync) {
    Write-Host "`n[1/5] Checking GPU Server Connection..." -ForegroundColor Yellow
    
    $tcpCheck = Test-NetConnection -ComputerName $ServerHost -Port 22 -WarningAction SilentlyContinue
    
    if ($tcpCheck.TcpTestSucceeded) {
        Write-Host "  [OK] GPU server is REACHABLE. Syncing latest combo results..." -ForegroundColor Green
        
        $localComboDir = Join-Path $PROJECT_DIR "results\combo_experiment"
        if (-not (Test-Path $localComboDir)) {
            New-Item -ItemType Directory -Path $localComboDir -Force | Out-Null
        }
        
        scp "${ServerUser}@${ServerHost}:~/representation-invariant-object-recognition/results/combo_experiment/combo_results.csv" "$localComboDir/"
        scp "${ServerUser}@${ServerHost}:~/representation-invariant-object-recognition/results/combo_experiment/combo_classwise.csv" "$localComboDir/"
        
        Write-Host "  [OK] Data synchronized from GPU server." -ForegroundColor Green
    } else {
        Write-Host "  [NOTICE] GPU server $ServerHost is currently offline / not on campus VPN." -ForegroundColor Yellow
        Write-Host "  Using cached local 511-experiment results and dataset." -ForegroundColor Cyan
    }
} else {
    Write-Host "`n[1/5] Skipping server sync (-SkipSync specified)." -ForegroundColor Yellow
}

# ─────────────────────────────────────────────────────────────
# Step 2: Multi-Criteria Top 10 Selection
# ─────────────────────────────────────────────────────────────
Write-Host "`n[2/5] Running Script 12: Multi-Criteria Top 10 Selection..." -ForegroundColor Yellow
python scripts/12_top10_selection.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Script 12 failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# ─────────────────────────────────────────────────────────────
# Step 3: TTA Calibration
# ─────────────────────────────────────────────────────────────
Write-Host "`n[3/5] Running Script 13: Test-Time Augmentation (TTA) Calibration..." -ForegroundColor Yellow
python scripts/13_tta_calibration.py --views 10 --samples 60
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Script 13 failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# ─────────────────────────────────────────────────────────────
# Step 4: Feedback Loop
# ─────────────────────────────────────────────────────────────
Write-Host "`n[4/5] Running Script 14: Iterative Feedback Loop & Failure Diagnosis..." -ForegroundColor Yellow
python scripts/14_feedback_loop.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Script 14 failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# ─────────────────────────────────────────────────────────────
# Step 5: Report & Dashboard Generation
# ─────────────────────────────────────────────────────────────
Write-Host "`n[5/5] Running Script 15: Phase 6 Report & Interactive Dashboard Generator..." -ForegroundColor Yellow
python scripts/15_generate_phase6_report.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Script 15 failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "  PIPELINE EXECUTION COMPLETE" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Deliverables ready at:" -ForegroundColor Cyan
Write-Host "    - Dashboard:  results\phase6_report\phase6_dashboard.html" -ForegroundColor White
Write-Host "    - Report:     results\phase6_report\phase6_report.md" -ForegroundColor White
Write-Host "    - Tables:     results\phase6_report\phase6_tables.csv" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan

