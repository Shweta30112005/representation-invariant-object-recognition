# =================================================================
# run_combo_experiment.ps1
# =================================================================
# Runs the 511-combination training experiment on the GPU server
# and syncs results back to the local machine.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_combo_experiment.ps1
#
# Or for background/long-running mode:
#   powershell -ExecutionPolicy Bypass -File scripts\run_combo_experiment.ps1 -Background
# =================================================================

param(
    [switch]$Background,
    [switch]$SyncOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Configuration
$SSH_USER = "akanksha"
$SSH_HOST = "172.25.0.179"
$SSH_TARGET = "${SSH_USER}@${SSH_HOST}"
$REMOTE_PROJECT_DIR = "~/representation-invariant-object-recognition"

# Local project directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOCAL_PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  511-COMBINATION EXPERIMENT - GPU RUNNER (PowerShell)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Local project:  $LOCAL_PROJECT_DIR"
Write-Host "  Remote target:  ${SSH_TARGET}:${REMOTE_PROJECT_DIR}"
Write-Host "=================================================================" -ForegroundColor Cyan

# ─────────────────────────────────────────────────────────────
# Step 1: Sync project TO the GPU server
# ─────────────────────────────────────────────────────────────
Write-Host "`n[1/5] Syncing project to GPU server..." -ForegroundColor Yellow

# Use scp to copy scripts (scp is available on Windows 10+)
scp -r "${LOCAL_PROJECT_DIR}\scripts\10_combo_train_eval.py" "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/scripts/"
scp -r "${LOCAL_PROJECT_DIR}\scripts\11_analyze_combo_results.py" "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/scripts/"
scp "${LOCAL_PROJECT_DIR}\requirements.txt" "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/"

Write-Host "[OK] Scripts synced to server." -ForegroundColor Green

if ($SyncOnly) {
    Write-Host "`nSync-only mode. Skipping training." -ForegroundColor Yellow
    
    Write-Host "`n[SYNC] Pulling results back..." -ForegroundColor Yellow
    
    $localResultsDir = Join-Path $LOCAL_PROJECT_DIR "results\combo_experiment"
    if (-not (Test-Path $localResultsDir)) {
        New-Item -ItemType Directory -Path $localResultsDir -Force | Out-Null
    }
    
    scp -r "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/results/combo_experiment/*" "$localResultsDir/"
    
    Write-Host "[OK] Results synced." -ForegroundColor Green
    exit 0
}

if ($DryRun) {
    Write-Host "`n[DRY RUN] Would run:" -ForegroundColor Yellow
    Write-Host "  ssh $SSH_TARGET 'cd $REMOTE_PROJECT_DIR && python scripts/10_combo_train_eval.py --epochs 10 --resume'"
    exit 0
}

# ─────────────────────────────────────────────────────────────
# Step 2: Install dependencies on server
# ─────────────────────────────────────────────────────────────
Write-Host "`n[2/5] Checking dependencies on server..." -ForegroundColor Yellow

ssh $SSH_TARGET "cd $REMOTE_PROJECT_DIR && pip install -q -r requirements.txt 2>&1 | tail -3"

Write-Host "[OK] Dependencies ready." -ForegroundColor Green

# ─────────────────────────────────────────────────────────────
# Step 3: Run experiment on GPU
# ─────────────────────────────────────────────────────────────
Write-Host "`n[3/5] Starting 511-combination experiment on GPU..." -ForegroundColor Yellow

if ($Background) {
    Write-Host "  Running in background (nohup) mode..." -ForegroundColor Cyan
    Write-Host "  You can disconnect and results will keep running." -ForegroundColor Cyan
    
    ssh $SSH_TARGET @'
cd ~/representation-invariant-object-recognition
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate ml-base 2>/dev/null || true
PYTHON_BIN=$(which python 2>/dev/null || echo '/opt/conda/envs/ml-base/bin/python')
if [ ! -x "$PYTHON_BIN" ] && [ -x "/opt/conda/envs/ml-base/bin/python" ]; then
    PYTHON_BIN="/opt/conda/envs/ml-base/bin/python"
fi

mkdir -p results/combo_experiment
nohup $PYTHON_BIN scripts/10_combo_train_eval.py \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --resume \
    > results/combo_experiment/experiment.log 2>&1 &
echo "PID: $!"
echo "Experiment started in background. Monitor with:"
echo "  tail -f ~/representation-invariant-object-recognition/results/combo_experiment/experiment.log"
'@
    
    Write-Host "`n[OK] Experiment launched in background on server." -ForegroundColor Green
    Write-Host "  To monitor: ssh $SSH_TARGET 'tail -f $REMOTE_PROJECT_DIR/results/combo_experiment/experiment.log'" -ForegroundColor Cyan
    Write-Host "  To sync results later: run this script with -SyncOnly flag" -ForegroundColor Cyan

} else {
    Write-Host "  Running in foreground (will block until complete)..." -ForegroundColor Cyan
    Write-Host "  WARNING: This may take 10-30+ hours depending on GPU." -ForegroundColor Red
    
    ssh $SSH_TARGET @'
cd ~/representation-invariant-object-recognition
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate ml-base 2>/dev/null || true
PYTHON_BIN=$(which python 2>/dev/null || echo '/opt/conda/envs/ml-base/bin/python')
if [ ! -x "$PYTHON_BIN" ] && [ -x "/opt/conda/envs/ml-base/bin/python" ]; then
    PYTHON_BIN="/opt/conda/envs/ml-base/bin/python"
fi

mkdir -p results/combo_experiment
echo "Starting experiment at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU info"

$PYTHON_BIN scripts/10_combo_train_eval.py \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --resume \
    2>&1 | tee results/combo_experiment/experiment.log

echo ""
echo "Training complete. Running analysis..."
$PYTHON_BIN scripts/11_analyze_combo_results.py 2>&1 | tee -a results/combo_experiment/experiment.log
echo "All done at $(date)"
'@

    Write-Host "[OK] Experiment completed on server." -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────
# Step 4: Sync results back
# ─────────────────────────────────────────────────────────────
if (-not $Background) {
    Write-Host "`n[4/5] Syncing results back from server..." -ForegroundColor Yellow
    
    $localResultsDir = Join-Path $LOCAL_PROJECT_DIR "results\combo_experiment"
    if (-not (Test-Path $localResultsDir)) {
        New-Item -ItemType Directory -Path $localResultsDir -Force | Out-Null
    }
    
    # Create analysis subdirectory
    $analysisDir = Join-Path $localResultsDir "analysis"
    if (-not (Test-Path $analysisDir)) {
        New-Item -ItemType Directory -Path $analysisDir -Force | Out-Null
    }
    
    scp -r "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/results/combo_experiment/*" "$localResultsDir/"
    
    Write-Host "[OK] Results synced to local machine." -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────
# Step 5: Summary
# ─────────────────────────────────────────────────────────────
Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "  EXPERIMENT COMPLETE" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Results location: $LOCAL_PROJECT_DIR\results\combo_experiment\"
Write-Host ""
Write-Host "  Key files:"
Write-Host "    combo_results.csv              - Full 511x9 result matrix"
Write-Host "    combo_classwise.csv            - Per-class metrics"
Write-Host "    combo_inference.csv             - Inference benchmarks"
Write-Host "    analysis\experiment_report.md   - Detailed analysis report"
Write-Host ""
Write-Host "  To re-run analysis locally:"
Write-Host "    python scripts\11_analyze_combo_results.py"
Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
