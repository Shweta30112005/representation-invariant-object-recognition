#!/bin/bash
# =================================================================
# run_combo_experiment.sh
# =================================================================
# Runs the 511-combination training experiment on the GPU server
# and syncs results back to the local machine.
#
# Usage (from the project root):
#   bash scripts/run_combo_experiment.sh
#
# The script assumes:
#   1. The project is already synced to the GPU server
#   2. SSH key auth is set up for akanksha@172.25.0.179
#   3. Required Python packages are installed on the server
# =================================================================

set -e

# Configuration
SSH_USER="akanksha"
SSH_HOST="172.25.0.179"
SSH_TARGET="${SSH_USER}@${SSH_HOST}"

# Remote project path (adjust if different on the server)
REMOTE_PROJECT_DIR="~/representation-invariant-object-recognition"

# Local project directory (this script's parent's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================================="
echo "  511-COMBINATION EXPERIMENT - GPU RUNNER"
echo "================================================================="
echo "  Local project:  $LOCAL_PROJECT_DIR"
echo "  Remote target:   $SSH_TARGET:$REMOTE_PROJECT_DIR"
echo "================================================================="

# ─────────────────────────────────────────────────────────────
# Step 1: Sync project TO the GPU server
# ─────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Syncing project to GPU server..."

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'results/combo_experiment/analysis' \
    "$LOCAL_PROJECT_DIR/" \
    "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/"

echo "[OK] Project synced to server."

# ─────────────────────────────────────────────────────────────
# Step 2: Install dependencies on server (if needed)
# ─────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Checking dependencies on server..."

ssh "$SSH_TARGET" "cd $REMOTE_PROJECT_DIR && pip install -q -r requirements.txt 2>&1 | tail -3"

echo "[OK] Dependencies ready."

# ─────────────────────────────────────────────────────────────
# Step 3: Run the 511-combination experiment
# ─────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Starting 511-combination experiment on GPU..."
echo "      This will take several hours. Using nohup + screen."
echo ""

# Run via nohup so it survives SSH disconnection
ssh "$SSH_TARGET" << 'REMOTE_SCRIPT'
cd ~/representation-invariant-object-recognition

# Activate conda environment with PyTorch and CUDA support
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate ml-base 2>/dev/null || true
PYTHON_BIN=$(which python 2>/dev/null || echo "/opt/conda/envs/ml-base/bin/python")
if [ ! -x "$PYTHON_BIN" ] && [ -x "/opt/conda/envs/ml-base/bin/python" ]; then
    PYTHON_BIN="/opt/conda/envs/ml-base/bin/python"
fi

# Create a log file
LOG_FILE="results/combo_experiment/experiment.log"
mkdir -p results/combo_experiment

echo "Starting experiment at $(date)" | tee "$LOG_FILE"
echo "GPU info:" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG_FILE" || echo "No GPU detected" | tee -a "$LOG_FILE"

# Run with resume support so it can recover from interruptions
$PYTHON_BIN scripts/10_combo_train_eval.py \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --resume \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Training complete at $(date)" | tee -a "$LOG_FILE"

# Run analysis
echo "" | tee -a "$LOG_FILE"
echo "Running analysis..." | tee -a "$LOG_FILE"
$PYTHON_BIN scripts/11_analyze_combo_results.py 2>&1 | tee -a "$LOG_FILE"

echo "All done at $(date)" | tee -a "$LOG_FILE"
REMOTE_SCRIPT

echo "[OK] Experiment completed on server."

# ─────────────────────────────────────────────────────────────
# Step 4: Sync results BACK from the server
# ─────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Syncing results back from server..."

rsync -avz --progress \
    "${SSH_TARGET}:${REMOTE_PROJECT_DIR}/results/combo_experiment/" \
    "$LOCAL_PROJECT_DIR/results/combo_experiment/"

echo "[OK] Results synced to local machine."

# ─────────────────────────────────────────────────────────────
# Step 5: Summary
# ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
echo "  EXPERIMENT COMPLETE"
echo "================================================================="
echo ""
echo "  Results are in: $LOCAL_PROJECT_DIR/results/combo_experiment/"
echo ""
echo "  Key files:"
echo "    combo_results.csv          - Full 511×9 result matrix"
echo "    combo_classwise.csv        - Per-class metrics"
echo "    combo_inference.csv        - Inference benchmarks"
echo "    analysis/experiment_report.md - Detailed analysis report"
echo ""
echo "  To re-run analysis locally:"
echo "    python scripts/11_analyze_combo_results.py"
echo ""
echo "================================================================="
