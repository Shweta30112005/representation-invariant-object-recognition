#!/bin/bash
# =================================================================
# sync_and_run_phase6.sh
# =================================================================
# Automated Sync & Execution Runner for Phase 6 Pipeline:
#   1. Attempts SSH sync with GPU server (172.25.0.179) if online
#   2. Executes Script 12: Multi-Criteria Top-10 Selection
#   3. Executes Script 13: Test-Time Augmentation (TTA) Calibration
#   4. Executes Script 14: Iterative Feedback Loop & Failure Diagnosis
#   5. Executes Script 15: Phase 6 Report & Interactive Dashboard
# =================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================="
echo "  PHASE 6: MASTER SYNC & PIPELINE EXECUTION RUNNER (Bash)"
echo "================================================================="

echo ""
echo "[1/4] Running Script 12: Multi-Criteria Top 10 Selection..."
python scripts/12_top10_selection.py

echo ""
echo "[2/4] Running Script 13: Test-Time Augmentation (TTA) Calibration..."
python scripts/13_tta_calibration.py --views 10 --samples 60

echo ""
echo "[3/4] Running Script 14: Iterative Feedback Loop & Failure Diagnosis..."
python scripts/14_feedback_loop.py

echo ""
echo "[4/4] Running Script 15: Phase 6 Report & Dashboard Generator..."
python scripts/15_generate_phase6_report.py

echo ""
echo "================================================================="
echo "  ALL PHASE 6 DELIVERABLES SUCCESSFULLY UPDATED"
echo "================================================================="

