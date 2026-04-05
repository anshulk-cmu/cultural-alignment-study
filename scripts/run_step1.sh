#!/bin/bash
#SBATCH --job-name=step1_eval
#SBATCH --output=/home/anshulk/cultural-mi/logs/step1_slurm_%j.out
#SBATCH --error=/home/anshulk/cultural-mi/logs/step1_slurm_%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# Handle preemption: SLURM sends SIGUSR1 120s before kill.
# Requeue the job so it restarts automatically.
handle_preempt() {
    echo ""
    echo "============================================================"
    echo "PREEMPTED — requeueing job $SLURM_JOB_ID at $(date)"
    echo "Resume logic will skip completed batches on restart."
    echo "============================================================"
    scontrol requeue "$SLURM_JOB_ID"
}
trap 'handle_preempt' USR1

# ============================================================================
# Step 1 — Behavioral Evaluation + Activation Extraction
# ============================================================================
# Runs LLaMA-3.1-8B base and instruct on all 21,726 Sanskriti questions.
# Single forward pass per question: logit-based answer extraction + activation
# hooks at 8 points (embed + layers 4,8,14,20,26,30,31).
#
# Base runs on cuda:0, instruct on cuda:1 (parallel via torch.multiprocessing).
# Checkpoints every 100 batches; resumes from partial CSV on restart.
#
# After eval completes, merge_step1.py combines results, assigns behavioral
# labels (suppression/enhancement/control), runs 11 sanity checks, and
# generates 7 plots.
#
# Expected runtime: ~8–10 min eval (parallel), ~15s merge.
#   7-day time limit is generous for preemption retries.
#
# Outputs:
#   Results:     /data/user_data/anshulk/cultural-mi/results/step1/
#   Activations: /data/user_data/anshulk/cultural-mi/activations/{base,instruct}/
#   Checkpoints: /data/user_data/anshulk/cultural-mi/checkpoints/
#   Logs:        /home/anshulk/cultural-mi/logs/
# ============================================================================

echo "============================================================"
echo "Step 1 — Behavioral Evaluation + Activation Extraction"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

cd /home/anshulk/cultural-mi || { echo "Failed to cd to workspace"; exit 1; }

echo "Activating conda environment..."
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate cultural || { echo "Failed to activate cultural environment"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# Python packages
echo "  Checking Python packages..."
REQUIRED="torch transformers datasets pandas numpy matplotlib yaml"
for pkg in $REQUIRED; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  ERROR: Missing package: $pkg"
        exit 1
    fi
done
echo "  All packages available"

# Config
echo "  Checking config..."
if [ ! -f "configs/config.yaml" ]; then
    echo "  ERROR: configs/config.yaml not found"
    exit 1
fi

# Models
echo "  Checking model weights..."
BASE_DIR="/data/user_data/anshulk/cultural-mi/models/base"
INST_DIR="/data/user_data/anshulk/cultural-mi/models/instruct"

if [ ! -d "$BASE_DIR" ] || [ -z "$(ls -A $BASE_DIR/*.safetensors 2>/dev/null)" ]; then
    echo "  ERROR: Base model not found at $BASE_DIR"
    exit 1
fi
echo "  Base model: OK"

if [ ! -d "$INST_DIR" ] || [ -z "$(ls -A $INST_DIR/*.safetensors 2>/dev/null)" ]; then
    echo "  ERROR: Instruct model not found at $INST_DIR"
    exit 1
fi
echo "  Instruct model: OK"

# Dataset
echo "  Checking dataset..."
DATASET_DIR="/data/user_data/anshulk/cultural-mi/dataset"
if [ ! -d "$DATASET_DIR" ]; then
    echo "  WARNING: Dataset cache not found at $DATASET_DIR (will download)"
fi
echo "  Dataset dir: OK"

# GPU check
echo "  Checking GPUs..."
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "  GPUs visible: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  WARNING: Expected 2 GPUs, found $GPU_COUNT. Will run sequentially."
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check for existing checkpoints
echo "  Checking for existing checkpoints..."
CKPT_DIR="/data/user_data/anshulk/cultural-mi/checkpoints"
for model in base instruct; do
    partial="$CKPT_DIR/step1_${model}_results_partial.csv"
    if [ -f "$partial" ]; then
        ROWS=$(wc -l < "$partial")
        echo "  Found checkpoint: $model — $((ROWS - 1)) rows completed"
    else
        echo "  No checkpoint: $model — starting fresh"
    fi
done

# Output directories
echo "  Creating output directories..."
mkdir -p /home/anshulk/cultural-mi/logs
mkdir -p /data/user_data/anshulk/cultural-mi/results/step1
mkdir -p /data/user_data/anshulk/cultural-mi/activations/base/mean_pool
mkdir -p /data/user_data/anshulk/cultural-mi/activations/base/last_token
mkdir -p /data/user_data/anshulk/cultural-mi/activations/instruct/mean_pool
mkdir -p /data/user_data/anshulk/cultural-mi/activations/instruct/last_token
mkdir -p /data/user_data/anshulk/cultural-mi/checkpoints
echo "  Output directories ready"

echo ""
echo "Pre-flight checks passed!"
echo ""

# ============================================================================
# RUN EVALUATION
# ============================================================================

echo "============================================================"
echo "Running Step 1 — Evaluation"
echo "  Models: base (cuda:0) + instruct (cuda:1)"
echo "  Questions: 21,726"
echo "  Batch size: 24 (safe for 48GB GPUs; increase to 64 for 96GB)"
echo "  Hooks: embed + layers [4, 8, 14, 20, 26, 30, 31]"
echo "  Expected: ~906 batches/model, ~15 min total (parallel)"
echo "============================================================"
echo ""

# Batch size 24: safe for 48GB GPUs (A6000, L40S). ~22GB peak per model.
# Increase to 64 if you know you have 96GB GPUs (RTX PRO 6000).
python scripts/eval_step1.py --batch-size 24 "$@"
EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo ""
    echo "EVALUATION FAILED (exit code $EVAL_EXIT) — check logs/"
    echo "If preempted, job will requeue and resume from checkpoint."
    exit $EVAL_EXIT
fi

# ============================================================================
# RUN MERGE + ANALYSIS
# ============================================================================

echo ""
echo "============================================================"
echo "Running Step 1 — Merge & Analysis"
echo "============================================================"
echo ""

python scripts/merge_step1.py
MERGE_EXIT=$?

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Post-Run Validation"
echo "============================================================"

RESULTS_DIR="/data/user_data/anshulk/cultural-mi/results/step1"
ACT_DIR="/data/user_data/anshulk/cultural-mi/activations"

echo "--- Result CSVs ---"
for f in base_results.csv instruct_results.csv sanskriti_behavioral_labels.csv sanskriti_prepared.csv; do
    path="$RESULTS_DIR/$f"
    if [ -f "$path" ]; then
        ROWS=$(wc -l < "$path")
        SIZE=$(du -h "$path" | cut -f1)
        echo "  $f: $((ROWS - 1)) rows, $SIZE"
    else
        echo "  $f: MISSING"
    fi
done

echo ""
echo "--- Analysis outputs ---"
for f in step1_aggregate_stats.json accuracy_by_question_type.csv suppression_by_attribute.csv suppression_by_state.csv position_distribution.csv behavioral_label_counts.csv entity_behavioral_labels.csv confidence_distribution.csv; do
    path="$RESULTS_DIR/$f"
    if [ -f "$path" ]; then
        echo "  $f: OK"
    else
        echo "  $f: MISSING"
    fi
done

echo ""
echo "--- Activation files ---"
for model in base instruct; do
    for pool in mean_pool last_token; do
        COUNT=$(ls "$ACT_DIR/$model/$pool"/*.npy 2>/dev/null | wc -l)
        echo "  $model/$pool: $COUNT / 8 files"
    done
done
TOTAL_ACT=$(find "$ACT_DIR" -name "*.npy" | wc -l)
TOTAL_SIZE=$(du -sh "$ACT_DIR" 2>/dev/null | cut -f1)
echo "  Total: $TOTAL_ACT / 32 files, $TOTAL_SIZE"

echo ""
echo "--- Plots ---"
PLOT_COUNT=$(ls "$RESULTS_DIR"/*.png 2>/dev/null | wc -l)
echo "  Plots generated: $PLOT_COUNT"

echo ""
echo "--- Sanity check summary ---"
MERGE_LOG=$(ls -t /home/anshulk/cultural-mi/logs/step1_merge_*.log 2>/dev/null | head -1)
if [ -n "$MERGE_LOG" ]; then
    grep -E "\[PASS\]|\[FAIL\]|\[WARN\]|\[INFO\]" "$MERGE_LOG" | tail -20
else
    echo "  No merge log found"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Job Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Eval exit code: $EVAL_EXIT"
echo "Merge exit code: $MERGE_EXIT"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""

if [ $EVAL_EXIT -eq 0 ] && [ $MERGE_EXIT -eq 0 ]; then
    echo "Step 1 completed successfully"
    exit 0
else
    echo "STEP 1 HAD ERRORS — check logs/"
    exit 1
fi
