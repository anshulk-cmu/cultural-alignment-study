#!/bin/bash
#SBATCH --job-name=phase2_5_qwen3_validate
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_%j.out
#SBATCH --error=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#SBATCH --requeue

set -e

mkdir -p /home/anshulk/cultural-alignment-study/outputs/logs

echo "=================================="
echo "SLURM Job Information"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

source ~/.bashrc
conda activate rq1

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'rq1'"
    exit 1
fi

echo "=================================="
echo "Environment"
echo "=================================="
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

export HF_HOME=/data/hf_cache
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/transformers
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/anshulk/cultural-alignment-study

echo "=================================="
echo "Pre-flight Validation"
echo "=================================="

MODEL_PATH="/data/user_data/anshulk/models/Qwen3-30B-A3B-Instruct-2507"
if [ -d "$MODEL_PATH" ]; then
    echo "Model found: $MODEL_PATH"
else
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path

labels_file = SAE_OUTPUT_ROOT / 'labels_qwen_initial.json'
if not labels_file.exists():
    print(f'ERROR: Initial labels not found: {labels_file}')
    sys.exit(1)

examples_dir = SAE_OUTPUT_ROOT / 'feature_examples'
if not examples_dir.exists():
    print(f'ERROR: Examples directory not found: {examples_dir}')
    sys.exit(1)

print('Validation prerequisites complete')
"

if [ $? -ne 0 ]; then
    echo "Pre-flight validation failed"
    exit 1
fi

echo ""
echo "=================================="
echo "Phase 2.5: Qwen3 Validation"
echo "=================================="
echo "Model: Qwen3-30B-A3B-Instruct-2507"
echo "GPUs: 2x L40S (8-bit quantization)"
echo "Input: labels_qwen_initial.json"
echo "Output: labels_qwen3_validated.json"
echo "Estimated time: 3-5 hours"
echo ""

START_TIME=$(date +%s)
echo "Started: $(date)"
echo ""

python scripts/phase2_5_qwen_validate.py 2>&1 | tee -a /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_${SLURM_JOB_ID}.log

EXIT_STATUS=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=================================="
echo "Job Summary"
echo "=================================="
echo "Exit status: $EXIT_STATUS"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

if [ $EXIT_STATUS -eq 0 ]; then
    echo "Validation completed successfully"
    
    python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
import json

output_file = SAE_OUTPUT_ROOT / 'labels_qwen3_validated.json'
if output_file.exists():
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    stats = {
        'KEEP': sum(1 for r in results if r.get('validation_action') == 'KEEP'),
        'REVISE': sum(1 for r in results if r.get('validation_action') == 'REVISE'),
        'INVALIDATE': sum(1 for r in results if r.get('validation_action') == 'INVALIDATE'),
    }
    
    print(f'Results: {output_file}')
    print(f'Total: {len(results)}')
    print(f'KEEP: {stats[\"KEEP\"]} ({stats[\"KEEP\"]/len(results)*100:.1f}%)')
    print(f'REVISE: {stats[\"REVISE\"]} ({stats[\"REVISE\"]/len(results)*100:.1f}%)')
    print(f'INVALIDATE: {stats[\"INVALIDATE\"]} ({stats[\"INVALIDATE\"]/len(results)*100:.1f}%)')
    print(f'Valid: {stats[\"KEEP\"] + stats[\"REVISE\"]} ({(stats[\"KEEP\"] + stats[\"REVISE\"])/len(results)*100:.1f}%)')
"
else
    echo "Validation failed or incomplete"
    echo "Check logs:"
    echo "  /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_${SLURM_JOB_ID}.err"
    echo "  /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_${SLURM_JOB_ID}.log"
fi

echo ""
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
echo ""

exit $EXIT_STATUS
