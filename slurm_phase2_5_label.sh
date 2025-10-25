#!/bin/bash
#SBATCH --job-name=phase2_5_qwen_label
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_label_%j.out
#SBATCH --error=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_label_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#SBATCH --requeue

# ============================================================================
# PHASE 2.5: QWEN FEATURE LABELING
# Labeling ~3,600 features using Qwen1.5-72B-Chat
# Expected duration: 6-10 hours
# Resources: 4x L40S 48GB GPUs, 48 CPUs, All available memory
# ============================================================================

set -e  # Exit on error

# Create logs directory
mkdir -p /home/anshulk/cultural-alignment-study/outputs/logs

# Print job information
echo "=================================="
echo "SLURM Job Information"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Print resource allocation
echo "=================================="
echo "Resource Allocation"
echo "=================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "GPUs allocated: $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: All available on node"
echo "Time limit: $SLURM_JOB_TIME"
echo ""

# Activate conda environment
echo "=================================="
echo "Environment Setup"
echo "=================================="
source ~/.bashrc
conda activate rq1

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'rq1'"
    exit 1
fi

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Transformers installed: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

# Print GPU information
echo "=================================="
echo "GPU Information"
echo "=================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu --format=csv,noheader
echo ""

# Set environment variables
export HF_HOME=/data/hf_cache
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/transformers
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment Variables:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Navigate to project directory
cd /home/anshulk/cultural-alignment-study

# Validate prerequisites
echo "=================================="
echo "Pre-flight Validation"
echo "=================================="

echo "Checking Qwen model..."
MODEL_PATH="/data/models/huggingface/qwen/Qwen1.5-72B-Chat"
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model found: $MODEL_PATH"
else
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo ""
echo "Checking Phase 2 SAE outputs..."
python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path

# Check for SAE training outputs
runs = sorted(SAE_OUTPUT_ROOT.glob('triple_sae_*'))
if not runs:
    print('ERROR: No SAE training runs found')
    sys.exit(1)

latest_run = runs[-1]
print(f'✓ Latest SAE run: {latest_run.name}')

# Check for trained SAE models
sae_dirs = [d for d in latest_run.iterdir() if d.is_dir()]
if len(sae_dirs) < 9:
    print(f'ERROR: Expected 9 SAE models, found {len(sae_dirs)}')
    sys.exit(1)

print(f'✓ Found {len(sae_dirs)} trained SAE models')
"

VALIDATION_STATUS=$?
if [ $VALIDATION_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Pre-flight validation failed!"
    echo "Please ensure Phase 2 (SAE training) completed successfully."
    exit 1
fi

echo ""
echo "Checking extracted examples..."
python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path

examples_dir = SAE_OUTPUT_ROOT / 'feature_examples'
if not examples_dir.exists():
    print(f'ERROR: Examples directory not found: {examples_dir}')
    print('Please run: python scripts/phase2_5_extract_examples.py')
    sys.exit(1)

example_files = list(examples_dir.glob('*_examples.json'))
if len(example_files) < 9:
    print(f'ERROR: Expected 9 example files, found {len(example_files)}')
    sys.exit(1)

print(f'✓ Found {len(example_files)} example files')

# Count total features
import json
total_features = 0
for f in example_files:
    with open(f, 'r') as fp:
        data = json.load(fp)
        total_features += len(data)

print(f'✓ Total features to label: {total_features:,}')
"

VALIDATION_STATUS=$?
if [ $VALIDATION_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Pre-flight validation failed!"
    echo "Please run: python scripts/phase2_5_extract_examples.py"
    exit 1
fi

echo ""
echo "=================================="
echo "Starting Phase 2.5: Feature Labeling"
echo "=================================="
echo "Configuration:"
echo "  Model: Qwen1.5-72B-Chat (4-bit quantized)"
echo "  GPUs: 4x L40S 48GB (parallel workers)"
echo "  Features per file: ~400 (top features)"
echo "  Total files: 9 (split across 4 GPUs)"
echo "  Checkpoint saves: Every 100 features"
echo "  Memory cleanup: Every 50 features"
echo "  Estimated time: 6-10 hours"
echo ""

# Record start time
START_TIME=$(date +%s)
echo "Job started at: $(date)"
echo ""

# Run Phase 2.5 labeling with error handling
python scripts/phase2_5_qwen_label.py 2>&1 | tee -a /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_label_${SLURM_JOB_ID}.log

# Capture exit status
EXIT_STATUS=${PIPESTATUS[0]}

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=================================="
echo "Job Summary"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Exit status: $EXIT_STATUS"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Started: $(date -d @$START_TIME)"
echo "Completed: $(date -d @$END_TIME)"
echo ""

# Print output location if successful
if [ $EXIT_STATUS -eq 0 ]; then
    echo "=================================="
    echo "Labeling Completed Successfully!"
    echo "=================================="
    python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path
import json

output_file = SAE_OUTPUT_ROOT / 'labels_qwen_initial.json'
if output_file.exists():
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    coherent = sum(1 for r in results if r.get('is_coherent', False))
    coherent_pct = (coherent / len(results)) * 100 if results else 0
    
    print(f'Labels saved to:')
    print(f'  {output_file}')
    print(f'')
    print(f'Summary:')
    print(f'  Total features labeled: {len(results):,}')
    print(f'  Coherent features: {coherent:,} ({coherent_pct:.1f}%)')
    print(f'  Incoherent features: {len(results) - coherent:,} ({100 - coherent_pct:.1f}%)')
    print(f'')
    print(f'GPU outputs:')
    for gpu_id in [0, 1, 2, 3]:
        gpu_file = SAE_OUTPUT_ROOT / f'labels_qwen_initial_gpu{gpu_id}.json'
        if gpu_file.exists():
            with open(gpu_file, 'r') as gf:
                gpu_results = json.load(gf)
            print(f'  ✓ GPU {gpu_id}: {len(gpu_results):,} features')
        else:
            print(f'  ✗ GPU {gpu_id}: output file missing')
else:
    print('WARNING: No output file found')
    print(f'Expected: {output_file}')
"
else
    echo "=================================="
    echo "Labeling Failed or Incomplete"
    echo "=================================="
    echo "Exit status: $EXIT_STATUS"
    echo ""
    echo "Checking for checkpoint files..."
    python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path
import json

checkpoints_found = False
for gpu_id in [0, 1, 2, 3]:
    checkpoint_file = SAE_OUTPUT_ROOT / f'labels_qwen_initial_gpu{gpu_id}_checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f'  ✓ GPU {gpu_id} checkpoint: {len(data):,} features')
        checkpoints_found = True

if checkpoints_found:
    print('')
    print('Checkpoint files exist - you can resume by rerunning:')
    print('  sbatch slurm_phase2_5_label.sh')
else:
    print('  No checkpoint files found')
"
    echo ""
    echo "Check logs:"
    echo "  Error log: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_label_${SLURM_JOB_ID}.err"
    echo "  Full log: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_label_${SLURM_JOB_ID}.log"
    echo ""
    echo "Individual GPU logs:"
    for i in 0 1 2 3; do
        echo "  GPU $i: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_qwen_gpu${i}.log"
    done
fi

# Print final GPU state
echo ""
echo "=================================="
echo "Final GPU State"
echo "=================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
echo ""

# Print resource usage statistics
echo "=================================="
echo "Job Resource Usage"
echo "=================================="
sstat --format=JobID,MaxRSS,MaxVMSize,AveCPU -j $SLURM_JOB_ID 2>/dev/null || echo "Resource stats will be available after job completion (use: sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,AveCPU)"

echo ""
echo "For detailed efficiency report after completion, run:"
echo "  seff $SLURM_JOB_ID"
echo ""

if [ $EXIT_STATUS -eq 0 ]; then
    echo "=================================="
    echo "Next Steps"
    echo "=================================="
    echo "1. Review labeling results:"
    echo "   less /home/anshulk/cultural-alignment-study/outputs/sae_models/labels_qwen_initial.json"
    echo ""
    echo "2. Run validation script:"
    echo "   sbatch slurm_phase2_5_validate.sh"
    echo ""
    echo "   OR interactively:"
    echo "   python scripts/phase2_5_qwen_validate.py"
    echo ""
fi

# Exit with the status from the Python script
exit $EXIT_STATUS
