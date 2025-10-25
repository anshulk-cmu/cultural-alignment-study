#!/bin/bash
#SBATCH --job-name=phase2_sae_training
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_sae_%j.out
#SBATCH --error=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_sae_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#SBATCH --requeue

# ============================================================================
# PHASE 2: SAE TRAINING
# Training 9 SAEs (3 model types × 3 layers) with 100 epochs each
# Expected duration: 24-30 hours
# Resources: 4x A6000 GPUs, 48 CPUs, All available memory
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

# Validate Phase 1 completion
echo "=================================="
echo "Pre-flight Validation"
echo "=================================="
echo "Checking Phase 1 outputs..."

python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import ACTIVATION_ROOT, TARGET_LAYERS
from pathlib import Path

print('Searching for Phase 1 runs...')
runs = sorted(ACTIVATION_ROOT.glob('run_*'))
if not runs:
    print('ERROR: No Phase 1 runs found in', ACTIVATION_ROOT)
    sys.exit(1)

latest_run = runs[-1]
print(f'✓ Latest run: {latest_run.name}')
print(f'  Path: {latest_run}')

# Check for required datasets
datasets = ['updesh_beta', 'snli_control', 'hindi_control']
model_types = ['base', 'chat', 'delta']
missing = []

print(f'\nValidating data for {len(TARGET_LAYERS)} layers...')
for ds in datasets:
    for mt in model_types:
        path = latest_run / ds / mt
        if not path.exists():
            missing.append(f'{ds}/{mt}')
        else:
            chunks = list(path.glob('chunk_*.npz'))
            if chunks:
                print(f'  ✓ {ds}/{mt}: {len(chunks)} chunks')
            else:
                missing.append(f'{ds}/{mt} (no chunks)')

if missing:
    print(f'\nERROR: Missing or incomplete directories:')
    for m in missing:
        print(f'  - {m}')
    sys.exit(1)

print(f'\n✓ All Phase 1 data validated successfully')
print(f'✓ Ready to train {len(model_types)} × {len(TARGET_LAYERS)} = {len(model_types) * len(TARGET_LAYERS)} SAEs')
"

VALIDATION_STATUS=$?
if [ $VALIDATION_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Pre-flight validation failed!"
    echo "Please ensure Phase 1 (activation extraction) completed successfully."
    exit 1
fi

echo ""
echo "=================================="
echo "Starting Phase 2: SAE Training"
echo "=================================="
echo "Configuration:"
echo "  Total SAEs: 9 (3 model types × 3 layers)"
echo "  Model types: base, chat, delta"
echo "  Layers: 6, 12, 18"
echo "  Epochs per SAE: 100"
echo "  Batch size: 256"
echo "  Dictionary size: 8192"
echo "  Sparsity K: 256"
echo "  GPUs: 3 (DataParallel on GPUs 0,1,2)"
echo "  Estimated time: 24-30 hours"
echo ""

# Record start time
START_TIME=$(date +%s)
echo "Job started at: $(date)"
echo ""

# Run Phase 2 training with error handling
python scripts/phase2_train_saes.py 2>&1 | tee -a /home/anshulk/cultural-alignment-study/outputs/logs/phase2_training_${SLURM_JOB_ID}.log

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
    echo "Training Completed Successfully!"
    echo "=================================="
    python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path

runs = sorted(SAE_OUTPUT_ROOT.glob('triple_sae_*'))
if runs:
    latest = runs[-1]
    print(f'SAE models saved to:')
    print(f'  {latest}')

    sae_dirs = [d for d in latest.iterdir() if d.is_dir()]
    print(f'\nTrained SAEs: {len(sae_dirs)}')
    for sae_dir in sorted(sae_dirs):
        best_model = sae_dir / 'best_model.pt'
        if best_model.exists():
            size_mb = best_model.stat().st_size / (1024**2)
            print(f'  ✓ {sae_dir.name} ({size_mb:.1f} MB)')
        else:
            print(f'  ✗ {sae_dir.name} (model missing)')
else:
    print('WARNING: No SAE output found')
"
else
    echo "=================================="
    echo "Training Failed"
    echo "=================================="
    echo "Exit status: $EXIT_STATUS"
    echo "Check error log: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_sae_${SLURM_JOB_ID}.err"
    echo "Check training log: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_training_${SLURM_JOB_ID}.log"
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

# Exit with the status from the Python script
exit $EXIT_STATUS
