#!/bin/bash
#SBATCH --job-name=phase2_5_qwen3_validate
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_%j.out
#SBATCH --error=/home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#SBATCH --requeue

# ============================================================================
# PHASE 2.5: QWEN3 FEATURE VALIDATION (4 GPUs - REPRODUCIBLE)
# Validating feature labels using Qwen3-30B-A3B-Instruct-2507
# Expected duration: 8-10 hours with 4 GPUs
# Resources: 4x L40S 48GB GPUs, 48 CPUs, All available memory
# Generation: GREEDY DECODING (deterministic), SEED=42
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
export PYTHONHASHSEED=42

echo "Environment Variables:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  PYTHONHASHSEED: 42 (reproducibility)"
echo ""

# Navigate to project directory
cd /home/anshulk/cultural-alignment-study

# Validate prerequisites
echo "=================================="
echo "Pre-flight Validation"
echo "=================================="

echo "Checking Qwen3 model..."
MODEL_PATH="/data/user_data/anshulk/data/models/Qwen3-30B-A3B-Instruct-2507"
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model found: $MODEL_PATH"
else
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo ""
echo "Checking initial labels..."
python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path
import json

# Check labels file on compute node storage
labels_file = SAE_OUTPUT_ROOT / 'labels_qwen_initial.json'
print(f'Looking for: {labels_file}')

if not labels_file.exists():
    print(f'ERROR: Initial labels not found: {labels_file}')
    sys.exit(1)

with open(labels_file, 'r') as f:
    labels = json.load(f)
print(f'✓ Initial labels found: {len(labels)} features')

# Check examples directory on compute node storage
examples_dir = SAE_OUTPUT_ROOT / 'feature_examples'
print(f'Looking for examples in: {examples_dir}')

if not examples_dir.exists():
    print(f'ERROR: Examples directory not found: {examples_dir}')
    sys.exit(1)

example_files = list(examples_dir.glob('*_examples.json'))
if len(example_files) < 9:
    print(f'ERROR: Expected 9 example files, found {len(example_files)}')
    sys.exit(1)

print(f'✓ Examples directory found: {len(example_files)} files')
"

VALIDATION_STATUS=$?
if [ $VALIDATION_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Pre-flight validation failed!"
    echo "Please ensure Phase 2.5 labeling completed successfully."
    exit 1
fi

# Check for existing checkpoints (4 GPUs)
echo ""
echo "Checking for existing checkpoints..."
python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path
import json

checkpoints_found = False
total_checkpointed = 0

for gpu_id in [0, 1, 2, 3]:
    checkpoint_file = SAE_OUTPUT_ROOT / f'labels_qwen3_validated_gpu{gpu_id}_checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f'  ✓ GPU {gpu_id} checkpoint exists: {len(data)} features already validated')
        total_checkpointed += len(data)
        checkpoints_found = True

if checkpoints_found:
    print(f'  Total checkpointed across all GPUs: {total_checkpointed} features')
    print('  Will resume from checkpoints and redistribute work across 4 GPUs')
else:
    print('  No existing checkpoints found - starting fresh')
"

echo ""
echo "=================================="
echo "Starting Phase 2.5: Feature Validation"
echo "=================================="
echo "Configuration:"
echo "  Model: Qwen3-30B-A3B-Instruct-2507 (8-bit quantized)"
echo "  GPUs: 4x L40S 48GB (parallel workers)"
echo "  Input: SAE_OUTPUT_ROOT/labels_qwen_initial.json"
echo "  Output: SAE_OUTPUT_ROOT/labels_qwen3_validated.json"
echo "  Storage: /data/user_data/anshulk/.../sae_models/ (compute node)"
echo "  Generation: GREEDY DECODING (deterministic)"
echo "  Seed: 42 (reproducibility)"
echo "  Checkpoint saves: Every 100 features"
echo "  Memory cleanup: Every 50 features"
echo "  Resume supported: Rerun if interrupted"
echo "  Estimated time: 8-10 hours with 4 GPUs"
echo ""

# Record start time
START_TIME=$(date +%s)
echo "Job started at: $(date)"
echo ""

# Run Phase 2.5 validation with error handling
python scripts/phase2_5_qwen_validate.py 2>&1 | tee -a /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_${SLURM_JOB_ID}.log

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
    echo "Validation Completed Successfully!"
    echo "=================================="
    python -c "
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT
from pathlib import Path
import json

output_file = SAE_OUTPUT_ROOT / 'labels_qwen3_validated.json'
if output_file.exists():
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    stats = {
        'KEEP': sum(1 for r in results if r.get('validation_action') == 'KEEP'),
        'REVISE': sum(1 for r in results if r.get('validation_action') == 'REVISE'),
        'INVALIDATE': sum(1 for r in results if r.get('validation_action') == 'INVALIDATE'),
        'ERROR': sum(1 for r in results if r.get('validation_action') == 'ERROR'),
    }
    
    final_valid = stats['KEEP'] + stats['REVISE']
    
    print(f'Labels saved to:')
    print(f'  {output_file}')
    print(f'')
    print(f'Summary:')
    print(f'  Seed: 42 (reproducible)')
    print(f'  Total features validated: {len(results):,}')
    print(f'  KEEP:       {stats[\"KEEP\"]:>6,} ({stats[\"KEEP\"]/len(results)*100:>5.1f}%)')
    print(f'  REVISE:     {stats[\"REVISE\"]:>6,} ({stats[\"REVISE\"]/len(results)*100:>5.1f}%)')
    print(f'  INVALIDATE: {stats[\"INVALIDATE\"]:>6,} ({stats[\"INVALIDATE\"]/len(results)*100:>5.1f}%)')
    print(f'  ERROR:      {stats[\"ERROR\"]:>6,} ({stats[\"ERROR\"]/len(results)*100:>5.1f}%)')
    print(f'  ---')
    print(f'  Final Valid (Keep+Revise): {final_valid:>6,} ({final_valid/len(results)*100:>5.1f}%)')
    print(f'')
    print(f'GPU outputs:')
    for gpu_id in [0, 1, 2, 3]:
        gpu_file = SAE_OUTPUT_ROOT / f'labels_qwen3_validated_gpu{gpu_id}.json'
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
    echo "Validation Failed or Incomplete"
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
total_checkpointed = 0

for gpu_id in [0, 1, 2, 3]:
    checkpoint_file = SAE_OUTPUT_ROOT / f'labels_qwen3_validated_gpu{gpu_id}_checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f'  ✓ GPU {gpu_id} checkpoint: {len(data):,} features')
        total_checkpointed += len(data)
        checkpoints_found = True

if checkpoints_found:
    print(f'')
    print(f'Total checkpointed: {total_checkpointed:,} features')
    print(f'')
    print('Checkpoint files exist - you can resume by rerunning:')
    print('  sbatch slurm_phase2_5_validate.sh')
    print('')
    print('The script will automatically:')
    print('  - Load existing checkpoints from all 4 GPUs')
    print('  - Skip already validated features')
    print('  - Redistribute remaining work across 4 GPUs')
    print('  - Continue from where it left off')
else:
    print('  No checkpoint files found')
    print('  This was likely an early failure (model loading, etc.)')
"
    echo ""
    echo "Check logs for details:"
    echo "  Error log: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_${SLURM_JOB_ID}.err"
    echo "  Full log: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_${SLURM_JOB_ID}.log"
    echo ""
    echo "Individual GPU logs:"
    for i in 0 1 2 3; do
        echo "  GPU $i: /home/anshulk/cultural-alignment-study/outputs/logs/phase2_5_validate_gpu${i}.log"
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
    echo "1. Review validation results:"
    echo "   python -c 'import sys; sys.path.append(\"/home/anshulk/cultural-alignment-study\"); from configs.config import SAE_OUTPUT_ROOT; import json; f=open(SAE_OUTPUT_ROOT/\"labels_qwen3_validated.json\"); data=json.load(f); print(f\"Total: {len(data)}\"); print(f\"Sample: {data[0]}\")'"
    echo ""
    echo "2. Analyze validated features:"
    echo "   python scripts/analyze_validated_features.py"
    echo ""
    echo "3. Continue with Phase 3 (Cultural Feature Prioritization)"
    echo ""
fi

# Exit with the status from the Python script
exit $EXIT_STATUS
