#!/bin/bash
#SBATCH --job-name=dosa_prep
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/anshulk/cultural-alignment-study/outputs/logs/dosa_prep_%j.out
#SBATCH --error=/home/anshulk/cultural-alignment-study/outputs/logs/dosa_prep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#SBATCH --requeue

# ============================================================================
# DOSA Dataset Preparation for CST
# Download all DOSA artifacts and generate natural language statements
# Expected duration: 2-4 hours
# Resources: 1x A6000 GPU, 16 CPUs, 64GB RAM
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
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "Time limit: 4 hours"
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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment Variables:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Navigate to project directory
cd /home/anshulk/cultural-alignment-study

# Pre-flight checks
echo "=================================="
echo "Pre-flight Validation"
echo "=================================="
echo "Checking output directory..."
mkdir -p /data/user_data/anshulk/cultural-alignment-study/data
echo "✓ Output directory ready"
echo ""

echo "Checking model availability..."
python -c "
from pathlib import Path
model_path = Path('/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct')
if not model_path.exists():
    print(f'ERROR: Model not found at {model_path}')
    exit(1)
print(f'✓ Model found: {model_path}')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Pre-flight validation failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "Starting DOSA Data Preparation"
echo "=================================="
echo "Pipeline:"
echo "  1. Download all DOSA artifacts from GitHub"
echo "     - 18 states × (CSV files + TXT files)"
echo "     - Expected: ~615 unique artifacts"
echo ""
echo "  2. Consolidate and deduplicate"
echo "     - Handle different file formats"
echo "     - Merge duplicate artifacts"
echo "     - Save: dosa_consolidated_original.csv"
echo ""
echo "  3. Generate natural language statements"
echo "     - Model: Llama-3.2-3B-Instruct"
echo "     - Convert clues → natural paragraphs"
echo "     - Save: dosa_cst_samples.csv"
echo ""

# Record start time
START_TIME=$(date +%s)
echo "Job started at: $(date)"
echo ""

# Run DOSA preparation script
python scripts/prepare_dosa_for_cst.py 2>&1 | tee -a /home/anshulk/cultural-alignment-study/outputs/logs/dosa_prep_${SLURM_JOB_ID}.log

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

# Print results if successful
if [ $EXIT_STATUS -eq 0 ]; then
    echo "=================================="
    echo "Processing Completed Successfully!"
    echo "=================================="
    
    python -c "
import pandas as pd
from pathlib import Path
import json

data_dir = Path('/data/user_data/anshulk/cultural-alignment-study/data')

# Check consolidated file
consolidated = data_dir / 'dosa_consolidated_original.csv'
if consolidated.exists():
    df = pd.read_csv(consolidated)
    print(f'✓ Consolidated dataset: {len(df)} artifacts')
    print(f'  File: {consolidated}')
    print(f'  Size: {consolidated.stat().st_size / (1024**2):.2f} MB')
    print()
    
    # Print per-state counts
    print('  Artifacts by state:')
    state_counts = df['state'].value_counts().sort_index()
    for state, count in state_counts.items():
        print(f'    {state:20} {count:3} artifacts')
else:
    print('✗ Consolidated file not found')

print()

# Check CST samples file
cst = data_dir / 'dosa_cst_samples.csv'
if cst.exists():
    df = pd.read_csv(cst)
    print(f'✓ CST samples: {len(df)} statements generated')
    print(f'  File: {cst}')
    print(f'  Size: {cst.stat().st_size / (1024**2):.2f} MB')
else:
    print('✗ CST samples file not found')

print()

# Check report
report = data_dir / 'dosa_processing_report.json'
if report.exists():
    with open(report) as f:
        data = json.load(f)
    print(f'✓ Processing report:')
    print(f'  Total downloaded: {data.get(\"total_artifacts_downloaded\", \"N/A\")}')
    print(f'  Unique artifacts: {data.get(\"total_unique_artifacts\", \"N/A\")}')
    print(f'  Statements generated: {data.get(\"total_statements_generated\", \"N/A\")}')
    print(f'  File: {report}')
else:
    print('✗ Report file not found')
"
    
    echo ""
    echo "=================================="
    echo "Next Steps"
    echo "=================================="
    echo "1. Review generated statements in dosa_cst_samples.csv"
    echo "2. Use for Causal Sufficiency Testing (CST)"
    echo "3. Run samples through Qwen models"
    echo "4. Perform feature ablation analysis"
    
else
    echo "=================================="
    echo "Processing Failed"
    echo "=================================="
    echo "Exit status: $EXIT_STATUS"
    echo ""
    echo "Check logs:"
    echo "  Error log: /home/anshulk/cultural-alignment-study/outputs/logs/dosa_prep_${SLURM_JOB_ID}.err"
    echo "  Full log: /home/anshulk/cultural-alignment-study/outputs/logs/dosa_prep_${SLURM_JOB_ID}.log"
    echo ""
    echo "Common issues:"
    echo "  - Network connectivity (GitHub download failures)"
    echo "  - Model loading errors (check model path)"
    echo "  - Insufficient memory (GPU OOM)"
    echo "  - Disk space (check /data partition)"
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
echo "For detailed resource usage, run after completion:"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,AveCPU,MaxDiskRead,MaxDiskWrite"
echo "  seff $SLURM_JOB_ID"
echo ""

# Exit with the status from the Python script
exit $EXIT_STATUS
