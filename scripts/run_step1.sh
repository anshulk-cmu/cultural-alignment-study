#!/bin/bash
#SBATCH --job-name=step1_eval
#SBATCH --partition=preempt
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/anshulk/cultural-mi/logs/step1_slurm_%j.out

echo "=========================================="
echo "Step 1: Behavioral Evaluation"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Time:      $(date)"
echo "=========================================="

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate cultural

nvidia-smi

python /home/anshulk/cultural-mi/scripts/eval_step1.py "$@"

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
