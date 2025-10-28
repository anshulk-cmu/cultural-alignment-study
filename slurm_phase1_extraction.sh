#!/bin/bash
#SBATCH --job-name=phase1_extract
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/anshulk/cultural-alignment-study/outputs/logs/phase1_%j.out
#SBATCH --error=/home/anshulk/cultural-alignment-study/outputs/logs/phase1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

source ~/.bashrc
conda activate rq1
cd /home/anshulk/cultural-alignment-study
python scripts/phase1_extract_activations.py
