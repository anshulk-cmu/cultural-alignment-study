import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import torch
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

from configs.config import (
    SAE_OUTPUT_ROOT,
    TARGET_LAYERS,
    VALIDATION_THRESHOLDS,
    setup_logger
)

logger = setup_logger('phase2_analysis', 'phase2_analysis.log')


def find_latest_run(output_root: Path) -> Path:
    run_dirs = sorted(output_root.glob("triple_sae_*"))
    if not run_dirs:
        raise ValueError(f"No training runs found in {output_root}")
    return run_dirs[-1]


def load_best_model(save_dir: Path):
    best_model_path = save_dir / "best_model.pt"
    if not best_model_path.exists():
        return None
    
    checkpoint = torch.load(best_model_path, map_location='cpu')
    return checkpoint


def analyze_training_run(run_dir: Path):
    logger.info(f"Analyzing run: {run_dir.name}")
    
    results = []
    
    model_types = ['base', 'chat', 'delta']
    
    for model_type in model_types:
        for layer_idx in TARGET_LAYERS:
            save_dir = run_dir / f"{model_type}_layer{layer_idx}"
            
            if not save_dir.exists():
                logger.warning(f"Directory not found: {save_dir}")
                continue
            
            checkpoint = load_best_model(save_dir)
            
            if checkpoint is None:
                logger.warning(f"No checkpoint found in {save_dir}")
                continue
            
            val_history = checkpoint.get('val_history', [])
            
            if not val_history:
                logger.warning(f"No validation history for {model_type} layer {layer_idx}")
                continue
            
            final_val = val_history[-1]
            
            recon_loss = final_val.get('recon_loss', float('inf'))
            l0_sparsity = final_val.get('l0_sparsity', 0.0)
            aux_loss = final_val.get('aux_loss', 0.0)
            
            l0_baseline = 1.0
            sparsity_ratio = l0_baseline / l0_sparsity if l0_sparsity > 0 else 0
            
            passes_recon = recon_loss < VALIDATION_THRESHOLDS['reconstruction_loss']
            passes_sparsity = sparsity_ratio > VALIDATION_THRESHOLDS['l0_sparsity_ratio']
            passes_validation = passes_recon and passes_sparsity
            
            results.append({
                'model_type': model_type,
                'layer': layer_idx,
                'best_val_loss': checkpoint['best_val_loss'],
                'final_val_recon': recon_loss,
                'final_val_aux': aux_loss,
                'final_val_l0': l0_sparsity,
                'epochs_trained': checkpoint['epoch'],
                'sparsity_ratio': sparsity_ratio,
                'passes_recon': passes_recon,
                'passes_sparsity': passes_sparsity,
                'passes_validation': passes_validation
            })
            
            logger.info(f"{model_type} layer {layer_idx}:")
            logger.info(f"  Recon loss: {recon_loss:.6f} ({'PASS' if passes_recon else 'FAIL'})")
            logger.info(f"  Aux loss: {aux_loss:.6f}")
            logger.info(f"  L0 sparsity: {l0_sparsity:.4f} ({sparsity_ratio:.1f}x)")
            logger.info(f"  Validation: {'PASS' if passes_validation else 'FAIL'}")
    
    df = pd.DataFrame(results)
    
    summary_path = run_dir / "PHASE2_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: TRIPLE SAE TRAINING WITH AUXILIARY LOSS - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Run: {run_dir.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total SAEs Trained: {len(df)}\n")
        f.write(f"SAEs Passing Validation: {df['passes_validation'].sum()}/{len(df)}\n")
        f.write(f"Success Rate: {100*df['passes_validation'].mean():.1f}%\n\n")
        
        f.write(f"Reconstruction Loss (Threshold: <{VALIDATION_THRESHOLDS['reconstruction_loss']}):\n")
        f.write(f"  Mean: {df['final_val_recon'].mean():.6f}\n")
        f.write(f"  Median: {df['final_val_recon'].median():.6f}\n")
        f.write(f"  Min: {df['final_val_recon'].min():.6f}\n")
        f.write(f"  Max: {df['final_val_recon'].max():.6f}\n")
        f.write(f"  SAEs Meeting Threshold: {df['passes_recon'].sum()}/{len(df)}\n\n")
        
        f.write(f"Auxiliary Loss:\n")
        f.write(f"  Mean: {df['final_val_aux'].mean():.6f}\n")
        f.write(f"  Median: {df['final_val_aux'].median():.6f}\n\n")
        
        f.write(f"L0 Sparsity:\n")
        f.write(f"  Mean L0: {df['final_val_l0'].mean():.4f}\n")
        f.write(f"  Mean Sparsity Ratio: {df['sparsity_ratio'].mean():.2f}×\n\n")
        
        f.write("="*80 + "\n")
        f.write("PER-MODEL TYPE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for model_type in ['base', 'chat', 'delta']:
            model_df = df[df['model_type'] == model_type]
            f.write(f"{model_type.upper()} Model:\n")
            f.write(f"  Avg Reconstruction Loss: {model_df['final_val_recon'].mean():.6f}\n")
            f.write(f"  Avg Auxiliary Loss: {model_df['final_val_aux'].mean():.6f}\n")
            f.write(f"  Avg L0 Sparsity: {model_df['final_val_l0'].mean():.4f}\n")
            f.write(f"  Passing Validation: {model_df['passes_validation'].sum()}/{len(model_df)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("INDIVIDUAL SAE DETAILS\n")
        f.write("="*80 + "\n\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['model_type'].upper()} Layer {row['layer']}:\n")
            f.write(f"  Epochs: {row['epochs_trained']}\n")
            f.write(f"  Best Val Loss: {row['best_val_loss']:.6f}\n")
            f.write(f"  Final Val Recon: {row['final_val_recon']:.6f}\n")
            f.write(f"  Final Val Aux: {row['final_val_aux']:.6f}\n")
            f.write(f"  Final Val L0: {row['final_val_l0']:.4f}\n")
            f.write(f"  Sparsity Ratio: {row['sparsity_ratio']:.2f}×\n")
            f.write(f"  Passes Validation: {'✓' if row['passes_validation'] else '✗'}\n\n")
    
    logger.info(f"\nSummary saved to: {summary_path}")
    
    csv_path = run_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to: {csv_path}")
    
    return df


def main():
    logger.info("="*80)
    logger.info("PHASE 2: RESULTS ANALYSIS")
    logger.info("="*80)
    
    try:
        run_dir = find_latest_run(SAE_OUTPUT_ROOT)
        logger.info(f"\nAnalyzing run: {run_dir}")
        
        df = analyze_training_run(run_dir)
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"\nSummary:")
        logger.info(f"  Total SAEs: {len(df)}")
        logger.info(f"  Passing: {df['passes_validation'].sum()}")
        logger.info(f"  Success Rate: {100*df['passes_validation'].mean():.1f}%")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
