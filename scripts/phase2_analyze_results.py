"""
Phase 2 Results Analysis
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

from configs.config import (
    SAE_OUTPUT_ROOT,
    TARGET_LAYERS,
    VALIDATION_THRESHOLDS,
    setup_logger
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger('phase2_analysis', f'phase2_analysis_{timestamp}.log')


def find_latest_sae_run(sae_output_root: Path) -> Path:
    run_dirs = sorted(sae_output_root.glob("triple_sae_*"))
    if not run_dirs:
        raise ValueError(f"No SAE runs found in {sae_output_root}")
    return run_dirs[-1]


def load_sae_checkpoint(checkpoint_path: Path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        train_history = checkpoint.get('train_history', [])
        val_history = checkpoint.get('val_history', [])
        
        final_train = train_history[-1] if train_history else {}
        final_val = val_history[-1] if val_history else {}
        
        return {
            'epoch': checkpoint.get('epoch', -1),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'final_train_loss': final_train.get('total_loss', None),
            'final_train_recon': final_train.get('recon_loss', None),
            'final_train_l0': final_train.get('l0_sparsity', None),
            'final_val_loss': final_val.get('total_loss', None),
            'final_val_recon': final_val.get('recon_loss', None),
            'final_val_l0': final_val.get('l0_sparsity', None),
        }
    except Exception as e:
        logger.error(f"Failed to load {checkpoint_path}: {e}")
        return None


def aggregate_sae_results(run_dir: Path):
    model_types = ['base', 'chat', 'delta']
    results = []
    
    for model_type in model_types:
        for layer_idx in TARGET_LAYERS:
            sae_dir = run_dir / f"{model_type}_layer{layer_idx}"
            best_model_path = sae_dir / "best_model.pt"
            
            if not best_model_path.exists():
                logger.warning(f"Missing: {best_model_path}")
                continue
            
            logger.info(f"Loading: {model_type} layer {layer_idx}")
            metrics = load_sae_checkpoint(best_model_path)
            
            if metrics:
                results.append({
                    'model_type': model_type,
                    'layer': layer_idx,
                    'best_val_loss': metrics['best_val_loss'],
                    'final_train_loss': metrics['final_train_loss'],
                    'final_train_recon': metrics['final_train_recon'],
                    'final_train_l0': metrics['final_train_l0'],
                    'final_val_loss': metrics['final_val_loss'],
                    'final_val_recon': metrics['final_val_recon'],
                    'final_val_l0': metrics['final_val_l0'],
                    'epochs_trained': metrics['epoch'],
                })
    
    return pd.DataFrame(results)


def compute_validation_status(df):
    recon_threshold = VALIDATION_THRESHOLDS['reconstruction_loss']
    
    df['meets_recon_threshold'] = df['final_val_recon'] < recon_threshold
    df['sparsity_ratio'] = 1.0 / df['final_val_l0']
    df['meets_sparsity_target'] = df['sparsity_ratio'] >= 10.0
    df['passes_validation'] = df['meets_recon_threshold'] & df['meets_sparsity_target']
    
    return df


def generate_summary_report(df, run_dir: Path):
    report_path = run_dir / "PHASE2_SUMMARY.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: TRIPLE SAE TRAINING - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Training Run: {run_dir.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        total_saes = len(df)
        passing_saes = df['passes_validation'].sum()
        
        f.write(f"Total SAEs Trained: {total_saes}\n")
        f.write(f"SAEs Passing Validation: {passing_saes}/{total_saes}\n")
        f.write(f"Success Rate: {passing_saes/total_saes*100:.1f}%\n\n")
        
        f.write("Reconstruction Loss (Threshold: <0.05):\n")
        f.write(f"  Mean: {df['final_val_recon'].mean():.6f}\n")
        f.write(f"  Median: {df['final_val_recon'].median():.6f}\n")
        f.write(f"  Min: {df['final_val_recon'].min():.6f}\n")
        f.write(f"  Max: {df['final_val_recon'].max():.6f}\n")
        f.write(f"  SAEs Meeting Threshold: {df['meets_recon_threshold'].sum()}/{total_saes}\n\n")
        
        f.write("L0 Sparsity (Target: 0.1 for 10× baseline):\n")
        f.write(f"  Mean L0: {df['final_val_l0'].mean():.3f}\n")
        f.write(f"  Median L0: {df['final_val_l0'].median():.3f}\n")
        f.write(f"  Mean Sparsity Ratio: {df['sparsity_ratio'].mean():.2f}×\n")
        f.write(f"  SAEs Meeting Target: {df['meets_sparsity_target'].sum()}/{total_saes}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PER-MODEL TYPE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for model_type in ['base', 'chat', 'delta']:
            subset = df[df['model_type'] == model_type]
            f.write(f"{model_type.upper()} Model:\n")
            f.write(f"  Avg Reconstruction Loss: {subset['final_val_recon'].mean():.6f}\n")
            f.write(f"  Avg L0 Sparsity: {subset['final_val_l0'].mean():.3f}\n")
            f.write(f"  Avg Sparsity Ratio: {subset['sparsity_ratio'].mean():.2f}×\n")
            f.write(f"  Passing Validation: {subset['passes_validation'].sum()}/{len(subset)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PER-LAYER SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for layer in TARGET_LAYERS:
            subset = df[df['layer'] == layer]
            f.write(f"Layer {layer}:\n")
            f.write(f"  Avg Reconstruction Loss: {subset['final_val_recon'].mean():.6f}\n")
            f.write(f"  Avg L0 Sparsity: {subset['final_val_l0'].mean():.3f}\n")
            f.write(f"  Avg Sparsity Ratio: {subset['sparsity_ratio'].mean():.2f}×\n")
            f.write(f"  Passing Validation: {subset['passes_validation'].sum()}/{len(subset)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("INDIVIDUAL SAE DETAILS\n")
        f.write("="*80 + "\n\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['model_type'].upper()} Layer {row['layer']}:\n")
            f.write(f"  Epochs: {row['epochs_trained']}\n")
            f.write(f"  Best Val Loss: {row['best_val_loss']:.6f}\n")
            f.write(f"  Final Val Recon: {row['final_val_recon']:.6f}\n")
            f.write(f"  Final Val L0: {row['final_val_l0']:.4f}\n")
            f.write(f"  Sparsity Ratio: {row['sparsity_ratio']:.2f}×\n")
            f.write(f"  Passes Validation: {'✓' if row['passes_validation'] else '✗'}\n\n")
    
    logger.info(f"Summary report saved: {report_path}")


def export_results_csv(df, run_dir: Path):
    csv_path = run_dir / "sae_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results exported to: {csv_path}")


def main():
    try:
        logger.info("="*80)
        logger.info("PHASE 2 RESULTS ANALYSIS")
        logger.info("="*80)
        
        logger.info("\nFinding latest SAE training run...")
        run_dir = find_latest_sae_run(SAE_OUTPUT_ROOT)
        logger.info(f"Analyzing: {run_dir.name}")
        
        logger.info("\nAggregating results from all SAEs...")
        df = aggregate_sae_results(run_dir)
        
        if df.empty:
            logger.error("No results found!")
            return
        
        logger.info(f"Loaded {len(df)} SAE results")
        
        logger.info("\nComputing validation status...")
        df = compute_validation_status(df)
        
        logger.info("\nGenerating summary report...")
        generate_summary_report(df, run_dir)
        
        logger.info("\nExporting results...")
        export_results_csv(df, run_dir)
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {run_dir}/")
        logger.info(f"  - Summary report: PHASE2_SUMMARY.txt")
        logger.info(f"  - CSV export: sae_results.csv")
        
        passing = df['passes_validation'].sum()
        total = len(df)
        logger.info(f"\nValidation Status: {passing}/{total} SAEs passing")
        
        if passing < total:
            logger.warning("\n⚠️  Some SAEs did not meet validation criteria")
            logger.warning("   Review summary report for details")
        else:
            logger.info("\n✓  All SAEs passed validation!")
            logger.info("   Ready for Phase 2.5: Feature Labeling")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
