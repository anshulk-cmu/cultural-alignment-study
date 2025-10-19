"""
Phase 2 Results Analysis: Aggregate metrics from all 9 trained SAEs
Generates comprehensive summary report and visualizations
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

from configs.config import (
    SAE_OUTPUT_ROOT,
    TARGET_LAYERS,
    VALIDATION_THRESHOLDS,
    setup_logger
)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger('phase2_analysis', f'phase2_analysis_{timestamp}.log')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def find_latest_sae_run(sae_output_root: Path) -> Path:
    """Find most recent SAE training run"""
    run_dirs = sorted(sae_output_root.glob("triple_sae_*"))
    if not run_dirs:
        raise ValueError(f"No SAE runs found in {sae_output_root}")
    return run_dirs[-1]


def load_sae_checkpoint(checkpoint_path: Path) -> Dict:
    """Load SAE checkpoint and extract metrics"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract final metrics
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
            'train_history': train_history,
            'val_history': val_history
        }
    except Exception as e:
        logger.error(f"Failed to load {checkpoint_path}: {e}")
        return None


def aggregate_sae_results(run_dir: Path) -> pd.DataFrame:
    """Aggregate results from all 9 SAEs"""
    
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
                    'checkpoint_path': str(best_model_path)
                })
    
    return pd.DataFrame(results)


def compute_validation_status(df: pd.DataFrame) -> pd.DataFrame:
    """Check which SAEs meet validation thresholds"""
    
    recon_threshold = VALIDATION_THRESHOLDS['reconstruction_loss']
    sparsity_target = 0.1  # 10% active = 10× baseline sparsity
    
    df['meets_recon_threshold'] = df['final_val_recon'] < recon_threshold
    df['sparsity_ratio'] = 1.0 / df['final_val_l0']  # Actual sparsity ratio
    df['meets_sparsity_target'] = df['sparsity_ratio'] >= 10.0
    df['passes_validation'] = df['meets_recon_threshold'] & df['meets_sparsity_target']
    
    return df


def generate_summary_report(df: pd.DataFrame, run_dir: Path):
    """Generate comprehensive text summary"""
    
    report_path = run_dir / "PHASE2_SUMMARY.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: TRIPLE SAE TRAINING - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Training Run: {run_dir.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        total_saes = len(df)
        passing_saes = df['passes_validation'].sum()
        
        f.write(f"Total SAEs Trained: {total_saes}\n")
        f.write(f"SAEs Passing Validation: {passing_saes}/{total_saes}\n")
        f.write(f"Success Rate: {passing_saes/total_saes*100:.1f}%\n\n")
        
        # Reconstruction statistics
        f.write("Reconstruction Loss (Threshold: <0.05):\n")
        f.write(f"  Mean: {df['final_val_recon'].mean():.6f}\n")
        f.write(f"  Median: {df['final_val_recon'].median():.6f}\n")
        f.write(f"  Min: {df['final_val_recon'].min():.6f}\n")
        f.write(f"  Max: {df['final_val_recon'].max():.6f}\n")
        f.write(f"  SAEs Meeting Threshold: {df['meets_recon_threshold'].sum()}/{total_saes}\n\n")
        
        # Sparsity statistics
        f.write("L0 Sparsity (Target: 0.1 for 10× baseline):\n")
        f.write(f"  Mean L0: {df['final_val_l0'].mean():.3f}\n")
        f.write(f"  Median L0: {df['final_val_l0'].median():.3f}\n")
        f.write(f"  Mean Sparsity Ratio: {df['sparsity_ratio'].mean():.2f}×\n")
        f.write(f"  SAEs Meeting Target: {df['meets_sparsity_target'].sum()}/{total_saes}\n\n")
        
        # Per-model type summary
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
        
        # Per-layer summary
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
        
        # Detailed results table
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS (ALL 9 SAEs)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Model':<8} {'Layer':<6} {'Val Recon':<12} {'L0':<8} {'Ratio':<8} {'Pass':<6}\n")
        f.write("-"*60 + "\n")
        
        for _, row in df.iterrows():
            status = "✓" if row['passes_validation'] else "✗"
            f.write(
                f"{row['model_type']:<8} "
                f"{row['layer']:<6} "
                f"{row['final_val_recon']:<12.6f} "
                f"{row['final_val_l0']:<8.3f} "
                f"{row['sparsity_ratio']:<8.2f} "
                f"{status:<6}\n"
            )
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        if passing_saes == 0:
            f.write("⚠️  NO SAEs meet validation criteria!\n")
            f.write("   Primary issue: L0 sparsity too high (features too dense)\n")
            f.write("   Recommendation: Increase SAE_SPARSITY_COEF or use top-k activation\n\n")
        elif passing_saes < total_saes:
            f.write(f"⚠️  Only {passing_saes}/{total_saes} SAEs pass validation\n")
            f.write("   Review sparsity coefficient and architecture\n\n")
        else:
            f.write(f"✓  All {total_saes} SAEs meet validation criteria!\n")
            f.write("   Ready for Phase 2.5: Feature Labeling\n\n")
        
        f.write("="*80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("="*80 + "\n\n")
        f.write("1. Review visualizations in {run_dir}/analysis/\n")
        f.write("2. If sparsity insufficient, retrain with higher sparsity coefficient\n")
        f.write("3. If validation passed, proceed to Phase 2.5: LLM Feature Labeling\n")
        f.write("4. Extract top-activating examples for interpretable features\n\n")
    
    logger.info(f"Summary report saved: {report_path}")
    return report_path


def create_visualizations(df: pd.DataFrame, run_dir: Path):
    """Generate visualization plots"""
    
    viz_dir = run_dir / "analysis"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Reconstruction loss heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_recon = df.pivot(index='layer', columns='model_type', values='final_val_recon')
    sns.heatmap(pivot_recon, annot=True, fmt='.6f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('Validation Reconstruction Loss (Lower = Better)', fontsize=14, weight='bold')
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    plt.tight_layout()
    plt.savefig(viz_dir / 'reconstruction_loss_heatmap.png', dpi=300)
    plt.close()
    
    # 2. L0 sparsity heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_l0 = df.pivot(index='layer', columns='model_type', values='final_val_l0')
    sns.heatmap(pivot_l0, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
    ax.set_title('L0 Sparsity (Lower = Better, Target: 0.1)', fontsize=14, weight='bold')
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    plt.tight_layout()
    plt.savefig(viz_dir / 'l0_sparsity_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Sparsity ratio comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['label'] = df_plot['model_type'] + '_L' + df_plot['layer'].astype(str)
    
    bars = ax.bar(df_plot['label'], df_plot['sparsity_ratio'])
    ax.axhline(y=10, color='r', linestyle='--', label='Target: 10× sparsity')
    ax.set_xlabel('SAE Model', fontsize=12)
    ax.set_ylabel('Sparsity Ratio (1/L0)', fontsize=12)
    ax.set_title('Sparsity Ratio by SAE Model', fontsize=14, weight='bold')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(viz_dir / 'sparsity_ratio_comparison.png', dpi=300)
    plt.close()
    
    # 4. Validation status overview
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reconstruction threshold
    recon_pass = df['meets_recon_threshold'].value_counts()
    axes[0].pie(recon_pass, labels=['Pass', 'Fail'], autopct='%1.0f%%', 
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0].set_title('Reconstruction Loss\n(< 0.05 threshold)', fontsize=12, weight='bold')
    
    # Sparsity threshold
    sparsity_pass = df['meets_sparsity_target'].value_counts()
    axes[1].pie(sparsity_pass, labels=['Pass', 'Fail'], autopct='%1.0f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('L0 Sparsity\n(10× target)', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'validation_status_overview.png', dpi=300)
    plt.close()
    
    logger.info(f"Visualizations saved to: {viz_dir}")


def export_results_csv(df: pd.DataFrame, run_dir: Path):
    """Export results to CSV for further analysis"""
    csv_path = run_dir / "sae_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results exported to: {csv_path}")


def main():
    """Main analysis execution"""
    
    try:
        logger.info("="*80)
        logger.info("PHASE 2 RESULTS ANALYSIS")
        logger.info("="*80)
        
        # Find latest run
        logger.info("\nFinding latest SAE training run...")
        run_dir = find_latest_sae_run(SAE_OUTPUT_ROOT)
        logger.info(f"Analyzing: {run_dir.name}")
        
        # Aggregate results
        logger.info("\nAggregating results from all SAEs...")
        df = aggregate_sae_results(run_dir)
        
        if df.empty:
            logger.error("No results found!")
            return
        
        logger.info(f"Loaded {len(df)} SAE results")
        
        # Compute validation status
        logger.info("\nComputing validation status...")
        df = compute_validation_status(df)
        
        # Generate summary report
        logger.info("\nGenerating summary report...")
        report_path = generate_summary_report(df, run_dir)
        
        # Create visualizations
        logger.info("\nCreating visualizations...")
        create_visualizations(df, run_dir)
        
        # Export CSV
        logger.info("\nExporting results...")
        export_results_csv(df, run_dir)
        
        # Print summary to console
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {run_dir}/")
        logger.info(f"  - Summary report: PHASE2_SUMMARY.txt")
        logger.info(f"  - CSV export: sae_results.csv")
        logger.info(f"  - Visualizations: analysis/")
        
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
