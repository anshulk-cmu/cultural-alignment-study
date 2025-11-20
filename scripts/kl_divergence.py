#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KL Divergence Analysis: Layer-wise Distribution Shift Between Base and Instruct Models

This script computes KL divergence between base and instruct model activation distributions
at each layer to localize where RLHF alignment creates representational shifts.

Analysis Levels:
- Overall: Across all data
- Group-level: Suppression, Enhancement, Control  
- Attribute-level: 16 cultural attributes
- State-level: 36 Indian states
- Question-type-level: 4 question types

Metric: KL(Base || Instruct) = measure of distributional shift
Low KL -> representations preserved
High KL -> alignment modified representations
"""

import os
import gc
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy import stats
from scipy.spatial import distance
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Input paths
    ACTIVATION_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/activations")
    METADATA_PATH = Path("/home/anshulk/cultural-alignment-study/outputs/eda_results/tables/enhanced_dataset.csv")

    # Output paths
    LIGHT_OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/kl_divergence")
    HEAVY_OUTPUT_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/kl_divergence")

    # Analysis parameters
    LAYERS = [8, 16, 24, 28]
    HIDDEN_SIZE = 1536

    # Regularization for numerical stability
    REGULARIZATION = 1e-6

    # Random seed
    SEED = 42

    @staticmethod
    def setup():
        """Initialize directories"""
        Config.LIGHT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.HEAVY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for subdir in ['results', 'plots/overall', 'plots/group_level',
                       'plots/attribute_level', 'plots/state_level',
                       'plots/question_type_level', 'logs']:
            (Config.LIGHT_OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

        np.random.seed(Config.SEED)

Config.setup()

# ==============================================================================
# LOGGING
# ==============================================================================

log_file = Config.LIGHT_OUTPUT_DIR / "logs" / f"kl_divergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(log_file, "a") as f:
        f.write(formatted + "\n")

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data():
    """Load activations and metadata"""
    log("\n" + "="*80)
    log("LOADING DATA")
    log("="*80)

    # Load metadata
    log("Loading metadata...")
    df = pd.read_csv(Config.METADATA_PATH)
    log(f"  Total sentences: {len(df)}")

    # Load activations for all layers
    activations = {'base': {}, 'instruct': {}}

    for model in ['base', 'instruct']:
        log(f"\nLoading {model} activations...")
        for layer in Config.LAYERS:
            file_path = Config.ACTIVATION_DIR / f"{model}_layer{layer}_activations.npy"
            activations[model][layer] = np.load(file_path)
            log(f"  Layer {layer}: {activations[model][layer].shape}")

    # Add question_type if not present
    if 'question_type' not in df.columns:
        log("\nInferring question types from sentences...")
        df['question_type'] = df['sentence'].apply(infer_question_type)

    log(f"\nData loading complete!")
    log(f"  Groups: {df['group_type'].unique()}")
    log(f"  Attributes: {df['attribute'].nunique()}")
    log(f"  States: {df['state'].nunique()}")
    log(f"  Question types: {df['question_type'].nunique() if 'question_type' in df.columns else 'N/A'}")

    return df, activations


def infer_question_type(sentence):
    """Infer question type from sentence structure"""
    sentence_lower = sentence.lower()

    if any(word in sentence_lower for word in ['known for', 'famous for', 'celebrated for']):
        return 'known_for'
    elif any(word in sentence_lower for word in ['traditional', 'cuisine', 'art', 'dance', 'music']):
        return 'cultural_practice'
    elif any(word in sentence_lower for word in ['festival', 'celebration', 'event']):
        return 'festival'
    else:
        return 'general'

# ==============================================================================
# KL DIVERGENCE COMPUTATION
# ==============================================================================

def compute_kl_divergence_gaussian(mean1, cov1, mean2, cov2):
    """
    Compute KL divergence between two multivariate Gaussians
    KL(P1 || P2) where P1 ~ N(mean1, cov1), P2 ~ N(mean2, cov2)
    """
    k = len(mean1)

    # Compute inverse and log determinant of cov2
    cov2_inv = np.linalg.inv(cov2)

    # Compute log determinants using sign and logdet
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2)

    # Mean difference
    mean_diff = mean2 - mean1

    # Compute KL divergence
    trace_term = np.trace(cov2_inv @ cov1)
    mahalanobis_term = mean_diff.T @ cov2_inv @ mean_diff
    log_det_term = logdet2 - logdet1

    kl = 0.5 * (trace_term + mahalanobis_term - k + log_det_term)

    return kl


def fit_gaussian_with_regularization(activations):
    """
    Fit multivariate Gaussian to activations with regularization
    Uses Ledoit-Wolf shrinkage for stable covariance estimation
    """
    mean = np.mean(activations, axis=0)

    # Use Ledoit-Wolf for stable covariance estimation
    lw = LedoitWolf()
    lw.fit(activations)
    cov = lw.covariance_

    # Add small regularization to diagonal for numerical stability
    cov = cov + Config.REGULARIZATION * np.eye(cov.shape[0])

    return mean, cov


def compute_kl_for_subset(base_acts, instruct_acts, name=""):
    """Compute KL divergence for a subset of data"""
    if len(base_acts) < 10 or len(instruct_acts) < 10:
        log(f"  Skipping {name}: insufficient samples ({len(base_acts)}, {len(instruct_acts)})")
        return None

    try:
        # Fit Gaussians
        mean_base, cov_base = fit_gaussian_with_regularization(base_acts)
        mean_instruct, cov_instruct = fit_gaussian_with_regularization(instruct_acts)

        # Compute KL divergence
        kl_div = compute_kl_divergence_gaussian(mean_base, cov_base, mean_instruct, cov_instruct)

        # Also compute reverse KL for symmetry check
        kl_div_reverse = compute_kl_divergence_gaussian(mean_instruct, cov_instruct, mean_base, cov_base)

        # Compute Jensen-Shannon divergence (symmetric)
        js_div = 0.5 * (kl_div + kl_div_reverse)

        return {
            'kl_divergence': float(kl_div),
            'kl_divergence_reverse': float(kl_div_reverse),
            'js_divergence': float(js_div),
            'n_samples': len(base_acts)
        }
    except Exception as e:
        log(f"  Error computing KL for {name}: {str(e)}")
        return None

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_overall(df, activations):
    """Compute overall KL divergence across all data"""
    log("\n" + "="*80)
    log("OVERALL KL DIVERGENCE ANALYSIS")
    log("="*80)

    results = []

    for layer in Config.LAYERS:
        log(f"\nLayer {layer}...")

        base_acts = activations['base'][layer]
        instruct_acts = activations['instruct'][layer]

        kl_result = compute_kl_for_subset(base_acts, instruct_acts, f"Layer {layer}")

        if kl_result:
            results.append({
                'layer': layer,
                **kl_result
            })
            log(f"  KL divergence: {kl_result['kl_divergence']:.6f}")
            log(f"  JS divergence: {kl_result['js_divergence']:.6f}")

    return pd.DataFrame(results)


def analyze_by_group(df, activations):
    """Compute KL divergence by group type"""
    log("\n" + "="*80)
    log("GROUP-LEVEL KL DIVERGENCE ANALYSIS")
    log("="*80)

    results = []

    for group in ['suppression', 'enhancement', 'control']:
        log(f"\n{group.upper()} GROUP:")
        group_mask = df['group_type'] == group
        group_indices = np.where(group_mask)[0]

        log(f"  Samples: {len(group_indices)}")

        for layer in Config.LAYERS:
            base_acts = activations['base'][layer][group_indices]
            instruct_acts = activations['instruct'][layer][group_indices]

            kl_result = compute_kl_for_subset(base_acts, instruct_acts, f"{group} - Layer {layer}")

            if kl_result:
                results.append({
                    'group': group,
                    'layer': layer,
                    **kl_result
                })
                log(f"  Layer {layer} - KL: {kl_result['kl_divergence']:.6f}")

    return pd.DataFrame(results)


def analyze_by_attribute(df, activations):
    """Compute KL divergence by attribute"""
    log("\n" + "="*80)
    log("ATTRIBUTE-LEVEL KL DIVERGENCE ANALYSIS")
    log("="*80)

    results = []
    attributes = sorted(df['attribute'].unique())

    for attribute in tqdm(attributes, desc="Attributes"):
        attr_mask = df['attribute'] == attribute
        attr_indices = np.where(attr_mask)[0]

        for layer in Config.LAYERS:
            base_acts = activations['base'][layer][attr_indices]
            instruct_acts = activations['instruct'][layer][attr_indices]

            kl_result = compute_kl_for_subset(base_acts, instruct_acts, f"{attribute} - Layer {layer}")

            if kl_result:
                results.append({
                    'attribute': attribute,
                    'layer': layer,
                    **kl_result
                })

    log(f"Completed {len(attributes)} attributes")
    return pd.DataFrame(results)


def analyze_by_state(df, activations):
    """Compute KL divergence by state"""
    log("\n" + "="*80)
    log("STATE-LEVEL KL DIVERGENCE ANALYSIS")
    log("="*80)

    results = []
    states = sorted(df['state'].unique())

    for state in tqdm(states, desc="States"):
        state_mask = df['state'] == state
        state_indices = np.where(state_mask)[0]

        for layer in Config.LAYERS:
            base_acts = activations['base'][layer][state_indices]
            instruct_acts = activations['instruct'][layer][state_indices]

            kl_result = compute_kl_for_subset(base_acts, instruct_acts, f"{state} - Layer {layer}")

            if kl_result:
                results.append({
                    'state': state,
                    'layer': layer,
                    **kl_result
                })

    log(f"Completed {len(states)} states")
    return pd.DataFrame(results)


def analyze_by_question_type(df, activations):
    """Compute KL divergence by question type"""
    log("\n" + "="*80)
    log("QUESTION-TYPE-LEVEL KL DIVERGENCE ANALYSIS")
    log("="*80)

    if 'question_type' not in df.columns:
        log("  No question_type column found, skipping...")
        return pd.DataFrame()

    results = []
    question_types = sorted(df['question_type'].unique())

    for qtype in question_types:
        log(f"\n{qtype.upper()}:")
        qtype_mask = df['question_type'] == qtype
        qtype_indices = np.where(qtype_mask)[0]

        log(f"  Samples: {len(qtype_indices)}")

        for layer in Config.LAYERS:
            base_acts = activations['base'][layer][qtype_indices]
            instruct_acts = activations['instruct'][layer][qtype_indices]

            kl_result = compute_kl_for_subset(base_acts, instruct_acts, f"{qtype} - Layer {layer}")

            if kl_result:
                results.append({
                    'question_type': qtype,
                    'layer': layer,
                    **kl_result
                })
                log(f"  Layer {layer} - KL: {kl_result['kl_divergence']:.6f}")

    return pd.DataFrame(results)

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_overall_kl(df_overall):
    """Plot overall KL divergence across layers"""
    log("\nGenerating overall KL divergence plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # KL divergence
    ax1.plot(df_overall['layer'], df_overall['kl_divergence'],
             marker='o', linewidth=2, markersize=10, color='steelblue', label='KL(Base || Instruct)')
    ax1.plot(df_overall['layer'], df_overall['kl_divergence_reverse'],
             marker='s', linewidth=2, markersize=8, color='coral', label='KL(Instruct || Base)', alpha=0.7)
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Distributional Shift: Base vs Instruct', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(Config.LAYERS)

    # JS divergence (symmetric)
    ax2.plot(df_overall['layer'], df_overall['js_divergence'],
             marker='D', linewidth=2, markersize=10, color='forestgreen')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('JS Divergence (Symmetric)', fontsize=12, fontweight='bold')
    ax2.set_title('Jensen-Shannon Divergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(Config.LAYERS)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/overall/overall_kl_divergence.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: overall_kl_divergence.png")


def plot_group_level_kl(df_group):
    """Plot KL divergence by group"""
    log("\nGenerating group-level KL divergence plots...")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'suppression': 'crimson', 'enhancement': 'forestgreen', 'control': 'steelblue'}

    for group in ['suppression', 'enhancement', 'control']:
        group_data = df_group[df_group['group'] == group]
        ax.plot(group_data['layer'], group_data['kl_divergence'],
                marker='o', linewidth=2.5, markersize=10,
                color=colors[group], label=group.capitalize(), alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Group-Level Distributional Shift Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Config.LAYERS)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/group_level/group_kl_divergence.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: group_kl_divergence.png")


def plot_attribute_level_kl(df_attr):
    """Plot KL divergence heatmap by attribute"""
    log("\nGenerating attribute-level KL divergence heatmap...")

    # Create pivot table
    pivot = df_attr.pivot(index='attribute', columns='layer', values='kl_divergence')

    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'KL Divergence'})

    ax.set_title('Attribute-Level KL Divergence Across Layers',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attribute', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/attribute_level/attribute_kl_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: attribute_kl_heatmap.png")

    # Top attributes with highest KL at each layer
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, layer in enumerate(Config.LAYERS):
        layer_data = df_attr[df_attr['layer'] == layer].sort_values('kl_divergence', ascending=False).head(10)

        axes[idx].barh(range(len(layer_data)), layer_data['kl_divergence'], color='coral', alpha=0.8)
        axes[idx].set_yticks(range(len(layer_data)))
        axes[idx].set_yticklabels(layer_data['attribute'], fontsize=9)
        axes[idx].set_xlabel('KL Divergence', fontsize=10, fontweight='bold')
        axes[idx].set_title(f'Layer {layer}: Top 10 Attributes', fontsize=11, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/attribute_level/top_attributes_by_layer.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: top_attributes_by_layer.png")


def plot_state_level_kl(df_state):
    """Plot KL divergence heatmap by state"""
    log("\nGenerating state-level KL divergence heatmap...")

    # Create pivot table
    pivot = df_state.pivot(index='state', columns='layer', values='kl_divergence')

    fig, ax = plt.subplots(figsize=(10, 16))

    sns.heatmap(pivot, annot=False, cmap='RdYlBu_r',
                ax=ax, cbar_kws={'label': 'KL Divergence'})

    ax.set_title('State-Level KL Divergence Across Layers',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('State', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/state_level/state_kl_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: state_kl_heatmap.png")

    # Top states with highest KL at layer 24
    layer_24_data = df_state[df_state['layer'] == 24].sort_values('kl_divergence', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(layer_24_data)), layer_24_data['kl_divergence'], color='indianred', alpha=0.8)
    ax.set_yticks(range(len(layer_24_data)))
    ax.set_yticklabels(layer_24_data['state'], fontsize=10)
    ax.set_xlabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 States by KL Divergence (Layer 24)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/state_level/top_states_layer24.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: top_states_layer24.png")


def plot_question_type_kl(df_qtype):
    """Plot KL divergence by question type"""
    if df_qtype.empty:
        return

    log("\nGenerating question-type-level KL divergence plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    qtypes = df_qtype['question_type'].unique()
    colors = plt.cm.Set2(range(len(qtypes)))

    for idx, qtype in enumerate(qtypes):
        qtype_data = df_qtype[df_qtype['question_type'] == qtype]
        ax.plot(qtype_data['layer'], qtype_data['kl_divergence'],
                marker='o', linewidth=2, markersize=8,
                color=colors[idx], label=qtype, alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Question-Type-Level KL Divergence Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Config.LAYERS)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/question_type_level/question_type_kl_divergence.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: question_type_kl_divergence.png")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    log("="*80)
    log("KL DIVERGENCE ANALYSIS PIPELINE")
    log("="*80)
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    df, activations = load_data()

    # Overall analysis
    df_overall = analyze_overall(df, activations)
    df_overall.to_csv(Config.LIGHT_OUTPUT_DIR / 'results/overall_kl_divergence.csv', index=False)
    plot_overall_kl(df_overall)

    # Group-level analysis
    df_group = analyze_by_group(df, activations)
    df_group.to_csv(Config.LIGHT_OUTPUT_DIR / 'results/group_kl_divergence.csv', index=False)
    plot_group_level_kl(df_group)

    # Attribute-level analysis
    df_attr = analyze_by_attribute(df, activations)
    df_attr.to_csv(Config.LIGHT_OUTPUT_DIR / 'results/attribute_kl_divergence.csv', index=False)
    # Save to heavy storage as well
    df_attr.to_csv(Config.HEAVY_OUTPUT_DIR / 'attribute_kl_divergence_full.csv', index=False)
    plot_attribute_level_kl(df_attr)

    # State-level analysis
    df_state = analyze_by_state(df, activations)
    df_state.to_csv(Config.LIGHT_OUTPUT_DIR / 'results/state_kl_divergence.csv', index=False)
    # Save to heavy storage as well
    df_state.to_csv(Config.HEAVY_OUTPUT_DIR / 'state_kl_divergence_full.csv', index=False)
    plot_state_level_kl(df_state)

    # Question-type analysis
    df_qtype = analyze_by_question_type(df, activations)
    if not df_qtype.empty:
        df_qtype.to_csv(Config.LIGHT_OUTPUT_DIR / 'results/question_type_kl_divergence.csv', index=False)
        plot_question_type_kl(df_qtype)

    # Generate summary statistics
    log("\n" + "="*80)
    log("SUMMARY STATISTICS")
    log("="*80)

    log("\nOverall KL Divergence by Layer:")
    for _, row in df_overall.iterrows():
        log(f"  Layer {int(row['layer'])}: {row['kl_divergence']:.6f}")

    log("\nGroup-Level KL Divergence at Layer 24:")
    layer_24 = df_group[df_group['layer'] == 24]
    for _, row in layer_24.iterrows():
        log(f"  {row['group'].capitalize()}: {row['kl_divergence']:.6f}")

    # Find layer with maximum KL increase for suppression group
    supp_data = df_group[df_group['group'] == 'suppression'].sort_values('layer')
    kl_diffs = supp_data['kl_divergence'].diff()
    max_increase_idx = kl_diffs.idxmax()
    if pd.notna(max_increase_idx):
        max_increase_layer = supp_data.loc[max_increase_idx, 'layer']
        log(f"\nMaximum KL increase for suppression group: Layer {int(max_increase_layer)}")

    log("\n" + "="*80)
    log("PIPELINE COMPLETE")
    log("="*80)
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"\nResults saved to:")
    log(f"  Light outputs: {Config.LIGHT_OUTPUT_DIR}")
    log(f"  Heavy data: {Config.HEAVY_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
