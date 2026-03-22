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
import argparse
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
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
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
    N_BOOT = 400  # bootstrap samples for CI
    N_JOBS = 10   # parallel workers for bootstrap
    PCA_ENABLED = False  # toggled per run
    PCA_DIM = 100

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

# Track skipped slices for transparency
SKIPPED_SLICES = []
STATS = {'attempted': 0}

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(log_file, "a") as f:
        f.write(formatted + "\n")

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

    # Use solve instead of inv for numerical stability: solve(A, B) = A^{-1} @ B
    # cov2_inv @ cov1 = solve(cov2, cov1)
    cov2_inv_cov1 = np.linalg.solve(cov2, cov1)

    # Compute log determinants using sign and logdet
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2)

    # Validate signs - covariance matrices must be positive definite
    if sign1 <= 0 or sign2 <= 0:
        # Return a large value to indicate numerical issues
        return np.inf

    # Mean difference
    mean_diff = mean2 - mean1

    # Compute KL divergence
    # trace_term = trace(cov2_inv @ cov1)
    trace_term = np.trace(cov2_inv_cov1)
    # mahalanobis_term = mean_diff.T @ cov2_inv @ mean_diff = mean_diff.T @ solve(cov2, mean_diff)
    mahalanobis_term = mean_diff.T @ np.linalg.solve(cov2, mean_diff)
    log_det_term = logdet2 - logdet1

    kl = 0.5 * (trace_term + mahalanobis_term - k + log_det_term)

    return kl


def bootstrap_ci_scalar(values, n_boot=500, ci=0.95, random_state=42):
    """Bootstrap CI for a 1D array."""
    if len(values) == 0:
        return {'point_estimate': None, 'ci_lower': None, 'ci_upper': None, 'std_error': None, 'n_bootstrap': n_boot}
    rng = np.random.RandomState(random_state)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(len(values), size=len(values), replace=True)
        boot.append(np.mean(np.array(values)[idx]))
    alpha = 1 - ci
    return {
        'point_estimate': float(np.mean(values)),
        'ci_lower': float(np.percentile(boot, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(boot, 100 * (1 - alpha / 2))),
        'std_error': float(np.std(boot)),
        'n_bootstrap': n_boot
    }


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
    """Compute KL/JS with optional PCA and bootstrap CIs."""
    STATS['attempted'] += 1
    if len(base_acts) < 10 or len(instruct_acts) < 10:
        log(f"  Skipping {name}: insufficient samples ({len(base_acts)}, {len(instruct_acts)})")
        SKIPPED_SLICES.append({'name': name, 'n_base': len(base_acts), 'n_instruct': len(instruct_acts)})
        return None

    try:
        Xb = base_acts
        Xi = instruct_acts

        if Config.PCA_ENABLED:
            pca = PCA(n_components=min(Config.PCA_DIM, Xb.shape[1], Xb.shape[0]-1, Xi.shape[0]-1))
            X_concat = np.vstack([Xb, Xi])
            pca.fit(X_concat)
            Xb = pca.transform(Xb)
            Xi = pca.transform(Xi)

        # Fit Gaussians
        mean_base, cov_base = fit_gaussian_with_regularization(Xb)
        mean_instruct, cov_instruct = fit_gaussian_with_regularization(Xi)

        # Point estimates
        kl_div = compute_kl_divergence_gaussian(mean_base, cov_base, mean_instruct, cov_instruct)
        kl_div_reverse = compute_kl_divergence_gaussian(mean_instruct, cov_instruct, mean_base, cov_base)

        # Proper Jensen-Shannon divergence: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) where M = 0.5*(P+Q)
        # For Gaussians, M has mean = 0.5*(mean_base + mean_instruct) and cov = 0.5*(cov_base + cov_instruct)
        mean_m = 0.5 * (mean_base + mean_instruct)
        cov_m = 0.5 * (cov_base + cov_instruct) + Config.REGULARIZATION * np.eye(cov_base.shape[0])
        js_div = 0.5 * (compute_kl_divergence_gaussian(mean_base, cov_base, mean_m, cov_m) +
                        compute_kl_divergence_gaussian(mean_instruct, cov_instruct, mean_m, cov_m))

        # Bootstrap CIs (resample rows from both sets independently) - parallelized
        n_boot = Config.N_BOOT
        n_b = len(Xb)
        n_i = len(Xi)

        def _single_bootstrap(seed):
            rng = np.random.RandomState(seed)
            idx_b = rng.choice(n_b, size=n_b, replace=True)
            idx_i = rng.choice(n_i, size=n_i, replace=True)
            mb, cb = fit_gaussian_with_regularization(Xb[idx_b])
            mi, ci = fit_gaussian_with_regularization(Xi[idx_i])
            kl_s = compute_kl_divergence_gaussian(mb, cb, mi, ci)
            # Proper JS: use mixture distribution M = 0.5*(P + Q)
            mm = 0.5 * (mb + mi)
            cm = 0.5 * (cb + ci) + Config.REGULARIZATION * np.eye(cb.shape[0])
            js_s = 0.5 * (compute_kl_divergence_gaussian(mb, cb, mm, cm) +
                          compute_kl_divergence_gaussian(mi, ci, mm, cm))
            return kl_s, js_s

        seeds = [Config.SEED + i for i in range(n_boot)]
        results = Parallel(n_jobs=Config.N_JOBS)(delayed(_single_bootstrap)(s) for s in seeds)
        kl_samples, js_samples = zip(*results)

        kl_ci = bootstrap_ci_scalar(list(kl_samples), n_boot=n_boot, ci=0.95, random_state=Config.SEED)
        js_ci = bootstrap_ci_scalar(list(js_samples), n_boot=n_boot, ci=0.95, random_state=Config.SEED)

        return {
            'kl_divergence': float(kl_div),
            'kl_divergence_reverse': float(kl_div_reverse),
            'js_divergence': float(js_div),
            'kl_ci_lower': kl_ci['ci_lower'],
            'kl_ci_upper': kl_ci['ci_upper'],
            'js_ci_lower': js_ci['ci_lower'],
            'js_ci_upper': js_ci['ci_upper'],
            'kl_std_error': kl_ci['std_error'],
            'js_std_error': js_ci['std_error'],
            'n_samples': len(base_acts),
            'pca_enabled': Config.PCA_ENABLED,
            'pca_dim': Config.PCA_DIM if Config.PCA_ENABLED else None
        }
    except Exception as e:
        log(f"  Error computing KL for {name}: {str(e)}")
        return None

# ==============================================================================
# HIERARCHICAL AGGREGATION FOR LOW-COUNT SLICES
# ==============================================================================

# Geographic regions for state aggregation
REGION_MAPPING = {
    # South
    'Tamil_Nadu': 'South', 'Karnataka': 'South', 'Kerala': 'South',
    'Andhra_Pradesh': 'South', 'Telangana': 'South', 'Puducherry': 'South',
    # North
    'Delhi': 'North', 'Uttar_Pradesh': 'North', 'Uttarakhand': 'North',
    'Haryana': 'North', 'Punjab': 'North', 'Himachal_Pradesh': 'North',
    'Jammu_and_Kashmir': 'North', 'Ladakh': 'North', 'Chandigarh': 'North',
    # East
    'West_Bengal': 'East', 'Odisha': 'East', 'Bihar': 'East',
    'Jharkhand': 'East',
    # West
    'Maharashtra': 'West', 'Gujarat': 'West', 'Goa': 'West',
    'Rajasthan': 'West', 'Madhya_Pradesh': 'West', 'Chhattisgarh': 'West',
    'Dadra_and_Nagar_Haveli': 'West', 'Daman_and_Diu': 'West',
    # Northeast
    'Assam': 'Northeast', 'Meghalaya': 'Northeast', 'Manipur': 'Northeast',
    'Mizoram': 'Northeast', 'Tripura': 'Northeast', 'Nagaland': 'Northeast',
    'Arunachal_Pradesh': 'Northeast', 'Sikkim': 'Northeast',
    # Island
    'Andaman_and_Nicobar': 'Island', 'Lakshadweep': 'Island'
}

# Attribute categories for semantic aggregation
ATTRIBUTE_CATEGORY_MAPPING = {
    # Cultural Practice
    'Dance_and_Music': 'Cultural_Practice', 'Art_and_Craft': 'Cultural_Practice',
    'Festivals': 'Cultural_Practice', 'Rituals_and_Ceremonies': 'Cultural_Practice',
    'Art': 'Cultural_Practice',
    # Identity
    'Religion': 'Identity', 'Language': 'Identity', 'Costume': 'Identity',
    'Folklore': 'Identity',
    # Lifestyle
    'Cuisine': 'Lifestyle', 'Sports': 'Lifestyle', 'Nightlife': 'Lifestyle',
    'Entertainment': 'Lifestyle',
    # Infrastructure
    'Architecture': 'Infrastructure', 'Transport': 'Infrastructure',
    'Medicine': 'Infrastructure', 'Economy': 'Infrastructure',
    # Heritage & Tourism
    'Tourism': 'Heritage', 'History': 'Heritage', 'Historical_Monuments': 'Heritage',
    'Natural_Landmarks': 'Heritage', 'Wildlife': 'Heritage',
    # Education & Knowledge
    'Education': 'Education', 'Literature': 'Education',
    # Social
    'Social_Customs': 'Social', 'Family_Structure': 'Social',
}

# Minimum samples for different analysis tiers
MIN_SAMPLES_TIER1 = 10   # Individual slice analysis
MIN_SAMPLES_TIER2 = 30   # Aggregated region/category analysis
MIN_SAMPLES_TIER3 = 50   # Interaction analysis (region × category)


def get_region(state):
    """Map state to geographic region"""
    return REGION_MAPPING.get(state, 'Other')


def get_attribute_category(attribute):
    """Map attribute to semantic category"""
    return ATTRIBUTE_CATEGORY_MAPPING.get(attribute, 'Other')


def analyze_with_hierarchical_fallback(df, activations, slice_col, slice_values, layer,
                                       fallback_mapping=None, fallback_col_name=None,
                                       min_samples=MIN_SAMPLES_TIER1):
    """
    Analyze KL divergence with hierarchical fallback for low-count slices.

    If individual slices have < min_samples, aggregate them using fallback_mapping
    and report at the aggregated level instead.

    Returns:
        results: list of dicts with KL results
        skipped_individual: list of slices that were aggregated
        aggregated_results: list of dicts with aggregated KL results
    """
    results = []
    skipped_individual = []
    aggregated_results = []

    # Track which slices need aggregation
    low_count_slices = {}

    for slice_val in slice_values:
        slice_mask = df[slice_col] == slice_val
        slice_indices = np.where(slice_mask)[0]
        n_samples = len(slice_indices)

        if n_samples >= min_samples:
            # Sufficient samples - analyze directly
            base_acts = activations['base'][layer][slice_indices]
            instruct_acts = activations['instruct'][layer][slice_indices]

            kl_result = compute_kl_for_subset(base_acts, instruct_acts,
                                              f"{slice_col}={slice_val} - Layer {layer}")
            if kl_result:
                results.append({
                    slice_col: slice_val,
                    'layer': layer,
                    'analysis_level': 'individual',
                    'n_samples': n_samples,
                    **kl_result
                })
        else:
            # Track for aggregation
            skipped_individual.append({
                slice_col: slice_val,
                'n_samples': n_samples,
                'layer': layer
            })

            if fallback_mapping is not None:
                agg_key = fallback_mapping.get(slice_val, 'Other')
                if agg_key not in low_count_slices:
                    low_count_slices[agg_key] = []
                low_count_slices[agg_key].extend(slice_indices.tolist())

    # Perform aggregated analysis for low-count slices
    if fallback_mapping is not None and low_count_slices:
        for agg_key, indices in low_count_slices.items():
            indices = np.array(indices)
            n_samples = len(indices)

            if n_samples >= MIN_SAMPLES_TIER2:
                base_acts = activations['base'][layer][indices]
                instruct_acts = activations['instruct'][layer][indices]

                kl_result = compute_kl_for_subset(base_acts, instruct_acts,
                                                  f"{fallback_col_name}={agg_key} - Layer {layer}")
                if kl_result:
                    aggregated_results.append({
                        fallback_col_name: agg_key,
                        'layer': layer,
                        'analysis_level': 'aggregated',
                        'n_samples': n_samples,
                        'aggregated_from': [s[slice_col] for s in skipped_individual
                                           if fallback_mapping.get(s[slice_col]) == agg_key],
                        **kl_result
                    })

    return results, skipped_individual, aggregated_results


def analyze_interaction_slices(df, activations, layer,
                               min_samples=MIN_SAMPLES_TIER3):
    """
    Analyze State × Attribute interactions with hierarchical fallback.

    Tier 1: Individual state × attribute (if n >= 50)
    Tier 2: Region × attribute (if tier 1 fails)
    Tier 3: Region × attribute_category (if tier 2 fails)
    """
    results = []

    states = df['state'].unique()
    attributes = df['attribute'].unique()

    # Add region and category columns if not present (single copy)
    needs_copy = 'region' not in df.columns or 'attribute_category' not in df.columns
    if needs_copy:
        df = df.copy()
    if 'region' not in df.columns:
        df['region'] = df['state'].apply(get_region)
    if 'attribute_category' not in df.columns:
        df['attribute_category'] = df['attribute'].apply(get_attribute_category)

    # Tier 1: Individual interactions (state × attribute)
    tier1_analyzed = set()
    for state in states:
        for attr in attributes:
            mask = (df['state'] == state) & (df['attribute'] == attr)
            indices = np.where(mask)[0]

            if len(indices) >= min_samples:
                base_acts = activations['base'][layer][indices]
                instruct_acts = activations['instruct'][layer][indices]

                kl_result = compute_kl_for_subset(
                    base_acts, instruct_acts,
                    f"{state}×{attr} - Layer {layer}"
                )
                if kl_result:
                    results.append({
                        'state': state,
                        'attribute': attr,
                        'region': get_region(state),
                        'attribute_category': get_attribute_category(attr),
                        'layer': layer,
                        'analysis_tier': 'tier1_individual',
                        'n_samples': len(indices),
                        **kl_result
                    })
                    tier1_analyzed.add((state, attr))

    # Tier 2: Region × attribute (for remaining)
    regions = df['region'].unique()
    tier2_analyzed = set()
    for region in regions:
        for attr in attributes:
            # Skip if all individual state×attr in this region were analyzed
            states_in_region = [s for s in states if get_region(s) == region]
            if all((s, attr) in tier1_analyzed for s in states_in_region):
                continue

            mask = (df['region'] == region) & (df['attribute'] == attr)
            indices = np.where(mask)[0]

            if len(indices) >= MIN_SAMPLES_TIER2:
                base_acts = activations['base'][layer][indices]
                instruct_acts = activations['instruct'][layer][indices]

                kl_result = compute_kl_for_subset(
                    base_acts, instruct_acts,
                    f"{region}×{attr} - Layer {layer}"
                )
                if kl_result:
                    results.append({
                        'region': region,
                        'attribute': attr,
                        'attribute_category': get_attribute_category(attr),
                        'layer': layer,
                        'analysis_tier': 'tier2_region_attr',
                        'n_samples': len(indices),
                        **kl_result
                    })
                    tier2_analyzed.add((region, attr))

    # Tier 3: Region × attribute_category (coarsest level)
    attr_categories = df['attribute_category'].unique()
    for region in regions:
        for attr_cat in attr_categories:
            # Skip if tier2 covered this
            attrs_in_cat = [a for a in attributes if get_attribute_category(a) == attr_cat]
            if all((region, a) in tier2_analyzed for a in attrs_in_cat):
                continue

            mask = (df['region'] == region) & (df['attribute_category'] == attr_cat)
            indices = np.where(mask)[0]

            if len(indices) >= MIN_SAMPLES_TIER2:
                base_acts = activations['base'][layer][indices]
                instruct_acts = activations['instruct'][layer][indices]

                kl_result = compute_kl_for_subset(
                    base_acts, instruct_acts,
                    f"{region}×{attr_cat} - Layer {layer}"
                )
                if kl_result:
                    results.append({
                        'region': region,
                        'attribute_category': attr_cat,
                        'layer': layer,
                        'analysis_tier': 'tier3_region_category',
                        'n_samples': len(indices),
                        **kl_result
                    })

    return pd.DataFrame(results) if results else pd.DataFrame()


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


def analyze_by_region(df, activations):
    """Compute KL divergence by geographic region (aggregated states)"""
    log("\n" + "="*80)
    log("REGION-LEVEL KL DIVERGENCE ANALYSIS")
    log("="*80)

    # Add region column
    df = df.copy()
    df['region'] = df['state'].apply(get_region)

    results = []
    regions = sorted(df['region'].unique())

    for region in regions:
        log(f"\n{region.upper()} REGION:")
        region_mask = df['region'] == region
        region_indices = np.where(region_mask)[0]

        log(f"  Samples: {len(region_indices)}")

        for layer in Config.LAYERS:
            base_acts = activations['base'][layer][region_indices]
            instruct_acts = activations['instruct'][layer][region_indices]

            kl_result = compute_kl_for_subset(base_acts, instruct_acts, f"{region} - Layer {layer}")

            if kl_result:
                results.append({
                    'region': region,
                    'layer': layer,
                    'n_states': len(df[region_mask]['state'].unique()),
                    **kl_result
                })
                log(f"  Layer {layer} - KL: {kl_result['kl_divergence']:.6f}")

    log(f"Completed {len(regions)} regions")
    return pd.DataFrame(results)


def analyze_by_attribute_category(df, activations):
    """Compute KL divergence by attribute category (aggregated attributes)"""
    log("\n" + "="*80)
    log("ATTRIBUTE-CATEGORY-LEVEL KL DIVERGENCE ANALYSIS")
    log("="*80)

    # Add attribute_category column
    df = df.copy()
    df['attribute_category'] = df['attribute'].apply(get_attribute_category)

    results = []
    categories = sorted(df['attribute_category'].unique())

    for category in categories:
        log(f"\n{category.upper()}:")
        cat_mask = df['attribute_category'] == category
        cat_indices = np.where(cat_mask)[0]

        log(f"  Samples: {len(cat_indices)}")

        for layer in Config.LAYERS:
            base_acts = activations['base'][layer][cat_indices]
            instruct_acts = activations['instruct'][layer][cat_indices]

            kl_result = compute_kl_for_subset(base_acts, instruct_acts, f"{category} - Layer {layer}")

            if kl_result:
                results.append({
                    'attribute_category': category,
                    'layer': layer,
                    'n_attributes': len(df[cat_mask]['attribute'].unique()),
                    **kl_result
                })
                log(f"  Layer {layer} - KL: {kl_result['kl_divergence']:.6f}")

    log(f"Completed {len(categories)} attribute categories")
    return pd.DataFrame(results)


def analyze_interactions(df, activations):
    """Analyze State × Attribute interactions with hierarchical fallback"""
    log("\n" + "="*80)
    log("INTERACTION ANALYSIS (State × Attribute) WITH HIERARCHICAL FALLBACK")
    log("="*80)

    all_results = []

    for layer in Config.LAYERS:
        log(f"\nLayer {layer}...")
        df_interactions = analyze_interaction_slices(df, activations, layer,
                                                      min_samples=MIN_SAMPLES_TIER3)
        if len(df_interactions) > 0:
            all_results.append(df_interactions)

            # Summary stats
            tier_counts = df_interactions['analysis_tier'].value_counts()
            log(f"  Tier 1 (individual): {tier_counts.get('tier1_individual', 0)}")
            log(f"  Tier 2 (region×attr): {tier_counts.get('tier2_region_attr', 0)}")
            log(f"  Tier 3 (region×cat): {tier_counts.get('tier3_region_category', 0)}")

    if all_results:
        df_final = pd.concat(all_results, ignore_index=True)
        log(f"\nTotal interaction analyses: {len(df_final)}")
        return df_final
    else:
        log("No interaction analyses completed")
        return pd.DataFrame()


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


def add_kl_labels(df, value_col='kl_divergence'):
    """Add heuristic high/moderate labels based on z-like threshold."""
    if df is None or len(df) == 0:
        return df
    mean_val = df[value_col].mean()
    std_val = df[value_col].std() if df[value_col].std() > 0 else 1e-6
    def label(v):
        if v >= mean_val + std_val:
            return 'high'
        elif v >= mean_val + 0.5 * std_val:
            return 'moderate'
        return 'baseline'
    df = df.copy()
    df[value_col + '_label'] = df[value_col].apply(label)
    return df


def compute_layer_deltas(df, level_cols=None, layer_low=24, layer_high=28):
    """Compute KL deltas between two layers for given grouping columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if level_cols is None:
        level_cols = []
    records = []
    for keys, group in df.groupby(level_cols if level_cols else lambda _: True):
        if level_cols:
            key_dict = dict(zip(level_cols, keys if isinstance(keys, tuple) else (keys,)))
        else:
            key_dict = {}
        low = group[group['layer'] == layer_low]
        high = group[group['layer'] == layer_high]
        if len(low) == 0 or len(high) == 0:
            continue
        low_row = low.iloc[0]
        high_row = high.iloc[0]
        records.append({
            **key_dict,
            'layer_low': layer_low,
            'layer_high': layer_high,
            'kl_low': low_row['kl_divergence'],
            'kl_high': high_row['kl_divergence'],
            'kl_delta': high_row['kl_divergence'] - low_row['kl_divergence'],
            'js_low': low_row.get('js_divergence'),
            'js_high': high_row.get('js_divergence'),
            'js_delta': (high_row.get('js_divergence') - low_row.get('js_divergence')) if pd.notna(low_row.get('js_divergence')) and pd.notna(high_row.get('js_divergence')) else None
        })
    return pd.DataFrame.from_records(records)

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_overall_kl(df_overall, file_suffix=""):
    """Plot overall KL divergence across layers"""
    log("\nGenerating overall KL divergence plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # KL divergence
    ax1.plot(df_overall['layer'], df_overall['kl_divergence'],
             marker='o', linewidth=2, markersize=10, color='steelblue', label='KL(Base || Instruct)')
    ax1.plot(df_overall['layer'], df_overall['kl_divergence_reverse'],
             marker='s', linewidth=2, markersize=8, color='coral', label='KL(Instruct || Base)', alpha=0.7)
    if 'kl_ci_lower' in df_overall and 'kl_ci_upper' in df_overall:
        ax1.fill_between(df_overall['layer'], df_overall['kl_ci_lower'], df_overall['kl_ci_upper'],
                         color='steelblue', alpha=0.18, label='KL 95% CI')
    if 'kl_divergence_label' in df_overall.columns:
        hi = df_overall[df_overall['kl_divergence_label'] == 'high']
        ax1.scatter(hi['layer'], hi['kl_divergence'], color='steelblue', edgecolor='black',
                    linewidth=1.0, s=120, zorder=6, label='High KL')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Distributional Shift: Base vs Instruct', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(Config.LAYERS)

    # JS divergence (symmetric)
    ax2.plot(df_overall['layer'], df_overall['js_divergence'],
             marker='D', linewidth=2, markersize=10, color='forestgreen')
    if 'js_ci_lower' in df_overall and 'js_ci_upper' in df_overall:
        ax2.fill_between(df_overall['layer'], df_overall['js_ci_lower'], df_overall['js_ci_upper'],
                         color='forestgreen', alpha=0.18, label='JS 95% CI')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('JS Divergence (Symmetric)', fontsize=12, fontweight='bold')
    ax2.set_title('Jensen-Shannon Divergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xticks(Config.LAYERS)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/overall/overall_kl_divergence{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: overall_kl_divergence.png")


def plot_group_level_kl(df_group, file_suffix=""):
    """Plot KL divergence by group"""
    log("\nGenerating group-level KL divergence plots...")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'suppression': 'crimson', 'enhancement': 'forestgreen', 'control': 'steelblue'}

    for group in ['suppression', 'enhancement', 'control']:
        group_data = df_group[df_group['group'] == group]
        ax.plot(group_data['layer'], group_data['kl_divergence'],
                marker='o', linewidth=2.5, markersize=10,
                color=colors[group], label=group.capitalize(), alpha=0.8)
        if {'kl_ci_lower', 'kl_ci_upper'}.issubset(group_data.columns):
            ax.fill_between(group_data['layer'], group_data['kl_ci_lower'], group_data['kl_ci_upper'],
                            color=colors[group], alpha=0.16)
        # Highlight high-labeled points if present
        if 'kl_divergence_label' in group_data.columns:
            high_pts = group_data[group_data['kl_divergence_label'] == 'high']
            ax.scatter(high_pts['layer'], high_pts['kl_divergence'], color=colors[group],
                       edgecolor='black', linewidth=1.0, s=110, zorder=5, label=f"{group.capitalize()} high")

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Group-Level Distributional Shift Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Config.LAYERS)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/group_level/group_kl_divergence{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: group_kl_divergence.png")


def plot_attribute_level_kl(df_attr, file_suffix=""):
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
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/attribute_level/attribute_kl_heatmap{file_suffix}.png',
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
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/attribute_level/top_attributes_by_layer{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: top_attributes_by_layer.png")


def plot_state_level_kl(df_state, file_suffix=""):
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
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/state_level/state_kl_heatmap{file_suffix}.png',
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
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/state_level/top_states_layer24{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: top_states_layer24.png")


def plot_question_type_kl(df_qtype, file_suffix=""):
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
        if {'kl_ci_lower', 'kl_ci_upper'}.issubset(qtype_data.columns):
            ax.fill_between(qtype_data['layer'], qtype_data['kl_ci_lower'], qtype_data['kl_ci_upper'],
                            color=colors[idx], alpha=0.15)
        if 'kl_divergence_label' in qtype_data.columns:
            hi = qtype_data[qtype_data['kl_divergence_label'] == 'high']
            ax.scatter(hi['layer'], hi['kl_divergence'], color=colors[idx], edgecolor='black',
                       linewidth=0.9, s=90, zorder=5)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Question-Type-Level KL Divergence Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Config.LAYERS)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/question_type_level/question_type_kl_divergence{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: question_type_kl_divergence.png")


def plot_region_level_kl(df_region, file_suffix=""):
    """Plot KL divergence by geographic region"""
    if df_region.empty:
        return

    log("\nGenerating region-level KL divergence plot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Line plot across layers
    ax = axes[0]
    regions = df_region['region'].unique()
    colors = plt.cm.Set1(range(len(regions)))

    for idx, region in enumerate(regions):
        region_data = df_region[df_region['region'] == region]
        ax.plot(region_data['layer'], region_data['kl_divergence'],
                marker='o', linewidth=2.5, markersize=10,
                color=colors[idx], label=region, alpha=0.8)
        if {'kl_ci_lower', 'kl_ci_upper'}.issubset(region_data.columns):
            ax.fill_between(region_data['layer'], region_data['kl_ci_lower'],
                           region_data['kl_ci_upper'], color=colors[idx], alpha=0.15)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Region-Level KL Divergence Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Config.LAYERS)

    # Bar chart at layer 28
    ax = axes[1]
    layer_28 = df_region[df_region['layer'] == 28].sort_values('kl_divergence', ascending=True)
    if len(layer_28) > 0:
        # Build color map from the regions list to ensure consistent indexing
        region_to_color = {r: colors[i] for i, r in enumerate(regions)}
        bar_colors = [region_to_color.get(r, 'gray') for r in layer_28['region']]
        ax.barh(range(len(layer_28)), layer_28['kl_divergence'],
                color=bar_colors, alpha=0.8)
        ax.set_yticks(range(len(layer_28)))
        ax.set_yticklabels(layer_28['region'], fontsize=10)
        ax.set_xlabel('KL Divergence', fontsize=12, fontweight='bold')
        ax.set_title('Region KL Divergence (Layer 28)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/state_level/region_kl_divergence{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: region_kl_divergence.png")


def plot_attribute_category_kl(df_attr_cat, file_suffix=""):
    """Plot KL divergence by attribute category"""
    if df_attr_cat.empty:
        return

    log("\nGenerating attribute-category-level KL divergence plot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Line plot across layers
    ax = axes[0]
    categories = df_attr_cat['attribute_category'].unique()
    colors = plt.cm.Set2(range(len(categories)))

    for idx, cat in enumerate(categories):
        cat_data = df_attr_cat[df_attr_cat['attribute_category'] == cat]
        ax.plot(cat_data['layer'], cat_data['kl_divergence'],
                marker='o', linewidth=2.5, markersize=10,
                color=colors[idx], label=cat.replace('_', ' '), alpha=0.8)
        if {'kl_ci_lower', 'kl_ci_upper'}.issubset(cat_data.columns):
            ax.fill_between(cat_data['layer'], cat_data['kl_ci_lower'],
                           cat_data['kl_ci_upper'], color=colors[idx], alpha=0.15)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Attribute Category KL Divergence Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Config.LAYERS)

    # Bar chart at layer 28
    ax = axes[1]
    layer_28 = df_attr_cat[df_attr_cat['layer'] == 28].sort_values('kl_divergence', ascending=True)
    if len(layer_28) > 0:
        # Build color map from the categories list to ensure consistent indexing
        cat_to_color = {c: colors[i] for i, c in enumerate(categories)}
        bar_colors = [cat_to_color.get(c, 'gray') for c in layer_28['attribute_category']]
        ax.barh(range(len(layer_28)), layer_28['kl_divergence'],
                color=bar_colors, alpha=0.8)
        ax.set_yticks(range(len(layer_28)))
        ax.set_yticklabels([c.replace('_', ' ') for c in layer_28['attribute_category']], fontsize=10)
        ax.set_xlabel('KL Divergence', fontsize=12, fontweight='bold')
        ax.set_title('Attribute Category KL (Layer 28)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/attribute_level/attribute_category_kl{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: attribute_category_kl.png")


def plot_interaction_analysis(df_interactions, file_suffix=""):
    """Plot hierarchical interaction analysis results"""
    if df_interactions.empty:
        return

    log("\nGenerating interaction analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: Tier distribution per layer
    ax = axes[0, 0]
    tier_counts = df_interactions.groupby(['layer', 'analysis_tier']).size().unstack(fill_value=0)
    tier_counts.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Analysis Tier Distribution', fontsize=14, fontweight='bold')
    ax.legend(title='Tier', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: KL by tier at layer 28
    ax = axes[0, 1]
    layer_28 = df_interactions[df_interactions['layer'] == 28]
    if len(layer_28) > 0:
        tiers = layer_28['analysis_tier'].unique()
        tier_kl_means = [layer_28[layer_28['analysis_tier'] == t]['kl_divergence'].mean() for t in tiers]
        tier_kl_stds = [layer_28[layer_28['analysis_tier'] == t]['kl_divergence'].std() for t in tiers]

        x = range(len(tiers))
        bars = ax.bar(x, tier_kl_means, yerr=tier_kl_stds, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in tiers], fontsize=9)
        ax.set_ylabel('Mean KL Divergence', fontsize=12, fontweight='bold')
        ax.set_title('KL by Analysis Tier (Layer 28)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Panel 3: Top interactions (Tier 1 individual)
    ax = axes[1, 0]
    tier1 = df_interactions[(df_interactions['analysis_tier'] == 'tier1_individual') &
                            (df_interactions['layer'] == 28)]
    if len(tier1) > 0:
        top_interactions = tier1.nlargest(15, 'kl_divergence')
        labels = [f"{r.get('state', r.get('region', 'N/A'))}×{r.get('attribute', r.get('attribute_category', 'N/A'))}"
                 for _, r in top_interactions.iterrows()]
        ax.barh(range(len(top_interactions)), top_interactions['kl_divergence'],
                color='coral', alpha=0.8)
        ax.set_yticks(range(len(top_interactions)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('KL Divergence', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 State×Attribute Interactions (Layer 28)', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Tier 1 interactions available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Panel 4: Region × Category heatmap (Tier 3)
    ax = axes[1, 1]
    tier3 = df_interactions[(df_interactions['analysis_tier'] == 'tier3_region_category') &
                            (df_interactions['layer'] == 28)]
    if len(tier3) > 0 and 'region' in tier3.columns and 'attribute_category' in tier3.columns:
        pivot = tier3.pivot(index='region', columns='attribute_category', values='kl_divergence')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'KL Divergence'})
        ax.set_title('Region × Attribute Category (Layer 28)', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Tier 3 interactions available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / f'plots/overall/interaction_analysis{file_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: interaction_analysis.png")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def run_pipeline(df, activations, run_suffix="", pca_enabled=None, probe_csv=None, mdl_csv=None):
    """Execute KL analysis with optional PCA toggle and output suffix."""
    suffix = f"_{run_suffix}" if run_suffix else ""
    prev_pca = Config.PCA_ENABLED
    if pca_enabled is not None:
        Config.PCA_ENABLED = pca_enabled
    STATS['attempted'] = 0
    SKIPPED_SLICES.clear()

    log("="*80)
    log(f"RUN start | label='{run_suffix or 'default'}' | PCA={Config.PCA_ENABLED} dim={Config.PCA_DIM}")
    log("="*80)

    # Overall analysis
    df_overall = analyze_overall(df, activations)
    df_overall = add_kl_labels(df_overall)
    df_overall.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/overall_kl_divergence{suffix}.csv', index=False)
    plot_overall_kl(df_overall, file_suffix=suffix)
    gc.collect()

    # Group-level analysis
    df_group = analyze_by_group(df, activations)
    df_group = add_kl_labels(df_group)
    df_group.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/group_kl_divergence{suffix}.csv', index=False)
    plot_group_level_kl(df_group, file_suffix=suffix)
    gc.collect()

    # Attribute-level analysis
    df_attr = analyze_by_attribute(df, activations)
    df_attr = add_kl_labels(df_attr)
    df_attr.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/attribute_kl_divergence{suffix}.csv', index=False)
    df_attr.to_csv(Config.HEAVY_OUTPUT_DIR / f'attribute_kl_divergence_full{suffix}.csv', index=False)
    plot_attribute_level_kl(df_attr, file_suffix=suffix)
    gc.collect()

    # State-level analysis
    df_state = analyze_by_state(df, activations)
    df_state = add_kl_labels(df_state)
    df_state.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/state_kl_divergence{suffix}.csv', index=False)
    df_state.to_csv(Config.HEAVY_OUTPUT_DIR / f'state_kl_divergence_full{suffix}.csv', index=False)
    plot_state_level_kl(df_state, file_suffix=suffix)
    gc.collect()

    # Question-type analysis
    df_qtype = analyze_by_question_type(df, activations)
    if not df_qtype.empty:
        df_qtype = add_kl_labels(df_qtype)
        df_qtype.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/question_type_kl_divergence{suffix}.csv', index=False)
        plot_question_type_kl(df_qtype, file_suffix=suffix)
    gc.collect()

    # Region-level analysis (aggregated states)
    df_region = analyze_by_region(df, activations)
    if not df_region.empty:
        df_region = add_kl_labels(df_region)
        df_region.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/region_kl_divergence{suffix}.csv', index=False)
        plot_region_level_kl(df_region, file_suffix=suffix)
    gc.collect()

    # Attribute-category analysis (aggregated attributes)
    df_attr_cat = analyze_by_attribute_category(df, activations)
    if not df_attr_cat.empty:
        df_attr_cat = add_kl_labels(df_attr_cat)
        df_attr_cat.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/attribute_category_kl_divergence{suffix}.csv', index=False)
        plot_attribute_category_kl(df_attr_cat, file_suffix=suffix)
    gc.collect()

    # Interaction analysis (State × Attribute with hierarchical fallback)
    df_interactions = analyze_interactions(df, activations)
    if not df_interactions.empty:
        df_interactions.to_csv(Config.HEAVY_OUTPUT_DIR / f'interaction_kl_divergence{suffix}.csv', index=False)
        plot_interaction_analysis(df_interactions, file_suffix=suffix)

        # Summary of interaction tiers
        log("\n" + "="*80)
        log("INTERACTION ANALYSIS SUMMARY")
        log("="*80)
        tier_summary = df_interactions.groupby('analysis_tier').agg({
            'kl_divergence': ['mean', 'std', 'count']
        }).round(4)
        log(f"\n{tier_summary.to_string()}")

    # Summary statistics
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

    # Max KL jump for suppression group
    supp_data = df_group[df_group['group'] == 'suppression'].sort_values('layer')
    kl_diffs = supp_data['kl_divergence'].diff()
    max_increase_idx = kl_diffs.idxmax()
    if pd.notna(max_increase_idx):
        max_increase_layer = supp_data.loc[max_increase_idx, 'layer']
        log(f"\nMaximum KL increase for suppression group: Layer {int(max_increase_layer)}")

    # L24->L28 deltas
    delta_overall = compute_layer_deltas(df_overall)
    delta_group = compute_layer_deltas(df_group, level_cols=['group'])
    if len(delta_overall) > 0:
        delta_overall.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/layer_delta_overall{suffix}.csv', index=False)
        log("\nL24->L28 Overall Deltas:")
        for _, r in delta_overall.iterrows():
            js_delta = r['js_delta'] if pd.notna(r['js_delta']) else 'NA'
            log(f"  dKL={r['kl_delta']:+.6f} | dJS={js_delta}")
    if len(delta_group) > 0:
        delta_group.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/layer_delta_group{suffix}.csv', index=False)
        log("\nL24->L28 Group Deltas:")
        for _, r in delta_group.iterrows():
            js_delta = r['js_delta'] if pd.notna(r['js_delta']) else 'NA'
            log(f"  {r['group'].capitalize()}: dKL={r['kl_delta']:+.6f} | dJS={js_delta}")

    # Layer deltas for finer slices
    delta_attr = compute_layer_deltas(df_attr, level_cols=['attribute'])
    if len(delta_attr) > 0:
        delta_attr.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/layer_delta_attribute{suffix}.csv', index=False)
    delta_state = compute_layer_deltas(df_state, level_cols=['state'])
    if len(delta_state) > 0:
        delta_state.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/layer_delta_state{suffix}.csv', index=False)
    delta_qtype = compute_layer_deltas(df_qtype, level_cols=['question_type']) if not df_qtype.empty else pd.DataFrame()
    if len(delta_qtype) > 0:
        delta_qtype.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/layer_delta_question_type{suffix}.csv', index=False)

    # Layer deltas for aggregated analyses
    delta_region = compute_layer_deltas(df_region, level_cols=['region']) if not df_region.empty else pd.DataFrame()
    if len(delta_region) > 0:
        delta_region.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/layer_delta_region{suffix}.csv', index=False)
        log("\nL24->L28 Region Deltas:")
        for _, r in delta_region.iterrows():
            log(f"  {r['region']}: dKL={r['kl_delta']:+.6f}")

    delta_attr_cat = compute_layer_deltas(df_attr_cat, level_cols=['attribute_category']) if not df_attr_cat.empty else pd.DataFrame()
    if len(delta_attr_cat) > 0:
        delta_attr_cat.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/layer_delta_attribute_category{suffix}.csv', index=False)
        log("\nL24->L28 Attribute Category Deltas:")
        for _, r in delta_attr_cat.iterrows():
            log(f"  {r['attribute_category']}: dKL={r['kl_delta']:+.6f}")

    # Skipped slices audit
    if SKIPPED_SLICES:
        skipped_path = Config.LIGHT_OUTPUT_DIR / f'results/skipped_slices{suffix}.csv'
        pd.DataFrame(SKIPPED_SLICES).to_csv(skipped_path, index=False)
        rate = len(SKIPPED_SLICES) / max(STATS['attempted'], 1)
        log(f"\nSkipped slices: {len(SKIPPED_SLICES)} of {STATS['attempted']} attempted ({rate:.2%}); saved to {skipped_path}")

    # Top KL slices at final layer
    if len(df_attr) > 0:
        top_attr = df_attr[df_attr['layer'] == 28].sort_values('kl_divergence', ascending=False).head(15)
        top_attr.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/top_attributes_layer28{suffix}.csv', index=False)
    if len(df_state) > 0:
        top_state = df_state[df_state['layer'] == 28].sort_values('kl_divergence', ascending=False).head(15)
        top_state.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/top_states_layer28{suffix}.csv', index=False)

    # Convergence table
    conv_cols = ['layer', 'kl_divergence', 'kl_ci_lower', 'kl_ci_upper', 'js_divergence']
    conv = df_overall[conv_cols].copy()

    def merge_optional(path, label):
        if not path:
            return None
        try:
            df_ext = pd.read_csv(path)
            if 'layer' not in df_ext.columns:
                log(f"  Skipping {label} merge: no 'layer' column in {path}")
                return None
            return df_ext
        except Exception as exc:
            log(f"  Skipping {label} merge: {exc}")
            return None

    probe_df = merge_optional(probe_csv, 'probe')
    if probe_df is not None:
        conv = conv.merge(probe_df, on='layer', how='left', suffixes=('', '_probe'))
    mdl_df = merge_optional(mdl_csv, 'mdl')
    if mdl_df is not None:
        conv = conv.merge(mdl_df, on='layer', how='left', suffixes=('', '_mdl'))
    conv.to_csv(Config.LIGHT_OUTPUT_DIR / f'results/convergence_summary{suffix}.csv', index=False)

    log("\n" + "="*80)
    log(f"RUN COMPLETE | label='{run_suffix or 'default'}'")
    log("="*80)

    Config.PCA_ENABLED = prev_pca
    return {
        'df_overall': df_overall,
        'df_group': df_group,
        'df_attr': df_attr,
        'df_state': df_state,
        'df_qtype': df_qtype,
        'delta_overall': delta_overall,
        'delta_group': delta_group,
        'delta_attr': delta_attr,
        'delta_state': delta_state,
        'delta_qtype': delta_qtype
    }


def parse_args():
    parser = argparse.ArgumentParser(description="KL divergence analysis between base and instruct models")
    parser.add_argument('--pca-dim', type=int, default=Config.PCA_DIM, help='PCA dimensionality when enabled')
    parser.add_argument('--run-suffix', type=str, default='', help='Suffix appended to outputs')
    parser.add_argument('--probe-csv', type=str, default=None, help='Optional probe results CSV with a layer column')
    parser.add_argument('--mdl-csv', type=str, default=None, help='Optional MDL results CSV with a layer column')
    return parser.parse_args()


def main():
    args = parse_args()
    Config.PCA_DIM = args.pca_dim

    log("="*80)
    log("KL DIVERGENCE ANALYSIS PIPELINE")
    log("="*80)
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load once
    df, activations = load_data()

    # Full-dim run
    run_pipeline(df, activations, run_suffix=args.run_suffix, pca_enabled=False,
                 probe_csv=args.probe_csv, mdl_csv=args.mdl_csv)

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
