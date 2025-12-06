#!/usr/bin/env python3
"""
Linear Probing for Cultural Knowledge Suppression Detection

Comprehensive probing suite for analyzing RLHF-induced suppression in language models:
- Attribute probing (16-class cultural categories)
- Cross-model correctness probing (base↔instruct prediction)
- State probing (36-class geographic regions)
- Cross-model transfer analysis
- Suppression-predictive probing
- Multi-task joint probing
- Per-attribute/state breakdown analysis
- Layer-wise information flow analysis

Includes proper CV scaling, baseline comparisons, and statistical rigor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.multioutput import MultiOutputClassifier
from scipy import stats
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input paths
    ACTIVATION_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/activations")
    INDEX_FILE = ACTIVATION_DIR / "activation_index.csv"
    ENHANCED_DATA = Path("/home/anshulk/cultural-alignment-study/outputs/eda_results/tables/enhanced_dataset.csv")
    
    # Output paths
    OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/linear_probing")
    HEAVY_DATA_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/Linear_Probing/v2")
    
    # Models and layers
    MODELS = ['base', 'instruct']
    LAYERS = [8, 16, 24, 28]
    HIDDEN_SIZE = 1536
    
    # Probing settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.25
    CV_FOLDS = 5
    MAX_ITER = 2000
    
    # Probe types
    PROBE_TYPES = [
        'attribute',      # 16-class
        'correctness',    # binary (separate for base/instruct)
        'state',          # 36-class
        'multitask'       # joint attribute+correctness+state
    ]
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.HEAVY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        (self.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "reports").mkdir(exist_ok=True)
        (self.HEAVY_DATA_DIR / "models").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "results").mkdir(exist_ok=True)

config = Config()

# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        
    def section(self, title):
        msg = f"\n{'='*80}\n{title.upper()}\n{'='*80}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
    
    def result(self, key, value):
        msg = f"  • {key}: {value}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

log = Logger(config.OUTPUT_DIR / "probing_log.txt")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load activations and metadata."""
    log.section("Loading Data")
    
    # Load metadata
    log.log("Loading metadata...")
    df = pd.read_csv(config.ENHANCED_DATA)
    log.result("Total sentences", len(df))
    
    # Load activations
    activations = {}
    for model in config.MODELS:
        activations[model] = {}
        for layer in config.LAYERS:
            file_path = config.ACTIVATION_DIR / f"{model}_layer{layer}_activations.npy"
            log.log(f"Loading {model} layer {layer}...")
            activations[model][layer] = np.load(file_path)
            log.result(f"  Shape", activations[model][layer].shape)
    
    # Prepare labels
    log.log("\nPreparing labels...")
    
    # Encode categorical labels
    label_encoders = {}
    
    # Attribute (16 classes)
    le_attr = LabelEncoder()
    df['attribute_label'] = le_attr.fit_transform(df['attribute'])
    label_encoders['attribute'] = le_attr
    log.result("Attributes", len(le_attr.classes_))
    
    # State (36 classes)
    le_state = LabelEncoder()
    df['state_label'] = le_state.fit_transform(df['state'])
    label_encoders['state'] = le_state
    log.result("States", len(le_state.classes_))
    
    # Group type (for stratification)
    le_group = LabelEncoder()
    df['group_label'] = le_group.fit_transform(df['group_type'])
    label_encoders['group'] = le_group
    
    # Correctness (already binary)
    df['base_correct_label'] = df['base_correct'].astype(int)
    df['instruct_correct_label'] = df['instruct_correct'].astype(int)
    
    log.log("\nLabel distributions:")
    log.result("Attribute balance", df['attribute'].value_counts().to_dict())
    log.result("State balance", df['state'].value_counts().head(10).to_dict())
    log.result("Base correctness", f"{df['base_correct'].sum()}/{len(df)}")
    log.result("Instruct correctness", f"{df['instruct_correct'].sum()}/{len(df)}")
    
    return df, activations, label_encoders


def create_splits(df: pd.DataFrame, stratify_col: str = 'group_type'):
    """Create stratified train/test splits."""
    log.section("Creating Train/Test Splits")
    
    # Stratified split
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[stratify_col]
    )
    
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()
    
    log.result("Train size", len(train_df))
    log.result("Test size", len(test_df))
    
    # Check balance
    log.log("\nGroup distribution in splits:")
    for group in df[stratify_col].unique():
        train_pct = (train_df[stratify_col] == group).mean()
        test_pct = (test_df[stratify_col] == group).mean()
        log.result(f"  {group}", f"Train: {train_pct:.2%}, Test: {test_pct:.2%}")
    
    return train_idx, test_idx, train_df, test_df


# ============================================================================
# PROBE TRAINING
# ============================================================================

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe_type: str,
    scale: bool = True
) -> Tuple[object, float, Dict]:
    """Train a single linear probe."""
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Determine task type
    n_classes = len(np.unique(y_train))
    
    if n_classes == 2:
        # Binary classification
        probe = LogisticRegression(
            max_iter=config.MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        )
    else:
        # Multi-class classification
        probe = LogisticRegression(
            max_iter=config.MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            multi_class='multinomial'
        )
    
    # Train
    probe.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = probe.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_classes': int(n_classes)
    }
    
    # ROC-AUC for binary
    if n_classes == 2:
        try:
            y_proba = probe.predict_proba(X_test_scaled)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    result = {
        'probe': probe,
        'scaler': scaler,
        'accuracy': accuracy,
        'metrics': metrics
    }
    
    return probe, accuracy, result


def cross_validate_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5
) -> Dict:
    """Cross-validation for probe with scaling inside CV loop via Pipeline."""
    n_classes = len(np.unique(y))

    if n_classes == 2:
        probe = LogisticRegression(
            max_iter=config.MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        )
    else:
        probe = LogisticRegression(
            max_iter=config.MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            multi_class='multinomial'
        )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', probe)
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    return {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'scores': scores.tolist()
    }


def compute_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_permutations: int = 10
) -> Dict:
    """Compute baseline accuracies for proper context.

    Returns:
    - chance_level: Expected accuracy from random guessing (1/n_classes)
    - majority_class: Accuracy from always predicting most common class
    - random_features: Accuracy from probe trained on shuffled activations
                       (averaged over n_permutations to reduce variance)
    """
    n_classes = len(np.unique(y_train))

    # 1. Chance level (random guessing)
    chance_level = 1.0 / n_classes

    # 2. Majority class baseline
    from collections import Counter
    class_counts = Counter(y_train)
    majority_class = class_counts.most_common(1)[0][0]
    majority_accuracy = (y_test == majority_class).mean()

    # 3. Random features baseline (permutation test)
    # Train probe on shuffled X to establish null distribution
    random_accuracies = []
    rng = np.random.RandomState(config.RANDOM_STATE)

    for i in range(n_permutations):
        # Shuffle activations independently of labels
        shuffle_idx = rng.permutation(len(X_train))
        X_train_shuffled = X_train[shuffle_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_shuffled)
        X_test_scaled = scaler.transform(X_test)  # Test set not shuffled

        # Train probe
        if n_classes == 2:
            probe = LogisticRegression(
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE + i,
                class_weight='balanced'
            )
        else:
            probe = LogisticRegression(
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE + i,
                class_weight='balanced',
                multi_class='multinomial'
            )

        probe.fit(X_train_scaled, y_train)
        y_pred = probe.predict(X_test_scaled)
        random_accuracies.append(accuracy_score(y_test, y_pred))

    random_feature_mean = float(np.mean(random_accuracies))
    random_feature_std = float(np.std(random_accuracies))

    return {
        'chance_level': float(chance_level),
        'majority_class_accuracy': float(majority_accuracy),
        'random_features_mean': random_feature_mean,
        'random_features_std': random_feature_std,
        'n_classes': int(n_classes)
    }


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn=accuracy_score,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Dict:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_fn: Scoring function (default: accuracy_score)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default: 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dict with point estimate, CI lower, CI upper, and standard error
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap resampling
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute CI using percentile method
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    std_error = np.std(bootstrap_scores)

    return {
        'point_estimate': float(point_estimate),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_level': ci,
        'std_error': float(std_error),
        'n_bootstrap': n_bootstrap
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation for unequal sample sizes.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def paired_comparison(
    scores1: List[float],
    scores2: List[float],
    names: Tuple[str, str] = ('Model1', 'Model2'),
    alpha: float = 0.05,
    n_comparisons: int = 1
) -> Dict:
    """Perform paired statistical comparison between two sets of scores.

    Args:
        scores1, scores2: Paired scores (e.g., CV fold accuracies)
        names: Names for the two conditions
        alpha: Significance level (default: 0.05)
        n_comparisons: Number of comparisons for Bonferroni correction

    Returns:
        Dict with t-statistic, p-value, effect size, and significance
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # Bonferroni correction
    corrected_alpha = alpha / n_comparisons
    is_significant = p_value < corrected_alpha

    # Effect size (Cohen's d for paired samples)
    diff = scores1 - scores2
    effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0

    # Interpret effect size
    abs_d = abs(effect_size)
    if abs_d < 0.2:
        effect_interpretation = 'negligible'
    elif abs_d < 0.5:
        effect_interpretation = 'small'
    elif abs_d < 0.8:
        effect_interpretation = 'medium'
    else:
        effect_interpretation = 'large'

    return {
        'comparison': f"{names[0]} vs {names[1]}",
        'mean_diff': float(np.mean(diff)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'corrected_alpha': float(corrected_alpha),
        'is_significant': bool(is_significant),
        'cohens_d': float(effect_size),
        'effect_interpretation': effect_interpretation,
        'n_comparisons': n_comparisons
    }


def compute_statistical_summary(
    results: Dict,
    task_name: str
) -> Dict:
    """Compute comprehensive statistical summary across models and layers.

    Args:
        results: Dict with structure {model: {layer: {accuracy, cv: {scores}}}}
        task_name: Name of the probing task

    Returns:
        Dict with cross-model comparisons and effect sizes
    """
    summary = {
        'task': task_name,
        'comparisons': [],
        'layer_effects': {}
    }

    # Number of comparisons for Bonferroni (layers × model pairs)
    n_layers = len(config.LAYERS)
    n_comparisons = n_layers  # base vs instruct at each layer

    # Compare base vs instruct at each layer
    for layer in config.LAYERS:
        if 'base' in results and 'instruct' in results:
            base_scores = results['base'][layer]['cv']['scores']
            instruct_scores = results['instruct'][layer]['cv']['scores']

            comparison = paired_comparison(
                base_scores,
                instruct_scores,
                names=('base', 'instruct'),
                n_comparisons=n_comparisons
            )
            comparison['layer'] = layer
            summary['comparisons'].append(comparison)

            # Store layer-specific effect
            summary['layer_effects'][layer] = {
                'base_mean': float(np.mean(base_scores)),
                'instruct_mean': float(np.mean(instruct_scores)),
                'difference': float(np.mean(base_scores) - np.mean(instruct_scores)),
                'cohens_d': comparison['cohens_d'],
                'significant': comparison['is_significant']
            }

    return summary


# ============================================================================
# PROBE 1: ATTRIBUTE (16-CLASS)
# ============================================================================

def probe_attribute(df, train_idx, test_idx, activations):
    """Attribute probing (16-class classification)."""
    log.section("Probe 1: Attribute Classification (16-class)")

    results = {}
    baselines_computed = False
    baselines = None

    for model in config.MODELS:
        results[model] = {}

        for layer in config.LAYERS:
            log.log(f"\n{model.upper()} Layer {layer}")

            # Get activations
            X = activations[model][layer]
            y = df['attribute_label'].values

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Compute baselines once (same labels across all models/layers)
            if not baselines_computed:
                baselines = compute_baselines(X_train, y_train, X_test, y_test)
                log.log("\n--- Baselines (16-class attribute) ---")
                log.result("Chance level", f"{baselines['chance_level']:.4f} (1/{baselines['n_classes']})")
                log.result("Majority class", f"{baselines['majority_class_accuracy']:.4f}")
                log.result("Random features", f"{baselines['random_features_mean']:.4f} ± {baselines['random_features_std']:.4f}")
                baselines_computed = True

            # Train probe
            probe, accuracy, result = train_probe(
                X_train, y_train, X_test, y_test, 'attribute'
            )

            # Compute accuracy gain over baselines
            gain_over_chance = accuracy - baselines['chance_level']
            gain_over_random = accuracy - baselines['random_features_mean']

            log.result(f"Accuracy", f"{accuracy:.4f}")
            log.result(f"Gain over chance", f"+{gain_over_chance:.4f}")
            log.result(f"Gain over random", f"+{gain_over_random:.4f}")
            log.result(f"Precision", f"{result['metrics']['precision']:.4f}")
            log.result(f"Recall", f"{result['metrics']['recall']:.4f}")
            log.result(f"F1", f"{result['metrics']['f1']:.4f}")

            # Cross-validation
            cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
            log.result(f"CV Accuracy", f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

            # Bootstrap CI for test set accuracy (fit on train, transform test)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_pred = probe.predict(X_test_scaled)
            bootstrap_result = bootstrap_ci(y_test, y_pred)
            log.result(f"95% CI", f"[{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")

            result['cv'] = cv_result
            result['bootstrap_ci'] = bootstrap_result
            result['baselines'] = baselines
            result['gain_over_chance'] = float(gain_over_chance)
            result['gain_over_random'] = float(gain_over_random)
            results[model][layer] = result

            # Save probe
            probe_file = config.HEAVY_DATA_DIR / "models" / f"attribute_{model}_layer{layer}.pkl"
            with open(probe_file, 'wb') as f:
                pickle.dump({'probe': probe, 'scaler': result['scaler']}, f)

    # Compute statistical summary (base vs instruct comparisons)
    log.log("\n--- Statistical Comparisons (Base vs Instruct) ---")
    stat_summary = compute_statistical_summary(results, 'attribute')
    for comp in stat_summary['comparisons']:
        sig_marker = "*" if comp['is_significant'] else ""
        log.result(
            f"Layer {comp['layer']}",
            f"Δ={comp['mean_diff']:+.4f}, t={comp['t_statistic']:.2f}, "
            f"p={comp['p_value']:.4f}{sig_marker}, d={comp['cohens_d']:.2f} ({comp['effect_interpretation']})"
        )

    # Save results
    with open(config.OUTPUT_DIR / "results" / "attribute_probing.json", 'w') as f:
        results_serializable = {
            model: {
                layer: {k: v for k, v in result.items() if k not in ['probe', 'scaler']}
                for layer, result in layer_results.items()
            }
            for model, layer_results in results.items()
        }
        results_serializable['statistical_summary'] = stat_summary
        json.dump(results_serializable, f, indent=2)

    return results


# ============================================================================
# PROBE 2: CROSS-MODEL CORRECTNESS (BINARY)
# ============================================================================

def probe_correctness(df, train_idx, test_idx, activations):
    """Cross-model correctness probing (binary classification).

    Tests whether one model's representations contain information about
    the other model's behavioral outcomes:
    - Base activations → predict INSTRUCT correctness
    - Instruct activations → predict BASE correctness
    """
    log.section("Probe 2: Cross-Model Correctness Prediction (Binary)")

    results = {}

    # Define cross-model pairs: (source_model, target_correctness)
    cross_pairs = [
        ('base', 'instruct'),  # Base activations predict instruct correctness
        ('instruct', 'base'),  # Instruct activations predict base correctness
    ]

    for source_model, target_model in cross_pairs:
        pair_key = f"{source_model}_to_{target_model}"
        results[pair_key] = {}

        for layer in config.LAYERS:
            log.log(f"\n{source_model.upper()} activations → {target_model.upper()} correctness | Layer {layer}")

            # Get SOURCE model activations
            X = activations[source_model][layer]

            # Predict TARGET model correctness (cross-model, non-circular)
            y = df[f'{target_model}_correct_label'].values

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train probe
            probe, accuracy, result = train_probe(
                X_train, y_train, X_test, y_test, 'correctness'
            )

            log.result(f"Accuracy", f"{accuracy:.4f}")
            log.result(f"Precision", f"{result['metrics']['precision']:.4f}")
            log.result(f"Recall", f"{result['metrics']['recall']:.4f}")
            log.result(f"F1", f"{result['metrics']['f1']:.4f}")
            if 'roc_auc' in result['metrics']:
                log.result(f"ROC-AUC", f"{result['metrics']['roc_auc']:.4f}")

            # Cross-validation
            cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
            log.result(f"CV Accuracy", f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

            result['cv'] = cv_result
            results[pair_key][layer] = result

            # Save probe
            probe_file = config.HEAVY_DATA_DIR / "models" / f"correctness_{pair_key}_layer{layer}.pkl"
            with open(probe_file, 'wb') as f:
                pickle.dump({'probe': probe, 'scaler': result['scaler']}, f)

    # Also compute self-correctness for comparison (labeled as baseline)
    log.log("\n--- Self-Correctness Baseline (for comparison only) ---")
    for model in config.MODELS:
        self_key = f"{model}_self"
        results[self_key] = {}

        for layer in config.LAYERS:
            X = activations[model][layer]
            y = df[f'{model}_correct_label'].values

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            probe, accuracy, result = train_probe(
                X_train, y_train, X_test, y_test, 'correctness'
            )

            log.log(f"{model.upper()} self L{layer}: {accuracy:.4f} (baseline)")
            result['cv'] = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
            results[self_key][layer] = result
    
    # Save results
    with open(config.OUTPUT_DIR / "results" / "correctness_probing.json", 'w') as f:
        results_serializable = {
            model: {
                layer: {k: v for k, v in result.items() if k not in ['probe', 'scaler']}
                for layer, result in layer_results.items()
            }
            for model, layer_results in results.items()
        }
        json.dump(results_serializable, f, indent=2)
    
    return results


# ============================================================================
# PROBE 3: STATE (36-CLASS)
# ============================================================================

def probe_state(df, train_idx, test_idx, activations):
    """State probing (36-class classification)."""
    log.section("Probe 3: State Classification (36-class)")

    results = {}
    baselines_computed = False
    baselines = None

    for model in config.MODELS:
        results[model] = {}

        for layer in config.LAYERS:
            log.log(f"\n{model.upper()} Layer {layer}")

            # Get activations
            X = activations[model][layer]
            y = df['state_label'].values

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Compute baselines once (same labels across all models/layers)
            if not baselines_computed:
                baselines = compute_baselines(X_train, y_train, X_test, y_test)
                log.log("\n--- Baselines (36-class state) ---")
                log.result("Chance level", f"{baselines['chance_level']:.4f} (1/{baselines['n_classes']})")
                log.result("Majority class", f"{baselines['majority_class_accuracy']:.4f}")
                log.result("Random features", f"{baselines['random_features_mean']:.4f} ± {baselines['random_features_std']:.4f}")
                baselines_computed = True

            # Train probe
            probe, accuracy, result = train_probe(
                X_train, y_train, X_test, y_test, 'state'
            )

            # Compute accuracy gain over baselines
            gain_over_chance = accuracy - baselines['chance_level']
            gain_over_random = accuracy - baselines['random_features_mean']

            log.result(f"Accuracy", f"{accuracy:.4f}")
            log.result(f"Gain over chance", f"+{gain_over_chance:.4f}")
            log.result(f"Gain over random", f"+{gain_over_random:.4f}")
            log.result(f"Precision", f"{result['metrics']['precision']:.4f}")
            log.result(f"Recall", f"{result['metrics']['recall']:.4f}")
            log.result(f"F1", f"{result['metrics']['f1']:.4f}")

            # Cross-validation
            cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
            log.result(f"CV Accuracy", f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

            # Bootstrap CI for test set accuracy (fit on train, transform test)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_pred = probe.predict(X_test_scaled)
            bootstrap_result = bootstrap_ci(y_test, y_pred)
            log.result(f"95% CI", f"[{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")

            result['cv'] = cv_result
            result['bootstrap_ci'] = bootstrap_result
            result['baselines'] = baselines
            result['gain_over_chance'] = float(gain_over_chance)
            result['gain_over_random'] = float(gain_over_random)
            results[model][layer] = result

            # Save probe
            probe_file = config.HEAVY_DATA_DIR / "models" / f"state_{model}_layer{layer}.pkl"
            with open(probe_file, 'wb') as f:
                pickle.dump({'probe': probe, 'scaler': result['scaler']}, f)

    # Compute statistical summary (base vs instruct comparisons)
    log.log("\n--- Statistical Comparisons (Base vs Instruct) ---")
    stat_summary = compute_statistical_summary(results, 'state')
    for comp in stat_summary['comparisons']:
        sig_marker = "*" if comp['is_significant'] else ""
        log.result(
            f"Layer {comp['layer']}",
            f"Δ={comp['mean_diff']:+.4f}, t={comp['t_statistic']:.2f}, "
            f"p={comp['p_value']:.4f}{sig_marker}, d={comp['cohens_d']:.2f} ({comp['effect_interpretation']})"
        )

    # Save results
    with open(config.OUTPUT_DIR / "results" / "state_probing.json", 'w') as f:
        results_serializable = {
            model: {
                layer: {k: v for k, v in result.items() if k not in ['probe', 'scaler']}
                for layer, result in layer_results.items()
            }
            for model, layer_results in results.items()
        }
        results_serializable['statistical_summary'] = stat_summary
        json.dump(results_serializable, f, indent=2)

    return results


# ============================================================================
# PROBE 4: CROSS-MODEL TRANSFER
# ============================================================================

def probe_cross_model_transfer(df, train_idx, test_idx, activations):
    """Cross-model transfer probing."""
    log.section("Probe 4: Cross-Model Transfer (CRITICAL TEST)")
    
    results = {}
    
    # For each layer, train on Base and test on Instruct
    for layer in config.LAYERS:
        log.log(f"\nLayer {layer}: Train on Base → Test on Instruct")
        
        results[layer] = {}
        
        # Get activations
        X_base = activations['base'][layer]
        X_instruct = activations['instruct'][layer]
        
        # Test 1: Attribute transfer
        log.log("\n  Attribute Transfer:")
        y = df['attribute_label'].values
        
        X_train_base = X_base[train_idx]
        y_train = y[train_idx]
        
        X_test_base = X_base[test_idx]
        X_test_instruct = X_instruct[test_idx]
        y_test = y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_base)
        X_test_base_scaled = scaler.transform(X_test_base)
        X_test_instruct_scaled = scaler.transform(X_test_instruct)
        
        # Train probe on Base
        probe = LogisticRegression(
            max_iter=config.MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            multi_class='multinomial'
        )
        probe.fit(X_train_scaled, y_train)
        
        # Test on Base (in-model)
        y_pred_base = probe.predict(X_test_base_scaled)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        # Test on Instruct (cross-model)
        y_pred_instruct = probe.predict(X_test_instruct_scaled)
        acc_instruct = accuracy_score(y_test, y_pred_instruct)
        
        transfer_rate = acc_instruct / acc_base if acc_base > 0 else 0
        
        log.result("  Base → Base", f"{acc_base:.4f}")
        log.result("  Base → Instruct", f"{acc_instruct:.4f}")
        log.result("  Transfer Rate", f"{transfer_rate:.4f} ({(transfer_rate-1)*100:+.2f}%)")
        
        results[layer]['attribute'] = {
            'base_to_base': float(acc_base),
            'base_to_instruct': float(acc_instruct),
            'transfer_rate': float(transfer_rate),
            'absolute_drop': float(acc_base - acc_instruct)
        }
        
        # Test 2: Correctness transfer
        log.log("\n  Correctness Transfer (Base):")
        y = df['base_correct_label'].values
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        probe_correct = LogisticRegression(
            max_iter=config.MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        )
        probe_correct.fit(X_train_scaled, y_train)
        
        y_pred_base = probe_correct.predict(X_test_base_scaled)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        y_pred_instruct = probe_correct.predict(X_test_instruct_scaled)
        acc_instruct = accuracy_score(y_test, y_pred_instruct)
        
        transfer_rate = acc_instruct / acc_base if acc_base > 0 else 0
        
        log.result("  Base → Base", f"{acc_base:.4f}")
        log.result("  Base → Instruct", f"{acc_instruct:.4f}")
        log.result("  Transfer Rate", f"{transfer_rate:.4f}")
        
        results[layer]['base_correctness'] = {
            'base_to_base': float(acc_base),
            'base_to_instruct': float(acc_instruct),
            'transfer_rate': float(transfer_rate),
            'absolute_drop': float(acc_base - acc_instruct)
        }
        
        # Test 3: State transfer
        log.log("\n  State Transfer:")
        y = df['state_label'].values
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        probe_state = LogisticRegression(
            max_iter=config.MAX_ITER,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            multi_class='multinomial'
        )
        probe_state.fit(X_train_scaled, y_train)
        
        y_pred_base = probe_state.predict(X_test_base_scaled)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        y_pred_instruct = probe_state.predict(X_test_instruct_scaled)
        acc_instruct = accuracy_score(y_test, y_pred_instruct)
        
        transfer_rate = acc_instruct / acc_base if acc_base > 0 else 0
        
        log.result("  Base → Base", f"{acc_base:.4f}")
        log.result("  Base → Instruct", f"{acc_instruct:.4f}")
        log.result("  Transfer Rate", f"{transfer_rate:.4f}")
        
        results[layer]['state'] = {
            'base_to_base': float(acc_base),
            'base_to_instruct': float(acc_instruct),
            'transfer_rate': float(transfer_rate),
            'absolute_drop': float(acc_base - acc_instruct)
        }
    
    # Save results
    with open(config.OUTPUT_DIR / "results" / "cross_model_transfer.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# PROBE 5: MULTI-TASK (JOINT)
# ============================================================================

def probe_multitask(df, train_idx, test_idx, activations):
    """Multi-task joint probing: attribute + correctness + state."""
    log.section("Probe 5: Multi-Task Joint Probing")
    
    results = {}
    
    for model in config.MODELS:
        results[model] = {}
        
        for layer in config.LAYERS:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            # Get activations
            X = activations[model][layer]
            
            # Prepare multi-task labels
            y_attribute = df['attribute_label'].values
            y_correctness = df[f'{model}_correct_label'].values
            y_state = df['state_label'].values
            
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            y_attr_train, y_attr_test = y_attribute[train_idx], y_attribute[test_idx]
            y_corr_train, y_corr_test = y_correctness[train_idx], y_correctness[test_idx]
            y_state_train, y_state_test = y_state[train_idx], y_state[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Strategy 1: Separate probes (baseline)
            log.log("  Strategy 1: Independent Probes")
            
            probe_attr = LogisticRegression(
                max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
                class_weight='balanced', multi_class='multinomial'
            )
            probe_corr = LogisticRegression(
                max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
                class_weight='balanced'
            )
            probe_state = LogisticRegression(
                max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
                class_weight='balanced', multi_class='multinomial'
            )
            
            probe_attr.fit(X_train_scaled, y_attr_train)
            probe_corr.fit(X_train_scaled, y_corr_train)
            probe_state.fit(X_train_scaled, y_state_train)
            
            acc_attr = accuracy_score(y_attr_test, probe_attr.predict(X_test_scaled))
            acc_corr = accuracy_score(y_corr_test, probe_corr.predict(X_test_scaled))
            acc_state = accuracy_score(y_state_test, probe_state.predict(X_test_scaled))
            
            log.result("    Attribute", f"{acc_attr:.4f}")
            log.result("    Correctness", f"{acc_corr:.4f}")
            log.result("    State", f"{acc_state:.4f}")
            log.result("    Average", f"{(acc_attr + acc_corr + acc_state)/3:.4f}")
            
            # Strategy 2: Shared representation (concatenated predictions)
            log.log("  Strategy 2: Joint Multi-Task")
            
            # Create multi-output target
            y_multi_train = np.column_stack([y_attr_train, y_corr_train, y_state_train])
            y_multi_test = np.column_stack([y_attr_test, y_corr_test, y_state_test])
            
            # Multi-output classifier
            base_clf = LogisticRegression(
                max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
                class_weight='balanced'
            )
            multi_clf = MultiOutputClassifier(base_clf, n_jobs=-1)
            
            multi_clf.fit(X_train_scaled, y_multi_train)
            y_pred_multi = multi_clf.predict(X_test_scaled)
            
            acc_attr_joint = accuracy_score(y_attr_test, y_pred_multi[:, 0])
            acc_corr_joint = accuracy_score(y_corr_test, y_pred_multi[:, 1])
            acc_state_joint = accuracy_score(y_state_test, y_pred_multi[:, 2])
            
            log.result("    Attribute", f"{acc_attr_joint:.4f}")
            log.result("    Correctness", f"{acc_corr_joint:.4f}")
            log.result("    State", f"{acc_state_joint:.4f}")
            log.result("    Average", f"{(acc_attr_joint + acc_corr_joint + acc_state_joint)/3:.4f}")
            
            # Compare strategies
            log.log("  Comparison:")
            attr_diff = acc_attr_joint - acc_attr
            corr_diff = acc_corr_joint - acc_corr
            state_diff = acc_state_joint - acc_state
            
            log.result("    Attribute Δ", f"{attr_diff:+.4f}")
            log.result("    Correctness Δ", f"{corr_diff:+.4f}")
            log.result("    State Δ", f"{state_diff:+.4f}")
            
            results[model][layer] = {
                'independent': {
                    'attribute': float(acc_attr),
                    'correctness': float(acc_corr),
                    'state': float(acc_state),
                    'average': float((acc_attr + acc_corr + acc_state)/3)
                },
                'joint': {
                    'attribute': float(acc_attr_joint),
                    'correctness': float(acc_corr_joint),
                    'state': float(acc_state_joint),
                    'average': float((acc_attr_joint + acc_corr_joint + acc_state_joint)/3)
                },
                'differences': {
                    'attribute': float(attr_diff),
                    'correctness': float(corr_diff),
                    'state': float(state_diff)
                }
            }
            
            # Save joint model
            model_file = config.HEAVY_DATA_DIR / "models" / f"multitask_{model}_layer{layer}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'multi_clf': multi_clf,
                    'scaler': scaler,
                    'separate_probes': {
                        'attribute': probe_attr,
                        'correctness': probe_corr,
                        'state': probe_state
                    }
                }, f)
    
    # Save results
    with open(config.OUTPUT_DIR / "results" / "multitask_probing.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# PROBE 6: SUPPRESSION-PREDICTIVE PROBING
# ============================================================================

def probe_suppression_prediction(df, train_idx, test_idx, activations):
    """Suppression-predictive probing: Can we predict suppression from activations?

    Tests whether suppression is predictable from representations:
    - Base activations → predict suppression (base correct, instruct wrong)
    - Instruct activations → predict suppression (comparison)
    - Base activations → predict enhancement (control)

    High accuracy from base would indicate suppression targets are identifiable
    from content before RLHF acts.
    """
    log.section("Probe 6: Suppression-Predictive Probing")

    # Create suppression label
    df['suppression_label'] = (
        (df['base_correct'] == True) & (df['instruct_correct'] == False)
    ).astype(int)

    # Also create enhancement label for comparison
    df['enhancement_label'] = (
        (df['base_correct'] == False) & (df['instruct_correct'] == True)
    ).astype(int)

    results = {}

    # Log class distribution
    n_suppression = df['suppression_label'].sum()
    n_enhancement = df['enhancement_label'].sum()
    n_total = len(df)
    log.log(f"\nClass distribution:")
    log.result("Suppression cases", f"{n_suppression} ({n_suppression/n_total*100:.1f}%)")
    log.result("Enhancement cases", f"{n_enhancement} ({n_enhancement/n_total*100:.1f}%)")
    log.result("Other cases", f"{n_total - n_suppression - n_enhancement} ({(n_total - n_suppression - n_enhancement)/n_total*100:.1f}%)")

    # Test 1: Predict suppression from BASE activations
    log.log("\n--- Predicting SUPPRESSION from BASE activations ---")
    results['suppression_from_base'] = {}

    for layer in config.LAYERS:
        log.log(f"\nLayer {layer}")

        X = activations['base'][layer]
        y = df['suppression_label'].values

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Compute baselines
        baselines = compute_baselines(X_train, y_train, X_test, y_test, n_permutations=5)

        # Train probe
        probe, accuracy, result = train_probe(
            X_train, y_train, X_test, y_test, 'suppression'
        )

        gain_over_chance = accuracy - baselines['chance_level']
        gain_over_random = accuracy - baselines['random_features_mean']

        log.result(f"Accuracy", f"{accuracy:.4f}")
        log.result(f"Chance level", f"{baselines['chance_level']:.4f}")
        log.result(f"Gain over chance", f"+{gain_over_chance:.4f}")
        log.result(f"Gain over random", f"+{gain_over_random:.4f}")

        if 'roc_auc' in result['metrics']:
            log.result(f"ROC-AUC", f"{result['metrics']['roc_auc']:.4f}")

        # CV
        cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
        log.result(f"CV Accuracy", f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

        result['cv'] = cv_result
        result['baselines'] = baselines
        result['gain_over_chance'] = float(gain_over_chance)
        result['gain_over_random'] = float(gain_over_random)
        results['suppression_from_base'][layer] = result

    # Test 2: Predict enhancement from BASE activations (control comparison)
    log.log("\n--- Predicting ENHANCEMENT from BASE activations (control) ---")
    results['enhancement_from_base'] = {}

    for layer in config.LAYERS:
        log.log(f"\nLayer {layer}")

        X = activations['base'][layer]
        y = df['enhancement_label'].values

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        baselines = compute_baselines(X_train, y_train, X_test, y_test, n_permutations=5)

        probe, accuracy, result = train_probe(
            X_train, y_train, X_test, y_test, 'enhancement'
        )

        log.result(f"Accuracy", f"{accuracy:.4f}")
        log.result(f"Gain over chance", f"+{accuracy - baselines['chance_level']:.4f}")

        if 'roc_auc' in result['metrics']:
            log.result(f"ROC-AUC", f"{result['metrics']['roc_auc']:.4f}")

        cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
        result['cv'] = cv_result
        result['baselines'] = baselines
        results['enhancement_from_base'][layer] = result

    # Test 3: Predict suppression from INSTRUCT activations (should be easier)
    log.log("\n--- Predicting SUPPRESSION from INSTRUCT activations ---")
    results['suppression_from_instruct'] = {}

    for layer in config.LAYERS:
        log.log(f"\nLayer {layer}")

        X = activations['instruct'][layer]
        y = df['suppression_label'].values

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        probe, accuracy, result = train_probe(
            X_train, y_train, X_test, y_test, 'suppression'
        )

        log.result(f"Accuracy", f"{accuracy:.4f}")
        if 'roc_auc' in result['metrics']:
            log.result(f"ROC-AUC", f"{result['metrics']['roc_auc']:.4f}")

        cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
        result['cv'] = cv_result
        results['suppression_from_instruct'][layer] = result

    # Summary comparison
    log.log("\n--- Summary: Suppression Predictability ---")
    for layer in config.LAYERS:
        base_acc = results['suppression_from_base'][layer]['accuracy']
        inst_acc = results['suppression_from_instruct'][layer]['accuracy']
        base_auc = results['suppression_from_base'][layer]['metrics'].get('roc_auc', 0)
        inst_auc = results['suppression_from_instruct'][layer]['metrics'].get('roc_auc', 0)

        log.result(
            f"Layer {layer}",
            f"Base→Supp: {base_acc:.4f} (AUC:{base_auc:.3f}) | "
            f"Inst→Supp: {inst_acc:.4f} (AUC:{inst_auc:.3f}) | "
            f"Δ={inst_acc - base_acc:+.4f}"
        )

    # Save results
    with open(config.OUTPUT_DIR / "results" / "suppression_predictive_probing.json", 'w') as f:
        results_serializable = {
            key: {
                layer: {k: v for k, v in result.items() if k not in ['probe', 'scaler']}
                for layer, result in layer_results.items()
            }
            for key, layer_results in results.items()
        }
        json.dump(results_serializable, f, indent=2)

    return results


# ============================================================================
# GROUP-WISE ANALYSIS
# ============================================================================

def analyze_by_group(df, train_idx, test_idx, activations):
    """Analyze probing performance by group type."""
    log.section("Group-Wise Analysis")
    
    results = {}
    
    for model in config.MODELS:
        results[model] = {}
        
        for layer in config.LAYERS:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            results[model][layer] = {}
            
            # Get activations
            X = activations[model][layer]
            
            # Only use test set for group analysis
            X_test = X[test_idx]
            test_df = df.loc[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train = X[train_idx]
            scaler.fit(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train probes on full training set
            # Attribute
            y_attr_train = df.loc[train_idx, 'attribute_label'].values
            probe_attr = LogisticRegression(
                max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
                class_weight='balanced', multi_class='multinomial'
            )
            probe_attr.fit(scaler.transform(X_train), y_attr_train)
            
            # Correctness
            y_corr_train = df.loc[train_idx, f'{model}_correct_label'].values
            probe_corr = LogisticRegression(
                max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
                class_weight='balanced'
            )
            probe_corr.fit(scaler.transform(X_train), y_corr_train)
            
            # Analyze by group
            for group in ['suppression', 'enhancement', 'control']:
                group_mask = test_df['group_type'] == group
                
                if group_mask.sum() == 0:
                    continue
                
                X_group = X_test_scaled[group_mask]
                
                # Attribute accuracy
                y_attr_group = test_df.loc[group_mask, 'attribute_label'].values
                pred_attr = probe_attr.predict(X_group)
                acc_attr = accuracy_score(y_attr_group, pred_attr)
                
                # Correctness accuracy
                y_corr_group = test_df.loc[group_mask, f'{model}_correct_label'].values
                pred_corr = probe_corr.predict(X_group)
                acc_corr = accuracy_score(y_corr_group, pred_corr)
                
                log.result(f"  {group}", f"Attr: {acc_attr:.4f}, Corr: {acc_corr:.4f}")
                
                results[model][layer][group] = {
                    'n_samples': int(group_mask.sum()),
                    'attribute_accuracy': float(acc_attr),
                    'correctness_accuracy': float(acc_corr)
                }
    
    # Save results
    with open(config.OUTPUT_DIR / "results" / "group_wise_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# PER-ATTRIBUTE/STATE BREAKDOWN
# ============================================================================

def analyze_by_cultural_category(df, train_idx, test_idx, activations, label_encoders):
    """Analyze probing performance broken down by attribute and state.

    Tests the selectivity hypothesis by comparing probe accuracy across:
    - High-suppression categories (Religion, Rituals, Costume, Dance/Music)
    - Low-suppression categories (Nightlife, Transport, Medicine, Sports)
    - High-suppression states (Tamil Nadu, Karnataka, Kerala, Mizoram)
    - Low-suppression states (Delhi, Punjab, Haryana, Chandigarh)
    """
    log.section("Per-Attribute/State Breakdown Analysis")

    results = {
        'by_attribute': {},
        'by_state': {},
        'high_vs_low_suppression': {}
    }

    # Define high/low suppression categories based on prior analysis
    HIGH_SUPPRESSION_ATTRIBUTES = ['Religion', 'Rituals_and_Ceremonies', 'Costume', 'Dance_and_Music']
    LOW_SUPPRESSION_ATTRIBUTES = ['Nightlife', 'Transport', 'Medicine', 'Sports']

    HIGH_SUPPRESSION_STATES = ['Tamil_Nadu', 'Karnataka', 'Kerala', 'Mizoram', 'Arunachal_Pradesh']
    LOW_SUPPRESSION_STATES = ['Delhi', 'Punjab', 'Haryana', 'Chandigarh']

    # Use Layer 28 for analysis (most relevant for suppression)
    layer = 28
    log.log(f"\nAnalyzing Layer {layer} (primary suppression layer)")

    for model in config.MODELS:
        log.log(f"\n--- {model.upper()} Model ---")

        X = activations[model][layer]
        X_train, X_test = X[train_idx], X[test_idx]
        test_df = df.loc[test_idx].copy()

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train attribute probe
        y_attr_train = df.loc[train_idx, 'attribute_label'].values
        probe_attr = LogisticRegression(
            max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
            class_weight='balanced', multi_class='multinomial'
        )
        probe_attr.fit(X_train_scaled, y_attr_train)

        # Train state probe
        y_state_train = df.loc[train_idx, 'state_label'].values
        probe_state = LogisticRegression(
            max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE,
            class_weight='balanced', multi_class='multinomial'
        )
        probe_state.fit(X_train_scaled, y_state_train)

        # Per-attribute analysis
        log.log("\n  Per-Attribute Accuracy:")
        results['by_attribute'][model] = {}

        for attr_name in df['attribute'].unique():
            mask = test_df['attribute'] == attr_name
            if mask.sum() < 10:
                continue

            X_attr = X_test_scaled[mask.values]
            y_attr = test_df.loc[mask, 'attribute_label'].values

            pred = probe_attr.predict(X_attr)
            acc = accuracy_score(y_attr, pred)

            # Determine suppression category
            category = 'high' if attr_name in HIGH_SUPPRESSION_ATTRIBUTES else (
                'low' if attr_name in LOW_SUPPRESSION_ATTRIBUTES else 'medium'
            )

            results['by_attribute'][model][attr_name] = {
                'accuracy': float(acc),
                'n_samples': int(mask.sum()),
                'suppression_category': category
            }

            if category in ['high', 'low']:
                log.result(f"    {attr_name} [{category}]", f"{acc:.4f} (n={mask.sum()})")

        # Per-state analysis
        log.log("\n  Per-State Accuracy (selected):")
        results['by_state'][model] = {}

        for state_name in df['state'].unique():
            mask = test_df['state'] == state_name
            if mask.sum() < 10:
                continue

            X_state = X_test_scaled[mask.values]
            y_state = test_df.loc[mask, 'state_label'].values

            pred = probe_state.predict(X_state)
            acc = accuracy_score(y_state, pred)

            category = 'high' if state_name in HIGH_SUPPRESSION_STATES else (
                'low' if state_name in LOW_SUPPRESSION_STATES else 'medium'
            )

            results['by_state'][model][state_name] = {
                'accuracy': float(acc),
                'n_samples': int(mask.sum()),
                'suppression_category': category
            }

            if category in ['high', 'low']:
                log.result(f"    {state_name} [{category}]", f"{acc:.4f} (n={mask.sum()})")

        # Aggregate high vs low suppression comparison
        high_attr_accs = [
            v['accuracy'] for k, v in results['by_attribute'][model].items()
            if v['suppression_category'] == 'high'
        ]
        low_attr_accs = [
            v['accuracy'] for k, v in results['by_attribute'][model].items()
            if v['suppression_category'] == 'low'
        ]

        high_state_accs = [
            v['accuracy'] for k, v in results['by_state'][model].items()
            if v['suppression_category'] == 'high'
        ]
        low_state_accs = [
            v['accuracy'] for k, v in results['by_state'][model].items()
            if v['suppression_category'] == 'low'
        ]

        results['high_vs_low_suppression'][model] = {
            'attribute': {
                'high_mean': float(np.mean(high_attr_accs)) if high_attr_accs else None,
                'low_mean': float(np.mean(low_attr_accs)) if low_attr_accs else None,
                'difference': float(np.mean(high_attr_accs) - np.mean(low_attr_accs)) if high_attr_accs and low_attr_accs else None
            },
            'state': {
                'high_mean': float(np.mean(high_state_accs)) if high_state_accs else None,
                'low_mean': float(np.mean(low_state_accs)) if low_state_accs else None,
                'difference': float(np.mean(high_state_accs) - np.mean(low_state_accs)) if high_state_accs and low_state_accs else None
            }
        }

        log.log("\n  High vs Low Suppression Summary:")
        if high_attr_accs and low_attr_accs:
            diff = np.mean(high_attr_accs) - np.mean(low_attr_accs)
            log.result("    Attribute", f"High: {np.mean(high_attr_accs):.4f}, Low: {np.mean(low_attr_accs):.4f}, Δ={diff:+.4f}")

            # Statistical test
            if len(high_attr_accs) >= 2 and len(low_attr_accs) >= 2:
                t_stat, p_val = stats.ttest_ind(high_attr_accs, low_attr_accs)
                d = cohens_d(np.array(high_attr_accs), np.array(low_attr_accs))
                log.result("    ", f"t={t_stat:.2f}, p={p_val:.4f}, Cohen's d={d:.2f}")

        if high_state_accs and low_state_accs:
            diff = np.mean(high_state_accs) - np.mean(low_state_accs)
            log.result("    State", f"High: {np.mean(high_state_accs):.4f}, Low: {np.mean(low_state_accs):.4f}, Δ={diff:+.4f}")

    # Save results
    with open(config.OUTPUT_DIR / "results" / "cultural_category_breakdown.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# LAYER-WISE INFORMATION FLOW ANALYSIS
# ============================================================================

def analyze_layer_information_flow(df, train_idx, test_idx, activations):
    """Analyze how information flows and transforms across layers.

    Tracks probe accuracy for each task across all layers to:
    - Compute layer-to-layer accuracy deltas
    - Identify where base-instruct divergence emerges
    - Locate the decision bottleneck layer
    """
    log.section("Layer-Wise Information Flow Analysis")

    results = {
        'layer_accuracies': {},
        'layer_deltas': {},
        'base_instruct_divergence': {},
        'information_peaks': {}
    }

    tasks = {
        'attribute': ('attribute_label', 16),
        'state': ('state_label', 36),
    }

    # Collect accuracies for all layers
    for task_name, (label_col, n_classes) in tasks.items():
        log.log(f"\n--- {task_name.upper()} Information Flow ---")

        results['layer_accuracies'][task_name] = {}
        results['layer_deltas'][task_name] = {}
        results['base_instruct_divergence'][task_name] = {}

        y = df[label_col].values
        y_train, y_test = y[train_idx], y[test_idx]

        for model in config.MODELS:
            results['layer_accuracies'][task_name][model] = {}
            prev_acc = None

            for layer in config.LAYERS:
                X = activations[model][layer]
                X_train, X_test = X[train_idx], X[test_idx]

                # Scale and train probe
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                probe = LogisticRegression(
                    max_iter=config.MAX_ITER,
                    random_state=config.RANDOM_STATE,
                    class_weight='balanced',
                    multi_class='multinomial' if n_classes > 2 else 'auto'
                )
                probe.fit(X_train_scaled, y_train)
                y_pred = probe.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)

                results['layer_accuracies'][task_name][model][layer] = float(acc)

                # Compute layer delta
                if prev_acc is not None:
                    delta = acc - prev_acc
                    if model not in results['layer_deltas'][task_name]:
                        results['layer_deltas'][task_name][model] = {}
                    results['layer_deltas'][task_name][model][layer] = float(delta)
                prev_acc = acc

        # Compute base-instruct divergence per layer
        log.log(f"\n  Base-Instruct Divergence by Layer:")
        for layer in config.LAYERS:
            base_acc = results['layer_accuracies'][task_name]['base'][layer]
            inst_acc = results['layer_accuracies'][task_name]['instruct'][layer]
            divergence = abs(base_acc - inst_acc)

            results['base_instruct_divergence'][task_name][layer] = {
                'base': float(base_acc),
                'instruct': float(inst_acc),
                'absolute_divergence': float(divergence),
                'relative_divergence': float(divergence / base_acc) if base_acc > 0 else 0
            }

            log.result(
                f"    Layer {layer}",
                f"Base: {base_acc:.4f}, Inst: {inst_acc:.4f}, |Δ|={divergence:.4f}"
            )

        # Find peak information layer
        base_peak = max(config.LAYERS, key=lambda l: results['layer_accuracies'][task_name]['base'][l])
        inst_peak = max(config.LAYERS, key=lambda l: results['layer_accuracies'][task_name]['instruct'][l])

        results['information_peaks'][task_name] = {
            'base_peak_layer': int(base_peak),
            'base_peak_accuracy': float(results['layer_accuracies'][task_name]['base'][base_peak]),
            'instruct_peak_layer': int(inst_peak),
            'instruct_peak_accuracy': float(results['layer_accuracies'][task_name]['instruct'][inst_peak])
        }

        log.result(f"  Peak layer (base)", f"Layer {base_peak} ({results['layer_accuracies'][task_name]['base'][base_peak]:.4f})")
        log.result(f"  Peak layer (instruct)", f"Layer {inst_peak} ({results['layer_accuracies'][task_name]['instruct'][inst_peak]:.4f})")

    # Summary: Layer 28 bottleneck analysis
    log.log("\n--- Layer 28 Bottleneck Analysis ---")

    # Compare layer 24 vs 28 divergence
    for task_name in tasks.keys():
        div_24 = results['base_instruct_divergence'][task_name].get(24, {}).get('absolute_divergence', 0)
        div_28 = results['base_instruct_divergence'][task_name].get(28, {}).get('absolute_divergence', 0)

        if div_24 > 0:
            amplification = div_28 / div_24
        else:
            amplification = float('inf') if div_28 > 0 else 1.0

        log.result(
            f"  {task_name}",
            f"L24 div: {div_24:.4f}, L28 div: {div_28:.4f}, Amplification: {amplification:.2f}x"
        )

        results['base_instruct_divergence'][task_name]['layer_28_amplification'] = float(amplification) if amplification != float('inf') else None

    # Save results
    with open(config.OUTPUT_DIR / "results" / "layer_information_flow.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(
    attr_results, corr_results, state_results,
    transfer_results, multitask_results, group_results,
    suppression_results=None, cultural_breakdown=None, layer_flow_results=None
):
    """Create comprehensive visualizations for all probing analyses."""
    log.section("Creating Visualizations")
    
    # 1. Accuracy comparison across probes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Attribute
    ax = axes[0, 0]
    for model in config.MODELS:
        accs = [attr_results[model][layer]['accuracy'] for layer in config.LAYERS]
        ax.plot(config.LAYERS, accs, marker='o', label=model, linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title('Attribute Probing (16-class)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(config.LAYERS)
    
    # Correctness (self-prediction for each model)
    ax = axes[0, 1]
    for model in config.MODELS:
        self_key = f"{model}_self"
        accs = [corr_results[self_key][layer]['accuracy'] for layer in config.LAYERS]
        ax.plot(config.LAYERS, accs, marker='o', label=f"{model} self", linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title('Correctness Probing (Binary)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(config.LAYERS)
    
    # State
    ax = axes[1, 0]
    for model in config.MODELS:
        accs = [state_results[model][layer]['accuracy'] for layer in config.LAYERS]
        ax.plot(config.LAYERS, accs, marker='o', label=model, linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title('State Probing (36-class)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(config.LAYERS)
    
    # Multi-task comparison
    ax = axes[1, 1]
    for model in config.MODELS:
        ind_accs = [multitask_results[model][layer]['independent']['average'] for layer in config.LAYERS]
        joint_accs = [multitask_results[model][layer]['joint']['average'] for layer in config.LAYERS]
        
        x = np.arange(len(config.LAYERS))
        width = 0.35
        
        if model == 'base':
            offset = -width/2
        else:
            offset = width/2
        
        ax.bar(x + offset, ind_accs, width/2, label=f'{model} (ind)', alpha=0.7)
        ax.bar(x + offset + width/2, joint_accs, width/2, label=f'{model} (joint)', alpha=0.7)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Multi-Task: Independent vs Joint', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config.LAYERS)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "probe_accuracies.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved probe accuracies plot")
    
    # 2. Cross-model transfer
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    tasks = ['attribute', 'base_correctness', 'state']
    titles = ['Attribute Transfer', 'Correctness Transfer', 'State Transfer']
    
    for ax, task, title in zip(axes, tasks, titles):
        base_to_base = [transfer_results[layer][task]['base_to_base'] for layer in config.LAYERS]
        base_to_instruct = [transfer_results[layer][task]['base_to_instruct'] for layer in config.LAYERS]
        
        x = np.arange(len(config.LAYERS))
        width = 0.35
        
        ax.bar(x - width/2, base_to_base, width, label='Base → Base', alpha=0.8)
        ax.bar(x + width/2, base_to_instruct, width, label='Base → Instruct', alpha=0.8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config.LAYERS)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "cross_model_transfer.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved cross-model transfer plot")
    
    # 3. Group-wise performance
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    groups = ['suppression', 'enhancement', 'control']
    
    for col, group in enumerate(groups):
        # Attribute by group
        ax = axes[0, col]
        for model in config.MODELS:
            accs = [group_results[model][layer][group]['attribute_accuracy'] 
                   for layer in config.LAYERS if group in group_results[model][layer]]
            ax.plot(config.LAYERS[:len(accs)], accs, marker='o', label=model, linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Attribute Accuracy')
        ax.set_title(f'{group.capitalize()} - Attribute', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Correctness by group
        ax = axes[1, col]
        for model in config.MODELS:
            accs = [group_results[model][layer][group]['correctness_accuracy'] 
                   for layer in config.LAYERS if group in group_results[model][layer]]
            ax.plot(config.LAYERS[:len(accs)], accs, marker='o', label=model, linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Correctness Accuracy')
        ax.set_title(f'{group.capitalize()} - Correctness', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "group_wise_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved group-wise performance plot")
    
    # 4. Transfer rate heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    transfer_matrix = np.zeros((3, len(config.LAYERS)))
    for i, task in enumerate(['attribute', 'base_correctness', 'state']):
        for j, layer in enumerate(config.LAYERS):
            transfer_matrix[i, j] = transfer_results[layer][task]['transfer_rate']
    
    im = ax.imshow(transfer_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0, aspect='auto')
    
    ax.set_xticks(np.arange(len(config.LAYERS)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(config.LAYERS)
    ax.set_yticklabels(['Attribute', 'Correctness', 'State'])
    
    # Annotate cells
    for i in range(3):
        for j in range(len(config.LAYERS)):
            text = ax.text(j, i, f'{transfer_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Cross-Model Transfer Rate (Base → Instruct)', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Transfer Rate')
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "transfer_rate_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved transfer rate heatmap")

    # 5. Suppression-Predictive Probing
    if suppression_results is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Suppression prediction accuracy (base vs instruct)
        ax = axes[0]
        base_accs = [suppression_results['suppression_from_base'][layer]['accuracy']
                     for layer in config.LAYERS]
        inst_accs = [suppression_results['suppression_from_instruct'][layer]['accuracy']
                     for layer in config.LAYERS]

        ax.plot(config.LAYERS, base_accs, marker='o', label='Base → Suppression', linewidth=2, color='#1f77b4')
        ax.plot(config.LAYERS, inst_accs, marker='s', label='Instruct → Suppression', linewidth=2, color='#ff7f0e')

        # Add chance level
        chance = suppression_results['suppression_from_base'][config.LAYERS[0]]['baselines']['chance_level']
        ax.axhline(y=chance, color='gray', linestyle='--', label=f'Chance ({chance:.3f})', alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title('Suppression Prediction by Model', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(config.LAYERS)

        # Panel 2: ROC-AUC comparison
        ax = axes[1]
        base_aucs = [suppression_results['suppression_from_base'][layer]['metrics'].get('roc_auc', 0.5)
                     for layer in config.LAYERS]
        inst_aucs = [suppression_results['suppression_from_instruct'][layer]['metrics'].get('roc_auc', 0.5)
                     for layer in config.LAYERS]

        x = np.arange(len(config.LAYERS))
        width = 0.35

        ax.bar(x - width/2, base_aucs, width, label='Base', alpha=0.8, color='#1f77b4')
        ax.bar(x + width/2, inst_aucs, width, label='Instruct', alpha=0.8, color='#ff7f0e')
        ax.axhline(y=0.5, color='gray', linestyle='--', label='Random (0.5)', alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('Suppression Prediction ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config.LAYERS)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Panel 3: Suppression vs Enhancement (control comparison)
        ax = axes[2]
        supp_accs = [suppression_results['suppression_from_base'][layer]['accuracy']
                     for layer in config.LAYERS]
        enh_accs = [suppression_results['enhancement_from_base'][layer]['accuracy']
                    for layer in config.LAYERS]

        ax.plot(config.LAYERS, supp_accs, marker='o', label='Suppression', linewidth=2, color='#d62728')
        ax.plot(config.LAYERS, enh_accs, marker='s', label='Enhancement', linewidth=2, color='#2ca02c')
        ax.axhline(y=chance, color='gray', linestyle='--', label=f'Chance', alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title('Suppression vs Enhancement (from Base)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(config.LAYERS)

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "plots" / "suppression_predictive.png", dpi=300, bbox_inches='tight')
        plt.close()
        log.log("✓ Saved suppression-predictive plot")

    # 6. Cultural Category Breakdown
    if cultural_breakdown is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Attribute accuracy by suppression category (Base)
        ax = axes[0, 0]
        if 'base' in cultural_breakdown['by_attribute']:
            high_attrs = {k: v for k, v in cultural_breakdown['by_attribute']['base'].items()
                         if v['suppression_category'] == 'high'}
            low_attrs = {k: v for k, v in cultural_breakdown['by_attribute']['base'].items()
                        if v['suppression_category'] == 'low'}

            all_attrs = list(high_attrs.keys()) + list(low_attrs.keys())
            all_accs = [high_attrs[k]['accuracy'] for k in high_attrs] + [low_attrs[k]['accuracy'] for k in low_attrs]
            colors = ['#d62728'] * len(high_attrs) + ['#2ca02c'] * len(low_attrs)

            bars = ax.barh(range(len(all_attrs)), all_accs, color=colors, alpha=0.8)
            ax.set_yticks(range(len(all_attrs)))
            ax.set_yticklabels([a.replace('_', ' ') for a in all_attrs], fontsize=9)
            ax.set_xlabel('Accuracy')
            ax.set_title('Base: Attribute Accuracy by Category', fontsize=12, fontweight='bold')
            ax.axvline(x=1/16, color='gray', linestyle='--', alpha=0.7, label='Chance')

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#d62728', alpha=0.8, label='High Suppression'),
                              Patch(facecolor='#2ca02c', alpha=0.8, label='Low Suppression')]
            ax.legend(handles=legend_elements, loc='lower right')
            ax.grid(alpha=0.3, axis='x')

        # Panel 2: Attribute accuracy by suppression category (Instruct)
        ax = axes[0, 1]
        if 'instruct' in cultural_breakdown['by_attribute']:
            high_attrs = {k: v for k, v in cultural_breakdown['by_attribute']['instruct'].items()
                         if v['suppression_category'] == 'high'}
            low_attrs = {k: v for k, v in cultural_breakdown['by_attribute']['instruct'].items()
                        if v['suppression_category'] == 'low'}

            all_attrs = list(high_attrs.keys()) + list(low_attrs.keys())
            all_accs = [high_attrs[k]['accuracy'] for k in high_attrs] + [low_attrs[k]['accuracy'] for k in low_attrs]
            colors = ['#d62728'] * len(high_attrs) + ['#2ca02c'] * len(low_attrs)

            ax.barh(range(len(all_attrs)), all_accs, color=colors, alpha=0.8)
            ax.set_yticks(range(len(all_attrs)))
            ax.set_yticklabels([a.replace('_', ' ') for a in all_attrs], fontsize=9)
            ax.set_xlabel('Accuracy')
            ax.set_title('Instruct: Attribute Accuracy by Category', fontsize=12, fontweight='bold')
            ax.axvline(x=1/16, color='gray', linestyle='--', alpha=0.7)
            ax.legend(handles=legend_elements, loc='lower right')
            ax.grid(alpha=0.3, axis='x')

        # Panel 3: High vs Low suppression comparison
        ax = axes[1, 0]
        categories = ['Attribute\n(High Supp)', 'Attribute\n(Low Supp)', 'State\n(High Supp)', 'State\n(Low Supp)']

        base_vals = []
        inst_vals = []
        for model in ['base', 'instruct']:
            hvl = cultural_breakdown['high_vs_low_suppression'].get(model, {})
            base_vals if model == 'base' else inst_vals
            vals = base_vals if model == 'base' else inst_vals
            vals.append(hvl.get('attribute', {}).get('high_mean', 0) or 0)
            vals.append(hvl.get('attribute', {}).get('low_mean', 0) or 0)
            vals.append(hvl.get('state', {}).get('high_mean', 0) or 0)
            vals.append(hvl.get('state', {}).get('low_mean', 0) or 0)

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, base_vals, width, label='Base', alpha=0.8, color='#1f77b4')
        ax.bar(x + width/2, inst_vals, width, label='Instruct', alpha=0.8, color='#ff7f0e')

        ax.set_ylabel('Mean Accuracy')
        ax.set_title('High vs Low Suppression Categories', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Panel 4: State accuracy heatmap (selected states)
        ax = axes[1, 1]
        if 'base' in cultural_breakdown['by_state']:
            # Get states sorted by suppression category
            states_data = cultural_breakdown['by_state']['base']
            high_states = {k: v for k, v in states_data.items() if v['suppression_category'] == 'high'}
            low_states = {k: v for k, v in states_data.items() if v['suppression_category'] == 'low'}

            selected_states = list(high_states.keys())[:5] + list(low_states.keys())[:4]

            if selected_states:
                base_accs = [cultural_breakdown['by_state']['base'].get(s, {}).get('accuracy', 0)
                            for s in selected_states]
                inst_accs = [cultural_breakdown['by_state']['instruct'].get(s, {}).get('accuracy', 0)
                            for s in selected_states]

                x = np.arange(len(selected_states))
                width = 0.35

                ax.bar(x - width/2, base_accs, width, label='Base', alpha=0.8, color='#1f77b4')
                ax.bar(x + width/2, inst_accs, width, label='Instruct', alpha=0.8, color='#ff7f0e')

                ax.set_ylabel('Accuracy')
                ax.set_title('State Accuracy (High vs Low Suppression)', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([s.replace('_', '\n') for s in selected_states], fontsize=8, rotation=45, ha='right')
                ax.legend()
                ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "plots" / "cultural_category_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        log.log("✓ Saved cultural category breakdown plot")

    # 7. Layer-wise Information Flow
    if layer_flow_results is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Attribute information flow
        ax = axes[0, 0]
        if 'attribute' in layer_flow_results['layer_accuracies']:
            for model in config.MODELS:
                accs = [layer_flow_results['layer_accuracies']['attribute'][model][layer]
                       for layer in config.LAYERS]
                ax.plot(config.LAYERS, accs, marker='o', label=model.capitalize(), linewidth=2)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Probe Accuracy')
            ax.set_title('Attribute Information Flow', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xticks(config.LAYERS)

        # Panel 2: State information flow
        ax = axes[0, 1]
        if 'state' in layer_flow_results['layer_accuracies']:
            for model in config.MODELS:
                accs = [layer_flow_results['layer_accuracies']['state'][model][layer]
                       for layer in config.LAYERS]
                ax.plot(config.LAYERS, accs, marker='o', label=model.capitalize(), linewidth=2)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Probe Accuracy')
            ax.set_title('State Information Flow', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xticks(config.LAYERS)

        # Panel 3: Base-Instruct Divergence
        ax = axes[1, 0]
        for task in ['attribute', 'state']:
            divergences = [layer_flow_results['base_instruct_divergence'][task][layer]['absolute_divergence']
                          for layer in config.LAYERS]
            ax.plot(config.LAYERS, divergences, marker='o', label=task.capitalize(), linewidth=2)

        ax.set_xlabel('Layer')
        ax.set_ylabel('|Base - Instruct| Accuracy')
        ax.set_title('Base-Instruct Divergence by Layer', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(config.LAYERS)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Panel 4: Information peaks and bottleneck
        ax = axes[1, 1]

        # Create bar chart showing peak layers
        tasks = list(layer_flow_results['information_peaks'].keys())
        x = np.arange(len(tasks))
        width = 0.35

        base_peaks = [layer_flow_results['information_peaks'][t]['base_peak_accuracy'] for t in tasks]
        inst_peaks = [layer_flow_results['information_peaks'][t]['instruct_peak_accuracy'] for t in tasks]

        bars1 = ax.bar(x - width/2, base_peaks, width, label='Base Peak', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x + width/2, inst_peaks, width, label='Instruct Peak', alpha=0.8, color='#ff7f0e')

        # Add layer annotations
        for i, t in enumerate(tasks):
            base_layer = layer_flow_results['information_peaks'][t]['base_peak_layer']
            inst_layer = layer_flow_results['information_peaks'][t]['instruct_peak_layer']
            ax.annotate(f'L{base_layer}', xy=(x[i] - width/2, base_peaks[i]),
                       ha='center', va='bottom', fontsize=9)
            ax.annotate(f'L{inst_layer}', xy=(x[i] + width/2, inst_peaks[i]),
                       ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Peak Accuracy')
        ax.set_title('Information Peaks by Task', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tasks])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "plots" / "layer_information_flow.png", dpi=300, bbox_inches='tight')
        plt.close()
        log.log("✓ Saved layer information flow plot")

    # 8. Baseline Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Attribute baselines
    ax = axes[0]
    if 'baselines' in attr_results['base'][config.LAYERS[0]]:
        baselines = attr_results['base'][config.LAYERS[0]]['baselines']

        # Get probe accuracies
        base_accs = [attr_results['base'][layer]['accuracy'] for layer in config.LAYERS]
        inst_accs = [attr_results['instruct'][layer]['accuracy'] for layer in config.LAYERS]

        ax.plot(config.LAYERS, base_accs, marker='o', label='Base Probe', linewidth=2, color='#1f77b4')
        ax.plot(config.LAYERS, inst_accs, marker='s', label='Instruct Probe', linewidth=2, color='#ff7f0e')

        # Baseline lines
        ax.axhline(y=baselines['chance_level'], color='red', linestyle='--',
                  label=f"Chance ({baselines['chance_level']:.3f})", alpha=0.7)
        ax.axhline(y=baselines['majority_class_accuracy'], color='green', linestyle='--',
                  label=f"Majority ({baselines['majority_class_accuracy']:.3f})", alpha=0.7)
        ax.axhline(y=baselines['random_features_mean'], color='purple', linestyle='--',
                  label=f"Random ({baselines['random_features_mean']:.3f})", alpha=0.7)

        ax.fill_between(config.LAYERS,
                       [baselines['random_features_mean'] - baselines['random_features_std']] * len(config.LAYERS),
                       [baselines['random_features_mean'] + baselines['random_features_std']] * len(config.LAYERS),
                       color='purple', alpha=0.1)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title('Attribute Probing vs Baselines', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(config.LAYERS)

    # State baselines
    ax = axes[1]
    if 'baselines' in state_results['base'][config.LAYERS[0]]:
        baselines = state_results['base'][config.LAYERS[0]]['baselines']

        base_accs = [state_results['base'][layer]['accuracy'] for layer in config.LAYERS]
        inst_accs = [state_results['instruct'][layer]['accuracy'] for layer in config.LAYERS]

        ax.plot(config.LAYERS, base_accs, marker='o', label='Base Probe', linewidth=2, color='#1f77b4')
        ax.plot(config.LAYERS, inst_accs, marker='s', label='Instruct Probe', linewidth=2, color='#ff7f0e')

        ax.axhline(y=baselines['chance_level'], color='red', linestyle='--',
                  label=f"Chance ({baselines['chance_level']:.3f})", alpha=0.7)
        ax.axhline(y=baselines['majority_class_accuracy'], color='green', linestyle='--',
                  label=f"Majority ({baselines['majority_class_accuracy']:.3f})", alpha=0.7)
        ax.axhline(y=baselines['random_features_mean'], color='purple', linestyle='--',
                  label=f"Random ({baselines['random_features_mean']:.3f})", alpha=0.7)

        ax.fill_between(config.LAYERS,
                       [baselines['random_features_mean'] - baselines['random_features_std']] * len(config.LAYERS),
                       [baselines['random_features_mean'] + baselines['random_features_std']] * len(config.LAYERS),
                       color='purple', alpha=0.1)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title('State Probing vs Baselines', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(config.LAYERS)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved baseline comparison plot")

    # 9. Cross-Model Correctness Probing (new visualization)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Cross-model prediction accuracy
    ax = axes[0]
    if 'base_to_instruct' in corr_results and 'instruct_to_base' in corr_results:
        b2i_accs = [corr_results['base_to_instruct'][layer]['accuracy'] for layer in config.LAYERS]
        i2b_accs = [corr_results['instruct_to_base'][layer]['accuracy'] for layer in config.LAYERS]

        ax.plot(config.LAYERS, b2i_accs, marker='o', label='Base → Instruct Correct', linewidth=2, color='#1f77b4')
        ax.plot(config.LAYERS, i2b_accs, marker='s', label='Instruct → Base Correct', linewidth=2, color='#ff7f0e')
        ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance', alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cross-Model Correctness Prediction', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(config.LAYERS)

    # Panel 2: Self vs Cross comparison
    ax = axes[1]
    if 'base_self' in corr_results and 'instruct_self' in corr_results:
        base_self = [corr_results['base_self'][layer]['accuracy'] for layer in config.LAYERS]
        inst_self = [corr_results['instruct_self'][layer]['accuracy'] for layer in config.LAYERS]
        b2i = [corr_results['base_to_instruct'][layer]['accuracy'] for layer in config.LAYERS]
        i2b = [corr_results['instruct_to_base'][layer]['accuracy'] for layer in config.LAYERS]

        x = np.arange(len(config.LAYERS))
        width = 0.2

        ax.bar(x - 1.5*width, base_self, width, label='Base Self', alpha=0.8, color='#1f77b4')
        ax.bar(x - 0.5*width, b2i, width, label='Base→Inst', alpha=0.8, color='#aec7e8')
        ax.bar(x + 0.5*width, inst_self, width, label='Inst Self', alpha=0.8, color='#ff7f0e')
        ax.bar(x + 1.5*width, i2b, width, label='Inst→Base', alpha=0.8, color='#ffbb78')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title('Self vs Cross-Model Correctness', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config.LAYERS)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "cross_model_correctness.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved cross-model correctness plot")

    # 10. Statistical Summary Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Pre-compute effect sizes, p-values, and significance markers per layer
    attr_effects, state_effects = [], []
    attr_sig, state_sig = [], []
    attr_pvals, state_pvals = [], []

    for layer in config.LAYERS:
        # Attribute
        base_attr_scores = np.array(attr_results['base'][layer]['cv']['scores'])
        inst_attr_scores = np.array(attr_results['instruct'][layer]['cv']['scores'])
        attr_effects.append(cohens_d(base_attr_scores, inst_attr_scores))
        attr_comp = paired_comparison(
            base_attr_scores, inst_attr_scores,
            names=('base', 'instruct'), n_comparisons=len(config.LAYERS)
        )
        attr_sig.append(attr_comp['is_significant'])
        attr_pvals.append(attr_comp['p_value'])

        # State
        base_state_scores = np.array(state_results['base'][layer]['cv']['scores'])
        inst_state_scores = np.array(state_results['instruct'][layer]['cv']['scores'])
        state_effects.append(cohens_d(base_state_scores, inst_state_scores))
        state_comp = paired_comparison(
            base_state_scores, inst_state_scores,
            names=('base', 'instruct'), n_comparisons=len(config.LAYERS)
        )
        state_sig.append(state_comp['is_significant'])
        state_pvals.append(state_comp['p_value'])

    x = np.arange(len(config.LAYERS))
    width = 0.35

    # Panel 1: Effect sizes (Cohen's d) for base vs instruct
    ax = axes[0, 0]
    bars_attr = ax.bar(x - width/2, attr_effects, width, label='Attribute', alpha=0.8, color='#1f77b4')
    bars_state = ax.bar(x + width/2, state_effects, width, label='State', alpha=0.8, color='#ff7f0e')

    # Effect size thresholds
    for thresh, color, label in [(0.2, 'green', 'Small (0.2)'), (0.5, 'orange', 'Medium (0.5)'), (0.8, 'red', 'Large (0.8)')]:
        ax.axhline(y=thresh, color=color, linestyle=':', alpha=0.5, label=label)
        ax.axhline(y=-thresh, color=color, linestyle=':', alpha=0.5)

    # Significance markers on bars
    for i, bar in enumerate(bars_attr):
        if attr_sig[i]:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.05 if y >= 0 else -0.05), '*',
                    ha='center', va='bottom' if y >= 0 else 'top', fontsize=10, color='black')
    for i, bar in enumerate(bars_state):
        if state_sig[i]:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.05 if y >= 0 else -0.05), '*',
                    ha='center', va='bottom' if y >= 0 else 'top', fontsize=10, color='black')

    ax.set_xlabel('Layer')
    ax.set_ylabel("Cohen's d (Base - Instruct)")
    ax.set_title('Effect Size: Base vs Instruct', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config.LAYERS)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Attribute accuracy with bootstrap CIs
    ax = axes[0, 1]

    base_attr = [attr_results['base'][layer]['accuracy'] for layer in config.LAYERS]
    base_attr_low = [attr_results['base'][layer]['bootstrap_ci']['ci_lower'] for layer in config.LAYERS]
    base_attr_up = [attr_results['base'][layer]['bootstrap_ci']['ci_upper'] for layer in config.LAYERS]

    inst_attr = [attr_results['instruct'][layer]['accuracy'] for layer in config.LAYERS]
    inst_attr_low = [attr_results['instruct'][layer]['bootstrap_ci']['ci_lower'] for layer in config.LAYERS]
    inst_attr_up = [attr_results['instruct'][layer]['bootstrap_ci']['ci_upper'] for layer in config.LAYERS]

    ax.plot(config.LAYERS, base_attr, marker='o', label='Base', linewidth=2, color='#1f77b4')
    ax.fill_between(config.LAYERS, base_attr_low, base_attr_up, color='#1f77b4', alpha=0.15)

    ax.plot(config.LAYERS, inst_attr, marker='s', label='Instruct', linewidth=2, color='#ff7f0e')
    ax.fill_between(config.LAYERS, inst_attr_low, inst_attr_up, color='#ff7f0e', alpha=0.15)

    # Significance markers (Bonferroni-corrected)
    y_max = np.maximum(inst_attr_up, base_attr_up)
    for idx, layer in enumerate(config.LAYERS):
        if attr_sig[idx]:
            ax.text(layer, y_max[idx] + 0.01, '*', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Test Accuracy (± bootstrap CI)')
    ax.set_title('Attribute Probing with Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(config.LAYERS)

    # Panel 3: State accuracy with bootstrap CIs
    ax = axes[1, 0]

    base_state = [state_results['base'][layer]['accuracy'] for layer in config.LAYERS]
    base_state_low = [state_results['base'][layer]['bootstrap_ci']['ci_lower'] for layer in config.LAYERS]
    base_state_up = [state_results['base'][layer]['bootstrap_ci']['ci_upper'] for layer in config.LAYERS]

    inst_state = [state_results['instruct'][layer]['accuracy'] for layer in config.LAYERS]
    inst_state_low = [state_results['instruct'][layer]['bootstrap_ci']['ci_lower'] for layer in config.LAYERS]
    inst_state_up = [state_results['instruct'][layer]['bootstrap_ci']['ci_upper'] for layer in config.LAYERS]

    ax.plot(config.LAYERS, base_state, marker='o', label='Base', linewidth=2, color='#1f77b4')
    ax.fill_between(config.LAYERS, base_state_low, base_state_up, color='#1f77b4', alpha=0.15)

    ax.plot(config.LAYERS, inst_state, marker='s', label='Instruct', linewidth=2, color='#ff7f0e')
    ax.fill_between(config.LAYERS, inst_state_low, inst_state_up, color='#ff7f0e', alpha=0.15)

    y_max_state = np.maximum(inst_state_up, base_state_up)
    for idx, layer in enumerate(config.LAYERS):
        if state_sig[idx]:
            ax.text(layer, y_max_state[idx] + 0.01, '*', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Test Accuracy (± bootstrap CI)')
    ax.set_title('State Probing with Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(config.LAYERS)

    # Panel 4: p-values (-log10) to visualize significance emergence
    ax = axes[1, 1]
    attr_logp = [-np.log10(p) if p > 0 else 0 for p in attr_pvals]
    state_logp = [-np.log10(p) if p > 0 else 0 for p in state_pvals]
    threshold = -np.log10(0.05 / len(config.LAYERS))

    ax.plot(config.LAYERS, attr_logp, marker='o', label='Attribute', linewidth=2, color='#1f77b4')
    ax.plot(config.LAYERS, state_logp, marker='s', label='State', linewidth=2, color='#ff7f0e')
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7, label='Bonferroni α')

    ax.set_xlabel('Layer')
    ax.set_ylabel('-log10(p)')
    ax.set_title('Significance Across Layers', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(config.LAYERS)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "statistical_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved statistical summary plot")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(
    attr_results, corr_results, state_results,
    transfer_results, multitask_results, group_results
):
    """Generate comprehensive summary report."""
    log.section("Generating Summary Report")
    
    lines = []
    
    lines.append("="*80)
    lines.append("LINEAR PROBING ANALYSIS - SUMMARY REPORT")
    lines.append("="*80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall results
    lines.append("\n" + "-"*80)
    lines.append("1. ATTRIBUTE PROBING (16-CLASS)")
    lines.append("-"*80)
    
    for model in config.MODELS:
        lines.append(f"\n{model.upper()}:")
        for layer in config.LAYERS:
            acc = attr_results[model][layer]['accuracy']
            cv_mean = attr_results[model][layer]['cv']['mean']
            cv_std = attr_results[model][layer]['cv']['std']
            lines.append(f"  Layer {layer}: {acc:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")
    
    lines.append("\n" + "-"*80)
    lines.append("2. CORRECTNESS PROBING (BINARY)")
    lines.append("-"*80)
    
    for model in config.MODELS:
        lines.append(f"\n{model.upper()} (self correctness):")
        self_key = f"{model}_self"
        for layer in config.LAYERS:
            acc = corr_results[self_key][layer]['accuracy']
            cv_mean = corr_results[self_key][layer]['cv']['mean']
            cv_std = corr_results[self_key][layer]['cv']['std']
            lines.append(f"  Layer {layer}: {acc:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")

    # Cross-model correctness (base→inst, inst→base)
    lines.append("\nCross-Model Correctness:")
    for pair_key, label in [("base_to_instruct", "Base → Instruct"), ("instruct_to_base", "Instruct → Base")]:
        lines.append(f"\n{label}:")
        for layer in config.LAYERS:
            acc = corr_results[pair_key][layer]['accuracy']
            cv_mean = corr_results[pair_key][layer]['cv']['mean']
            cv_std = corr_results[pair_key][layer]['cv']['std']
            lines.append(f"  Layer {layer}: {acc:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")
    
    lines.append("\n" + "-"*80)
    lines.append("3. STATE PROBING (36-CLASS)")
    lines.append("-"*80)
    
    for model in config.MODELS:
        lines.append(f"\n{model.upper()}:")
        for layer in config.LAYERS:
            acc = state_results[model][layer]['accuracy']
            cv_mean = state_results[model][layer]['cv']['mean']
            cv_std = state_results[model][layer]['cv']['std']
            lines.append(f"  Layer {layer}: {acc:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")
    
    lines.append("\n" + "-"*80)
    lines.append("4. CROSS-MODEL TRANSFER (CRITICAL)")
    lines.append("-"*80)
    
    for layer in config.LAYERS:
        lines.append(f"\nLayer {layer}:")
        
        for task in ['attribute', 'base_correctness', 'state']:
            task_name = task.replace('_', ' ').title()
            base_to_base = transfer_results[layer][task]['base_to_base']
            base_to_inst = transfer_results[layer][task]['base_to_instruct']
            transfer_rate = transfer_results[layer][task]['transfer_rate']
            drop = transfer_results[layer][task]['absolute_drop']
            
            lines.append(f"  {task_name}:")
            lines.append(f"    Base → Base:     {base_to_base:.4f}")
            lines.append(f"    Base → Instruct: {base_to_inst:.4f}")
            lines.append(f"    Transfer Rate:   {transfer_rate:.4f} ({(transfer_rate-1)*100:+.2f}%)")
            lines.append(f"    Absolute Drop:   {drop:+.4f}")
    
    lines.append("\n" + "-"*80)
    lines.append("5. MULTI-TASK PROBING")
    lines.append("-"*80)
    
    for model in config.MODELS:
        lines.append(f"\n{model.upper()}:")
        for layer in config.LAYERS:
            ind_avg = multitask_results[model][layer]['independent']['average']
            joint_avg = multitask_results[model][layer]['joint']['average']
            
            lines.append(f"  Layer {layer}:")
            lines.append(f"    Independent: {ind_avg:.4f}")
            lines.append(f"    Joint:       {joint_avg:.4f}")
            lines.append(f"    Difference:  {joint_avg - ind_avg:+.4f}")
    
    lines.append("\n" + "-"*80)
    lines.append("6. KEY FINDINGS")
    lines.append("-"*80)
    
    # Best attribute layer
    best_attr_layer = max(config.LAYERS, 
                         key=lambda l: attr_results['base'][l]['accuracy'])
    lines.append(f"\n• Best Attribute Layer: {best_attr_layer}")
    
    # Average transfer rate
    avg_transfer = np.mean([
        transfer_results[layer][task]['transfer_rate']
        for layer in config.LAYERS
        for task in ['attribute', 'base_correctness', 'state']
    ])
    lines.append(f"• Average Transfer Rate: {avg_transfer:.4f} ({(avg_transfer-1)*100:+.2f}%)")
    
    # Interpretation
    if avg_transfer > 0.95:
        lines.append("\n⚠️  HIGH TRANSFER RATE (>95%)")
        lines.append("Representations remain highly aligned despite behavioral divergence.")
        lines.append("Suppression likely operates via downstream decision boundaries,")
        lines.append("NOT through representational rewriting.")
    elif avg_transfer < 0.85:
        lines.append("\n⚠️  LOW TRANSFER RATE (<85%)")
        lines.append("Representations have significantly diverged.")
        lines.append("Instruction-tuning fundamentally rewrote internal representations.")
    else:
        lines.append("\n⚠️  MODERATE TRANSFER RATE (85-95%)")
        lines.append("Partial representation divergence detected.")
        lines.append("Suppression involves both representational and decision-level changes.")
    
    lines.append("\n" + "="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)
    
    report_text = '\n'.join(lines)
    
    # Save report
    with open(config.OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    log.log("✓ Summary report saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    start_time = datetime.now()

    log.section("Linear Probing Analysis Pipeline")
    log.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load data
        df, activations, label_encoders = load_data()

        # Create splits
        train_idx, test_idx, train_df, test_df = create_splits(df)

        # Probe 1: Attribute
        attr_results = probe_attribute(df, train_idx, test_idx, activations)

        # Probe 2: Cross-Model Correctness
        corr_results = probe_correctness(df, train_idx, test_idx, activations)

        # Probe 3: State
        state_results = probe_state(df, train_idx, test_idx, activations)

        # Probe 4: Cross-model transfer
        transfer_results = probe_cross_model_transfer(df, train_idx, test_idx, activations)

        # Probe 5: Multi-task
        multitask_results = probe_multitask(df, train_idx, test_idx, activations)

        # Probe 6: Suppression-Predictive
        suppression_results = probe_suppression_prediction(df, train_idx, test_idx, activations)

        # Group-wise analysis
        group_results = analyze_by_group(df, train_idx, test_idx, activations)

        # Per-attribute/state breakdown
        cultural_breakdown = analyze_by_cultural_category(
            df, train_idx, test_idx, activations, label_encoders
        )

        # Layer-wise information flow
        layer_flow_results = analyze_layer_information_flow(
            df, train_idx, test_idx, activations
        )

        # Visualizations
        create_visualizations(
            attr_results, corr_results, state_results,
            transfer_results, multitask_results, group_results,
            suppression_results, cultural_breakdown, layer_flow_results
        )

        # Summary report
        generate_summary_report(
            attr_results, corr_results, state_results,
            transfer_results, multitask_results, group_results
        )

        # Save label encoders
        with open(config.OUTPUT_DIR / "label_encoders.pkl", 'wb') as f:
            pickle.dump(label_encoders, f)

        log.log("✓ Label encoders saved")

    except Exception as e:
        log.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        log.log(traceback.format_exc())
        raise

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    log.section("Pipeline Complete")
    log.log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"Total duration: {duration/60:.1f} minutes")
    log.log(f"\nAll outputs saved to:")
    log.log(f"  {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()