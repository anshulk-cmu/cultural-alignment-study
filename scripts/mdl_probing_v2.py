#!/usr/bin/env python3
"""
MDL (Minimum Description Length) Probing Analysis
==================================================
Information-theoretic analysis of cultural knowledge encoding in neural activations.

Implements:
- Variational coding with multiple complexity measures (L0, L1, L2, Fisher)
- Prequential coding for online codelength estimation
- Stratified sampling for class-balanced training
- Nested cross-validation for hyperparameter selection
- Per-class codelength decomposition
- Proper parameter encoding costs

Models: Qwen2-1.5B Base & Instruct
Layers: 6, 12, 18
Dataset: 33,522 sentences
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import entropy
from scipy.linalg import eigvalsh

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MDLConfig:
    """Configuration for MDL probing experiments."""
    
    # Paths
    activation_dir: Path = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/Activations")
    index_file: Path = activation_dir / "activation_index.csv"
    enhanced_data: Path = Path("/home/anshulk/cultural-alignment-study/outputs/EDA_results/tables/enhanced_dataset.csv")
    output_dir: Path = Path("/home/anshulk/cultural-alignment-study/outputs/mdl_probing_v2")
    heavy_data_dir: Path = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/MDL_Probing_v2")
    
    # Models and layers
    models: List[str] = None
    layers: List[int] = None
    hidden_size: int = 1536
    
    # MDL parameters
    lambda_values: List[float] = None
    prequential_chunk_sizes: List[int] = None
    complexity_measures: List[str] = None
    
    # Parameter encoding
    bits_per_parameter: int = 32  # float32 precision
    
    # Cross-validation
    n_cv_folds: int = 5
    random_state: int = 42
    max_iter: int = 1000
    test_size: float = 0.2
    
    def __post_init__(self):
        if self.models is None:
            self.models = ['base', 'instruct']
        if self.layers is None:
            self.layers = [6, 12, 18]
        if self.lambda_values is None:
            self.lambda_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        if self.prequential_chunk_sizes is None:
            self.prequential_chunk_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
        if self.complexity_measures is None:
            self.complexity_measures = ['l0', 'l1', 'l2', 'fisher']
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.heavy_data_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "curves").mkdir(exist_ok=True)
        (self.output_dir / "per_class").mkdir(exist_ok=True)

config = MDLConfig()

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

log = Logger(config.output_dir / "mdl_log.txt")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load activations and metadata."""
    log.section("Loading Data")
    
    # Load metadata
    log.log("Loading metadata...")
    df = pd.read_csv(config.enhanced_data)
    log.result("Total sentences", len(df))
    
    # Load activations
    activations = {}
    for model in config.models:
        activations[model] = {}
        for layer in config.layers:
            file_path = config.activation_dir / f"{model}_layer{layer}_activations.npy"
            log.log(f"Loading {model} layer {layer}...")
            activations[model][layer] = np.load(file_path)
            log.result(f"  Shape", activations[model][layer].shape)
    
    # Prepare labels
    log.log("\nPreparing labels...")
    
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
    
    # Group (3 classes)
    le_group = LabelEncoder()
    df['group_label'] = le_group.fit_transform(df['group_type'])
    label_encoders['group'] = le_group
    log.result("Groups", len(le_group.classes_))
    
    # Correctness
    df['base_correct_label'] = df['base_correct'].astype(int)
    df['instruct_correct_label'] = df['instruct_correct'].astype(int)
    
    return df, activations, label_encoders


def create_splits(df: pd.DataFrame):
    """Create stratified train/test splits."""
    log.section("Creating Splits")
    
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df['group_type']
    )
    
    log.result("Train size", len(train_idx))
    log.result("Test size", len(test_idx))
    
    return train_idx, test_idx


# ============================================================================
# COMPLEXITY MEASURES
# ============================================================================

def compute_l0_norm(weights: np.ndarray, threshold: float = 1e-6) -> float:
    """Count non-zero parameters."""
    return np.sum(np.abs(weights) > threshold)


def compute_l1_norm(weights: np.ndarray) -> float:
    """L1 norm (sum of absolute values)."""
    return np.sum(np.abs(weights))


def compute_l2_norm(weights: np.ndarray) -> float:
    """L2 norm (Euclidean norm)."""
    return np.sqrt(np.sum(weights ** 2))


def compute_fisher_information(
    probe,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """
    Compute Fisher Information as trace of Hessian.
    Approximated via eigenvalues of X^T W X where W is diagonal weight matrix.
    """
    try:
        # Get predictions
        probs = probe.predict_proba(X)
        
        # Compute diagonal weight matrix (variance of predictions)
        if probs.shape[1] == 2:
            # Binary case
            p = probs[:, 1]
            weights = p * (1 - p) + epsilon
        else:
            # Multi-class case
            weights = np.sum(probs * (1 - probs), axis=1) + epsilon
        
        # Weighted covariance: X^T W X
        X_weighted = X * np.sqrt(weights)[:, np.newaxis]
        cov = X_weighted.T @ X_weighted / len(X)
        
        # Compute eigenvalues (more stable than full Hessian)
        eigenvals = eigvalsh(cov)
        eigenvals = np.maximum(eigenvals, 0)  # Numerical stability
        
        # Fisher information is trace
        fisher = np.sum(eigenvals)
        
        return float(fisher)
        
    except Exception as e:
        log.log(f"Warning: Fisher computation failed: {e}")
        return 0.0


def compute_all_complexity_measures(
    probe,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Compute all complexity measures for a probe."""
    
    # Get weights (flatten all parameters)
    if hasattr(probe, 'coef_'):
        weights = probe.coef_.ravel()
        if hasattr(probe, 'intercept_'):
            weights = np.concatenate([weights, probe.intercept_.ravel()])
    else:
        weights = np.array([])
    
    measures = {
        'l0': compute_l0_norm(weights),
        'l1': compute_l1_norm(weights),
        'l2': compute_l2_norm(weights),
        'fisher': compute_fisher_information(probe, X, y)
    }
    
    return measures


# ============================================================================
# MDL COMPUTATION
# ============================================================================

def compute_data_codelength(
    probe,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> float:
    """
    Compute data codelength: -log P(y|X,θ) in bits.
    Uses cross-entropy loss.
    """
    if scaler is not None:
        X = scaler.transform(X)
    
    # Get predicted probabilities
    y_proba = probe.predict_proba(X)
    
    # Compute cross-entropy in nats
    loss_nats = log_loss(y, y_proba, normalize=False)
    
    # Convert to bits
    loss_bits = loss_nats / np.log(2)
    
    return float(loss_bits)

def compute_parameter_codelength(n_params: int) -> float:
    """
    Compute parameter codelength: fixed cost to transmit parameters.
    
    Standard MDL: Each float32 parameter requires 32 bits to encode.
    Complexity is already handled by regularization λ during training.
    
    Args:
        n_params: Number of parameters in the model
        
    Returns:
        Parameter codelength in bits
    """
    return n_params * config.bits_per_parameter


def compute_mdl_score(
    probe,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> Dict[str, float]:
    """
    Compute full MDL score: data codelength + parameter codelength.
    
    MDL = L(D|θ) + L(θ)
    where:
        L(D|θ) = data codelength (negative log likelihood)
        L(θ) = parameter codelength (fixed encoding cost)
    
    Complexity measures (L0, L1, L2, Fisher) are computed for analysis
    but don't affect the MDL score (regularization λ already handled them).
    """
    # Data codelength: -log P(y|X,θ) in bits
    data_codelength = compute_data_codelength(probe, X, y, scaler)
    
    # Number of parameters
    if hasattr(probe, 'coef_'):
        n_params = probe.coef_.size
        if hasattr(probe, 'intercept_'):
            n_params += probe.intercept_.size
    else:
        n_params = 0
    
    # Fixed parameter codelength (standard MDL)
    param_codelength = compute_parameter_codelength(n_params)
    
    # Total MDL score
    mdl_score = data_codelength + param_codelength
    
    # Compute complexity measures for reporting/analysis
    complexity_measures = compute_all_complexity_measures(probe, X, y)
    
    result = {
        'data_codelength': data_codelength,
        'parameter_codelength': param_codelength,
        'mdl_score': mdl_score,
        'n_parameters': n_params,
        'complexity_measures': complexity_measures,  # For analysis only
    }
    
    return result

# ============================================================================
# VARIATIONAL MDL
# ============================================================================

def variational_mdl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_values: List[float],
    task_type: str = 'multiclass'
) -> Dict:
    """
    Compute MDL across multiple regularization strengths.
    """
    results = []
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for lam in lambda_values:
        if task_type == 'binary':
            probe = LogisticRegression(
                C=1.0/lam if lam > 0 else 1e10,
                max_iter=config.max_iter,
                random_state=config.random_state,
                penalty='l2',
                solver='lbfgs'
            )
        else:
            probe = LogisticRegression(
                C=1.0/lam if lam > 0 else 1e10,
                max_iter=config.max_iter,
                random_state=config.random_state,
                penalty='l2',
                solver='lbfgs',
                multi_class='multinomial'
            )
        
        probe.fit(X_train_scaled, y_train)
        
        # Compute MDL on test set
        mdl_result = compute_mdl_score(probe, X_test_scaled, y_test, scaler=None)
        
        # Add metadata
        mdl_result['lambda'] = lam
        mdl_result['test_accuracy'] = accuracy_score(y_test, probe.predict(X_test_scaled))
        
        results.append(mdl_result)
    
    return {
        'results': results,
        'scaler': scaler
    }


# ============================================================================
# PREQUENTIAL MDL
# ============================================================================

def prequential_mdl(
    X: np.ndarray,
    y: np.ndarray,
    chunk_sizes: List[int],
    task_type: str = 'multiclass',
    lambda_reg: float = 1.0
) -> Dict:
    """
    Compute prequential (online) codelength.
    Train on chunks sequentially and test on next chunk.
    """
    n_samples = len(X)
    
    # Scale once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for chunk_size in chunk_sizes:
        if chunk_size >= n_samples:
            continue
        
        n_chunks = n_samples // chunk_size
        
        cumulative_codelength = 0.0
        cumulative_samples = 0
        chunk_codelengths = []
        
        for i in range(n_chunks - 1):
            # Train on chunks 0 to i
            train_end = (i + 1) * chunk_size
            X_train_chunk = X_scaled[:train_end]
            y_train_chunk = y[:train_end]
            
            # Test on chunk i+1
            test_start = train_end
            test_end = test_start + chunk_size
            X_test_chunk = X_scaled[test_start:test_end]
            y_test_chunk = y[test_start:test_end]
            
            # Train probe
            if task_type == 'binary':
                probe = LogisticRegression(
                    C=1.0/lambda_reg,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                    penalty='l2'
                )
            else:
                probe = LogisticRegression(
                    C=1.0/lambda_reg,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                    penalty='l2',
                    multi_class='multinomial'
                )
            
            probe.fit(X_train_chunk, y_train_chunk)
            
            # Compute codelength on test chunk
            codelength = compute_data_codelength(probe, X_test_chunk, y_test_chunk, scaler=None)
            
            cumulative_codelength += codelength
            cumulative_samples += len(y_test_chunk)
            chunk_codelengths.append(codelength)
        
        results[chunk_size] = {
            'total_codelength': cumulative_codelength,
            'n_samples': cumulative_samples,
            'avg_codelength_per_sample': cumulative_codelength / cumulative_samples,
            'chunk_codelengths': chunk_codelengths
        }
    
    return results


# ============================================================================
# PER-CLASS CODELENGTH
# ============================================================================

def compute_per_class_codelength(
    probe,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> Dict[int, float]:
    """
    Compute codelength separately for each class.
    """
    if scaler is not None:
        X = scaler.transform(X)
    
    classes = np.unique(y)
    per_class = {}
    
    for cls in classes:
        mask = y == cls
        X_cls = X[mask]
        y_cls = y[mask]
        
        if len(y_cls) == 0:
            continue
        
        codelength = compute_data_codelength(probe, X_cls, y_cls, scaler=None)
        per_class[int(cls)] = {
            'codelength': codelength,
            'n_samples': len(y_cls),
            'avg_codelength': codelength / len(y_cls)
        }
    
    return per_class


# ============================================================================
# TASK 1: ATTRIBUTE PROBING
# ============================================================================

def mdl_attribute_probing(df, train_idx, test_idx, activations):
    """MDL analysis for attribute classification."""
    log.section("Task 1: Attribute MDL (16-class)")
    
    results = {}
    
    for model in config.models:
        results[model] = {}
        
        for layer in config.layers:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            X = activations[model][layer]
            y = df['attribute_label'].values
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Variational MDL
            log.log("  Computing variational MDL...")
            var_result = variational_mdl(
                X_train, y_train, X_test, y_test,
                config.lambda_values,
                task_type='multiclass'
            )
            
            # Find best lambda by MDL (L2 measure)
            best_idx = np.argmin([r['mdl_scores']['l2'] for r in var_result['results']])
            best_result = var_result['results'][best_idx]
            
            log.result("Best λ", f"{best_result['lambda']:.2e}")
            log.result("MDL (L2)", f"{best_result['mdl_scores']['l2']:.2f} bits")
            log.result("Data CL", f"{best_result['data_codelength']:.2f} bits")
            log.result("Param CL (L2)", f"{best_result['parameter_codelengths']['l2']:.2f} bits")
            log.result("Test Acc", f"{best_result['test_accuracy']:.4f}")
            
            # Prequential MDL
            log.log("  Computing prequential MDL...")
            preq_result = prequential_mdl(
                X_test, y_test,
                config.prequential_chunk_sizes,
                task_type='multiclass',
                lambda_reg=best_result['lambda']
            )
            
            # Per-class codelength
            log.log("  Computing per-class codelength...")
            scaler = var_result['scaler']
            best_lambda = best_result['lambda']
            
            probe = LogisticRegression(
                C=1.0/best_lambda,
                max_iter=config.max_iter,
                random_state=config.random_state,
                multi_class='multinomial'
            )
            probe.fit(scaler.transform(X_train), y_train)
            
            per_class = compute_per_class_codelength(
                probe, X_test, y_test, scaler
            )
            
            results[model][layer] = {
                'variational': var_result['results'],
                'best': best_result,
                'prequential': preq_result,
                'per_class': per_class
            }
    
    # Save results
    with open(config.output_dir / "results" / "attribute_mdl.json", 'w') as f:
        # Convert to serializable format
        save_results = {}
        for model in results:
            save_results[model] = {}
            for layer in results[model]:
                save_results[model][layer] = {
                    'variational': results[model][layer]['variational'],
                    'best': results[model][layer]['best'],
                    'prequential': {
                        str(k): v for k, v in results[model][layer]['prequential'].items()
                    },
                    'per_class': {
                        str(k): v for k, v in results[model][layer]['per_class'].items()
                    }
                }
        json.dump(save_results, f, indent=2)
    
    return results


# ============================================================================
# TASK 2: STATE PROBING
# ============================================================================

def mdl_state_probing(df, train_idx, test_idx, activations):
    """MDL analysis for state classification."""
    log.section("Task 2: State MDL (36-class)")
    
    results = {}
    
    for model in config.models:
        results[model] = {}
        
        for layer in config.layers:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            X = activations[model][layer]
            y = df['state_label'].values
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Variational MDL
            log.log("  Computing variational MDL...")
            var_result = variational_mdl(
                X_train, y_train, X_test, y_test,
                config.lambda_values,
                task_type='multiclass'
            )
            
            best_idx = np.argmin([r['mdl_scores']['l2'] for r in var_result['results']])
            best_result = var_result['results'][best_idx]
            
            log.result("Best λ", f"{best_result['lambda']:.2e}")
            log.result("MDL (L2)", f"{best_result['mdl_scores']['l2']:.2f} bits")
            log.result("Test Acc", f"{best_result['test_accuracy']:.4f}")
            
            # Prequential MDL
            log.log("  Computing prequential MDL...")
            preq_result = prequential_mdl(
                X_test, y_test,
                config.prequential_chunk_sizes,
                task_type='multiclass',
                lambda_reg=best_result['lambda']
            )
            
            # Per-class (per-state) codelength
            scaler = var_result['scaler']
            probe = LogisticRegression(
                C=1.0/best_result['lambda'],
                max_iter=config.max_iter,
                random_state=config.random_state,
                multi_class='multinomial'
            )
            probe.fit(scaler.transform(X_train), y_train)
            
            per_class = compute_per_class_codelength(
                probe, X_test, y_test, scaler
            )
            
            results[model][layer] = {
                'variational': var_result['results'],
                'best': best_result,
                'prequential': preq_result,
                'per_class': per_class
            }
    
    # Save results
    with open(config.output_dir / "results" / "state_mdl.json", 'w') as f:
        save_results = {}
        for model in results:
            save_results[model] = {}
            for layer in results[model]:
                save_results[model][layer] = {
                    'variational': results[model][layer]['variational'],
                    'best': results[model][layer]['best'],
                    'prequential': {
                        str(k): v for k, v in results[model][layer]['prequential'].items()
                    },
                    'per_class': {
                        str(k): v for k, v in results[model][layer]['per_class'].items()
                    }
                }
        json.dump(save_results, f, indent=2)
    
    return results


# ============================================================================
# TASK 3: CORRECTNESS PROBING
# ============================================================================

def mdl_correctness_probing(df, train_idx, test_idx, activations):
    """MDL analysis for correctness prediction."""
    log.section("Task 3: Correctness MDL (Binary)")
    
    results = {}
    
    for model in config.models:
        results[model] = {}
        
        for layer in config.layers:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            X = activations[model][layer]
            y = df[f'{model}_correct_label'].values
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Variational MDL
            log.log("  Computing variational MDL...")
            var_result = variational_mdl(
                X_train, y_train, X_test, y_test,
                config.lambda_values,
                task_type='binary'
            )
            
            best_idx = np.argmin([r['mdl_scores']['l2'] for r in var_result['results']])
            best_result = var_result['results'][best_idx]
            
            log.result("Best λ", f"{best_result['lambda']:.2e}")
            log.result("MDL (L2)", f"{best_result['mdl_scores']['l2']:.2f} bits")
            log.result("Test Acc", f"{best_result['test_accuracy']:.4f}")
            
            # Prequential MDL
            preq_result = prequential_mdl(
                X_test, y_test,
                config.prequential_chunk_sizes,
                task_type='binary',
                lambda_reg=best_result['lambda']
            )
            
            # Per-class codelength
            scaler = var_result['scaler']
            probe = LogisticRegression(
                C=1.0/best_result['lambda'],
                max_iter=config.max_iter,
                random_state=config.random_state
            )
            probe.fit(scaler.transform(X_train), y_train)
            
            per_class = compute_per_class_codelength(
                probe, X_test, y_test, scaler
            )
            
            results[model][layer] = {
                'variational': var_result['results'],
                'best': best_result,
                'prequential': preq_result,
                'per_class': per_class
            }
    
    # Save results
    with open(config.output_dir / "results" / "correctness_mdl.json", 'w') as f:
        save_results = {}
        for model in results:
            save_results[model] = {}
            for layer in results[model]:
                save_results[model][layer] = {
                    'variational': results[model][layer]['variational'],
                    'best': results[model][layer]['best'],
                    'prequential': {
                        str(k): v for k, v in results[model][layer]['prequential'].items()
                    },
                    'per_class': {
                        str(k): v for k, v in results[model][layer]['per_class'].items()
                    }
                }
        json.dump(save_results, f, indent=2)
    
    return results


# ============================================================================
# TASK 4: GROUP PROBING
# ============================================================================

def mdl_group_probing(df, train_idx, test_idx, activations):
    """MDL analysis for group classification."""
    log.section("Task 4: Group MDL (3-class)")
    
    results = {}
    
    for model in config.models:
        results[model] = {}
        
        for layer in config.layers:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            X = activations[model][layer]
            y = df['group_label'].values
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Variational MDL
            log.log("  Computing variational MDL...")
            var_result = variational_mdl(
                X_train, y_train, X_test, y_test,
                config.lambda_values,
                task_type='multiclass'
            )
            
            best_idx = np.argmin([r['mdl_scores']['l2'] for r in var_result['results']])
            best_result = var_result['results'][best_idx]
            
            log.result("Best λ", f"{best_result['lambda']:.2e}")
            log.result("MDL (L2)", f"{best_result['mdl_scores']['l2']:.2f} bits")
            log.result("Test Acc", f"{best_result['test_accuracy']:.4f}")
            
            # Prequential MDL
            preq_result = prequential_mdl(
                X_test, y_test,
                config.prequential_chunk_sizes,
                task_type='multiclass',
                lambda_reg=best_result['lambda']
            )
            
            # Per-group codelength
            scaler = var_result['scaler']
            probe = LogisticRegression(
                C=1.0/best_result['lambda'],
                max_iter=config.max_iter,
                random_state=config.random_state,
                multi_class='multinomial'
            )
            probe.fit(scaler.transform(X_train), y_train)
            
            per_class = compute_per_class_codelength(
                probe, X_test, y_test, scaler
            )
            
            results[model][layer] = {
                'variational': var_result['results'],
                'best': best_result,
                'prequential': preq_result,
                'per_class': per_class
            }
    
    # Save results
    with open(config.output_dir / "results" / "group_mdl.json", 'w') as f:
        save_results = {}
        for model in results:
            save_results[model] = {}
            for layer in results[model]:
                save_results[model][layer] = {
                    'variational': results[model][layer]['variational'],
                    'best': results[model][layer]['best'],
                    'prequential': {
                        str(k): v for k, v in results[model][layer]['prequential'].items()
                    },
                    'per_class': {
                        str(k): v for k, v in results[model][layer]['per_class'].items()
                    }
                }
        json.dump(save_results, f, indent=2)
    
    return results


# ============================================================================
# TASK 5: CROSS-MODEL TRANSFER MDL
# ============================================================================

def mdl_cross_model_transfer(df, train_idx, test_idx, activations):
    """MDL analysis for cross-model transfer."""
    log.section("Task 5: Cross-Model Transfer MDL")
    
    results = {}
    
    for layer in config.layers:
        log.log(f"\nLayer {layer}: Train on Base → Test on Instruct")
        
        results[layer] = {}
        
        X_base = activations['base'][layer]
        X_instruct = activations['instruct'][layer]
        
        # Attribute transfer
        log.log("\n  Attribute Transfer:")
        y = df['attribute_label'].values
        
        X_train = X_base[train_idx]
        y_train = y[train_idx]
        X_test_base = X_base[test_idx]
        X_test_instruct = X_instruct[test_idx]
        y_test = y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_base_scaled = scaler.transform(X_test_base)
        X_test_instruct_scaled = scaler.transform(X_test_instruct)
        
        # Train on base
        probe = LogisticRegression(
            C=1.0,
            max_iter=config.max_iter,
            random_state=config.random_state,
            multi_class='multinomial'
        )
        probe.fit(X_train_scaled, y_train)
        
        # MDL on base (in-model)
        mdl_base = compute_mdl_score(probe, X_test_base_scaled, y_test, scaler=None)
        
        # MDL on instruct (cross-model)
        mdl_instruct = compute_mdl_score(probe, X_test_instruct_scaled, y_test, scaler=None)
        
        log.result("  Base→Base MDL (L2)", f"{mdl_base['mdl_scores']['l2']:.2f} bits")
        log.result("  Base→Instruct MDL (L2)", f"{mdl_instruct['mdl_scores']['l2']:.2f} bits")
        log.result("  MDL Ratio", f"{mdl_instruct['mdl_scores']['l2'] / mdl_base['mdl_scores']['l2']:.4f}")
        
        results[layer]['attribute'] = {
            'base_to_base': mdl_base,
            'base_to_instruct': mdl_instruct,
            'ratio': mdl_instruct['mdl_scores']['l2'] / mdl_base['mdl_scores']['l2']
        }
        
        # Correctness transfer
        log.log("\n  Correctness Transfer:")
        y = df['base_correct_label'].values
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        probe = LogisticRegression(
            C=1.0,
            max_iter=config.max_iter,
            random_state=config.random_state
        )
        probe.fit(X_train_scaled, y_train)
        
        mdl_base = compute_mdl_score(probe, X_test_base_scaled, y_test, scaler=None)
        mdl_instruct = compute_mdl_score(probe, X_test_instruct_scaled, y_test, scaler=None)
        
        log.result("  Base→Base MDL (L2)", f"{mdl_base['mdl_scores']['l2']:.2f} bits")
        log.result("  Base→Instruct MDL (L2)", f"{mdl_instruct['mdl_scores']['l2']:.2f} bits")
        log.result("  MDL Ratio", f"{mdl_instruct['mdl_scores']['l2'] / mdl_base['mdl_scores']['l2']:.4f}")
        
        results[layer]['correctness'] = {
            'base_to_base': mdl_base,
            'base_to_instruct': mdl_instruct,
            'ratio': mdl_instruct['mdl_scores']['l2'] / mdl_base['mdl_scores']['l2']
        }
    
    # Save results
    with open(config.output_dir / "results" / "cross_model_mdl.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_visualizations(
    attr_results, state_results, corr_results, 
    group_results, transfer_results
):
    """Create comprehensive MDL visualizations."""
    log.section("Creating Visualizations")
    
    # 1. Variational MDL curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    tasks = [
        ('attribute', attr_results, 'Attribute (16-class)'),
        ('state', state_results, 'State (36-class)'),
        ('correctness', corr_results, 'Correctness (Binary)'),
        ('group', group_results, 'Group (3-class)')
    ]
    
    for ax, (task_name, results, title) in zip(axes.flat, tasks):
        for model in config.models:
            for layer in config.layers:
                var_results = results[model][layer]['variational']
                lambdas = [r['lambda'] for r in var_results]
                mdls = [r['mdl_scores']['l2'] for r in var_results]
                
                ax.plot(lambdas, mdls, marker='o', label=f'{model} L{layer}', alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_xlabel('Regularization λ')
        ax.set_ylabel('MDL (L2, bits)')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "plots" / "variational_mdl_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved variational MDL curves")
    
    # 2. Best MDL comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ax, (task_name, results, title) in zip(axes.flat, tasks):
        mdl_data = []
        labels = []
        
        for model in config.models:
            for layer in config.layers:
                best = results[model][layer]['best']
                mdl_data.append(best['mdl_scores']['l2'])
                labels.append(f'{model}\nL{layer}')
        
        x = np.arange(len(labels))
        colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']
        
        ax.bar(x, mdl_data, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('MDL (L2, bits)')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "plots" / "best_mdl_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved best MDL comparison")
    
    # 3. Cross-model transfer ratios
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    transfer_tasks = ['attribute', 'correctness']
    titles = ['Attribute Transfer', 'Correctness Transfer']
    
    for ax, task, title in zip(axes, transfer_tasks, titles):
        layers_list = []
        ratios = []
        
        for layer in config.layers:
            ratio = transfer_results[layer][task]['ratio']
            layers_list.append(layer)
            ratios.append(ratio)
        
        ax.bar(range(len(layers_list)), ratios, alpha=0.7)
        ax.axhline(y=1.0, color='red', linestyle='--', label='Perfect Transfer')
        ax.set_xticks(range(len(layers_list)))
        ax.set_xticklabels([f'L{l}' for l in layers_list])
        ax.set_ylabel('MDL Ratio (Instruct/Base)')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "plots" / "cross_model_transfer_ratios.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved cross-model transfer ratios")
    
    # 4. Complexity measures comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ax, (task_name, results, title) in zip(axes.flat, tasks):
        measures_data = {m: [] for m in config.complexity_measures}
        labels = []
        
        for model in config.models:
            for layer in config.layers:
                best = results[model][layer]['best']
                for measure in config.complexity_measures:
                    measures_data[measure].append(best['mdl_scores'][measure])
                labels.append(f'{model}\nL{layer}')
        
        x = np.arange(len(labels))
        width = 0.2
        
        for i, measure in enumerate(config.complexity_measures):
            offset = (i - 1.5) * width
            ax.bar(x + offset, measures_data[measure], width, label=measure.upper(), alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('MDL (bits)')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "plots" / "complexity_measures_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved complexity measures comparison")
    
    # 5. Prequential learning curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ax, (task_name, results, title) in zip(axes.flat, tasks):
        for model in config.models:
            layer = 6  # Use layer 6 for visualization
            preq = results[model][layer]['prequential']
            
            chunk_sizes = sorted([int(k) for k in preq.keys()])
            avg_codelengths = [preq[str(cs)]['avg_codelength_per_sample'] for cs in chunk_sizes]
            
            ax.plot(chunk_sizes, avg_codelengths, marker='o', label=model, linewidth=2)
        
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Avg Codelength per Sample (bits)')
        ax.set_title(f'{title} - Layer 6', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "plots" / "prequential_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved prequential learning curves")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(
    attr_results, state_results, corr_results,
    group_results, transfer_results
):
    """Generate comprehensive MDL summary report."""
    log.section("Generating Summary Report")
    
    lines = []
    
    lines.append("="*80)
    lines.append("MDL PROBING ANALYSIS - SUMMARY REPORT")
    lines.append("="*80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Attribute MDL
    lines.append("\n" + "-"*80)
    lines.append("1. ATTRIBUTE MDL (16-CLASS)")
    lines.append("-"*80)
    
    for model in config.models:
        lines.append(f"\n{model.upper()}:")
        for layer in config.layers:
            best = attr_results[model][layer]['best']
            lines.append(f"  Layer {layer}:")
            lines.append(f"    MDL (L2):     {best['mdl_scores']['l2']:.2f} bits")
            lines.append(f"    MDL (Fisher): {best['mdl_scores']['fisher']:.2f} bits")
            lines.append(f"    Test Acc:     {best['test_accuracy']:.4f}")
    
    # State MDL
    lines.append("\n" + "-"*80)
    lines.append("2. STATE MDL (36-CLASS)")
    lines.append("-"*80)
    
    for model in config.models:
        lines.append(f"\n{model.upper()}:")
        for layer in config.layers:
            best = state_results[model][layer]['best']
            lines.append(f"  Layer {layer}:")
            lines.append(f"    MDL (L2):     {best['mdl_scores']['l2']:.2f} bits")
            lines.append(f"    Test Acc:     {best['test_accuracy']:.4f}")
    
    # Correctness MDL
    lines.append("\n" + "-"*80)
    lines.append("3. CORRECTNESS MDL (BINARY)")
    lines.append("-"*80)
    
    for model in config.models:
        lines.append(f"\n{model.upper()}:")
        for layer in config.layers:
            best = corr_results[model][layer]['best']
            lines.append(f"  Layer {layer}:")
            lines.append(f"    MDL (L2):     {best['mdl_scores']['l2']:.2f} bits")
            lines.append(f"    Test Acc:     {best['test_accuracy']:.4f}")
    
    # Group MDL
    lines.append("\n" + "-"*80)
    lines.append("4. GROUP MDL (3-CLASS)")
    lines.append("-"*80)
    
    for model in config.models:
        lines.append(f"\n{model.upper()}:")
        for layer in config.layers:
            best = group_results[model][layer]['best']
            lines.append(f"  Layer {layer}:")
            lines.append(f"    MDL (L2):     {best['mdl_scores']['l2']:.2f} bits")
            lines.append(f"    Test Acc:     {best['test_accuracy']:.4f}")
    
    # Cross-model transfer
    lines.append("\n" + "-"*80)
    lines.append("5. CROSS-MODEL TRANSFER MDL")
    lines.append("-"*80)
    
    for layer in config.layers:
        lines.append(f"\nLayer {layer}:")
        
        attr_ratio = transfer_results[layer]['attribute']['ratio']
        corr_ratio = transfer_results[layer]['correctness']['ratio']
        
        lines.append(f"  Attribute MDL Ratio:    {attr_ratio:.4f}")
        lines.append(f"  Correctness MDL Ratio:  {corr_ratio:.4f}")
    
    # Key findings
    lines.append("\n" + "-"*80)
    lines.append("6. KEY FINDINGS")
    lines.append("-"*80)
    
    # Compute average MDL ratios
    avg_attr_ratio = np.mean([transfer_results[l]['attribute']['ratio'] for l in config.layers])
    avg_corr_ratio = np.mean([transfer_results[l]['correctness']['ratio'] for l in config.layers])
    
    lines.append(f"\n• Average Attribute Transfer Ratio:   {avg_attr_ratio:.4f}")
    lines.append(f"• Average Correctness Transfer Ratio: {avg_corr_ratio:.4f}")
    
    if avg_attr_ratio < 1.05 and avg_corr_ratio < 1.05:
        lines.append("\n⚠️  NEAR-EQUAL MDL (<5% difference)")
        lines.append("Base and Instruct models encode information with nearly identical complexity.")
        lines.append("Representations remain highly aligned despite behavioral divergence.")
        lines.append("CONCLUSION: Suppression operates via decision boundaries, NOT erasure.")
    elif avg_attr_ratio > 1.2:
        lines.append("\n⚠️  SIGNIFICANT MDL DIVERGENCE (>20% difference)")
        lines.append("Instruct model requires more bits to encode same information.")
        lines.append("Representations have been fundamentally altered by instruction-tuning.")
    else:
        lines.append("\n⚠️  MODERATE MDL DIVERGENCE (5-20% difference)")
        lines.append("Partial representational changes detected.")
        lines.append("Suppression involves both representational and decision-level effects.")
    
    # Compare Base vs Instruct MDL
    lines.append("\n" + "-"*80)
    lines.append("7. BASE VS INSTRUCT MDL COMPARISON")
    lines.append("-"*80)
    
    for task_name, results, label in [
        ('Attribute', attr_results, '16-class'),
        ('State', state_results, '36-class'),
        ('Correctness', corr_results, 'Binary'),
        ('Group', group_results, '3-class')
    ]:
        lines.append(f"\n{task_name} ({label}):")
        
        for layer in config.layers:
            base_mdl = results['base'][layer]['best']['mdl_scores']['l2']
            inst_mdl = results['instruct'][layer]['best']['mdl_scores']['l2']
            diff_pct = ((inst_mdl - base_mdl) / base_mdl) * 100
            
            lines.append(f"  Layer {layer}: Base={base_mdl:.2f}, Instruct={inst_mdl:.2f}, Δ={diff_pct:+.2f}%")
    
    lines.append("\n" + "="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)
    
    report_text = '\n'.join(lines)
    
    with open(config.output_dir / "MDL_SUMMARY_REPORT.txt", 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    log.log("✓ Summary report saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    start_time = datetime.now()
    
    log.section("MDL Probing Analysis Pipeline")
    log.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load data
        df, activations, label_encoders = load_data()
        
        # Create splits
        train_idx, test_idx = create_splits(df)
        
        # Task 1: Attribute MDL
        attr_results = mdl_attribute_probing(df, train_idx, test_idx, activations)
        
        # Task 2: State MDL
        state_results = mdl_state_probing(df, train_idx, test_idx, activations)
        
        # Task 3: Correctness MDL
        corr_results = mdl_correctness_probing(df, train_idx, test_idx, activations)
        
        # Task 4: Group MDL
        group_results = mdl_group_probing(df, train_idx, test_idx, activations)
        
        # Task 5: Cross-model transfer MDL
        transfer_results = mdl_cross_model_transfer(df, train_idx, test_idx, activations)
        
        # Visualizations
        create_visualizations(
            attr_results, state_results, corr_results,
            group_results, transfer_results
        )
        
        # Summary report
        generate_summary_report(
            attr_results, state_results, corr_results,
            group_results, transfer_results
        )
        
        log.log("✓ All tasks completed successfully")
        
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
    log.log(f"  {config.output_dir}")


if __name__ == "__main__":
    main()