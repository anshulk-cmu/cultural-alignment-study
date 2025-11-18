#!/usr/bin/env python3
"""
Linear Probing for Cultural Knowledge Suppression Detection
Comprehensive probing suite: Attribute, Correctness, State, Cross-Model Transfer, Multi-Task
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.multioutput import MultiOutputClassifier
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input paths
    ACTIVATION_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/Activations")
    INDEX_FILE = ACTIVATION_DIR / "activation_index.csv"
    ENHANCED_DATA = Path("/home/anshulk/cultural-alignment-study/outputs/EDA_results/tables/enhanced_dataset.csv")
    
    # Output paths
    OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/linear_probing/v2")
    HEAVY_DATA_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/Linear_Probing/v2")
    
    # Models and layers
    MODELS = ['base', 'instruct']
    LAYERS = [6, 12, 18]
    HIDDEN_SIZE = 1536
    
    # Probing settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
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
    """Cross-validation for probe."""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    
    scores = cross_val_score(probe, X_scaled, y, cv=cv, scoring='accuracy')
    
    return {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'scores': scores.tolist()
    }


# ============================================================================
# PROBE 1: ATTRIBUTE (16-CLASS)
# ============================================================================

def probe_attribute(df, train_idx, test_idx, activations):
    """Attribute probing (16-class classification)."""
    log.section("Probe 1: Attribute Classification (16-class)")
    
    results = {}
    
    for model in config.MODELS:
        results[model] = {}
        
        for layer in config.LAYERS:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            # Get activations
            X = activations[model][layer]
            y = df['attribute_label'].values
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Train probe
            probe, accuracy, result = train_probe(
                X_train, y_train, X_test, y_test, 'attribute'
            )
            
            log.result(f"Accuracy", f"{accuracy:.4f}")
            log.result(f"Precision", f"{result['metrics']['precision']:.4f}")
            log.result(f"Recall", f"{result['metrics']['recall']:.4f}")
            log.result(f"F1", f"{result['metrics']['f1']:.4f}")
            
            # Cross-validation
            cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
            log.result(f"CV Accuracy", f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}")
            
            result['cv'] = cv_result
            results[model][layer] = result
            
            # Save probe
            probe_file = config.HEAVY_DATA_DIR / "models" / f"attribute_{model}_layer{layer}.pkl"
            with open(probe_file, 'wb') as f:
                pickle.dump({'probe': probe, 'scaler': result['scaler']}, f)
    
    # Save results
    with open(config.OUTPUT_DIR / "results" / "attribute_probing.json", 'w') as f:
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
# PROBE 2: CORRECTNESS (BINARY)
# ============================================================================

def probe_correctness(df, train_idx, test_idx, activations):
    """Correctness probing (binary classification)."""
    log.section("Probe 2: Correctness Prediction (Binary)")
    
    results = {}
    
    for model in config.MODELS:
        results[model] = {}
        
        for layer in config.LAYERS:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            # Get activations
            X = activations[model][layer]
            
            # CRITICAL: Predict own correctness
            y = df[f'{model}_correct_label'].values
            
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
            results[model][layer] = result
            
            # Save probe
            probe_file = config.HEAVY_DATA_DIR / "models" / f"correctness_{model}_layer{layer}.pkl"
            with open(probe_file, 'wb') as f:
                pickle.dump({'probe': probe, 'scaler': result['scaler']}, f)
    
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
    
    for model in config.MODELS:
        results[model] = {}
        
        for layer in config.LAYERS:
            log.log(f"\n{model.upper()} Layer {layer}")
            
            # Get activations
            X = activations[model][layer]
            y = df['state_label'].values
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Train probe
            probe, accuracy, result = train_probe(
                X_train, y_train, X_test, y_test, 'state'
            )
            
            log.result(f"Accuracy", f"{accuracy:.4f}")
            log.result(f"Precision", f"{result['metrics']['precision']:.4f}")
            log.result(f"Recall", f"{result['metrics']['recall']:.4f}")
            log.result(f"F1", f"{result['metrics']['f1']:.4f}")
            
            # Cross-validation
            cv_result = cross_validate_probe(X, y, n_folds=config.CV_FOLDS)
            log.result(f"CV Accuracy", f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}")
            
            result['cv'] = cv_result
            results[model][layer] = result
            
            # Save probe
            probe_file = config.HEAVY_DATA_DIR / "models" / f"state_{model}_layer{layer}.pkl"
            with open(probe_file, 'wb') as f:
                pickle.dump({'probe': probe, 'scaler': result['scaler']}, f)
    
    # Save results
    with open(config.OUTPUT_DIR / "results" / "state_probing.json", 'w') as f:
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
# VISUALIZATION
# ============================================================================

def create_visualizations(
    attr_results, corr_results, state_results, 
    transfer_results, multitask_results, group_results
):
    """Create comprehensive visualizations."""
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
    
    # Correctness
    ax = axes[0, 1]
    for model in config.MODELS:
        accs = [corr_results[model][layer]['accuracy'] for layer in config.LAYERS]
        ax.plot(config.LAYERS, accs, marker='o', label=model, linewidth=2)
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
        lines.append(f"\n{model.upper()}:")
        for layer in config.LAYERS:
            acc = corr_results[model][layer]['accuracy']
            cv_mean = corr_results[model][layer]['cv']['mean']
            cv_std = corr_results[model][layer]['cv']['std']
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
        
        # Probe 2: Correctness
        corr_results = probe_correctness(df, train_idx, test_idx, activations)
        
        # Probe 3: State
        state_results = probe_state(df, train_idx, test_idx, activations)
        
        # Probe 4: Cross-model transfer
        transfer_results = probe_cross_model_transfer(df, train_idx, test_idx, activations)
        
        # Probe 5: Multi-task
        multitask_results = probe_multitask(df, train_idx, test_idx, activations)
        
        # Group-wise analysis
        group_results = analyze_by_group(df, train_idx, test_idx, activations)
        
        # Visualizations
        create_visualizations(
            attr_results, corr_results, state_results,
            transfer_results, multitask_results, group_results
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