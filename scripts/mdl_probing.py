#!/usr/bin/env python3
"""
MDL (Minimum Description Length) Probing Analysis
==================================================

Implements two MDL coding schemes:
1. Variational Coding: Model complexity + data misfit tradeoff
2. Online Coding: Sequential learning efficiency

Based on: Voita & Titov (2020) "Information-Theoretic Probing with MDL"

Models: Qwen2-1.5B Base & Instruct
Layers: 6, 12, 18
Dataset: 33,522 sentences across 11,174 questions
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
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input paths
    ACTIVATION_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/Activations")
    INDEX_FILE = ACTIVATION_DIR / "activation_index.csv"
    SPLIT_INFO = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/Linear_Probing/split_info.json")
    
    # Output paths
    OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/mdl_probing")
    HEAVY_DATA_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/MDL_Probing")
    
    # Models and layers
    MODELS = ['base', 'instruct']
    LAYERS = [6, 12, 18]
    
    # MDL parameters
    LAMBDA_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    ONLINE_CODING_SPLITS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    RANDOM_STATE = 42
    MAX_ITER = 1000
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.HEAVY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "results").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "curves").mkdir(exist_ok=True)

config = Config()

# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal_width = 80
        
    def section(self, title):
        msg = f"\n{'='*self.terminal_width}\n{title.upper()}\n{'='*self.terminal_width}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
    
    def subsection(self, title):
        msg = f"\n{'-'*60}\n{title}\n{'-'*60}"
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

log = Logger(config.OUTPUT_DIR / "mdl_log.txt")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load activation matrices and labels with existing splits."""
    log.section("Loading Data")
    
    log.log("Loading activation index...")
    df = pd.read_csv(config.INDEX_FILE)
    log.result("Total sentences", len(df))
    
    log.log("Loading train/dev/test splits...")
    with open(config.SPLIT_INFO, 'r') as f:
        split_info = json.load(f)
    
    train_questions = set(split_info['train_questions'])
    dev_questions = set(split_info['dev_questions'])
    test_questions = set(split_info['test_questions'])
    
    train_mask = df['row_id'].isin(train_questions).values
    dev_mask = df['row_id'].isin(dev_questions).values
    test_mask = df['row_id'].isin(test_questions).values
    
    log.result("Train sentences", train_mask.sum())
    log.result("Dev sentences", dev_mask.sum())
    log.result("Test sentences", test_mask.sum())
    
    activations = {}
    for model in config.MODELS:
        activations[model] = {}
        for layer in config.LAYERS:
            file_path = config.ACTIVATION_DIR / f"{model}_layer{layer}_activations.npy"
            log.log(f"Loading {model} layer {layer}...")
            acts = np.load(file_path)
            activations[model][layer] = acts
            log.result(f"  Shape", acts.shape)
    
    return df, activations, train_mask, dev_mask, test_mask

# ============================================================================
# CONTROL TASK GENERATION
# ============================================================================

def create_control_labels(df):
    """Create shuffled labels for control tasks."""
    log.section("Creating Control Task Labels")
    
    df_control = df.copy()
    np.random.seed(config.RANDOM_STATE)
    
    question_groups = df.groupby('row_id')['group_type'].first()
    shuffled_groups = question_groups.sample(frac=1, random_state=config.RANDOM_STATE)
    group_mapping = dict(zip(question_groups.index, shuffled_groups.values))
    df_control['group_type_control'] = df_control['row_id'].map(group_mapping)
    
    for label_col in ['base_correct', 'instruct_correct']:
        question_labels = df.groupby('row_id')[label_col].first()
        shuffled_labels = question_labels.sample(frac=1, random_state=config.RANDOM_STATE)
        label_mapping = dict(zip(question_labels.index, shuffled_labels.values))
        df_control[f'{label_col}_control'] = df_control['row_id'].map(label_mapping)
    
    log.log("✓ Control labels created")
    return df_control

# ============================================================================
# MDL TASK DEFINITIONS
# ============================================================================

def get_mdl_tasks(df, model):
    """Define probing tasks for MDL analysis."""
    tasks = {}
    
    # Task 1: Suppression vs Control
    suppression_mask = df['group_type'] == 'suppression'
    control_mask = df['group_type'] == 'control'
    supp_vs_ctrl_mask = suppression_mask | control_mask
    
    tasks['suppression_vs_control'] = {
        'label': (df['group_type'] == 'suppression').astype(int),
        'mask': supp_vs_ctrl_mask,
        'name': 'Suppression vs Control',
        'control': False
    }
    
    suppression_mask_ctrl = df['group_type_control'] == 'suppression'
    control_mask_ctrl = df['group_type_control'] == 'control'
    supp_vs_ctrl_mask_ctrl = suppression_mask_ctrl | control_mask_ctrl
    
    tasks['suppression_vs_control_control'] = {
        'label': (df['group_type_control'] == 'suppression').astype(int),
        'mask': supp_vs_ctrl_mask_ctrl,
        'name': 'Suppression vs Control (Control)',
        'control': True,
        'real_task': 'suppression_vs_control'
    }
    
    # Task 2: Enhancement vs Control
    enhancement_mask = df['group_type'] == 'enhancement'
    enh_vs_ctrl_mask = enhancement_mask | control_mask
    
    tasks['enhancement_vs_control'] = {
        'label': (df['group_type'] == 'enhancement').astype(int),
        'mask': enh_vs_ctrl_mask,
        'name': 'Enhancement vs Control',
        'control': False
    }
    
    enhancement_mask_ctrl = df['group_type_control'] == 'enhancement'
    enh_vs_ctrl_mask_ctrl = enhancement_mask_ctrl | control_mask_ctrl
    
    tasks['enhancement_vs_control_control'] = {
        'label': (df['group_type_control'] == 'enhancement').astype(int),
        'mask': enh_vs_ctrl_mask_ctrl,
        'name': 'Enhancement vs Control (Control)',
        'control': True,
        'real_task': 'enhancement_vs_control'
    }
    
    # Task 3: 3-way classification
    group_map = {'suppression': 0, 'enhancement': 1, 'control': 2}
    tasks['group_3way'] = {
        'label': df['group_type'].map(group_map),
        'name': 'Group Type (3-way)',
        'multiclass': True,
        'control': False
    }
    
    tasks['group_3way_control'] = {
        'label': df['group_type_control'].map(group_map),
        'name': 'Group Type (3-way) (Control)',
        'multiclass': True,
        'control': True,
        'real_task': 'group_3way'
    }
    
    return tasks

# ============================================================================
# VARIATIONAL MDL CODING
# ============================================================================

def compute_codelength_uniform(y_train):
    """Compute uniform codelength (transmitting labels without model)."""
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    return n_samples * np.log2(n_classes)

def compute_model_complexity(probe, n_features):
    """Compute model complexity (weight encoding cost)."""
    if hasattr(probe, 'coef_'):
        weights = probe.coef_
        if len(weights.shape) == 1:
            weights = weights.reshape(1, -1)
        complexity = np.sum(weights ** 2)
    else:
        complexity = 0.0
    return complexity

def compute_data_misfit(probe, scaler, X, y):
    """Compute data misfit (negative log likelihood in bits)."""
    X_scaled = scaler.transform(X)
    ce_loss = log_loss(y, probe.predict_proba(X_scaled))
    n_samples = len(y)
    return (ce_loss * n_samples) / np.log(2)

def variational_mdl_coding(X_train, y_train, X_test, y_test, multiclass=False):
    """
    Variational MDL: Find optimal lambda that minimizes total codelength.
    
    Codelength = Model Complexity + Data Misfit
    """
    results = []
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    uniform_codelength = compute_codelength_uniform(y_train)
    
    for lam in config.LAMBDA_VALUES:
        C = 1.0 / (2 * lam) if lam > 0 else 1e10
        
        if multiclass:
            probe = LogisticRegression(
                C=C,
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE,
                multi_class='multinomial',
                solver='lbfgs'
            )
        else:
            probe = LogisticRegression(
                C=C,
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE
            )
        
        probe.fit(X_train_scaled, y_train)
        
        n_features = X_train.shape[1]
        model_complexity = lam * compute_model_complexity(probe, n_features)
        data_misfit = compute_data_misfit(probe, scaler, X_train, y_train)
        test_misfit = compute_data_misfit(probe, scaler, X_test, y_test)
        
        total_codelength = model_complexity + data_misfit
        compression = uniform_codelength / total_codelength
        
        results.append({
            'lambda': lam,
            'C': C,
            'model_complexity': model_complexity,
            'train_misfit': data_misfit,
            'test_misfit': test_misfit,
            'total_codelength': total_codelength,
            'uniform_codelength': uniform_codelength,
            'compression_ratio': compression
        })
    
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['total_codelength'].idxmin()
    optimal_result = results_df.iloc[optimal_idx]
    
    return results_df, optimal_result

# ============================================================================
# ONLINE MDL CODING
# ============================================================================

def online_mdl_coding(X_train, y_train, X_test, y_test, optimal_C, multiclass=False):
    """
    Online MDL: Measure learning efficiency through sequential training.
    
    Train on increasing fractions of data and measure codelength growth.
    """
    results = []
    
    n_samples = len(y_train)
    
    # Shuffle data to ensure class balance in early chunks
    indices = np.arange(n_samples)
    np.random.seed(config.RANDOM_STATE)
    np.random.shuffle(indices)
    
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_shuffled)
    X_test_scaled = scaler.transform(X_test)
    
    for split_frac in config.ONLINE_CODING_SPLITS:
        split_size = int(n_samples * split_frac)
        if split_size < 10:
            continue
        
        X_partial = X_train_scaled[:split_size]
        y_partial = y_train_shuffled[:split_size]
        
        # Check class balance
        n_classes_present = len(np.unique(y_partial))
        
        if multiclass:
            min_classes_required = 3
        else:
            min_classes_required = 2
        
        if n_classes_present < min_classes_required:
            continue
        
        if multiclass:
            probe = LogisticRegression(
                C=optimal_C,
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE,
                multi_class='multinomial',
                solver='lbfgs'
            )
        else:
            probe = LogisticRegression(
                C=optimal_C,
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_STATE
            )
        
        probe.fit(X_partial, y_partial)
        
        test_misfit = compute_data_misfit(probe, scaler, X_test, y_test)
        
        train_acc = probe.score(X_partial, y_partial)
        test_acc = probe.score(X_test_scaled, y_test)
        
        results.append({
            'data_fraction': split_frac,
            'n_samples': split_size,
            'test_misfit': test_misfit,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        })
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN MDL PIPELINE
# ============================================================================

def run_mdl_analysis(df, activations, train_mask, dev_mask, test_mask):
    """Run complete MDL analysis for all model×layer×task combinations."""
    log.section("Running MDL Analysis")
    
    all_variational_results = []
    all_online_results = []
    
    for model in config.MODELS:
        log.subsection(f"Model: {model.upper()}")
        
        tasks = get_mdl_tasks(df, model)
        
        for layer in config.LAYERS:
            log.log(f"\nLayer {layer}:")
            
            X = activations[model][layer]
            
            for task_name, task_info in tasks.items():
                log.log(f"  Task: {task_info['name']}")
                
                y = task_info['label'].values
                
                if 'mask' in task_info:
                    task_mask = task_info['mask'].values
                    train_idx = train_mask & task_mask
                    dev_idx = dev_mask & task_mask
                    test_idx = test_mask & task_mask
                else:
                    train_idx = train_mask
                    dev_idx = dev_mask
                    test_idx = test_mask
                
                train_dev_idx = train_idx | dev_idx
                
                X_train = X[train_dev_idx]
                y_train = y[train_dev_idx]
                X_test = X[test_idx]
                y_test = y[test_idx]
                
                log.result("Train samples", len(y_train))
                log.result("Test samples", len(y_test))
                
                multiclass = task_info.get('multiclass', False)
                
                # 1. Variational MDL Coding
                log.log("    Running variational MDL coding...")
                var_results, optimal = variational_mdl_coding(
                    X_train, y_train, X_test, y_test, multiclass
                )
                
                log.result("Optimal λ", f"{optimal['lambda']:.2e}")
                log.result("Min codelength", f"{optimal['total_codelength']:.2f} bits")
                log.result("Compression ratio", f"{optimal['compression_ratio']:.3f}")
                
                for _, row in var_results.iterrows():
                    all_variational_results.append({
                        'model': model,
                        'layer': layer,
                        'task': task_name,
                        'task_name': task_info['name'],
                        'is_control': task_info.get('control', False),
                        'real_task': task_info.get('real_task', task_name),
                        'lambda': row['lambda'],
                        'C': row['C'],
                        'model_complexity': row['model_complexity'],
                        'train_misfit': row['train_misfit'],
                        'test_misfit': row['test_misfit'],
                        'total_codelength': row['total_codelength'],
                        'uniform_codelength': row['uniform_codelength'],
                        'compression_ratio': row['compression_ratio']
                    })
                
                # 2. Online MDL Coding
                log.log("    Running online MDL coding...")
                online_results = online_mdl_coding(
                    X_train, y_train, X_test, y_test, 
                    optimal['C'], multiclass
                )
                
                log.result("Online samples", len(online_results))
                
                for _, row in online_results.iterrows():
                    all_online_results.append({
                        'model': model,
                        'layer': layer,
                        'task': task_name,
                        'task_name': task_info['name'],
                        'is_control': task_info.get('control', False),
                        'real_task': task_info.get('real_task', task_name),
                        'data_fraction': row['data_fraction'],
                        'n_samples': row['n_samples'],
                        'test_misfit': row['test_misfit'],
                        'train_accuracy': row['train_accuracy'],
                        'test_accuracy': row['test_accuracy']
                    })
    
    var_df = pd.DataFrame(all_variational_results)
    online_df = pd.DataFrame(all_online_results)
    
    var_df.to_csv(config.OUTPUT_DIR / "results" / "variational_mdl_results.csv", index=False)
    online_df.to_csv(config.OUTPUT_DIR / "results" / "online_mdl_results.csv", index=False)
    
    log.log(f"\n✓ Saved MDL results")
    
    return var_df, online_df

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_variational_mdl_curves(var_df):
    """Plot MDL vs lambda curves for each task."""
    log.section("Creating Variational MDL Visualizations")
    
    real_tasks = var_df[~var_df['is_control']]['task'].unique()
    
    for task in real_tasks:
        task_name = var_df[var_df['task'] == task]['task_name'].iloc[0]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, layer in enumerate(config.LAYERS):
            ax = axes[idx]
            
            for model in config.MODELS:
                data = var_df[
                    (var_df['task'] == task) & 
                    (var_df['model'] == model) & 
                    (var_df['layer'] == layer)
                ]
                
                ax.plot(data['lambda'], data['total_codelength'], 
                       marker='o', label=f'{model.capitalize()}', linewidth=2)
            
            ax.set_xscale('log')
            ax.set_xlabel('λ (Regularization)', fontsize=11)
            ax.set_ylabel('Total Codelength (bits)', fontsize=11)
            ax.set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for idx, component in enumerate(['model_complexity', 'train_misfit', 'test_misfit']):
            ax = axes[3 + idx]
            
            for model in config.MODELS:
                data = var_df[
                    (var_df['task'] == task) & 
                    (var_df['model'] == model) & 
                    (var_df['layer'] == 18)
                ]
                
                ax.plot(data['lambda'], data[component], 
                       marker='o', label=f'{model.capitalize()}', linewidth=2)
            
            ax.set_xscale('log')
            ax.set_xlabel('λ (Regularization)', fontsize=11)
            ax.set_ylabel(f'{component.replace("_", " ").title()}', fontsize=11)
            ax.set_title(f'Layer 18: {component.replace("_", " ").title()}', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Variational MDL: {task_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "curves" / f"variational_mdl_{task}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        log.log(f"✓ Saved variational curve for {task}")

def plot_online_mdl_curves(online_df):
    """Plot online coding curves showing learning efficiency."""
    log.section("Creating Online MDL Visualizations")
    
    real_tasks = online_df[~online_df['is_control']]['task'].unique()
    
    for task in real_tasks:
        task_name = online_df[online_df['task'] == task]['task_name'].iloc[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, layer in enumerate(config.LAYERS):
            ax = axes[idx]
            
            for model in config.MODELS:
                data = online_df[
                    (online_df['task'] == task) & 
                    (online_df['model'] == model) & 
                    (online_df['layer'] == layer)
                ]
                
                ax.plot(data['data_fraction'] * 100, data['test_misfit'],
                       marker='o', label=f'{model.capitalize()}', linewidth=2)
            
            ax.set_xlabel('Training Data Used (%)', fontsize=11)
            ax.set_ylabel('Test Misfit (bits)', fontsize=11)
            ax.set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Online MDL: {task_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "curves" / f"online_mdl_{task}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        log.log(f"✓ Saved online curve for {task}")

def plot_mdl_comparison(var_df, online_df):
    """Create comparison plots for base vs instruct MDL."""
    log.section("Creating MDL Comparison Plots")
    
    optimal_mdl = var_df.loc[var_df.groupby(['model', 'layer', 'task'])['total_codelength'].idxmin()]
    
    real_tasks_mdl = optimal_mdl[~optimal_mdl['is_control']]
    
    # Plot 1: Optimal codelength comparison (layer 18)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    tasks = real_tasks_mdl['task'].unique()
    for idx, task in enumerate(tasks):
        if idx >= 3:
            break
            
        ax = axes[idx]
        task_data = real_tasks_mdl[(real_tasks_mdl['task'] == task) & 
                                    (real_tasks_mdl['layer'] == 18)]
        
        x = np.arange(len(config.MODELS))
        codelengths = [task_data[task_data['model'] == m]['total_codelength'].values[0] 
                      for m in config.MODELS]
        
        bars = ax.bar(x, codelengths, color=['steelblue', 'coral'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in config.MODELS])
        ax.set_ylabel('Optimal Codelength (bits)', fontsize=11)
        ax.set_title(task_data.iloc[0]['task_name'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, codelengths)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 50, 
                   f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Optimal MDL Codelength: Base vs Instruct (Layer 18)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "mdl_base_vs_instruct.png",
               dpi=300, bbox_inches='tight')
    plt.close()
    
    log.log("✓ Saved base vs instruct comparison")
    
    # Plot 2: Real vs Control MDL gap
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, task in enumerate(tasks):
        if idx >= 3:
            break
            
        ax = axes[idx]
        
        real_data = optimal_mdl[(optimal_mdl['task'] == task) & (optimal_mdl['layer'] == 18)]
        
        control_task = real_data.iloc[0]['task'] + '_control'
        control_data = optimal_mdl[(optimal_mdl['task'] == control_task) & 
                                    (optimal_mdl['layer'] == 18)]
        
        x = np.arange(len(config.MODELS))
        width = 0.35
        
        real_vals = [real_data[real_data['model'] == m]['total_codelength'].values[0] 
                    for m in config.MODELS]
        control_vals = [control_data[control_data['model'] == m]['total_codelength'].values[0] 
                       for m in config.MODELS]
        
        ax.bar(x - width/2, real_vals, width, label='Real Task', 
              alpha=0.8, color='steelblue')
        ax.bar(x + width/2, control_vals, width, label='Control Task',
              alpha=0.8, color='coral')
        
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in config.MODELS])
        ax.set_ylabel('Codelength (bits)', fontsize=11)
        ax.set_title(real_data.iloc[0]['task_name'], fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (real, ctrl) in enumerate(zip(real_vals, control_vals)):
            gap = real - ctrl
            ax.text(i, max(real, ctrl) + 100, f'Δ={gap:.0f}',
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Real vs Control MDL (Layer 18)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "mdl_real_vs_control.png",
               dpi=300, bbox_inches='tight')
    plt.close()
    
    log.log("✓ Saved real vs control comparison")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_mdl_summary(var_df, online_df):
    """Generate text summary of MDL findings."""
    log.section("Generating MDL Summary Report")
    
    optimal_mdl = var_df.loc[var_df.groupby(['model', 'layer', 'task'])['total_codelength'].idxmin()]
    
    lines = []
    lines.append("="*80)
    lines.append("MDL PROBING ANALYSIS - SUMMARY REPORT")
    lines.append("="*80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    lines.append("\n" + "-"*80)
    lines.append("VARIATIONAL MDL: OPTIMAL CODELENGTHS (Layer 18)")
    lines.append("-"*80)
    
    real_tasks = optimal_mdl[~optimal_mdl['is_control']]['task'].unique()
    
    for task in real_tasks:
        task_data = optimal_mdl[(optimal_mdl['task'] == task) & 
                                (optimal_mdl['layer'] == 18)]
        
        lines.append(f"\n{task_data.iloc[0]['task_name']}:")
        for _, row in task_data.iterrows():
            lines.append(f"  {row['model']:10s}: {row['total_codelength']:.1f} bits "
                        f"(λ={row['lambda']:.2e}, compression={row['compression_ratio']:.3f})")
    
    lines.append("\n" + "-"*80)
    lines.append("REAL VS CONTROL MDL GAPS (Layer 18)")
    lines.append("-"*80)
    
    for task in real_tasks:
        real_data = optimal_mdl[(optimal_mdl['task'] == task) & 
                                (optimal_mdl['layer'] == 18)]
        control_task = task + '_control'
        control_data = optimal_mdl[(optimal_mdl['task'] == control_task) & 
                                    (optimal_mdl['layer'] == 18)]
        
        if len(control_data) > 0:
            lines.append(f"\n{real_data.iloc[0]['task_name']}:")
            for model in config.MODELS:
                real_mdl = real_data[real_data['model'] == model]['total_codelength'].values[0]
                ctrl_mdl = control_data[control_data['model'] == model]['total_codelength'].values[0]
                gap = real_mdl - ctrl_mdl
                
                status = "✓ VALID" if gap < -100 else "⚠ WEAK" if gap < 0 else "✗ INVALID"
                lines.append(f"  {model:10s}: Real={real_mdl:.1f}, Control={ctrl_mdl:.1f}, "
                           f"Gap={gap:.1f} {status}")
    
    lines.append("\n" + "-"*80)
    lines.append("ONLINE MDL: LEARNING EFFICIENCY")
    lines.append("-"*80)
    
    for task in real_tasks:
        task_online = online_df[(online_df['task'] == task) & 
                                (online_df['data_fraction'] == 0.5) &
                                (online_df['layer'] == 18)]
        
        if len(task_online) > 0:
            lines.append(f"\n{task_online.iloc[0]['task_name']} (at 50% data):")
            for model in config.MODELS:
                model_data = task_online[task_online['model'] == model]
                if len(model_data) > 0:
                    misfit = model_data['test_misfit'].values[0]
                    acc = model_data['test_accuracy'].values[0]
                    lines.append(f"  {model:10s}: Misfit={misfit:.1f} bits, Accuracy={acc:.3f}")
    
    lines.append("\n" + "="*80)
    lines.append("KEY FINDINGS")
    lines.append("="*80)
    
    supp_task = 'suppression_vs_control'
    supp_data = optimal_mdl[(optimal_mdl['task'] == supp_task) & 
                            (optimal_mdl['layer'] == 18)]
    
    base_mdl = supp_data[supp_data['model'] == 'base']['total_codelength'].values[0]
    inst_mdl = supp_data[supp_data['model'] == 'instruct']['total_codelength'].values[0]
    diff = inst_mdl - base_mdl
    
    lines.append(f"\nSuppression Task (Layer 18):")
    lines.append(f"  Base MDL:     {base_mdl:.1f} bits")
    lines.append(f"  Instruct MDL: {inst_mdl:.1f} bits")
    lines.append(f"  Difference:   {diff:+.1f} bits")
    
    if abs(diff) < 50:
        lines.append(f"\n✓ IDENTICAL ENCODING EFFICIENCY")
        lines.append(f"  Base and instruct encode suppression groups with equal efficiency.")
        lines.append(f"  This supports the mechanistic hypothesis: knowledge is preserved,")
        lines.append(f"  but downstream access is blocked.")
    elif diff > 0:
        lines.append(f"\n⚠ DEGRADED ENCODING IN INSTRUCT")
        lines.append(f"  Instruct model requires more bits to encode suppression groups.")
        lines.append(f"  This suggests instruction-tuning may degrade representation quality.")
    else:
        lines.append(f"\n⚠ IMPROVED ENCODING IN INSTRUCT")
        lines.append(f"  Instruct model requires fewer bits to encode suppression groups.")
        lines.append(f"  Unexpected finding requiring further investigation.")
    
    lines.append("\n" + "="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)
    
    report_text = '\n'.join(lines)
    
    with open(config.OUTPUT_DIR / "MDL_SUMMARY.txt", 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    log.log("✓ Summary report saved")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution pipeline."""
    start_time = datetime.now()
    
    log.section("MDL Probing Pipeline")
    log.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        df, activations, train_mask, dev_mask, test_mask = load_data()
        
        df = create_control_labels(df)
        
        var_df, online_df = run_mdl_analysis(df, activations, train_mask, dev_mask, test_mask)
        
        plot_variational_mdl_curves(var_df)
        plot_online_mdl_curves(online_df)
        plot_mdl_comparison(var_df, online_df)
        
        generate_mdl_summary(var_df, online_df)
        
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
    log.log(f"  Results: {config.OUTPUT_DIR / 'results'}")
    log.log(f"  Curves: {config.OUTPUT_DIR / 'curves'}")
    log.log(f"  Plots: {config.OUTPUT_DIR / 'plots'}")

if __name__ == "__main__":
    main()