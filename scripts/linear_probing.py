#!/usr/bin/env python3
"""
Linear Probing Analysis for Cultural Knowledge Suppression
===========================================================

Trains simple linear classifiers on frozen neural activations to measure
the linear decodability of:
1. Knowledge presence (base_correct, instruct_correct)
2. Suppression/enhancement/control group membership
3. Control tasks with shuffled labels for validation

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input paths
    ACTIVATION_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/Activations")
    INDEX_FILE = ACTIVATION_DIR / "activation_index.csv"
    
    # Output paths
    OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/linear_probing")
    HEAVY_DATA_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/Linear_Probing")
    
    # Models and layers
    MODELS = ['base', 'instruct']
    LAYERS = [6, 12, 18]
    
    # Train/dev/test split (at question level)
    TRAIN_RATIO = 0.7
    DEV_RATIO = 0.1
    TEST_RATIO = 0.2
    RANDOM_STATE = 42
    CONTROL_RANDOM_STATE = 123  # Different seed for control tasks
    
    # Probe settings
    PROBE_MAX_ITER = 1000
    PROBE_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.HEAVY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "results").mkdir(exist_ok=True)

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

log = Logger(config.OUTPUT_DIR / "probing_log.txt")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load activation matrices and labels."""
    log.section("Loading Data")
    
    # Load index with labels
    log.log("Loading activation index...")
    df = pd.read_csv(config.INDEX_FILE)
    log.result("Total sentences", len(df))
    log.result("Unique questions (row_ids)", df['row_id'].nunique())
    
    # Load all activation matrices
    activations = {}
    for model in config.MODELS:
        activations[model] = {}
        for layer in config.LAYERS:
            file_path = config.ACTIVATION_DIR / f"{model}_layer{layer}_activations.npy"
            log.log(f"Loading {model} layer {layer}...")
            acts = np.load(file_path)
            activations[model][layer] = acts
            log.result(f"  Shape", acts.shape)
            log.result(f"  Mean", f"{acts.mean():.4f}")
            log.result(f"  Std", f"{acts.std():.4f}")
    
    return df, activations

# ============================================================================
# DATA SPLITTING
# ============================================================================

def create_splits(df):
    """Create train/dev/test splits at question level."""
    log.section("Creating Train/Dev/Test Splits")
    
    # Get unique question IDs
    unique_questions = df['row_id'].unique()
    log.result("Total unique questions", len(unique_questions))
    
    # Split questions into train/dev/test
    train_questions, temp_questions = train_test_split(
        unique_questions, 
        test_size=(1 - config.TRAIN_RATIO),
        random_state=config.RANDOM_STATE
    )
    
    dev_questions, test_questions = train_test_split(
        temp_questions,
        test_size=config.TEST_RATIO / (config.DEV_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_STATE
    )
    
    # Create sentence-level masks
    train_mask = df['row_id'].isin(train_questions)
    dev_mask = df['row_id'].isin(dev_questions)
    test_mask = df['row_id'].isin(test_questions)
    
    log.result("Train questions", len(train_questions))
    log.result("Train sentences", train_mask.sum())
    log.result("Dev questions", len(dev_questions))
    log.result("Dev sentences", dev_mask.sum())
    log.result("Test questions", len(test_questions))
    log.result("Test sentences", test_mask.sum())
    
    # Verify no overlap
    assert len(set(train_questions) & set(dev_questions)) == 0
    assert len(set(train_questions) & set(test_questions)) == 0
    assert len(set(dev_questions) & set(test_questions)) == 0
    log.log("✓ No overlap between splits verified")
    
    # Save split info
    split_info = {
        'train_questions': train_questions.tolist(),
        'dev_questions': dev_questions.tolist(),
        'test_questions': test_questions.tolist(),
        'train_size': len(train_questions),
        'dev_size': len(dev_questions),
        'test_size': len(test_questions)
    }
    
    with open(config.HEAVY_DATA_DIR / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return train_mask, dev_mask, test_mask

# ============================================================================
# CONTROL TASK GENERATION
# ============================================================================

def create_control_labels(df):
    """Create shuffled labels for control tasks while preserving question-level grouping."""
    log.section("Creating Control Task Labels")
    
    df_control = df.copy()
    
    # Shuffle group_type labels at question level
    np.random.seed(config.CONTROL_RANDOM_STATE)
    question_groups = df.groupby('row_id')['group_type'].first()
    shuffled_groups = question_groups.sample(frac=1, random_state=config.CONTROL_RANDOM_STATE)
    group_mapping = dict(zip(question_groups.index, shuffled_groups.values))
    df_control['group_type_control'] = df_control['row_id'].map(group_mapping)
    
    # Shuffle correctness labels at question level
    for label_col in ['base_correct', 'instruct_correct']:
        question_labels = df.groupby('row_id')[label_col].first()
        shuffled_labels = question_labels.sample(frac=1, random_state=config.CONTROL_RANDOM_STATE)
        label_mapping = dict(zip(question_labels.index, shuffled_labels.values))
        df_control[f'{label_col}_control'] = df_control['row_id'].map(label_mapping)
    
    log.result("Control labels created", "✓")
    log.result("Original group distribution", df['group_type'].value_counts().to_dict())
    log.result("Control group distribution", df_control['group_type_control'].value_counts().to_dict())
    
    return df_control

# ============================================================================
# PROBING TASKS
# ============================================================================

def get_probing_tasks(df, model):
    """Define all probing tasks including control tasks for a given model."""
    tasks = {}
    
    # Task 1: Knowledge presence (model-specific)
    if model == 'base':
        tasks['knowledge'] = {
            'label': df['base_correct'].astype(int),
            'name': 'Base Model Correctness',
            'description': 'Predicting whether base model answered correctly',
            'control': False
        }
        tasks['knowledge_control'] = {
            'label': df['base_correct_control'].astype(int),
            'name': 'Base Model Correctness (Control)',
            'description': 'Control task with shuffled correctness labels',
            'control': True,
            'real_task': 'knowledge'
        }
    else:
        tasks['knowledge'] = {
            'label': df['instruct_correct'].astype(int),
            'name': 'Instruct Model Correctness',
            'description': 'Predicting whether instruct model answered correctly',
            'control': False
        }
        tasks['knowledge_control'] = {
            'label': df['instruct_correct_control'].astype(int),
            'name': 'Instruct Model Correctness (Control)',
            'description': 'Control task with shuffled correctness labels',
            'control': True,
            'real_task': 'knowledge'
        }
    
    # Task 2: Suppression vs Control
    suppression_mask = df['group_type'] == 'suppression'
    control_mask = df['group_type'] == 'control'
    supp_vs_ctrl_mask = suppression_mask | control_mask
    
    tasks['suppression_vs_control'] = {
        'label': df['group_type'].apply(lambda x: 1 if x == 'suppression' else 0),
        'mask': supp_vs_ctrl_mask,
        'name': 'Suppression vs Control',
        'description': 'Binary classification: suppression=1, control=0',
        'control': False
    }
    
    suppression_mask_ctrl = df['group_type_control'] == 'suppression'
    control_mask_ctrl = df['group_type_control'] == 'control'
    supp_vs_ctrl_mask_ctrl = suppression_mask_ctrl | control_mask_ctrl
    
    tasks['suppression_vs_control_control'] = {
        'label': df['group_type_control'].apply(lambda x: 1 if x == 'suppression' else 0),
        'mask': supp_vs_ctrl_mask_ctrl,
        'name': 'Suppression vs Control (Control)',
        'description': 'Control task with shuffled group labels',
        'control': True,
        'real_task': 'suppression_vs_control'
    }
    
    # Task 3: Enhancement vs Control
    enhancement_mask = df['group_type'] == 'enhancement'
    enh_vs_ctrl_mask = enhancement_mask | control_mask
    
    tasks['enhancement_vs_control'] = {
        'label': df['group_type'].apply(lambda x: 1 if x == 'enhancement' else 0),
        'mask': enh_vs_ctrl_mask,
        'name': 'Enhancement vs Control',
        'description': 'Binary classification: enhancement=1, control=0',
        'control': False
    }
    
    enhancement_mask_ctrl = df['group_type_control'] == 'enhancement'
    enh_vs_ctrl_mask_ctrl = enhancement_mask_ctrl | control_mask_ctrl
    
    tasks['enhancement_vs_control_control'] = {
        'label': df['group_type_control'].apply(lambda x: 1 if x == 'enhancement' else 0),
        'mask': enh_vs_ctrl_mask_ctrl,
        'name': 'Enhancement vs Control (Control)',
        'description': 'Control task with shuffled group labels',
        'control': True,
        'real_task': 'enhancement_vs_control'
    }
    
    # Task 4: Suppression vs Enhancement
    supp_vs_enh_mask = suppression_mask | enhancement_mask
    
    tasks['suppression_vs_enhancement'] = {
        'label': df['group_type'].apply(lambda x: 1 if x == 'suppression' else 0),
        'mask': supp_vs_enh_mask,
        'name': 'Suppression vs Enhancement',
        'description': 'Binary classification: suppression=1, enhancement=0',
        'control': False
    }
    
    supp_vs_enh_mask_ctrl = suppression_mask_ctrl | enhancement_mask_ctrl
    
    tasks['suppression_vs_enhancement_control'] = {
        'label': df['group_type_control'].apply(lambda x: 1 if x == 'suppression' else 0),
        'mask': supp_vs_enh_mask_ctrl,
        'name': 'Suppression vs Enhancement (Control)',
        'description': 'Control task with shuffled group labels',
        'control': True,
        'real_task': 'suppression_vs_enhancement'
    }
    
    # Task 5: Three-way classification
    group_map = {'suppression': 0, 'enhancement': 1, 'control': 2}
    tasks['group_3way'] = {
        'label': df['group_type'].map(group_map),
        'name': 'Group Type (3-way)',
        'description': 'Multi-class: suppression=0, enhancement=1, control=2',
        'multiclass': True,
        'control': False
    }
    
    tasks['group_3way_control'] = {
        'label': df['group_type_control'].map(group_map),
        'name': 'Group Type (3-way) (Control)',
        'description': 'Control task with shuffled group labels',
        'multiclass': True,
        'control': True,
        'real_task': 'group_3way'
    }
    
    return tasks

# ============================================================================
# PROBING FUNCTIONS
# ============================================================================

def train_probe(X_train, y_train, X_dev, y_dev, multiclass=False):
    """Train logistic regression probe with hyperparameter tuning."""
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    best_score = 0
    best_C = None
    best_probe = None
    
    # Hyperparameter search
    for C in config.PROBE_C_VALUES:
        if multiclass:
            probe = LogisticRegression(
                C=C,
                max_iter=config.PROBE_MAX_ITER,
                random_state=config.RANDOM_STATE,
                multi_class='multinomial',
                solver='lbfgs'
            )
        else:
            probe = LogisticRegression(
                C=C,
                max_iter=config.PROBE_MAX_ITER,
                random_state=config.RANDOM_STATE
            )
        
        probe.fit(X_train_scaled, y_train)
        score = probe.score(X_dev_scaled, y_dev)
        
        if score > best_score:
            best_score = score
            best_C = C
            best_probe = probe
    
    return best_probe, scaler, best_C, best_score

def evaluate_probe(probe, scaler, X_test, y_test, multiclass=False):
    """Evaluate probe on test set."""
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = probe.predict(X_test_scaled)
    y_pred_proba = probe.predict_proba(X_test_scaled)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro' if multiclass else 'binary')
    }
    
    if not multiclass:
        results['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    return results, y_pred, y_pred_proba

# ============================================================================
# MAIN PROBING PIPELINE
# ============================================================================

def run_probing_analysis(df, activations, train_mask, dev_mask, test_mask):
    """Run complete probing analysis for all model×layer×task combinations."""
    log.section("Running Probing Analysis")
    
    all_results = []
    
    for model in config.MODELS:
        log.subsection(f"Model: {model.upper()}")
        
        # Get tasks for this model
        tasks = get_probing_tasks(df, model)
        
        for layer in config.LAYERS:
            log.log(f"\nLayer {layer}:")
            
            # Get activations for this layer
            X = activations[model][layer]
            
            for task_name, task_info in tasks.items():
                log.log(f"  Task: {task_info['name']}")
                
                # Get labels
                y = task_info['label'].values
                
                # Apply task-specific mask if present
                if 'mask' in task_info:
                    task_mask = task_info['mask'].values
                    train_idx = train_mask & task_mask
                    dev_idx = dev_mask & task_mask
                    test_idx = test_mask & task_mask
                else:
                    train_idx = train_mask
                    dev_idx = dev_mask
                    test_idx = test_mask
                
                # Get splits
                X_train, y_train = X[train_idx], y[train_idx]
                X_dev, y_dev = X[dev_idx], y[dev_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                log.result("Train samples", len(y_train))
                log.result("Dev samples", len(y_dev))
                log.result("Test samples", len(y_test))
                
                # Check class distribution
                unique, counts = np.unique(y_train, return_counts=True)
                log.result("Train class dist", dict(zip(unique, counts)))
                
                # Train probe
                multiclass = task_info.get('multiclass', False)
                probe, scaler, best_C, dev_acc = train_probe(
                    X_train, y_train, X_dev, y_dev, multiclass
                )
                
                log.result("Best C", best_C)
                log.result("Dev accuracy", f"{dev_acc:.4f}")
                
                # Evaluate on test
                test_results, y_pred, y_pred_proba = evaluate_probe(
                    probe, scaler, X_test, y_test, multiclass
                )
                
                log.result("Test accuracy", f"{test_results['accuracy']:.4f}")
                log.result("Test F1", f"{test_results['f1']:.4f}")
                if 'auc' in test_results:
                    log.result("Test AUC", f"{test_results['auc']:.4f}")
                
                # Store results
                result_record = {
                    'model': model,
                    'layer': layer,
                    'task': task_name,
                    'task_name': task_info['name'],
                    'best_C': best_C,
                    'dev_accuracy': dev_acc,
                    'test_accuracy': test_results['accuracy'],
                    'test_f1': test_results['f1'],
                    'test_auc': test_results.get('auc', np.nan),
                    'train_size': len(y_train),
                    'test_size': len(y_test),
                    'multiclass': multiclass,
                    'is_control': task_info.get('control', False),
                    'real_task': task_info.get('real_task', task_name)
                }
                
                all_results.append(result_record)
                
                # Save detailed results for this probe
                detailed_results = {
                    'config': result_record,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                result_file = config.OUTPUT_DIR / "results" / f"{model}_layer{layer}_{task_name}.json"
                with open(result_file, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save summary
    results_df.to_csv(config.OUTPUT_DIR / "results" / "probing_results_summary.csv", index=False)
    log.log(f"\n✓ Saved results summary to {config.OUTPUT_DIR / 'results' / 'probing_results_summary.csv'}")
    
    return results_df

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results_df):
    """Create visualization plots for probing results."""
    log.section("Creating Visualizations")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Get real tasks only for some plots
    real_tasks_df = results_df[~results_df['is_control']]
    tasks = real_tasks_df['task'].unique()
    
    # 1. Heatmap: Accuracy by Model × Layer for each task
    for task in tasks:
        task_df = real_tasks_df[real_tasks_df['task'] == task]
        
        # Pivot for heatmap
        pivot_data = task_df.pivot(index='layer', columns='model', values='test_accuracy')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Test Accuracy'})
        ax.set_title(f'Probing Accuracy: {task_df.iloc[0]["task_name"]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / "plots" / f"heatmap_{task}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        log.log(f"✓ Saved heatmap for {task}")
    
    # 2. Line plot: Accuracy across layers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_df = real_tasks_df[real_tasks_df['task'] == task]
        
        for model in config.MODELS:
            model_df = task_df[task_df['model'] == model]
            ax.plot(model_df['layer'], model_df['test_accuracy'], 
                   marker='o', label=model.capitalize(), linewidth=2)
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title(task_df.iloc[0]['task_name'], fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(config.LAYERS)
        ax.set_ylim([0.4, 1.0])
    
    plt.suptitle('Probing Accuracy Across Layers and Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "accuracy_by_layer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log.log("✓ Saved layer comparison plot")
    
    # 3. Real vs Control comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        
        # Get real and control results for layer 18
        real_df = results_df[(results_df['task'] == task) & (results_df['layer'] == 18)]
        control_df = results_df[(results_df['real_task'] == task) & (results_df['is_control']) & (results_df['layer'] == 18)]
        
        if len(control_df) > 0:
            x = np.arange(len(config.MODELS))
            width = 0.35
            
            real_accs = [real_df[real_df['model'] == m]['test_accuracy'].values[0] for m in config.MODELS]
            control_accs = [control_df[control_df['model'] == m]['test_accuracy'].values[0] for m in config.MODELS]
            
            ax.bar(x - width/2, real_accs, width, label='Real Task', alpha=0.8, color='steelblue')
            ax.bar(x + width/2, control_accs, width, label='Control Task', alpha=0.8, color='coral')
            
            ax.set_ylabel('Test Accuracy', fontsize=11)
            ax.set_title(real_df.iloc[0]['task_name'], fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([m.capitalize() for m in config.MODELS])
            ax.legend()
            ax.set_ylim([0, 1.0])
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add delta text
            for i, (real, ctrl) in enumerate(zip(real_accs, control_accs)):
                delta = real - ctrl
                ax.text(i, max(real, ctrl) + 0.05, f'Δ={delta:.3f}', 
                       ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Real vs Control Tasks (Layer 18)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "real_vs_control_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log.log("✓ Saved real vs control comparison plot")
    
    # 4. Control task gap across layers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        
        for model in config.MODELS:
            real_df = results_df[(results_df['task'] == task) & (results_df['model'] == model)]
            control_df = results_df[(results_df['real_task'] == task) & (results_df['is_control']) & (results_df['model'] == model)]
            
            if len(control_df) > 0:
                layers = real_df['layer'].values
                real_accs = real_df['test_accuracy'].values
                control_accs = control_df['test_accuracy'].values
                gaps = real_accs - control_accs
                
                ax.plot(layers, gaps, marker='o', label=model.capitalize(), linewidth=2)
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Accuracy Gap (Real - Control)', fontsize=11)
        ax.set_title(real_df.iloc[0]['task_name'], fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(config.LAYERS)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Target Gap')
    
    plt.suptitle('Real-Control Accuracy Gap Across Layers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "control_gap_by_layer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log.log("✓ Saved control gap analysis plot")
    
    # 5. Task comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, model in enumerate(config.MODELS):
        ax = axes[idx]
        model_df = real_tasks_df[real_tasks_df['model'] == model]
        
        # Group by task and get layer 18 results
        layer18_df = model_df[model_df['layer'] == 18]
        
        x = np.arange(len(layer18_df))
        ax.bar(x, layer18_df['test_accuracy'], alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(layer18_df['task'], rotation=45, ha='right')
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{model.capitalize()} Model - Layer 18', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Task Performance Comparison (Layer 18)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "task_comparison_layer18.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log.log("✓ Saved task comparison plot")
    
    # 6. Base vs Instruct difference
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_df = real_tasks_df[real_tasks_df['task'] == task]
        
        # Pivot for both models
        pivot_data = task_df.pivot_table(index='layer', columns='model', values='test_accuracy')
        
        # Calculate difference (base - instruct)
        if 'base' in pivot_data.columns and 'instruct' in pivot_data.columns:
            diff = pivot_data['base'] - pivot_data['instruct']
            
            # Create bar plot
            diff.plot(kind='bar', ax=ax, color=['green' if x > 0 else 'red' for x in diff])
            ax.set_title(task_df.iloc[0]['task_name'], fontsize=12, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=11)
            ax.set_ylabel('Base - Instruct Accuracy', fontsize=11)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Base vs Instruct: Accuracy Difference (Positive = Base Better)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "base_vs_instruct_diff.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log.log("✓ Saved base vs instruct comparison plot")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(results_df):
    """Generate text summary of key findings."""
    log.section("Generating Summary Report")
    
    lines = []
    lines.append("="*80)
    lines.append("LINEAR PROBING ANALYSIS - SUMMARY REPORT")
    lines.append("="*80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal probes trained: {len(results_df)}")
    lines.append(f"Models: {', '.join(config.MODELS)}")
    lines.append(f"Layers: {', '.join(map(str, config.LAYERS))}")
    
    # Split real and control tasks
    real_df = results_df[~results_df['is_control']]
    control_df = results_df[results_df['is_control']]
    
    lines.append("\n" + "-"*80)
    lines.append("KEY FINDINGS")
    lines.append("-"*80)
    
    # Find best and worst performing probes (real tasks only)
    best_row = real_df.loc[real_df['test_accuracy'].idxmax()]
    worst_row = real_df.loc[real_df['test_accuracy'].idxmin()]
    
    lines.append(f"\nBest probe (real tasks):")
    lines.append(f"  {best_row['model']} layer {best_row['layer']} - {best_row['task_name']}")
    lines.append(f"  Accuracy: {best_row['test_accuracy']:.4f}")
    
    lines.append(f"\nWorst probe (real tasks):")
    lines.append(f"  {worst_row['model']} layer {worst_row['layer']} - {worst_row['task_name']}")
    lines.append(f"  Accuracy: {worst_row['test_accuracy']:.4f}")
    
    # Control task analysis
    lines.append("\n" + "-"*80)
    lines.append("CONTROL TASK VALIDATION (Layer 18)")
    lines.append("-"*80)
    
    real_tasks = real_df['task'].unique()
    for task in real_tasks:
        real_task_df = real_df[(real_df['task'] == task) & (real_df['layer'] == 18)]
        control_task_df = control_df[(control_df['real_task'] == task) & (control_df['layer'] == 18)]
        
        if len(control_task_df) > 0:
            lines.append(f"\n{real_task_df.iloc[0]['task_name']}:")
            for model in config.MODELS:
                real_acc = real_task_df[real_task_df['model'] == model]['test_accuracy'].values[0]
                ctrl_acc = control_task_df[control_task_df['model'] == model]['test_accuracy'].values[0]
                gap = real_acc - ctrl_acc
                
                status = "✓ VALID" if gap >= 0.15 else "⚠ WEAK" if gap >= 0.05 else "✗ INVALID"
                lines.append(f"  {model:10s}: Real={real_acc:.4f}, Control={ctrl_acc:.4f}, "
                           f"Gap={gap:.4f} {status}")
    
    # Task-specific insights
    lines.append("\n" + "-"*80)
    lines.append("TASK-SPECIFIC RESULTS (Layer 18)")
    lines.append("-"*80)
    
    for task in real_tasks:
        task_df = real_df[(real_df['task'] == task) & (real_df['layer'] == 18)]
        
        if len(task_df) > 0:
            lines.append(f"\n{task_df.iloc[0]['task_name']}:")
            for _, row in task_df.iterrows():
                auc_str = f"{row['test_auc']:.4f}" if not np.isnan(row['test_auc']) else 'N/A'
                lines.append(f"  {str(row['model']):10s}: Acc={row['test_accuracy']:.4f}, AUC={auc_str}")
    
    # Layer progression analysis
    lines.append("\n" + "-"*80)
    lines.append("LAYER PROGRESSION (Knowledge Task)")
    lines.append("-"*80)
    
    knowledge_df = real_df[real_df['task'] == 'knowledge']
    for model in config.MODELS:
        model_df = knowledge_df[knowledge_df['model'] == model].sort_values('layer')
        lines.append(f"\n{model.capitalize()} Model:")
        for _, row in model_df.iterrows():
            lines.append(f"  Layer {row['layer']}: {row['test_accuracy']:.4f}")
    
    lines.append("\n" + "="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)
    
    report_text = '\n'.join(lines)
    
    # Save report
    with open(config.OUTPUT_DIR / "PROBING_SUMMARY.txt", 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    log.log("✓ Summary report saved")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution pipeline."""
    start_time = datetime.now()
    
    log.section("Linear Probing Pipeline")
    log.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load data
        df, activations = load_data()
        
        # Create control labels
        df = create_control_labels(df)
        
        # Create splits
        train_mask, dev_mask, test_mask = create_splits(df)
        
        # Run probing analysis
        results_df = run_probing_analysis(df, activations, train_mask, dev_mask, test_mask)
        
        # Create visualizations
        create_visualizations(results_df)
        
        # Generate summary
        generate_summary_report(results_df)
        
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
    log.log(f"  Plots: {config.OUTPUT_DIR / 'plots'}")

if __name__ == "__main__":
    main()