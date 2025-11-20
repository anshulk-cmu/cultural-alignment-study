#!/usr/bin/env python3
"""
Causal Intervention Analysis: Layer-wise Activation Patching for RLHF Suppression Localization

This script performs interchange interventions to identify which layers causally mediate
cultural knowledge suppression in RLHF-aligned models.

Multi-level Analysis:
- Group-level: Suppression, Enhancement, Control
- Attribute-level: 16 cultural attributes
- State-level: 36 Indian states
- Combined: Fine-grained attribute × state interactions

Evaluation Metric: Answer entity probability - measures whether the correct answer
appears in top-k predictions when processing knowledge-bearing sentences.
"""

import os
import gc
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import multiprocessing as mp

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Data paths
    CSV_PATH = Path("/home/anshulk/cultural-alignment-study/outputs/eda_results/tables/enhanced_dataset.csv")
    ACTIVATION_INDEX_PATH = Path("/data/user_data/anshulk/cultural-alignment-study/activations/activation_index.csv")
    QUESTIONS_PATH = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/data/sanskriti_12k_targeted.csv")
    ACTIVATION_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/activations")

    # Output paths
    HEAVY_OUTPUT_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/causal_intervention")
    LIGHT_OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/causal_intervention")

    # Model paths
    BASE_MODEL = "/data/user_data/anshulk/cultural-alignment-study/qwen_models/Qwen2-1.5B"
    INSTRUCT_MODEL = "/data/user_data/anshulk/cultural-alignment-study/qwen_models/Qwen2-1.5B-Instruct"

    # Experimental setup
    LAYERS = [8, 16, 24, 28]
    BATCH_SIZE = 512
    MAX_SEQ_LENGTH = 256
    TOP_K = 10  # Check if answer appears in top-10 predictions

    # GPU configuration
    GPU_0 = 0
    GPU_1 = 1

    SEED = 42

    @staticmethod
    def setup():
        """Initialize directories and random seeds"""
        Config.HEAVY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.LIGHT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for subdir in ['data/layer_wise_results', 'data/group_level',
                       'data/attribute_level', 'data/state_level', 'data/combined_analysis',
                       'plots/group_level', 'plots/attribute_level',
                       'plots/state_level', 'plots/combined_analysis', 'logs']:
            (Config.LIGHT_OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)

Config.setup()

log_file = Config.LIGHT_OUTPUT_DIR / "logs" / f"intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg, gpu_id=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    gpu_str = f"[GPU {gpu_id}] " if gpu_id is not None else ""
    formatted = f"[{timestamp}] {gpu_str}{msg}"
    print(formatted)
    with open(log_file, "a") as f:
        f.write(formatted + "\n")

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_metadata():
    """Load dataset with annotations and link to answer entities"""
    log("Loading metadata...")

    # Load enhanced dataset
    df_enhanced = pd.read_csv(Config.CSV_PATH)

    # Load activation index
    df_index = pd.read_csv(Config.ACTIVATION_INDEX_PATH)

    # Load questions with answers
    df_questions = pd.read_csv(Config.QUESTIONS_PATH)

    # Merge to get answers
    df_merged = df_enhanced.merge(
        df_index[['activation_idx', 'row_id']],
        on='activation_idx',
        how='left'
    )

    df_final = df_merged.merge(
        df_questions[['state', 'attribute', 'answer']],
        left_on=['state', 'attribute'],
        right_on=['state', 'attribute'],
        how='left'
    )

    # Handle potential duplicates by taking first match
    df_final = df_final.drop_duplicates(subset='activation_idx', keep='first')

    log(f"  Total sentences: {len(df_final)}")
    log(f"  Suppression: {(df_final['group_type'] == 'suppression').sum()}")
    log(f"  Enhancement: {(df_final['group_type'] == 'enhancement').sum()}")
    log(f"  Control: {(df_final['group_type'] == 'control').sum()}")
    log(f"  Sentences with answers: {df_final['answer'].notna().sum()}")

    return df_final


def load_preextracted_activations(layer):
    """Load pre-extracted activations for given layer"""
    base_path = Config.ACTIVATION_DIR / f"base_layer{layer}_activations.npy"
    inst_path = Config.ACTIVATION_DIR / f"instruct_layer{layer}_activations.npy"

    base_acts = np.load(base_path)
    inst_acts = np.load(inst_path)

    return {
        'base': torch.from_numpy(base_acts).float(),
        'instruct': torch.from_numpy(inst_acts).float()
    }

# ==============================================================================
# INTERVENTION HOOK
# ==============================================================================

class ActivationPatchingHook:
    """Hook to intercept and replace activations at specific layer"""

    def __init__(self, layer_idx, replacement_activations, device):
        self.layer_idx = layer_idx
        self.replacement_activations = replacement_activations.to(device)
        self.device = device
        self.hook_handle = None
        self.current_batch_idx = 0

    def __call__(self, module, input, output):
        """Replace activations during forward pass"""
        hidden_states = output[0]
        batch_size = hidden_states.size(0)

        batch_replacements = self.replacement_activations[
            self.current_batch_idx:self.current_batch_idx + batch_size
        ]

        return (batch_replacements.to(hidden_states.dtype), ) + output[1:]

    def update_batch_idx(self, idx):
        self.current_batch_idx = idx

# ==============================================================================
# MODEL WRAPPER FOR INTERVENTION
# ==============================================================================

class InterventionModel:
    """Wrapper for model with intervention capabilities"""

    def __init__(self, model_name, device, layer_idx=None, replacement_acts=None):
        self.device = device
        self.layer_idx = layer_idx

        log(f"Loading {model_name.split('/')[-1]} on GPU {device}...", gpu_id=device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        self.hook = None
        if layer_idx is not None and replacement_acts is not None:
            self.setup_intervention(layer_idx, replacement_acts)

    def setup_intervention(self, layer_idx, replacement_acts):
        """Attach intervention hook to specified layer"""
        self.hook = ActivationPatchingHook(layer_idx, replacement_acts, self.device)

        target_layer = self.model.model.layers[layer_idx]
        self.hook.hook_handle = target_layer.register_forward_hook(self.hook)

    def remove_intervention(self):
        """Remove intervention hook"""
        if self.hook and self.hook.hook_handle:
            self.hook.hook_handle.remove()
            self.hook = None

    @torch.no_grad()
    def predict_answer_probability(self, sentences, answer_entities, batch_indices=None):
        """
        Evaluate if answer entities appear in top-k predictions

        Args:
            sentences: List of sentences
            answer_entities: List of answer strings
            batch_indices: Indices for tracking activation replacements

        Returns:
            results: List of dicts with answer presence and rank
        """
        if self.hook and batch_indices is not None:
            self.hook.update_batch_idx(batch_indices[0])

        inputs = self.tokenizer(
            sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last token logits

        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, Config.TOP_K, dim=-1)

        results = []

        for i, answer in enumerate(answer_entities):
            if pd.isna(answer):
                results.append({
                    'answer_in_top_k': False,
                    'answer_rank': -1,
                    'answer_probability': 0.0
                })
                continue

            # Tokenize answer
            answer_tokens = self.tokenizer.encode(str(answer), add_special_tokens=False)

            # Check if any answer token in top-k
            top_k_ids = top_k_indices[i].cpu().tolist()

            answer_found = False
            answer_rank = -1
            answer_prob = 0.0

            for answer_token_id in answer_tokens:
                if answer_token_id in top_k_ids:
                    answer_found = True
                    answer_rank = top_k_ids.index(answer_token_id) + 1
                    answer_prob = top_k_probs[i][answer_rank - 1].item()
                    break

            results.append({
                'answer_in_top_k': answer_found,
                'answer_rank': answer_rank,
                'answer_probability': answer_prob
            })

        return results

    def __del__(self):
        """Cleanup"""
        self.remove_intervention()
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

# ==============================================================================
# INTERVENTION EXPERIMENT RUNNER
# ==============================================================================

def run_intervention_for_layer(layer_idx, df, gpu_id, split_range=None):
    """
    Run intervention experiment for one layer on one GPU

    Args:
        layer_idx: Layer to intervene at
        df: Full dataframe
        gpu_id: GPU device ID
        split_range: (start_idx, end_idx) for data splitting

    Returns:
        results: List of dicts with intervention results
    """
    log(f"Starting Layer {layer_idx} intervention...", gpu_id=gpu_id)

    log(f"Loading pre-extracted activations for layer {layer_idx}...", gpu_id=gpu_id)
    activations = load_preextracted_activations(layer_idx)
    base_acts = activations['base']

    if split_range:
        start_idx, end_idx = split_range
        df_split = df.iloc[start_idx:end_idx].reset_index(drop=True)
        base_acts_split = base_acts[start_idx:end_idx]
    else:
        df_split = df
        base_acts_split = base_acts

    model = InterventionModel(
        Config.INSTRUCT_MODEL,
        device=gpu_id,
        layer_idx=layer_idx,
        replacement_acts=base_acts_split
    )

    sentences = df_split['sentence'].tolist()
    answer_entities = df_split['answer'].tolist()

    results = []

    num_batches = (len(sentences) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc=f"GPU{gpu_id} Layer{layer_idx}", leave=False):
        start_idx = batch_idx * Config.BATCH_SIZE
        end_idx = min(start_idx + Config.BATCH_SIZE, len(sentences))

        batch_sentences = sentences[start_idx:end_idx]
        batch_answers = answer_entities[start_idx:end_idx]

        batch_results = model.predict_answer_probability(
            batch_sentences,
            batch_answers,
            batch_indices=[start_idx]
        )

        for i, result in enumerate(batch_results):
            global_idx = start_idx + i
            if split_range:
                global_idx += split_range[0]

            results.append({
                'sentence_idx': global_idx,
                'layer': layer_idx,
                'answer_in_top_k': result['answer_in_top_k'],
                'answer_rank': result['answer_rank'],
                'answer_probability': result['answer_probability']
            })

        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    log(f"Completed Layer {layer_idx} intervention", gpu_id=gpu_id)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def run_parallel_intervention(layer_idx, df):
    """
    Run intervention on 2 GPUs in parallel

    Args:
        layer_idx: Layer to intervene at
        df: Full dataframe

    Returns:
        combined_results: Combined results from both GPUs
    """
    log(f"\n{'='*80}")
    log(f"LAYER {layer_idx} INTERVENTION (2-GPU PARALLEL)")
    log(f"{'='*80}")

    total_size = len(df)
    split_point = total_size // 2

    ctx = mp.get_context('spawn')
    queue_0 = ctx.Queue()
    queue_1 = ctx.Queue()

    def gpu_worker(gpu_id, split_range, queue):
        results = run_intervention_for_layer(layer_idx, df, gpu_id, split_range)
        queue.put(results)

    process_0 = ctx.Process(target=gpu_worker, args=(Config.GPU_0, (0, split_point), queue_0))
    process_1 = ctx.Process(target=gpu_worker, args=(Config.GPU_1, (split_point, total_size), queue_1))

    process_0.start()
    process_1.start()

    results_0 = queue_0.get()
    results_1 = queue_1.get()

    process_0.join()
    process_1.join()

    combined_results = results_0 + results_1

    log(f"Layer {layer_idx} complete: {len(combined_results)} results collected")

    return combined_results

# ==============================================================================
# BASELINE MEASUREMENT
# ==============================================================================

def measure_baseline_accuracy(df):
    """Measure baseline accuracy for base and instruct models"""
    log("\n" + "="*80)
    log("MEASURING BASELINE ACCURACY")
    log("="*80)

    baselines = {
        'overall': {},
        'by_group': {},
        'by_attribute': {},
        'by_state': {}
    }

    baselines['overall']['base'] = df['base_correct'].mean()
    baselines['overall']['instruct'] = df['instruct_correct'].mean()

    log(f"Overall - Base: {baselines['overall']['base']:.4f}, Instruct: {baselines['overall']['instruct']:.4f}")

    for group in ['suppression', 'enhancement', 'control']:
        group_df = df[df['group_type'] == group]
        baselines['by_group'][group] = {
            'base': group_df['base_correct'].mean(),
            'instruct': group_df['instruct_correct'].mean(),
            'count': len(group_df)
        }
        log(f"{group.capitalize()} - Base: {baselines['by_group'][group]['base']:.4f}, "
            f"Instruct: {baselines['by_group'][group]['instruct']:.4f}, "
            f"N: {baselines['by_group'][group]['count']}")

    for attr in df['attribute'].unique():
        attr_df = df[df['attribute'] == attr]
        baselines['by_attribute'][attr] = {
            'base': attr_df['base_correct'].mean(),
            'instruct': attr_df['instruct_correct'].mean(),
            'count': len(attr_df)
        }

    for state in df['state'].unique():
        state_df = df[df['state'] == state]
        baselines['by_state'][state] = {
            'base': state_df['base_correct'].mean(),
            'instruct': state_df['instruct_correct'].mean(),
            'count': len(state_df)
        }

    with open(Config.LIGHT_OUTPUT_DIR / 'data' / 'baseline_accuracy.json', 'w') as f:
        json.dump(baselines, f, indent=2)

    return baselines

# ==============================================================================
# MULTI-LEVEL ANALYSIS
# ==============================================================================

def compute_multi_level_metrics(df, all_layer_results, baselines):
    """
    Compute metrics at group, attribute, state, and combined levels

    Args:
        df: Metadata dataframe
        all_layer_results: Dict of {layer_idx: results_list}
        baselines: Baseline accuracy dict

    Returns:
        metrics: Comprehensive metrics dict
    """
    log("\n" + "="*80)
    log("COMPUTING MULTI-LEVEL METRICS")
    log("="*80)

    metrics = {
        'group_level': defaultdict(lambda: defaultdict(dict)),
        'attribute_level': defaultdict(lambda: defaultdict(dict)),
        'state_level': defaultdict(lambda: defaultdict(dict)),
        'combined_level': []
    }

    for layer_idx, results in all_layer_results.items():
        log(f"Processing Layer {layer_idx}...")

        results_df = pd.DataFrame(results)
        df_merged = df.copy()
        df_merged[f'layer{layer_idx}_answer_found'] = results_df['answer_in_top_k'].values

        for group in ['suppression', 'enhancement', 'control']:
            group_mask = df_merged['group_type'] == group

            base_acc = baselines['by_group'][group]['base']
            inst_acc = baselines['by_group'][group]['instruct']
            intervention_acc = df_merged[group_mask][f'layer{layer_idx}_answer_found'].mean()

            if base_acc != inst_acc:
                recovery_rate = (intervention_acc - inst_acc) / (base_acc - inst_acc)
            else:
                recovery_rate = 0.0

            metrics['group_level'][layer_idx][group] = {
                'baseline_base': base_acc,
                'baseline_instruct': inst_acc,
                'intervention_accuracy': intervention_acc,
                'recovery_rate': recovery_rate,
                'absolute_change': intervention_acc - inst_acc
            }

        for attr in df['attribute'].unique():
            attr_mask = df_merged['attribute'] == attr

            base_acc = baselines['by_attribute'][attr]['base']
            inst_acc = baselines['by_attribute'][attr]['instruct']
            intervention_acc = df_merged[attr_mask][f'layer{layer_idx}_answer_found'].mean()

            if base_acc != inst_acc:
                recovery_rate = (intervention_acc - inst_acc) / (base_acc - inst_acc)
            else:
                recovery_rate = 0.0

            metrics['attribute_level'][layer_idx][attr] = {
                'baseline_base': base_acc,
                'baseline_instruct': inst_acc,
                'intervention_accuracy': intervention_acc,
                'recovery_rate': recovery_rate,
                'count': attr_mask.sum()
            }

        for state in df['state'].unique():
            state_mask = df_merged['state'] == state

            base_acc = baselines['by_state'][state]['base']
            inst_acc = baselines['by_state'][state]['instruct']
            intervention_acc = df_merged[state_mask][f'layer{layer_idx}_answer_found'].mean()

            if base_acc != inst_acc:
                recovery_rate = (intervention_acc - inst_acc) / (base_acc - inst_acc)
            else:
                recovery_rate = 0.0

            metrics['state_level'][layer_idx][state] = {
                'baseline_base': base_acc,
                'baseline_instruct': inst_acc,
                'intervention_accuracy': intervention_acc,
                'recovery_rate': recovery_rate,
                'count': state_mask.sum()
            }

        for group in ['suppression', 'enhancement', 'control']:
            group_mask = df_merged['group_type'] == group

            for attr in df['attribute'].unique():
                for state in df['state'].unique():
                    combo_mask = group_mask & (df_merged['attribute'] == attr) & (df_merged['state'] == state)

                    if combo_mask.sum() >= 5:
                        intervention_acc = df_merged[combo_mask][f'layer{layer_idx}_answer_found'].mean()

                        combo_base = df_merged[combo_mask]['base_correct'].mean()
                        combo_inst = df_merged[combo_mask]['instruct_correct'].mean()

                        if combo_base != combo_inst:
                            recovery_rate = (intervention_acc - combo_inst) / (combo_base - combo_inst)
                        else:
                            recovery_rate = 0.0

                        metrics['combined_level'].append({
                            'layer': layer_idx,
                            'group': group,
                            'attribute': attr,
                            'state': state,
                            'baseline_base': combo_base,
                            'baseline_instruct': combo_inst,
                            'intervention_accuracy': intervention_acc,
                            'recovery_rate': recovery_rate,
                            'suppression_strength': combo_base - combo_inst,
                            'count': combo_mask.sum()
                        })

    log("Multi-level metrics computed")

    return metrics

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(all_layer_results, metrics):
    """Save all results to disk"""
    log("\n" + "="*80)
    log("SAVING RESULTS")
    log("="*80)

    for layer_idx, results in all_layer_results.items():
        results_df = pd.DataFrame(results)
        heavy_path = Config.HEAVY_OUTPUT_DIR / f'layer_{layer_idx}_full_results.csv'
        results_df.to_csv(heavy_path, index=False)
        log(f"Saved: {heavy_path}")

    group_data = []
    for layer_idx, groups in metrics['group_level'].items():
        for group, vals in groups.items():
            group_data.append({
                'layer': layer_idx,
                'group': group,
                **vals
            })
    pd.DataFrame(group_data).to_csv(
        Config.LIGHT_OUTPUT_DIR / 'data/group_level/group_recovery_rates.csv',
        index=False
    )

    attr_data = []
    for layer_idx, attrs in metrics['attribute_level'].items():
        for attr, vals in attrs.items():
            attr_data.append({
                'layer': layer_idx,
                'attribute': attr,
                **vals
            })
    pd.DataFrame(attr_data).to_csv(
        Config.LIGHT_OUTPUT_DIR / 'data/attribute_level/attribute_recovery_rates.csv',
        index=False
    )

    state_data = []
    for layer_idx, states in metrics['state_level'].items():
        for state, vals in states.items():
            state_data.append({
                'layer': layer_idx,
                'state': state,
                **vals
            })
    pd.DataFrame(state_data).to_csv(
        Config.LIGHT_OUTPUT_DIR / 'data/state_level/state_recovery_rates.csv',
        index=False
    )

    pd.DataFrame(metrics['combined_level']).to_csv(
        Config.LIGHT_OUTPUT_DIR / 'data/combined_analysis/state_x_attribute_heatmap_data.csv',
        index=False
    )

    combined_df = pd.DataFrame(metrics['combined_level'])
    suppression_df = combined_df[combined_df['group'] == 'suppression']
    top_20 = suppression_df.nlargest(20, 'suppression_strength')
    top_20.to_csv(
        Config.LIGHT_OUTPUT_DIR / 'data/combined_analysis/top_20_combinations.csv',
        index=False
    )

    log("All results saved")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_group_level_results(metrics):
    """Plot group-level recovery rates"""
    log("Generating group-level plots...")

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ['suppression', 'enhancement', 'control']
    colors = {'suppression': 'red', 'enhancement': 'green', 'control': 'blue'}

    for group in groups:
        recovery_rates = [
            metrics['group_level'][layer][group]['recovery_rate']
            for layer in Config.LAYERS
        ]
        ax.plot(Config.LAYERS, recovery_rates, marker='o', label=group.capitalize(),
                color=colors[group], linewidth=2, markersize=8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Recovery Rate', fontsize=12)
    ax.set_title('Intervention Recovery Rate by Layer and Group', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/group_level/recovery_rate_by_layer.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: recovery_rate_by_layer.png")


def plot_attribute_level_results(metrics):
    """Plot attribute-level heatmap"""
    log("Generating attribute-level plots...")

    attributes = sorted(set(
        attr for layer_metrics in metrics['attribute_level'].values()
        for attr in layer_metrics.keys()
    ))

    heatmap_data = np.zeros((len(attributes), len(Config.LAYERS)))

    for i, attr in enumerate(attributes):
        for j, layer in enumerate(Config.LAYERS):
            if attr in metrics['attribute_level'][layer]:
                heatmap_data[i, j] = metrics['attribute_level'][layer][attr]['recovery_rate']

    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[f'L{l}' for l in Config.LAYERS],
                yticklabels=attributes, ax=ax, cbar_kws={'label': 'Recovery Rate'})

    ax.set_title('Attribute Recovery Rates Across Layers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attribute', fontsize=12)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/attribute_level/attribute_recovery_by_layer.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: attribute_recovery_by_layer.png")


def plot_state_level_results(metrics):
    """Plot top states by suppression"""
    log("Generating state-level plots...")

    layer_28_data = []
    for state, vals in metrics['state_level'][28].items():
        suppression = vals['baseline_base'] - vals['baseline_instruct']
        layer_28_data.append({
            'state': state,
            'suppression_strength': suppression,
            'recovery_rate': vals['recovery_rate']
        })

    state_df = pd.DataFrame(layer_28_data).sort_values('suppression_strength', ascending=False).head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(state_df['state'], state_df['suppression_strength'], color='crimson', alpha=0.7)
    ax1.set_xlabel('Suppression Strength', fontsize=12)
    ax1.set_title('Top 15 States by Suppression Strength', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    ax2.barh(state_df['state'], state_df['recovery_rate'], color='forestgreen', alpha=0.7)
    ax2.set_xlabel('Recovery Rate (Layer 28)', fontsize=12)
    ax2.set_title('Recovery Rates for Top Suppressed States', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/state_level/state_suppression_ranking.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: state_suppression_ranking.png")


def plot_combined_analysis(metrics):
    """Plot top 20 combinations"""
    log("Generating combined analysis plots...")

    combined_df = pd.DataFrame(metrics['combined_level'])

    top_20 = combined_df[combined_df['group'] == 'suppression'].nlargest(20, 'suppression_strength')

    fig, ax = plt.subplots(figsize=(12, 10))

    y_labels = [f"{row['attribute'][:15]} + {row['state'][:15]}"
                for _, row in top_20.iterrows()]

    x_pos = np.arange(len(y_labels))

    width = 0.35
    ax.barh(x_pos - width/2, top_20['suppression_strength'], width,
            label='Suppression Strength', color='crimson', alpha=0.7)
    ax.barh(x_pos + width/2, top_20['recovery_rate'], width,
            label='Recovery Rate (L28)', color='forestgreen', alpha=0.7)

    ax.set_yticks(x_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Top 20 Attribute×State Combinations with Highest Suppression',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.LIGHT_OUTPUT_DIR / 'plots/combined_analysis/top_20_combinations_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    log("  Saved: top_20_combinations_comparison.png")


def generate_all_plots(metrics):
    """Generate all visualizations"""
    log("\n" + "="*80)
    log("GENERATING VISUALIZATIONS")
    log("="*80)

    plot_group_level_results(metrics)
    plot_attribute_level_results(metrics)
    plot_state_level_results(metrics)
    plot_combined_analysis(metrics)

    log("All plots generated")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    log("="*80)
    log("CAUSAL INTERVENTION ANALYSIS PIPELINE")
    log("="*80)
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    df = load_metadata()

    baselines = measure_baseline_accuracy(df)

    all_layer_results = {}

    for layer_idx in Config.LAYERS:
        results = run_parallel_intervention(layer_idx, df)
        all_layer_results[layer_idx] = results

        pd.DataFrame(results).to_csv(
            Config.HEAVY_OUTPUT_DIR / f'layer_{layer_idx}_full_results.csv',
            index=False
        )

        gc.collect()

    metrics = compute_multi_level_metrics(df, all_layer_results, baselines)

    save_results(all_layer_results, metrics)

    generate_all_plots(metrics)

    log("\n" + "="*80)
    log("PIPELINE COMPLETE")
    log("="*80)
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"\nResults saved to:")
    log(f"  Heavy data: {Config.HEAVY_OUTPUT_DIR}")
    log(f"  Light outputs: {Config.LIGHT_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
