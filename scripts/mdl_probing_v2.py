#!/usr/bin/env python3
"""
MDL Probing: Information-Theoretic Analysis of RLHF Cultural Suppression

This script implements comprehensive MDL-based probing experiments to analyze
how RLHF affects cultural knowledge representations in language models.

Key experiments:
- Online Prequential Coding (Data Efficiency Analysis)
- Variational MDL with L0/L1/L2 Priors (Model Complexity)
- Fisher Information Matrix (Decision Boundary Analysis)
- Cross-Model Transfer MDL (Representational Isomorphism)
- Multi-Task Joint Compression (Single, Dual, and Triple Task Probing)
- Group-Stratified Analysis (Suppression/Enhancement/Control Groups)

Triple Entanglement Test:
Simultaneously probes State, Attribute, and Correctness predictions to test
whether aligned models maintain unified representations despite behavioral
suppression. Low joint compression with high semantic accuracy but low
correctness accuracy indicates policy-layer blocking rather than information
erasure.
"""

import os
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    CSV_PATH = Path("/home/anshulk/cultural-alignment-study/outputs/eda_results/tables/enhanced_dataset.csv")
    ACTIVATION_TEMPLATE = "/data/user_data/anshulk/cultural-alignment-study/activations/{model}_layer{layer}_activations.npy"
    OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/mdl_probing")
    
    MODELS = ['base', 'instruct']
    LAYERS = [8, 16, 24, 28]
    INPUT_DIM = 1536
    HIDDEN_DIM = 512
    
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ONLINE_CHUNKS = np.unique(np.concatenate([
        np.arange(0.02, 0.20, 0.02),
        np.arange(0.20, 1.01, 0.05)
    ]))
    
    PRIORS = ['l0', 'l1', 'l2']
    VAR_EPOCHS = 100
    VAR_BATCH = 1024
    VAR_LR = 1e-3
    L0_TEMP = 2.0 / 3.0
    L0_DROPRATE_INIT = 0.5
    
    ONLINE_TRAIN_ITERS = 5
    ONLINE_BATCH = 256
    ONLINE_LR = 0.05
    
    ISO_EPOCHS = 50
    ISO_BATCH = 1024
    
    @staticmethod
    def setup():
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (Config.OUTPUT_DIR / "data").mkdir(exist_ok=True)
        (Config.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
        (Config.OUTPUT_DIR / "logs").mkdir(exist_ok=True)
        
        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

Config.setup()

log_file = Config.OUTPUT_DIR / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(log_file, "a") as f:
        f.write(formatted + "\n")

log(f"Device: {Config.DEVICE}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==============================================================================
# PROBE ARCHITECTURES
# ==============================================================================

class L0VariationalLayer(nn.Module):
    """Concrete Dropout Layer for L0 Sparsity Regularization"""
    
    def __init__(self, in_dim, out_dim, weight_decay=1.0, droprate_init=0.5, temperature=2./3.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_decay = weight_decay
        self.temperature = temperature
        
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(0, 0.01))
        self.bias = nn.Parameter(torch.Tensor(out_dim).zero_())
        
        init_val = np.log(1 - droprate_init) - np.log(droprate_init)
        self.qz_loga = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(init_val, 1e-2))
    
    def quantile_concrete(self, u):
        """Inverse CDF for sampling from concrete distribution"""
        y = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.qz_loga) / self.temperature)
        return (y * 2 - 1.0).clamp(0, 1)
    
    def regularization(self):
        """Expected L0 norm for model complexity measurement"""
        target = torch.sigmoid(self.qz_loga - self.temperature * np.log(0.1 / 1.1))
        return torch.sum(target)
    
    def get_sparsity(self):
        """Calculate proportion of pruned parameters"""
        pruned = (self.qz_loga <= 0).float().sum()
        total = self.in_dim * self.out_dim
        return (pruned / total).item()
    
    def forward(self, x):
        if self.training:
            u = torch.rand_like(self.qz_loga)
            z = self.quantile_concrete(u)
        else:
            z = (self.qz_loga > 0).float()
        
        return torch.matmul(x, self.weight * z) + self.bias


class UniversalProbe(nn.Module):
    """Universal probe supporting single-task, multi-task, and all prior types"""
    
    def __init__(self, input_dim, task_dims, prior_type, dataset_size, use_bottleneck=False):
        super().__init__()
        self.prior_type = prior_type
        self.dataset_size = dataset_size
        self.task_dims = task_dims
        self.use_bottleneck = use_bottleneck
        
        if len(task_dims) > 1 and use_bottleneck:
            if prior_type == 'l0':
                self.body = L0VariationalLayer(input_dim, Config.HIDDEN_DIM, weight_decay=1.0/dataset_size)
            else:
                self.body = nn.Linear(input_dim, Config.HIDDEN_DIM)
            
            self.heads = nn.ModuleDict({
                task: nn.Linear(Config.HIDDEN_DIM, dim) for task, dim in task_dims.items()
            })
        else:
            self.body = nn.Identity()
            task_name = list(task_dims.keys())[0]
            task_dim = list(task_dims.values())[0]
            
            if prior_type == 'l0':
                self.heads = nn.ModuleDict({
                    task_name: L0VariationalLayer(input_dim, task_dim, weight_decay=1.0/dataset_size)
                })
            else:
                self.heads = nn.ModuleDict({
                    task_name: nn.Linear(input_dim, task_dim)
                })
    
    def forward(self, x):
        features = self.body(x)
        return {task: head(features) for task, head in self.heads.items()}
    
    def compute_loss(self, logits_dict, targets_dict):
        """Compute data cost (negative log-likelihood) and model cost (regularization)"""
        data_cost = 0
        for task, logits in logits_dict.items():
            data_cost += nn.functional.cross_entropy(logits, targets_dict[task], reduction='sum')
        
        model_cost = torch.tensor(0.0, device=data_cost.device)
        
        all_params = []
        
        if isinstance(self.body, L0VariationalLayer):
            model_cost += self.body.regularization()
        elif not isinstance(self.body, nn.Identity):
            all_params.extend(self.body.parameters())
        
        for head in self.heads.values():
            if isinstance(head, L0VariationalLayer):
                model_cost += head.regularization()
            else:
                all_params.extend(head.parameters())
        
        if self.prior_type == 'l1':
            l1_norm = sum(p.abs().sum() for p in all_params)
            model_cost += 1e-4 * self.dataset_size * l1_norm
        elif self.prior_type == 'l2':
            l2_norm = sum(p.pow(2).sum() for p in all_params)
            model_cost += 1e-4 * self.dataset_size * l2_norm
        
        return data_cost, model_cost
    
    def get_sparsity(self):
        """Calculate overall sparsity for L0 layers"""
        if self.prior_type != 'l0':
            return 0.0
        
        sparsities = []
        if isinstance(self.body, L0VariationalLayer):
            sparsities.append(self.body.get_sparsity())
        
        for head in self.heads.values():
            if isinstance(head, L0VariationalLayer):
                sparsities.append(head.get_sparsity())
        
        return np.mean(sparsities) if sparsities else 0.0


class SimpleOnlineProbe(nn.Module):
    """Lightweight linear probe for online coding experiments"""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# ==============================================================================
# DATA UTILITIES
# ==============================================================================

def load_layer_data(layer):
    """Load dataset and activations for specified layer"""
    log(f"Loading Layer {layer} data...")
    
    df = pd.read_csv(Config.CSV_PATH)
    
    if 'activation_idx' not in df.columns:
        raise ValueError("activation_idx column not found in dataset")
    
    if df['activation_idx'].isna().any():
        log(f"  WARNING: {df['activation_idx'].isna().sum()} rows missing activation indices")
        df = df.dropna(subset=['activation_idx'])
    
    df['activation_idx'] = df['activation_idx'].astype(int)
    
    activations = {}
    for model in Config.MODELS:
        path = Config.ACTIVATION_TEMPLATE.format(model=model, layer=layer)
        acts_full = np.load(path)
        activations[model] = acts_full[df['activation_idx'].values]
        log(f"  {model}: {activations[model].shape}")
    
    return df, activations


def encode_all_labels(df):
    """Encode categorical and binary task labels"""
    labels = {}
    dims = {}
    
    le_attr = LabelEncoder()
    labels['attribute'] = le_attr.fit_transform(df['attribute'])
    dims['attribute'] = len(le_attr.classes_)
    
    le_state = LabelEncoder()
    labels['state'] = le_state.fit_transform(df['state'])
    dims['state'] = len(le_state.classes_)
    
    labels['correctness_base'] = df['base_correct'].astype(int).values
    labels['correctness_instruct'] = df['instruct_correct'].astype(int).values
    dims['correctness'] = 2
    
    return labels, dims


def create_dataloaders(X, y_dict, batch_size, shuffle_data=True):
    """Create PyTorch dataloaders for multi-task learning"""
    X_tensor = torch.FloatTensor(X).to(Config.DEVICE)
    y_tensors = [torch.LongTensor(y).to(Config.DEVICE) for y in y_dict.values()]
    
    dataset = TensorDataset(X_tensor, *y_tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)
    
    return loader

# ==============================================================================
# METRIC COMPUTATIONS
# ==============================================================================

def compute_fisher_information(model, dataloader, task_names):
    """Compute diagonal Fisher Information Matrix for decision boundary analysis"""
    model.eval()
    fisher_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    for batch in dataloader:
        xb = batch[0]
        yb_dict = {task: batch[i+1] for i, task in enumerate(task_names)}
        
        model.zero_grad()
        logits = model(xb)
        
        loss = sum(nn.functional.cross_entropy(logits[task], yb_dict[task]) 
                   for task in task_names)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_diag[name] += param.grad.pow(2)
    
    num_samples = len(dataloader.dataset)
    fisher_means = {name: (f / num_samples).mean().item() for name, f in fisher_diag.items()}
    
    return np.mean(list(fisher_means.values()))


def compute_accuracy(model, dataloader, task_names):
    """Compute classification accuracy for all tasks"""
    model.eval()
    correct = {task: 0 for task in task_names}
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            xb = batch[0]
            yb_dict = {task: batch[i+1] for i, task in enumerate(task_names)}
            
            logits = model(xb)
            
            for task in task_names:
                preds = logits[task].argmax(dim=1)
                correct[task] += (preds == yb_dict[task]).sum().item()
            
            total += len(xb)
    
    return {task: correct[task] / total for task in task_names}

# ==============================================================================
# EXPERIMENT 1: ONLINE PREQUENTIAL CODING
# ==============================================================================

def experiment_online_coding(X, y_dict, task_dims, metadata):
    """Online MDL with prequential coding for data efficiency analysis"""
    X_shuf, indices = shuffle(X, np.arange(len(X)), random_state=Config.SEED)
    y_shuf = {task: y[indices] for task, y in y_dict.items()}
    
    X_tensor = torch.FloatTensor(X_shuf).to(Config.DEVICE)
    y_tensors = {task: torch.LongTensor(y).to(Config.DEVICE) for task, y in y_shuf.items()}
    
    task_names = list(task_dims.keys())
    num_classes = list(task_dims.values())[0] if len(task_dims) == 1 else Config.HIDDEN_DIM
    
    probe = SimpleOnlineProbe(Config.INPUT_DIM, num_classes).to(Config.DEVICE)
    optimizer = optim.SGD(probe.parameters(), lr=Config.ONLINE_LR)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    results = []
    cumulative_bits = 0.0
    prev_idx = 0
    
    N = len(X)
    
    for pct in tqdm(Config.ONLINE_CHUNKS, desc="Online Coding", leave=False):
        curr_idx = int(pct * N)
        if curr_idx <= prev_idx:
            continue
        
        X_eval = X_tensor[prev_idx:curr_idx]
        y_eval = list(y_tensors.values())[0][prev_idx:curr_idx]
        
        probe.eval()
        with torch.no_grad():
            logits = probe(X_eval)
            chunk_bits = criterion(logits, y_eval).item()
            acc = (logits.argmax(1) == y_eval).float().mean().item()
        
        cumulative_bits += chunk_bits
        
        probe.train()
        if curr_idx > 0:
            X_train = X_tensor[:curr_idx]
            y_train = list(y_tensors.values())[0][:curr_idx]
            
            for _ in range(Config.ONLINE_TRAIN_ITERS):
                if len(X_train) > Config.ONLINE_BATCH:
                    idx = torch.randperm(len(X_train))[:Config.ONLINE_BATCH]
                    X_batch = X_train[idx]
                    y_batch = y_train[idx]
                else:
                    X_batch = X_train
                    y_batch = y_train
                
                optimizer.zero_grad()
                loss = criterion(probe(X_batch), y_batch) / len(X_batch)
                loss.backward()
                optimizer.step()
        
        results.append({
            **metadata,
            'data_pct': pct,
            'chunk_bits': chunk_bits,
            'cumulative_bits': cumulative_bits,
            'bits_per_sample': cumulative_bits / curr_idx,
            'accuracy': acc
        })
        
        prev_idx = curr_idx
    
    return results

# ==============================================================================
# EXPERIMENT 2: VARIATIONAL MDL
# ==============================================================================

def experiment_variational_mdl(X, y_dict, task_dims, metadata):
    """Variational complexity analysis with L0/L1/L2 priors"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    task_names = list(task_dims.keys())
    loader = create_dataloaders(X_scaled, y_dict, Config.VAR_BATCH, shuffle_data=True)
    
    results = []
    
    for prior in Config.PRIORS:
        use_bottleneck = len(task_dims) > 1
        probe = UniversalProbe(Config.INPUT_DIM, task_dims, prior, len(X), use_bottleneck).to(Config.DEVICE)
        optimizer = optim.Adam(probe.parameters(), lr=Config.VAR_LR)
        
        epochs = Config.VAR_EPOCHS if prior == 'l0' else Config.VAR_EPOCHS // 2
        
        probe.train()
        for epoch in tqdm(range(epochs), desc=f"VAR-{prior}", leave=False):
            for batch in loader:
                xb = batch[0]
                yb_dict = {task: batch[i+1] for i, task in enumerate(task_names)}
                
                optimizer.zero_grad()
                logits = probe(xb)
                data_cost, model_cost = probe.compute_loss(logits, yb_dict)
                loss = data_cost + model_cost
                loss.backward()
                optimizer.step()
        
        probe.eval()
        
        total_data_cost = 0.0
        total_model_cost = 0.0
        
        with torch.no_grad():
            for batch in loader:
                xb = batch[0]
                yb_dict = {task: batch[i+1] for i, task in enumerate(task_names)}
                
                logits = probe(xb)
                dc, mc = probe.compute_loss(logits, yb_dict)
                total_data_cost += dc.item()
                total_model_cost += mc.item()
        
        accuracies = compute_accuracy(probe, loader, task_names)
        fisher_info = compute_fisher_information(probe, loader, task_names)
        sparsity = probe.get_sparsity()
        
        results.append({
            **metadata,
            'prior': prior,
            'data_cost': total_data_cost,
            'model_cost': total_model_cost,
            'total_mdl': total_data_cost + total_model_cost,
            'fisher_info': fisher_info,
            'sparsity': sparsity,
            **{f'acc_{task}': acc for task, acc in accuracies.items()}
        })
        
        del probe, optimizer
        torch.cuda.empty_cache()
    
    return results

# ==============================================================================
# EXPERIMENT 3: ISOMORPHISM TEST
# ==============================================================================

def experiment_isomorphism(df, activations, labels, layer):
    """Cross-model transfer MDL for representational isomorphism testing"""
    suppression_mask = df['group_type'] == 'suppression'
    
    if suppression_mask.sum() < 100:
        log(f"  Insufficient suppression samples ({suppression_mask.sum()}), skipping isomorphism")
        return []
    
    results = []
    
    tasks_to_test = [
        ('attribute', {'attribute': labels['attribute']}, {'attribute': 16}),
        ('state', {'state': labels['state']}, {'state': 36})
    ]
    
    for task_name, y_dict, task_dims in tasks_to_test:
        scaler = StandardScaler()
        X_base_scaled = scaler.fit_transform(activations['base'])
        
        task_list = list(task_dims.keys())
        y_base = {task: y for task, y in y_dict.items()}
        
        probe = UniversalProbe(Config.INPUT_DIM, task_dims, 'l0', len(X_base_scaled), use_bottleneck=False).to(Config.DEVICE)
        optimizer = optim.Adam(probe.parameters(), lr=Config.VAR_LR)
        
        loader = create_dataloaders(X_base_scaled, y_base, Config.ISO_BATCH, shuffle_data=True)
        
        probe.train()
        for epoch in tqdm(range(Config.ISO_EPOCHS), desc=f"ISO-{task_name}", leave=False):
            for batch in loader:
                xb = batch[0]
                yb_dict = {task: batch[i+1] for i, task in enumerate(task_list)}
                
                optimizer.zero_grad()
                logits = probe(xb)
                dc, mc = probe.compute_loss(logits, yb_dict)
                loss = dc + mc
                loss.backward()
                optimizer.step()
        
        probe.eval()
        
        X_instruct_scaled = scaler.transform(activations['instruct'])
        
        supp_indices = np.where(suppression_mask)[0]
        
        if len(supp_indices) == 0:
            log(f"  No suppression samples found for {task_name}, skipping")
            continue
        
        X_base_supp = torch.FloatTensor(X_base_scaled[supp_indices]).to(Config.DEVICE)
        X_instruct_supp = torch.FloatTensor(X_instruct_scaled[supp_indices]).to(Config.DEVICE)
        y_supp = {task: torch.LongTensor(y[supp_indices]).to(Config.DEVICE) for task, y in y_base.items()}
        
        with torch.no_grad():
            logits_base = probe(X_base_supp)
            dc_base, _ = probe.compute_loss(logits_base, y_supp)
            mdl_base = dc_base.item() / len(supp_indices)
            
            logits_instruct = probe(X_instruct_supp)
            dc_instruct, _ = probe.compute_loss(logits_instruct, y_supp)
            mdl_instruct = dc_instruct.item() / len(supp_indices)
            
            drift = mdl_instruct - mdl_base
        
        results.append({
            'layer': layer,
            'task': task_name,
            'mdl_base_self': mdl_base,
            'mdl_instruct_transfer': mdl_instruct,
            'drift': drift,
            'is_isomorphic': abs(drift) < 0.1
        })
        
        del probe, optimizer
        torch.cuda.empty_cache()
    
    return results

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_online_curves(df_online):
    """Plot online code length learning curves"""
    if len(df_online) == 0:
        log("  No online data to plot, skipping")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, layer in enumerate(Config.LAYERS):
        ax = axes[idx // 2, idx % 2]
        
        layer_data = df_online[df_online['layer'] == layer]
        
        if len(layer_data) == 0:
            ax.text(0.5, 0.5, f'No data for Layer {layer}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        for model in ['base', 'instruct']:
            for task in ['attribute', 'state', 'correctness_base', 'correctness_instruct']:
                subset = layer_data[(layer_data['model'] == model) & 
                                   (layer_data['task'] == task) &
                                   (layer_data['group'] == 'all')]
                
                if len(subset) > 0:
                    ax.plot(subset['data_pct'], subset['bits_per_sample'], 
                           label=f"{model}-{task}", alpha=0.7)
        
        ax.set_xlabel('Data Fraction', fontsize=10)
        ax.set_ylabel('Bits per Sample', fontsize=10)
        ax.set_title(f'Layer {layer}', fontsize=12)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "online_coding_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log("  Saved: online_coding_curves.png")


def plot_variational_comparison(df_var):
    """Plot variational MDL across priors"""
    if len(df_var) == 0:
        log("  No variational data to plot, skipping")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['data_cost', 'model_cost', 'total_mdl']
    titles = ['Data Cost (NLL)', 'Model Cost (Reg)', 'Total MDL']
    
    for ax, metric, title in zip(axes, metrics, titles):
        data = []
        labels_list = []
        
        for layer in Config.LAYERS:
            for model in Config.MODELS:
                subset = df_var[(df_var['layer'] == layer) & 
                               (df_var['model'] == model) &
                               (df_var['group'] == 'all') &
                               (df_var['task'] == 'attribute')]
                
                if len(subset) > 0:
                    for prior in Config.PRIORS:
                        val = subset[subset['prior'] == prior][metric].values
                        if len(val) > 0:
                            data.append(val[0])
                            labels_list.append(f"L{layer}\n{model}\n{prior}")
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        ax.bar(range(len(data)), data, color=sns.color_palette("husl", len(data)))
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels_list, rotation=60, ha='right', fontsize=6)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "variational_mdl_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log("  Saved: variational_mdl_comparison.png")


def plot_fisher_information(df_var):
    """Plot Fisher information across layers"""
    if len(df_var) == 0:
        log("  No variational data to plot Fisher, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plotted_any = False
    
    for model in Config.MODELS:
        for prior in Config.PRIORS:
            fisher_vals = []
            layers_list = []
            
            for layer in Config.LAYERS:
                for task in ['state', 'attribute']:
                    subset = df_var[(df_var['layer'] == layer) & 
                                   (df_var['model'] == model) &
                                   (df_var['prior'] == prior) &
                                   (df_var['group'] == 'all') &
                                   (df_var['task'] == task)]
                    
                    if len(subset) > 0:
                        fisher_vals.append(subset['fisher_info'].values[0])
                        layers_list.append(layer)
                        break
            
            if fisher_vals:
                ax.plot(layers_list, fisher_vals, marker='o', 
                       label=f"{model}-{prior}", linewidth=2)
                plotted_any = True
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'No data available', 
               ha='center', va='center', transform=ax.transAxes)
    else:
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Fisher Information', fontsize=12)
        ax.set_title('Decision Boundary Sharpness', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "fisher_information.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log("  Saved: fisher_information.png")


def plot_isomorphism_drift(df_iso):
    """Plot cross-model representational drift"""
    if len(df_iso) == 0:
        log("  No isomorphism data to plot, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = []
    drift_vals = []
    labels_list = []
    colors = []
    
    pos = 0
    for layer in Config.LAYERS:
        for task in ['attribute', 'state']:
            subset = df_iso[(df_iso['layer'] == layer) & (df_iso['task'] == task)]
            
            if len(subset) > 0:
                drift = subset['drift'].values[0]
                drift_vals.append(drift)
                x_pos.append(pos)
                labels_list.append(f"L{layer}\n{task}")
                colors.append('green' if abs(drift) < 0.1 else 'red')
                pos += 1
    
    if len(drift_vals) == 0:
        ax.text(0.5, 0.5, 'No data available', 
               ha='center', va='center', transform=ax.transAxes)
    else:
        ax.bar(x_pos, drift_vals, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=0.1, color='gray', linestyle=':', linewidth=1, label='Isomorphism Threshold')
        ax.axhline(y=-0.1, color='gray', linestyle=':', linewidth=1)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, rotation=45, ha='right')
        ax.set_ylabel('MDL Drift (Instruct - Base)', fontsize=12)
        ax.set_title('Cross-Model Representational Drift', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "isomorphism_drift.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log("  Saved: isomorphism_drift.png")


def plot_sparsity_analysis(df_var):
    """Plot L0 sparsity patterns across layers"""
    if len(df_var) == 0:
        log("  No variational data to plot sparsity, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    l0_data = df_var[df_var['prior'] == 'l0']
    
    if len(l0_data) == 0:
        ax.text(0.5, 0.5, 'No L0 data available', 
               ha='center', va='center', transform=ax.transAxes)
    else:
        plotted_any = False
        
        for model in Config.MODELS:
            sparsity_vals = []
            layers_list = []
            
            for layer in Config.LAYERS:
                for task in ['attribute', 'state']:
                    subset = l0_data[(l0_data['layer'] == layer) & 
                                    (l0_data['model'] == model) &
                                    (l0_data['group'] == 'all') &
                                    (l0_data['task'] == task)]
                    
                    if len(subset) > 0:
                        sparsity_vals.append(subset['sparsity'].values[0] * 100)
                        layers_list.append(layer)
                        break
            
            if sparsity_vals:
                ax.plot(layers_list, sparsity_vals, marker='s', 
                       label=model, linewidth=2, markersize=8)
                plotted_any = True
        
        if plotted_any:
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Sparsity (%)', fontsize=12)
            ax.set_title('L0 Pruning Patterns', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "sparsity_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log("  Saved: sparsity_analysis.png")


def plot_group_comparison(df_var):
    """Compare MDL across different sample groups"""
    if len(df_var) == 0:
        log("  No variational data to plot group comparison, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    available_layers = sorted(df_var['layer'].unique())
    layer = available_layers[len(available_layers)//2] if len(available_layers) > 0 else 16
    
    for idx, model in enumerate(Config.MODELS):
        ax = axes[idx]
        
        groups = ['all', 'suppression', 'enhancement']
        group_data = {g: [] for g in groups}
        
        for group in groups:
            for task in ['state', 'attribute']:
                subset = df_var[(df_var['layer'] == layer) & 
                               (df_var['model'] == model) &
                               (df_var['group'] == group) &
                               (df_var['prior'] == 'l0') &
                               (df_var['task'] == task)]
                
                if len(subset) > 0:
                    group_data[group].append(subset['total_mdl'].values[0])
                    break
            else:
                group_data[group].append(0)
        
        if any(group_data[g] for g in groups):
            x_pos = range(len(groups))
            heights = [group_data[g][0] if group_data[g] else 0 for g in groups]
            
            ax.bar(x_pos, heights, color=['blue', 'red', 'green'], alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(groups, rotation=45)
            ax.set_ylabel('Total MDL', fontsize=12)
            ax.set_title(f'{model.capitalize()} - Layer {layer}', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "group_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log("  Saved: group_comparison.png")


def plot_triple_entanglement(df_var):
    """
    Comprehensive analysis of joint compression for State, Attribute, and Correctness.
    
    Tests whether models maintain unified representations by examining compression
    efficiency when probing multiple tasks simultaneously versus independently.
    """
    if len(df_var) == 0:
        log("  No variational data to plot triple entanglement, skipping")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    available_layers = sorted(df_var['layer'].unique())
    layer = available_layers[len(available_layers)//2] if len(available_layers) > 0 else 16
    
    for col_idx, model in enumerate(Config.MODELS):
        ax = axes[0, col_idx]
        
        mdl_values = []
        task_labels = []
        colors = []
        
        for task in ['attribute', 'state']:
            for corr_task in [f'correctness_{model}']:
                combined_tasks = [task, corr_task]
                for t in combined_tasks:
                    subset = df_var[(df_var['layer'] == layer) & 
                                   (df_var['model'] == model) &
                                   (df_var['group'] == 'all') &
                                   (df_var['prior'] == 'l0') &
                                   (df_var['task'] == t)]
                    if len(subset) > 0:
                        mdl_values.append(subset['total_mdl'].values[0])
                        task_labels.append(t.replace('correctness_', 'corr_'))
                        colors.append('lightblue')
        
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == model) &
                       (df_var['group'] == 'all') &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == 'multitask_dual')]
        if len(subset) > 0:
            mdl_values.append(subset['total_mdl'].values[0])
            task_labels.append('Dual\n(S+A)')
            colors.append('orange')
        
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == model) &
                       (df_var['group'] == 'all') &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == f'multitask_triple_{model}')]
        if len(subset) > 0:
            mdl_values.append(subset['total_mdl'].values[0])
            task_labels.append('Triple\n(S+A+C)')
            colors.append('red')
        
        if mdl_values:
            x_pos = range(len(mdl_values))
            ax.bar(x_pos, mdl_values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Total MDL', fontsize=11)
            ax.set_title(f'{model.capitalize()} - Task Complexity', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    ax = axes[0, 2]
    
    compression_data = {'base': {}, 'instruct': {}}
    
    for model in Config.MODELS:
        single_sum = 0
        for task in ['attribute', 'state', f'correctness_{model}']:
            subset = df_var[(df_var['layer'] == layer) & 
                           (df_var['model'] == model) &
                           (df_var['group'] == 'all') &
                           (df_var['prior'] == 'l0') &
                           (df_var['task'] == task)]
            if len(subset) > 0:
                single_sum += subset['total_mdl'].values[0]
        
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == model) &
                       (df_var['group'] == 'all') &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == f'multitask_triple_{model}')]
        
        if len(subset) > 0 and single_sum > 0:
            triple_mdl = subset['total_mdl'].values[0]
            compression_data[model] = {
                'single_sum': single_sum,
                'triple': triple_mdl,
                'compression_ratio': triple_mdl / single_sum
            }
    
    if compression_data['base'] and compression_data['instruct']:
        models = ['Base', 'Instruct']
        x_pos = np.arange(len(models))
        width = 0.35
        
        single_sums = [compression_data['base']['single_sum'], 
                      compression_data['instruct']['single_sum']]
        triple_mdls = [compression_data['base']['triple'], 
                      compression_data['instruct']['triple']]
        
        ax.bar(x_pos - width/2, single_sums, width, label='Sum(Single Tasks)', 
              color='lightblue', alpha=0.7)
        ax.bar(x_pos + width/2, triple_mdls, width, label='Triple Task', 
              color='red', alpha=0.7)
        
        ax.set_ylabel('Total MDL', fontsize=11)
        ax.set_title('Compression: Single Sum vs Triple', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        for i, model in enumerate(['base', 'instruct']):
            ratio = compression_data[model]['compression_ratio']
            ax.text(i, max(single_sums + triple_mdls) * 1.05, 
                   f'Ratio: {ratio:.2f}', 
                   ha='center', fontsize=9, fontweight='bold',
                   color='green' if ratio < 0.9 else 'red')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', 
               ha='center', va='center', transform=ax.transAxes)
    
    for col_idx, model in enumerate(Config.MODELS):
        ax = axes[1, col_idx]
        
        layers_list = []
        triple_mdls = []
        
        for lyr in Config.LAYERS:
            subset = df_var[(df_var['layer'] == lyr) & 
                           (df_var['model'] == model) &
                           (df_var['group'] == 'all') &
                           (df_var['prior'] == 'l0') &
                           (df_var['task'] == f'multitask_triple_{model}')]
            
            if len(subset) > 0:
                layers_list.append(lyr)
                triple_mdls.append(subset['total_mdl'].values[0])
        
        if triple_mdls:
            ax.plot(layers_list, triple_mdls, marker='o', linewidth=3, 
                   markersize=10, color='red', label='Triple MDL')
            ax.set_xlabel('Layer', fontsize=11)
            ax.set_ylabel('Total MDL', fontsize=11)
            ax.set_title(f'{model.capitalize()} - Triple Task Across Layers', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    ax = axes[1, 2]
    
    analysis_data = []
    
    for group in ['all', 'suppression']:
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == 'instruct') &
                       (df_var['group'] == group) &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == 'multitask_triple_instruct')]
        
        if len(subset) > 0:
            row = subset.iloc[0]
            analysis_data.append({
                'group': group.capitalize(),
                'mdl': row['total_mdl'],
                'acc_attr': row.get('acc_attribute', 0) * 100,
                'acc_state': row.get('acc_state', 0) * 100,
                'acc_corr': row.get('acc_correctness_instruct', 0) * 100
            })
    
    if analysis_data:
        groups = [d['group'] for d in analysis_data]
        x_pos = np.arange(len(groups))
        width = 0.2
        
        mdls = [d['mdl'] for d in analysis_data]
        acc_attrs = [d['acc_attr'] for d in analysis_data]
        acc_states = [d['acc_state'] for d in analysis_data]
        acc_corrs = [d['acc_corr'] for d in analysis_data]
        
        max_mdl = max(mdls)
        mdls_norm = [(m/max_mdl)*100 for m in mdls]
        
        ax.bar(x_pos - 1.5*width, mdls_norm, width, label='MDL (norm)', 
              color='red', alpha=0.7)
        ax.bar(x_pos - 0.5*width, acc_attrs, width, label='Acc: Attribute', 
              color='blue', alpha=0.7)
        ax.bar(x_pos + 0.5*width, acc_states, width, label='Acc: State', 
              color='green', alpha=0.7)
        ax.bar(x_pos + 1.5*width, acc_corrs, width, label='Acc: Correctness', 
              color='orange', alpha=0.7)
        
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('Compression vs Accuracy Analysis\nSuppression Group', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups)
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        if len(analysis_data) > 1:
            supp_data = next((d for d in analysis_data if d['group'] == 'Suppression'), None)
            if supp_data and supp_data['acc_corr'] < 60 and mdls_norm[1] < 80:
                ax.text(0.5, 0.95, 
                       'Low MDL with low correctness detected',
                       transform=ax.transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No suppression data', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "triple_entanglement_analysis.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    log("  Saved: triple_entanglement_analysis.png")

# ==============================================================================
# TRIPLE ENTANGLEMENT METRICS
# ==============================================================================

def compute_triple_entanglement_metrics(df_var):
    """
    Compute comprehensive metrics for triple entanglement analysis.
    
    Extracts compression ratios, task accuracies, and group-stratified statistics
    for evaluating whether models maintain unified representations across semantic
    and behavioral dimensions.
    """
    if len(df_var) == 0:
        return {}
    
    available_layers = sorted(df_var['layer'].unique())
    layer = available_layers[len(available_layers)//2] if len(available_layers) > 0 else 16
    
    evidence = {}
    
    for model in Config.MODELS:
        model_evidence = {}
        
        single_sum = 0
        single_accs = {}
        
        for task in ['attribute', 'state', f'correctness_{model}']:
            subset = df_var[(df_var['layer'] == layer) & 
                           (df_var['model'] == model) &
                           (df_var['group'] == 'all') &
                           (df_var['prior'] == 'l0') &
                           (df_var['task'] == task)]
            
            if len(subset) > 0:
                single_sum += subset['total_mdl'].values[0]
                acc_col = f'acc_{task}'
                if acc_col in subset.columns:
                    single_accs[task] = subset[acc_col].values[0]
        
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == model) &
                       (df_var['group'] == 'all') &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == f'multitask_triple_{model}')]
        
        if len(subset) > 0 and single_sum > 0:
            row = subset.iloc[0]
            triple_mdl = row['total_mdl']
            
            model_evidence['compression_ratio'] = triple_mdl / single_sum
            model_evidence['triple_mdl'] = triple_mdl
            model_evidence['single_sum_mdl'] = single_sum
            
            model_evidence['acc_attribute'] = row.get('acc_attribute', 0)
            model_evidence['acc_state'] = row.get('acc_state', 0)
            model_evidence['acc_correctness'] = row.get(f'acc_correctness_{model}', 0)
        
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == model) &
                       (df_var['group'] == 'suppression') &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == f'multitask_triple_{model}')]
        
        if len(subset) > 0:
            row = subset.iloc[0]
            model_evidence['suppression_analysis'] = {
                'triple_mdl': row['total_mdl'],
                'acc_attribute': row.get('acc_attribute', 0),
                'acc_state': row.get('acc_state', 0),
                'acc_correctness': row.get(f'acc_correctness_{model}', 0),
                'data_cost': row.get('data_cost', 0),
                'model_cost': row.get('model_cost', 0)
            }
        
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == model) &
                       (df_var['group'] == 'enhancement') &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == f'multitask_triple_{model}')]
        
        if len(subset) > 0:
            row = subset.iloc[0]
            model_evidence['enhancement_analysis'] = {
                'triple_mdl': row['total_mdl'],
                'acc_attribute': row.get('acc_attribute', 0),
                'acc_state': row.get('acc_state', 0),
                'acc_correctness': row.get(f'acc_correctness_{model}', 0)
            }
        
        if model_evidence:
            evidence[model] = model_evidence
    
    if 'base' in evidence and 'instruct' in evidence:
        evidence['comparison'] = {
            'compression_ratio_diff': (
                evidence['instruct'].get('compression_ratio', 0) - 
                evidence['base'].get('compression_ratio', 0)
            ),
            'suppression_mdl_diff': 0,
            'suppression_correctness_diff': 0
        }
        
        if ('suppression_analysis' in evidence['base'] and 
            'suppression_analysis' in evidence['instruct']):
            
            base_supp = evidence['base']['suppression_analysis']
            inst_supp = evidence['instruct']['suppression_analysis']
            
            evidence['comparison']['suppression_mdl_diff'] = (
                inst_supp['triple_mdl'] - base_supp['triple_mdl']
            )
            evidence['comparison']['suppression_correctness_diff'] = (
                inst_supp['acc_correctness'] - base_supp['acc_correctness']
            )
            
            compression_similar = abs(evidence['comparison']['compression_ratio_diff']) < 0.15
            correctness_divergent = evidence['comparison']['suppression_correctness_diff'] < -0.2
            
            evidence['comparison']['policy_mask_indicator'] = bool(
                compression_similar and correctness_divergent
            )
    
    return evidence

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    log("="*80)
    log("STARTING MDL PROBING ")
    log("="*80)
    
    all_online = []
    all_variational = []
    all_isomorphism = []
    
    for layer in Config.LAYERS:
        log(f"\n{'='*60}")
        log(f"PROCESSING LAYER {layer}")
        log(f"{'='*60}")
        
        df, activations = load_layer_data(layer)
        labels, dims = encode_all_labels(df)
        
        groups_config = {
            'all': slice(None),
            'suppression': df['group_type'] == 'suppression',
            'enhancement': df['group_type'] == 'enhancement'
        }
        
        for group_name, mask in groups_config.items():
            if isinstance(mask, pd.Series) and mask.sum() < 50:
                log(f"  Skipping {group_name}: insufficient samples ({mask.sum()})")
                continue
            
            log(f"\n  Processing Group: {group_name}")
            
            if isinstance(mask, pd.Series):
                mask_array = mask.values
                group_acts = {m: acts[mask_array] for m, acts in activations.items()}
                group_labels = {k: v[mask_array] for k, v in labels.items()}
            else:
                group_acts = activations
                group_labels = labels
            
            tasks_config = [
                ('attribute', {'attribute': group_labels['attribute']}, {'attribute': dims['attribute']}),
                ('state', {'state': group_labels['state']}, {'state': dims['state']}),
                ('correctness_base', {'correctness_base': group_labels['correctness_base']}, 
                 {'correctness_base': dims['correctness']}),
                ('correctness_instruct', {'correctness_instruct': group_labels['correctness_instruct']}, 
                 {'correctness_instruct': dims['correctness']}),
                ('multitask_dual', {'attribute': group_labels['attribute'], 
                                    'state': group_labels['state']}, 
                 {'attribute': dims['attribute'], 'state': dims['state']}),
                ('multitask_triple_base', {
                    'attribute': group_labels['attribute'],
                    'state': group_labels['state'],
                    'correctness_base': group_labels['correctness_base']
                }, {
                    'attribute': dims['attribute'],
                    'state': dims['state'],
                    'correctness_base': dims['correctness']
                }),
                ('multitask_triple_instruct', {
                    'attribute': group_labels['attribute'],
                    'state': group_labels['state'],
                    'correctness_instruct': group_labels['correctness_instruct']
                }, {
                    'attribute': dims['attribute'],
                    'state': dims['state'],
                    'correctness_instruct': dims['correctness']
                })
            ]
            
            for model in Config.MODELS:
                X = group_acts[model]
                
                for task_name, y_dict, task_dims in tasks_config:
                    if 'correctness' in task_name and model not in task_name:
                        continue
                    
                    if 'multitask_triple' in task_name and model not in task_name:
                        continue
                    
                    log(f"    {model} - {task_name}")
                    
                    metadata = {
                        'layer': layer,
                        'model': model,
                        'task': task_name,
                        'group': group_name
                    }
                    
                    online_results = experiment_online_coding(X, y_dict, task_dims, metadata)
                    all_online.extend(online_results)
                    
                    var_results = experiment_variational_mdl(X, y_dict, task_dims, metadata)
                    all_variational.extend(var_results)
                    
                    if group_name == 'all' and task_name in ['attribute', 'state']:
                        y_control = {k: shuffle(v, random_state=Config.SEED) 
                                    for k, v in y_dict.items()}
                        
                        metadata_ctrl = metadata.copy()
                        metadata_ctrl['model'] = f"{model}_control"
                        
                        online_ctrl = experiment_online_coding(X, y_control, task_dims, metadata_ctrl)
                        all_online.extend(online_ctrl)
                        
                        var_ctrl = experiment_variational_mdl(X, y_control, task_dims, metadata_ctrl)
                        all_variational.extend(var_ctrl)
                    
                    gc.collect()
                    torch.cuda.empty_cache()
        
        log(f"\n  Running Isomorphism Test (Layer {layer})")
        iso_results = experiment_isomorphism(df, activations, labels, layer)
        all_isomorphism.extend(iso_results)
        
        del df, activations, labels
        gc.collect()
        torch.cuda.empty_cache()
    
    log("\n" + "="*80)
    log("SAVING RESULTS")
    log("="*80)
    
    df_online = pd.DataFrame(all_online)
    df_variational = pd.DataFrame(all_variational)
    df_isomorphism = pd.DataFrame(all_isomorphism)
    
    df_online.to_csv(Config.OUTPUT_DIR / "data" / "online_mdl.csv", index=False)
    df_variational.to_csv(Config.OUTPUT_DIR / "data" / "variational_mdl.csv", index=False)
    df_isomorphism.to_csv(Config.OUTPUT_DIR / "data" / "isomorphism.csv", index=False)
    
    log("  online_mdl.csv")
    log("  variational_mdl.csv")
    log("  isomorphism.csv")
    
    log("\n" + "="*80)
    log("GENERATING VISUALIZATIONS")
    log("="*80)
    
    plot_online_curves(df_online)
    plot_variational_comparison(df_variational)
    plot_fisher_information(df_variational)
    plot_isomorphism_drift(df_isomorphism)
    plot_sparsity_analysis(df_variational)
    plot_group_comparison(df_variational)
    plot_triple_entanglement(df_variational)
    
    log("\n" + "="*80)
    log("COMPUTING TRIPLE ENTANGLEMENT METRICS")
    log("="*80)
    
    entanglement_metrics = compute_triple_entanglement_metrics(df_variational)
    
    if entanglement_metrics:
        metrics_path = Config.OUTPUT_DIR / "data" / "triple_entanglement_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(entanglement_metrics, f, indent=2)
        
        log(f"  Saved: triple_entanglement_metrics.json")
        log("\n" + "="*60)
        log("TRIPLE ENTANGLEMENT ANALYSIS SUMMARY:")
        log("="*60)
        
        for model, stats in entanglement_metrics.items():
            if model in ['base', 'instruct']:
                log(f"\n{model.upper()} Model:")
                log(f"  Compression Ratio (Triple/Single): {stats.get('compression_ratio', 'N/A'):.3f}")
                log(f"  Triple Task Accuracy:")
                log(f"    - Attribute: {stats.get('acc_attribute', 0)*100:.1f}%")
                log(f"    - State: {stats.get('acc_state', 0)*100:.1f}%")
                log(f"    - Correctness: {stats.get('acc_correctness', 0)*100:.1f}%")
                
                if 'suppression_analysis' in stats:
                    supp = stats['suppression_analysis']
                    log(f"  Suppression Group:")
                    log(f"    - Triple MDL: {supp.get('triple_mdl', 'N/A'):.2f}")
                    log(f"    - Correctness Acc: {supp.get('acc_correctness', 0)*100:.1f}%")
                    
                    if model == 'instruct':
                        compression = stats.get('compression_ratio', 1.0)
                        corr_acc = supp.get('acc_correctness', 0)
                        
                        if compression < 0.9 and corr_acc < 0.6:
                            log(f"\n  Policy mask indicator detected:")
                            log(f"    Compression ratio: {compression:.3f} (efficient)")
                            log(f"    Correctness accuracy: {corr_acc*100:.1f}% (suppressed)")
                            log(f"    Pattern suggests output-layer blocking mechanism")
    else:
        log("  Insufficient data for triple entanglement analysis")
    
    log("\n" + "="*80)
    log("PIPELINE COMPLETE")
    log("="*80)
    
    log(f"\nTotal Experiments:")
    log(f"  Online Coding: {len(df_online)} data points")
    log(f"  Variational MDL: {len(df_variational)} configurations")
    log(f"  Isomorphism Tests: {len(df_isomorphism)} comparisons")
    
    log(f"\nOutputs saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise