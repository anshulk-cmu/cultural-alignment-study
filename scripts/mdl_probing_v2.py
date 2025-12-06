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
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
    TEST_SIZE = 0.25
    
    SEED = 42
    FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"

    @staticmethod
    def get_device():
        """Get available device with robust GPU check and fallback."""
        if Config.FORCE_CPU:
            return torch.device("cpu")

        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            # Test if GPU is actually usable
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"[WARNING] GPU unavailable ({e}), falling back to CPU")
            return torch.device("cpu")

    DEVICE = None  # Will be set during setup()
    
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

        # Set device with robust GPU check
        Config.DEVICE = Config.get_device()

        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        if Config.DEVICE.type == "cuda":
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
if Config.DEVICE.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    log("Running on CPU (GPU unavailable or FORCE_CPU=1)")

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

        # Task-specific loss weights to balance gradient contributions
        task_weights = {
            'attribute': 1.0,      # 16 classes
            'state': 2.25,         # 36 classes (36/16 = 2.25)
            'correctness_base': 0.125,   # 2 classes (2/16 = 0.125)
            'correctness_instruct': 0.125  # 2 classes
        }

        data_cost = 0
        for task, logits in logits_dict.items():
            task_loss = nn.functional.cross_entropy(logits, targets_dict[task], reduction='sum')

            # Apply weight based on task complexity (proportional to num_classes)
            weight = task_weights.get(task, 1.0)
            data_cost += weight * task_loss
        
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


class MultiHeadOnlineProbe(nn.Module):
    """Simple shared-body probe with separate linear heads for multi-task online coding"""

    def __init__(self, input_dim, task_dims):
        super().__init__()
        self.heads = nn.ModuleDict({task: nn.Linear(input_dim, dim) for task, dim in task_dims.items()})

    def forward(self, x):
        return {task: head(x) for task, head in self.heads.items()}

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


def create_train_test_split(df, stratify_col='group_type'):
    """Create deterministic stratified train/test split aligned with linear probing."""
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=Config.TEST_SIZE,
        random_state=Config.SEED,
        stratify=df[stratify_col]
    )
    return np.array(train_idx), np.array(test_idx)


def encode_all_labels(df):
    """Encode categorical and binary task labels and return encoders for name lookups"""
    labels = {}
    dims = {}
    encoders = {}
    
    le_attr = LabelEncoder()
    labels['attribute'] = le_attr.fit_transform(df['attribute'])
    dims['attribute'] = len(le_attr.classes_)
    encoders['attribute'] = le_attr
    
    le_state = LabelEncoder()
    labels['state'] = le_state.fit_transform(df['state'])
    dims['state'] = len(le_state.classes_)
    encoders['state'] = le_state
    
    labels['correctness_base'] = df['base_correct'].astype(int).values
    labels['correctness_instruct'] = df['instruct_correct'].astype(int).values
    dims['correctness'] = 2
    
    return labels, dims, encoders


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
# STATISTICAL UTILITIES
# ==============================================================================

def bootstrap_ci_array(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, random_state: int = 42) -> Dict:
    """Bootstrap CI for scalar values (e.g., MDL, Fisher).

    Returns point estimate, CI bounds, std_error.
    """
    rng = np.random.RandomState(random_state)
    values = np.array(values)
    if len(values) == 0:
        return {
            'point_estimate': None,
            'ci_lower': None,
            'ci_upper': None,
            'ci_level': ci,
            'std_error': None,
            'n_bootstrap': n_bootstrap
        }

    point_estimate = float(np.mean(values))
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(values), size=len(values), replace=True)
        boot.append(np.mean(values[idx]))
    boot = np.array(boot)
    alpha = 1 - ci
    return {
        'point_estimate': point_estimate,
        'ci_lower': float(np.percentile(boot, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(boot, 100 * (1 - alpha / 2))),
        'ci_level': ci,
        'std_error': float(np.std(boot)),
        'n_bootstrap': n_bootstrap
    }


def cohens_d_array(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)

# ==============================================================================
# EXPERIMENT 1: ONLINE PREQUENTIAL CODING
# ==============================================================================

def experiment_online_coding(
    X_train,
    y_train_dict,
    task_dims,
    metadata,
    X_test=None,
    y_test_dict=None
):
    """Online MDL with prequential coding for data efficiency analysis.

    Trains sequentially on the training set (shuffled once) and optionally
    evaluates final model on a held-out test set to avoid optimistic estimates.
    """
    X_shuf, indices = shuffle(X_train, np.arange(len(X_train)), random_state=Config.SEED)
    y_shuf = {task: y[indices] for task, y in y_train_dict.items()}

    X_tensor = torch.FloatTensor(X_shuf).to(Config.DEVICE)
    y_tensors = {task: torch.LongTensor(y).to(Config.DEVICE) for task, y in y_shuf.items()}
    
    task_names = list(task_dims.keys())
    is_multi = len(task_dims) > 1

    if is_multi:
        probe = MultiHeadOnlineProbe(Config.INPUT_DIM, task_dims).to(Config.DEVICE)
    else:
        num_classes = list(task_dims.values())[0]
        probe = SimpleOnlineProbe(Config.INPUT_DIM, num_classes).to(Config.DEVICE)

    optimizer = optim.SGD(probe.parameters(), lr=Config.ONLINE_LR)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    results = []
    cumulative_bits = 0.0
    prev_idx = 0
    
    N = len(X_train)
    
    for pct in tqdm(Config.ONLINE_CHUNKS, desc="Online Coding", leave=False):
        curr_idx = int(pct * N)
        if curr_idx <= prev_idx:
            continue
        
        X_eval = X_tensor[prev_idx:curr_idx]
        probe.eval()
        with torch.no_grad():
            if is_multi:
                logits = probe(X_eval)
                chunk_bits = 0.0
                accs = []
                for task in task_names:
                    y_eval_t = y_tensors[task][prev_idx:curr_idx]
                    lgt = logits[task]
                    chunk_bits += criterion(lgt, y_eval_t).item()
                    accs.append((lgt.argmax(1) == y_eval_t).float().mean().item())
                acc = float(np.mean(accs)) if accs else 0.0
            else:
                y_eval = list(y_tensors.values())[0][prev_idx:curr_idx]
                logits = probe(X_eval)
                chunk_bits = criterion(logits, y_eval).item()
                acc = (logits.argmax(1) == y_eval).float().mean().item()
        
        cumulative_bits += chunk_bits
        
        probe.train()
        if curr_idx > 0:
            X_train_t = X_tensor[:curr_idx]
            y_train_tensors = {task: y[:curr_idx] for task, y in y_tensors.items()}
            
            for _ in range(Config.ONLINE_TRAIN_ITERS):
                if len(X_train_t) > Config.ONLINE_BATCH:
                    idx = torch.randperm(len(X_train_t))[:Config.ONLINE_BATCH]
                    X_batch = X_train_t[idx]
                else:
                    idx = torch.arange(len(X_train_t))
                    X_batch = X_train_t

                optimizer.zero_grad()
                if is_multi:
                    logits = probe(X_batch)
                    loss = 0.0
                    for task in task_names:
                        y_batch = y_train_tensors[task][idx]
                        loss = loss + criterion(logits[task], y_batch) / len(X_batch)
                else:
                    y_train_single = list(y_tensors.values())[0][:curr_idx]
                    y_batch = y_train_single[idx]
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
    
    # Optional held-out evaluation using the final probe
    if X_test is not None and y_test_dict is not None and len(X_test) > 0:
        probe.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(Config.DEVICE)
        with torch.no_grad():
            if is_multi:
                logits = probe(X_test_tensor)
                test_bits = 0.0
                accs = []
                for task in task_names:
                    y_test_tensor = torch.LongTensor(y_test_dict[task]).to(Config.DEVICE)
                    lgt = logits[task]
                    test_bits += criterion(lgt, y_test_tensor).item()
                    accs.append((lgt.argmax(1) == y_test_tensor).float().mean().item())
                test_acc = float(np.mean(accs)) if accs else 0.0
            else:
                y_test_tensor = torch.LongTensor(list(y_test_dict.values())[0]).to(Config.DEVICE)
                logits = probe(X_test_tensor)
                test_bits = criterion(logits, y_test_tensor).item()
                test_acc = (logits.argmax(1) == y_test_tensor).float().mean().item()
        results.append({
            **metadata,
            'data_pct': 'test',
            'chunk_bits': test_bits,
            'cumulative_bits': cumulative_bits,
            'bits_per_sample': test_bits / len(X_test_tensor),
            'accuracy': test_acc,
            'eval_split': 'test'
        })

    return results

# ==============================================================================
# EXPERIMENT 2: VARIATIONAL MDL
# ==============================================================================

def experiment_variational_mdl(
    X_train,
    y_train_dict,
    X_test,
    y_test_dict,
    task_dims,
    metadata,
    attr_classes=None,
    state_classes=None
):
    """Variational complexity analysis with L0/L1/L2 priors using held-out test eval."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None and len(X_test) > 0 else None

    task_names = list(task_dims.keys())
    loader_train = create_dataloaders(X_train_scaled, y_train_dict, Config.VAR_BATCH, shuffle_data=True)
    loader_test = create_dataloaders(X_test_scaled, y_test_dict, Config.VAR_BATCH, shuffle_data=False) if X_test_scaled is not None else None
    
    results = []
    
    for prior in Config.PRIORS:
        use_bottleneck = len(task_dims) > 1
        probe = UniversalProbe(Config.INPUT_DIM, task_dims, prior, len(X_train), use_bottleneck).to(Config.DEVICE)
        optimizer = optim.Adam(probe.parameters(), lr=Config.VAR_LR)
        
        epochs = Config.VAR_EPOCHS if prior == 'l0' else Config.VAR_EPOCHS // 2
        
        probe.train()
        for epoch in tqdm(range(epochs), desc=f"VAR-{prior}", leave=False):
            for batch in loader_train:
                xb = batch[0]
                yb_dict = {task: batch[i+1] for i, task in enumerate(task_names)}
                
                optimizer.zero_grad()
                logits = probe(xb)
                data_cost, model_cost = probe.compute_loss(logits, yb_dict)
                loss = data_cost + model_cost
                loss.backward()
                optimizer.step()
        
        probe.eval()
        
        total_data_cost_train = 0.0
        total_model_cost = 0.0
        batch_bits_train = []
        batch_acc_train = []

        with torch.no_grad():
            for batch in loader_train:
                xb = batch[0]
                yb_dict = {task: batch[i+1] for i, task in enumerate(task_names)}
                
                logits = probe(xb)
                dc, mc = probe.compute_loss(logits, yb_dict)
                total_data_cost_train += dc.item()
                total_model_cost += mc.item()

                # Per-batch bits per sample and accuracy for CIs
                bits_per_sample = dc.item() / len(xb)
                batch_bits_train.append(bits_per_sample)
                batch_acc_train.append(np.mean([
                    (logits[task].argmax(dim=1) == yb_dict[task]).float().mean().item()
                    for task in task_names
                ]))

        # Evaluate on test split if available
        total_data_cost_test = None
        accuracies_test = None
        batch_bits_test = []
        batch_acc_test = []
        per_attr_acc = None
        per_state_acc = None
        if loader_test is not None:
            total_data_cost_test = 0.0
            y_true_attr, y_pred_attr = [], []
            y_true_state, y_pred_state = [], []
            with torch.no_grad():
                for batch in loader_test:
                    xb = batch[0]
                    yb_dict = {task: batch[i+1] for i, task in enumerate(task_names)}
                    logits = probe(xb)
                    dc, _ = probe.compute_loss(logits, yb_dict)
                    total_data_cost_test += dc.item()

                    bits_per_sample = dc.item() / len(xb)
                    batch_bits_test.append(bits_per_sample)
                    batch_acc_test.append(np.mean([
                        (logits[task].argmax(dim=1) == yb_dict[task]).float().mean().item()
                        for task in task_names
                    ]))

                    # Collect per-category preds for attribute/state if present
                    if 'attribute' in task_names:
                        y_true_attr.extend(yb_dict['attribute'].cpu().numpy())
                        y_pred_attr.extend(logits['attribute'].argmax(dim=1).cpu().numpy())
                    if 'state' in task_names:
                        y_true_state.extend(yb_dict['state'].cpu().numpy())
                        y_pred_state.extend(logits['state'].argmax(dim=1).cpu().numpy())
            accuracies_test = compute_accuracy(probe, loader_test, task_names)

            def per_class_acc(y_true, y_pred, class_names):
                if y_true is None or y_pred is None or class_names is None:
                    return None
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                out = {}
                for idx, name in enumerate(class_names):
                    mask = y_true == idx
                    if mask.sum() == 0:
                        continue
                    out[name] = float((y_pred[mask] == idx).mean())
                return out if out else None

            per_attr_acc = per_class_acc(y_true_attr, y_pred_attr, attr_classes)
            per_state_acc = per_class_acc(y_true_state, y_pred_state, state_classes)
        
        accuracies_train = compute_accuracy(probe, loader_train, task_names)
        fisher_info = compute_fisher_information(probe, loader_train, task_names)
        sparsity = probe.get_sparsity()

        bits_ci_train = bootstrap_ci_array(np.array(batch_bits_train)) if batch_bits_train else {}
        bits_ci_test = bootstrap_ci_array(np.array(batch_bits_test)) if batch_bits_test else {}
        acc_ci_train = bootstrap_ci_array(np.array(batch_acc_train)) if batch_acc_train else {}
        acc_ci_test = bootstrap_ci_array(np.array(batch_acc_test)) if batch_acc_test else {}
        
        results.append({
            **metadata,
            'prior': prior,
            'data_cost_train': total_data_cost_train,
            'data_cost_test': total_data_cost_test,
            'model_cost': total_model_cost,
            'total_mdl_train': total_data_cost_train + total_model_cost,
            'total_mdl_test': (total_data_cost_test + total_model_cost) if total_data_cost_test is not None else None,
            # Backward-compatible aggregate keys (prefer test if available)
            'data_cost': total_data_cost_test if total_data_cost_test is not None else total_data_cost_train,
            'total_mdl': (total_data_cost_test + total_model_cost) if total_data_cost_test is not None else (total_data_cost_train + total_model_cost),
            'fisher_info': fisher_info,
            'sparsity': sparsity,
            **{f'acc_train_{task}': acc for task, acc in accuracies_train.items()},
            **({f'acc_test_{task}': acc for task, acc in accuracies_test.items()} if accuracies_test else {}),
            'bits_per_sample_train_ci': bits_ci_train,
            'bits_per_sample_test_ci': bits_ci_test,
            'acc_train_ci': acc_ci_train,
            'acc_test_ci': acc_ci_test,
            'bits_per_sample_train_samples': batch_bits_train,
            'bits_per_sample_test_samples': batch_bits_test,
            'acc_train_samples': batch_acc_train,
            'acc_test_samples': batch_acc_test,
            'per_attribute_accuracy': per_attr_acc,
            'per_state_accuracy': per_state_acc
        })
        
        del probe, optimizer
        torch.cuda.empty_cache()
    
    return results

# ==============================================================================
# EXPERIMENT 3: ISOMORPHISM TEST
# ==============================================================================

def experiment_isomorphism(df, activations, labels, layer, test_indices=None):
    """Cross-model transfer MDL for representational isomorphism testing (held-out)."""
    suppression_mask = df['group_type'] == 'suppression'
    if test_indices is not None:
        mask_array = np.zeros(len(df), dtype=bool)
        mask_array[test_indices] = True
        suppression_mask = suppression_mask & mask_array
    
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
        
        # Bootstrap drift/transfer for robustness
        n_boot = 200
        rng = np.random.RandomState(Config.SEED)
        drift_samples = []
        ratio_samples = []
        acc_delta_samples = []

        with torch.no_grad():
            logits_base = probe(X_base_supp)
            logits_instruct = probe(X_instruct_supp)

            dc_base, _ = probe.compute_loss(logits_base, y_supp)
            dc_instruct, _ = probe.compute_loss(logits_instruct, y_supp)
            mdl_base = dc_base.item() / len(supp_indices)
            mdl_instruct = dc_instruct.item() / len(supp_indices)
            drift = mdl_instruct - mdl_base
            acc_base = (logits_base['attribute' if task_name == 'attribute' else 'state'].argmax(dim=1) == y_supp[list(y_supp.keys())[0]]).float().mean().item()
            acc_instruct = (logits_instruct['attribute' if task_name == 'attribute' else 'state'].argmax(dim=1) == y_supp[list(y_supp.keys())[0]]).float().mean().item()
            transfer_ratio = mdl_instruct / mdl_base if mdl_base != 0 else None

            # Bootstrap over suppression indices
            base_logits_np = logits_base['attribute' if task_name == 'attribute' else 'state'].cpu().numpy()
            inst_logits_np = logits_instruct['attribute' if task_name == 'attribute' else 'state'].cpu().numpy()
            y_np = y_supp[list(y_supp.keys())[0]].cpu().numpy()
            for _ in range(n_boot):
                idx = rng.choice(len(y_np), size=len(y_np), replace=True)
                lb = torch.tensor(base_logits_np[idx]).to(Config.DEVICE)
                li = torch.tensor(inst_logits_np[idx]).to(Config.DEVICE)
                yt = torch.tensor(y_np[idx]).to(Config.DEVICE)
                dcb = nn.functional.cross_entropy(lb, yt, reduction='sum').item() / len(idx)
                dci = nn.functional.cross_entropy(li, yt, reduction='sum').item() / len(idx)
                drift_samples.append(dci - dcb)
                ratio_samples.append((dci / dcb) if dcb != 0 else np.nan)
                acc_b = (lb.argmax(dim=1) == yt).float().mean().item()
                acc_i = (li.argmax(dim=1) == yt).float().mean().item()
                acc_delta_samples.append(acc_i - acc_b)

        drift_ci = bootstrap_ci_array(np.array(drift_samples)) if drift_samples else {}
        ratio_ci = bootstrap_ci_array(np.array([r for r in ratio_samples if not np.isnan(r)])) if ratio_samples else {}
        acc_delta_ci = bootstrap_ci_array(np.array(acc_delta_samples)) if acc_delta_samples else {}
        
        results.append({
            'layer': layer,
            'task': task_name,
            'mdl_base_self': mdl_base,
            'mdl_instruct_transfer': mdl_instruct,
            'drift': drift,
            'transfer_ratio': transfer_ratio,
            'acc_base': acc_base,
            'acc_instruct': acc_instruct,
            'acc_delta': acc_instruct - acc_base,
            'drift_ci': drift_ci,
            'transfer_ratio_ci': ratio_ci,
            'acc_delta_ci': acc_delta_ci,
            'is_isomorphic': (abs(drift) < 0.1) and (transfer_ratio is not None and 0.9 <= transfer_ratio <= 1.1)
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
        lowers = []
        uppers = []

        for layer in Config.LAYERS:
            for model in Config.MODELS:
                subset = df_var[(df_var['layer'] == layer) &
                               (df_var['model'] == model) &
                               (df_var['group'] == 'all') &
                               (df_var['task'] == 'attribute')]
                if len(subset) == 0:
                    continue

                for prior in Config.PRIORS:
                    row = subset[subset['prior'] == prior]
                    if len(row) == 0:
                        continue
                    row = row.iloc[0]
                    val = row.get(metric)
                    if val is None:
                        continue

                    data.append(val)
                    labels_list.append(f"L{layer}\n{model}\n{prior}")

                    # Use bits-per-sample CI as a proxy for uncertainty on data/total MDL
                    ci_lower = None
                    ci_upper = None
                    if metric in ['data_cost', 'total_mdl']:
                        ci_dict = row.get('bits_per_sample_test_ci') or row.get('bits_per_sample_train_ci')
                        if isinstance(ci_dict, dict):
                            ci_lower = ci_dict.get('ci_lower')
                            ci_upper = ci_dict.get('ci_upper')
                    lowers.append(ci_lower)
                    uppers.append(ci_upper)

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        x = np.arange(len(data))
        colors = sns.color_palette("husl", len(data))
        ax.bar(x, data, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)

        if any(lo is not None and up is not None for lo, up in zip(lowers, uppers)):
            err_low = [val - lo if lo is not None else 0 for val, lo in zip(data, lowers)]
            err_up = [up - val if up is not None else 0 for val, up in zip(data, uppers)]
            err = np.vstack([err_low, err_up])
            ax.errorbar(x, data, yerr=err, fmt='none', ecolor='black', elinewidth=1, capsize=3, alpha=0.7)

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


def plot_layer_deltas(delta_df):
    """Plot L24 vs L28 MDL deltas to mirror linear layer-comparison plots."""
    if delta_df is None or len(delta_df) == 0:
        log("  No layer delta data to plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    deltas = []
    colors = []

    for _, row in delta_df.iterrows():
        labels.append(f"{row['model']}-{row['task']}")
        deltas.append(row.get('mdl_delta_28_minus_24', 0))
        colors.append('#1f77b4' if row['model'] == 'base' else '#ff7f0e')

    x = np.arange(len(labels))
    ax.bar(x, deltas, color=colors, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MDL Δ (L28 - L24)')
    ax.set_title('Layer Localization: MDL Delta (L24→L28)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "layer_deltas.png", dpi=300, bbox_inches='tight')
    plt.close()
    log("  Saved: layer_deltas.png")


def plot_isomorphism_transfer(df_iso):
    """Scatter plot of drift vs transfer ratio to mirror transfer heatmaps."""
    if df_iso is None or len(df_iso) == 0:
        log("  No isomorphism data to plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'attribute': '#1f77b4', 'state': '#ff7f0e'}
    for _, row in df_iso.iterrows():
        ax.scatter(row.get('drift', 0), row.get('transfer_ratio', 0),
                   color=colors.get(row['task'], 'gray'), s=80, alpha=0.8,
                   label=row['task'] if row['task'] not in ax.get_legend_handles_labels()[1] else "")
        ax.text(row.get('drift', 0), row.get('transfer_ratio', 0)+0.005,
                f"L{int(row['layer'])}", fontsize=8, ha='center')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('MDL Drift (Instruct - Base)')
    ax.set_ylabel('Transfer Ratio (Instruct/Base)')
    ax.set_title('Isomorphism: Drift vs Transfer Ratio')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "isomorphism_transfer.png", dpi=300, bbox_inches='tight')
    plt.close()
    log("  Saved: isomorphism_transfer.png")


def plot_suppression_selectivity(selectivity_df):
    """Plot suppression vs enhancement MDL differences at Layer 28."""
    if selectivity_df is None or len(selectivity_df) == 0:
        log("  No suppression selectivity data to plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = []
    deltas = []
    colors = []
    for _, row in selectivity_df.iterrows():
        labels.append(f"{row['model']}-{row['task']}")
        deltas.append(row.get('mdl_delta_supp_minus_enh', 0))
        colors.append('#d62728' if row.get('mdl_delta_supp_minus_enh', 0) > 0 else '#2ca02c')

    x = np.arange(len(labels))
    ax.bar(x, deltas, color=colors, alpha=0.85)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('MDL Δ (Suppression - Enhancement)')
    ax.set_title('Suppression Selectivity at Layer 28')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / "plots" / "suppression_selectivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    log("  Saved: suppression_selectivity.png")


def compute_layer_delta_table(df_var):
    """Compute L24 vs L28 deltas to localize gating effects."""
    if len(df_var) == 0:
        return pd.DataFrame()

    records = []
    for model in Config.MODELS:
        for task in df_var['task'].unique():
            subset_24 = df_var[(df_var['layer'] == 24) & (df_var['model'] == model) & (df_var['task'] == task) & (df_var['group'] == 'all')]
            subset_28 = df_var[(df_var['layer'] == 28) & (df_var['model'] == model) & (df_var['task'] == task) & (df_var['group'] == 'all')]
            if len(subset_24) == 0 or len(subset_28) == 0:
                continue

            row24 = subset_24.iloc[0]
            row28 = subset_28.iloc[0]

            def pref_acc(row):
                # Prefer test accuracy if available
                acc_cols = [c for c in row.index if c.startswith('acc_test_')]
                if acc_cols:
                    return float(row[acc_cols[0]])
                acc_cols = [c for c in row.index if c.startswith('acc_train_')]
                return float(row[acc_cols[0]]) if acc_cols else None

            acc24 = pref_acc(row24)
            acc28 = pref_acc(row28)

            records.append({
                'model': model,
                'task': task,
                'mdl_24': row24.get('total_mdl'),
                'mdl_28': row28.get('total_mdl'),
                'mdl_delta_28_minus_24': (row28.get('total_mdl') - row24.get('total_mdl')) if row24.get('total_mdl') is not None and row28.get('total_mdl') is not None else None,
                'data_cost_24': row24.get('data_cost'),
                'data_cost_28': row28.get('data_cost'),
                'fisher_24': row24.get('fisher_info'),
                'fisher_28': row28.get('fisher_info'),
                'acc_24': acc24,
                'acc_28': acc28,
                'acc_delta_28_minus_24': (acc28 - acc24) if acc24 is not None and acc28 is not None else None
            })

    return pd.DataFrame.from_records(records)


def compute_effect_sizes(df_var):
    """Compute Cohen's d between base and instruct for bits-per-sample distributions.

    Uses per-batch bits (test if available else train) for prior l0, group=all.
    """
    if len(df_var) == 0:
        return pd.DataFrame()

    records = []
    for layer in Config.LAYERS:
        for task in df_var['task'].unique():
            for split in ['test', 'train']:
                col = f'bits_per_sample_{split}_samples'
                subset_base = df_var[(df_var['layer'] == layer) & (df_var['model'] == 'base') & (df_var['task'] == task) & (df_var['group'] == 'all') & (df_var['prior'] == 'l0')]
                subset_inst = df_var[(df_var['layer'] == layer) & (df_var['model'] == 'instruct') & (df_var['task'] == task) & (df_var['group'] == 'all') & (df_var['prior'] == 'l0')]
                if len(subset_base) == 0 or len(subset_inst) == 0:
                    continue
                vals_base = subset_base.iloc[0].get(col, [])
                vals_inst = subset_inst.iloc[0].get(col, [])
                if not vals_base or not vals_inst:
                    continue
                d = cohens_d_array(np.array(vals_base), np.array(vals_inst))
                records.append({
                    'layer': layer,
                    'task': task,
                    'split': split,
                    'cohens_d_bits': d,
                    'n_base': len(vals_base),
                    'n_instruct': len(vals_inst)
                })
                # Prefer test split if available; break to avoid duplicate when test present
                if split == 'test':
                    break

    return pd.DataFrame.from_records(records)


def compute_accuracy_effect_sizes(df_var):
    """Compute Cohen's d for accuracies using per-batch accuracy samples (base vs instruct).

    Prefers test split accuracies; falls back to train if test missing. Uses prior l0, group=all.
    """
    if len(df_var) == 0:
        return pd.DataFrame()

    records = []
    for layer in Config.LAYERS:
        for task in df_var['task'].unique():
            for split in ['test', 'train']:
                col = f'acc_{split}_samples'
                subset_base = df_var[(df_var['layer'] == layer) & (df_var['model'] == 'base') & (df_var['task'] == task) & (df_var['group'] == 'all') & (df_var['prior'] == 'l0')]
                subset_inst = df_var[(df_var['layer'] == layer) & (df_var['model'] == 'instruct') & (df_var['task'] == task) & (df_var['group'] == 'all') & (df_var['prior'] == 'l0')]
                if len(subset_base) == 0 or len(subset_inst) == 0:
                    continue
                vals_base = subset_base.iloc[0].get(col, [])
                vals_inst = subset_inst.iloc[0].get(col, [])
                if not vals_base or not vals_inst:
                    continue
                d = cohens_d_array(np.array(vals_base), np.array(vals_inst))
                records.append({
                    'layer': layer,
                    'task': task,
                    'split': split,
                    'cohens_d_acc': d,
                    'n_base': len(vals_base),
                    'n_instruct': len(vals_inst)
                })
                if split == 'test':
                    break
    return pd.DataFrame.from_records(records)


def compute_suppression_selectivity(df_var):
    """Compare suppression vs enhancement MDL at Layer 28 for attribute/state tasks."""
    if len(df_var) == 0:
        return pd.DataFrame()

    records = []
    for task in ['attribute', 'state']:
        for model in Config.MODELS:
            subset_supp = df_var[(df_var['layer'] == 28) & (df_var['task'] == task) & (df_var['model'] == model) & (df_var['group'] == 'suppression') & (df_var['prior'] == 'l0')]
            subset_enh = df_var[(df_var['layer'] == 28) & (df_var['task'] == task) & (df_var['model'] == model) & (df_var['group'] == 'enhancement') & (df_var['prior'] == 'l0')]
            if len(subset_supp) == 0 or len(subset_enh) == 0:
                continue
            mdl_supp = subset_supp.iloc[0].get('total_mdl')
            mdl_enh = subset_enh.iloc[0].get('total_mdl')
            records.append({
                'layer': 28,
                'task': task,
                'model': model,
                'mdl_suppression': mdl_supp,
                'mdl_enhancement': mdl_enh,
                'mdl_delta_supp_minus_enh': (mdl_supp - mdl_enh) if mdl_supp is not None and mdl_enh is not None else None
            })
    df_out = pd.DataFrame.from_records(records)

    # Per-attribute/state high/low suppression buckets using layer 28
    bucket_records = []
    # Use group=all, prior=l0
    layer28 = df_var[(df_var['layer'] == 28) & (df_var['prior'] == 'l0') & (df_var['group'] == 'all')]
    if len(layer28) > 0:
        # Define buckets (edit as needed)
        high_attrs = ['Religion', 'Rituals_and_Ceremonies', 'Costume', 'Dance_and_Music']
        low_attrs = ['Nightlife', 'Transport', 'Medicine', 'Sports']
        high_states = ['Tamil_Nadu', 'Karnataka', 'Kerala', 'Mizoram', 'Arunachal_Pradesh']
        low_states = ['Delhi', 'Punjab', 'Haryana', 'Chandigarh']

        for model in Config.MODELS:
            for task in ['attribute', 'state']:
                subset = layer28[(layer28['model'] == model) & (layer28['task'] == task)]
                if len(subset) == 0:
                    continue
                row = subset.iloc[0]
                # We need per-category accuracies; if absent, skip
                per_attr = row.get('per_attribute_accuracy') if 'per_attribute_accuracy' in row else None
                per_state = row.get('per_state_accuracy') if 'per_state_accuracy' in row else None
                if task == 'attribute' and per_attr:
                    highs = [per_attr.get(a) for a in high_attrs if a in per_attr]
                    lows = [per_attr.get(a) for a in low_attrs if a in per_attr]
                elif task == 'state' and per_state:
                    highs = [per_state.get(s) for s in high_states if s in per_state]
                    lows = [per_state.get(s) for s in low_states if s in per_state]
                else:
                    highs = []
                    lows = []
                if highs and lows:
                    bucket_records.append({
                        'layer': 28,
                        'task': task,
                        'model': model,
                        'high_mean': float(np.mean(highs)),
                        'low_mean': float(np.mean(lows)),
                        'diff_high_minus_low': float(np.mean(highs) - np.mean(lows))
                    })

    if bucket_records:
        return df_out, pd.DataFrame.from_records(bucket_records)
    return df_out, pd.DataFrame()

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

    def preferred_acc(row, task):
        """Prefer test accuracy when present; otherwise fall back to train."""
        test_col = f'acc_test_{task}'
        train_col = f'acc_train_{task}'
        if test_col in row:
            return row.get(test_col, None)
        if train_col in row:
            return row.get(train_col, None)
        return None
    
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
                acc_val = preferred_acc(subset.iloc[0], task)
                if acc_val is not None:
                    single_accs[task] = acc_val
        
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
            
            model_evidence['acc_attribute'] = preferred_acc(row, 'attribute')
            model_evidence['acc_state'] = preferred_acc(row, 'state')
            model_evidence['acc_correctness'] = preferred_acc(row, f'correctness_{model}')
        
        subset = df_var[(df_var['layer'] == layer) & 
                       (df_var['model'] == model) &
                       (df_var['group'] == 'suppression') &
                       (df_var['prior'] == 'l0') &
                       (df_var['task'] == f'multitask_triple_{model}')]
        
        if len(subset) > 0:
            row = subset.iloc[0]
            model_evidence['suppression_analysis'] = {
                'triple_mdl': row['total_mdl'],
                'acc_attribute': preferred_acc(row, 'attribute'),
                'acc_state': preferred_acc(row, 'state'),
                'acc_correctness': preferred_acc(row, f'correctness_{model}'),
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
                'acc_attribute': preferred_acc(row, 'attribute'),
                'acc_state': preferred_acc(row, 'state'),
                'acc_correctness': preferred_acc(row, f'correctness_{model}')
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


def compute_triple_compression_ci(df_var, n_boot: int = 500):
    """Bootstrap compression ratio using batch-level bits for triple vs sum of singles.

    Uses the middle layer (as in triple_entanglement plots), prior l0, group=all.
    """
    if len(df_var) == 0:
        return pd.DataFrame()

    available_layers = sorted(df_var['layer'].unique())
    if not available_layers:
        return pd.DataFrame()
    layer = available_layers[len(available_layers)//2]

    records = []
    rng = np.random.RandomState(Config.SEED)

    for model in Config.MODELS:
        triple_row = df_var[(df_var['layer'] == layer) &
                            (df_var['model'] == model) &
                            (df_var['group'] == 'all') &
                            (df_var['prior'] == 'l0') &
                            (df_var['task'] == f'multitask_triple_{model}')]
        if len(triple_row) == 0:
            continue
        triple_samples = triple_row.iloc[0].get('bits_per_sample_train_samples', [])
        if not triple_samples:
            continue

        single_tasks = ['attribute', 'state', f'correctness_{model}']
        single_samples = []
        for t in single_tasks:
            row = df_var[(df_var['layer'] == layer) &
                         (df_var['model'] == model) &
                         (df_var['group'] == 'all') &
                         (df_var['prior'] == 'l0') &
                         (df_var['task'] == t)]
            if len(row) == 0:
                single_samples.append([])
                continue
            single_samples.append(row.iloc[0].get('bits_per_sample_train_samples', []))

        if any(len(s) == 0 for s in single_samples):
            continue

        ratios = []
        for _ in range(n_boot):
            triple_val = rng.choice(triple_samples)
            single_sum = sum(rng.choice(s) for s in single_samples)
            if single_sum > 0:
                ratios.append(triple_val / single_sum)

        if ratios:
            ci = bootstrap_ci_array(np.array(ratios), n_bootstrap=n_boot)
            records.append({
                'model': model,
                'layer': layer,
                'compression_ratio_point': float(np.mean(ratios)),
                'compression_ratio_ci_lower': ci.get('ci_lower'),
                'compression_ratio_ci_upper': ci.get('ci_upper')
            })

    return pd.DataFrame.from_records(records)

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
        labels, dims, encoders = encode_all_labels(df)
        attr_classes = list(encoders['attribute'].classes_)
        state_classes = list(encoders['state'].classes_)

        # Global deterministic split (aligned with linear probing)
        train_idx_global, test_idx_global = create_train_test_split(df, stratify_col='group_type')
        
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
                group_indices = np.where(mask.values)[0]
            else:
                group_indices = np.arange(len(df))

            group_train_idx = np.intersect1d(train_idx_global, group_indices)
            group_test_idx = np.intersect1d(test_idx_global, group_indices)

            if len(group_train_idx) < 50 or len(group_test_idx) < 20:
                log(f"  Skipping {group_name}: insufficient split sizes (train={len(group_train_idx)}, test={len(group_test_idx)})")
                continue
            else:
                log(f"  Split sizes for {group_name}: train={len(group_train_idx)}, test={len(group_test_idx)}")
            
            tasks_config = [
                ('attribute', {'attribute': labels['attribute']}, {'attribute': dims['attribute']}),
                ('state', {'state': labels['state']}, {'state': dims['state']}),
                ('correctness_base', {'correctness_base': labels['correctness_base']}, 
                 {'correctness_base': dims['correctness']}),
                ('correctness_instruct', {'correctness_instruct': labels['correctness_instruct']}, 
                 {'correctness_instruct': dims['correctness']}),
                ('multitask_dual', {'attribute': labels['attribute'], 
                                    'state': labels['state']}, 
                 {'attribute': dims['attribute'], 'state': dims['state']}),
                ('multitask_triple_base', {
                    'attribute': labels['attribute'],
                    'state': labels['state'],
                    'correctness_base': labels['correctness_base']
                }, {
                    'attribute': dims['attribute'],
                    'state': dims['state'],
                    'correctness_base': dims['correctness']
                }),
                ('multitask_triple_instruct', {
                    'attribute': labels['attribute'],
                    'state': labels['state'],
                    'correctness_instruct': labels['correctness_instruct']
                }, {
                    'attribute': dims['attribute'],
                    'state': dims['state'],
                    'correctness_instruct': dims['correctness']
                })
            ]
            
            for model in Config.MODELS:
                X = activations[model]
                
                for task_name, y_dict, task_dims in tasks_config:
                    if 'correctness' in task_name and model not in task_name:
                        continue
                    
                    if 'multitask_triple' in task_name and model not in task_name:
                        continue
                    
                    log(f"    {model} - {task_name}")
                    
                    # Slice labels for current split/group
                    y_train_dict = {k: v[group_train_idx] for k, v in y_dict.items()}
                    y_test_dict = {k: v[group_test_idx] for k, v in y_dict.items()}
                    X_train = X[group_train_idx]
                    X_test = X[group_test_idx]

                    metadata = {
                        'layer': layer,
                        'model': model,
                        'task': task_name,
                        'group': group_name
                    }
                    
                    online_results = experiment_online_coding(X_train, y_train_dict, task_dims, metadata, X_test, y_test_dict)
                    all_online.extend(online_results)
                    
                    var_results = experiment_variational_mdl(
                        X_train,
                        y_train_dict,
                        X_test,
                        y_test_dict,
                        task_dims,
                        metadata,
                        attr_classes=attr_classes,
                        state_classes=state_classes
                    )
                    all_variational.extend(var_results)
                    
                    if group_name == 'all':
                        y_control_train = {k: shuffle(v[group_train_idx], random_state=Config.SEED) 
                                    for k, v in y_dict.items()}
                        y_control_test = {k: shuffle(v[group_test_idx], random_state=Config.SEED) 
                                    for k, v in y_dict.items()}
                        
                        metadata_ctrl = metadata.copy()
                        metadata_ctrl['model'] = f"{model}_control"
                        
                        online_ctrl = experiment_online_coding(X_train, y_control_train, task_dims, metadata_ctrl, X_test, y_control_test)
                        all_online.extend(online_ctrl)
                        
                        var_ctrl = experiment_variational_mdl(X_train, y_control_train, X_test, y_control_test, task_dims, metadata_ctrl)
                        all_variational.extend(var_ctrl)
                    
                    gc.collect()
                    torch.cuda.empty_cache()
        
        log(f"\n  Running Isomorphism Test (Layer {layer})")
        iso_results = experiment_isomorphism(df, activations, labels, layer, test_indices=test_idx_global)
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
    
    # Layer localization table (L24 vs L28)
    delta_table = compute_layer_delta_table(df_variational)
    if len(delta_table) > 0:
        delta_path = Config.OUTPUT_DIR / "data" / "layer_24_28_deltas.csv"
        delta_table.to_csv(delta_path, index=False)
        log(f"  Saved: {delta_path.name}")

    effect_df = compute_effect_sizes(df_variational)
    if len(effect_df) > 0:
        eff_path = Config.OUTPUT_DIR / "data" / "effect_sizes.csv"
        effect_df.to_csv(eff_path, index=False)
        log(f"  Saved: {eff_path.name}")

    acc_eff_df = compute_accuracy_effect_sizes(df_variational)
    if len(acc_eff_df) > 0:
        acc_eff_path = Config.OUTPUT_DIR / "data" / "accuracy_effect_sizes.csv"
        acc_eff_df.to_csv(acc_eff_path, index=False)
        log(f"  Saved: {acc_eff_path.name}")

    comp_ci_df = compute_triple_compression_ci(df_variational)
    if len(comp_ci_df) > 0:
        comp_ci_path = Config.OUTPUT_DIR / "data" / "triple_compression_ci.csv"
        comp_ci_df.to_csv(comp_ci_path, index=False)
        log(f"  Saved: {comp_ci_path.name}")

    selectivity_df, bucket_df = compute_suppression_selectivity(df_variational)
    if len(selectivity_df) > 0:
        sel_path = Config.OUTPUT_DIR / "data" / "suppression_selectivity.csv"
        selectivity_df.to_csv(sel_path, index=False)
        log(f"  Saved: {sel_path.name}")
    if bucket_df is not None and len(bucket_df) > 0:
        bucket_path = Config.OUTPUT_DIR / "data" / "suppression_buckets.csv"
        bucket_df.to_csv(bucket_path, index=False)
        log(f"  Saved: {bucket_path.name}")

    # Unified summary table for quick inspection
    summary_rows = []
    for _, row in df_variational.iterrows():
        if row.get('prior') != 'l0':
            continue
        if row.get('group') != 'all':
            continue
        acc_cols_test = [c for c in row.index if c.startswith('acc_test_')]
        acc_cols_train = [c for c in row.index if c.startswith('acc_train_')]
        acc_preferred = None
        if acc_cols_test:
            acc_preferred = float(row[acc_cols_test[0]])
        elif acc_cols_train:
            acc_preferred = float(row[acc_cols_train[0]])

        summary_rows.append({
            'layer': int(row['layer']),
            'model': row['model'],
            'task': row['task'],
            'total_mdl': row.get('total_mdl'),
            'data_cost': row.get('data_cost'),
            'model_cost': row.get('model_cost'),
            'fisher_info': row.get('fisher_info'),
            'sparsity': row.get('sparsity'),
            'accuracy': acc_preferred,
            'bits_ci_lower': row.get('bits_per_sample_test_ci', {}).get('ci_lower') if isinstance(row.get('bits_per_sample_test_ci'), dict) else None,
            'bits_ci_upper': row.get('bits_per_sample_test_ci', {}).get('ci_upper') if isinstance(row.get('bits_per_sample_test_ci'), dict) else None,
            'acc_ci_lower': row.get('acc_test_ci', {}).get('ci_lower') if isinstance(row.get('acc_test_ci'), dict) else None,
            'acc_ci_upper': row.get('acc_test_ci', {}).get('ci_upper') if isinstance(row.get('acc_test_ci'), dict) else None
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = Config.OUTPUT_DIR / "data" / "summary_table.csv"
        summary_df.to_csv(summary_path, index=False)
        log(f"  Saved: {summary_path.name}")

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
    plot_layer_deltas(delta_table)
    plot_isomorphism_transfer(df_isomorphism)
    plot_suppression_selectivity(selectivity_df)
    
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
        # Save a flat CSV for quick reference
        flat_rows = []
        for model, stats in entanglement_metrics.items():
            if model not in ['base', 'instruct']:
                continue
            flat_rows.append({
                'model': model,
                'compression_ratio': stats.get('compression_ratio'),
                'triple_mdl': stats.get('triple_mdl'),
                'single_sum_mdl': stats.get('single_sum_mdl'),
                'acc_attribute': stats.get('acc_attribute'),
                'acc_state': stats.get('acc_state'),
                'acc_correctness': stats.get('acc_correctness')
            })
        if flat_rows:
            pd.DataFrame(flat_rows).to_csv(Config.OUTPUT_DIR / "data" / "triple_entanglement_summary.csv", index=False)
            log("  Saved: triple_entanglement_summary.csv")
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