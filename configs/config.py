"""
Configuration file for RQ1 Cultural Features Discovery
Verified model names from Hugging Face on 2025-10-19
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path("/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features")
DATA_ROOT = Path("/datadrive/anshulk/data")
ACTIVATION_ROOT = Path("/datadrive/anshulk/activations")
MODEL_CACHE = Path("/datadrive/anshulk/models")

# Ensure directories exist
for path in [PROJECT_ROOT, DATA_ROOT, ACTIVATION_ROOT, MODEL_CACHE]:
    path.mkdir(parents=True, exist_ok=True)

# Model configurations - VERIFIED from HuggingFace
MODELS = {
    "base": "Qwen/Qwen1.5-1.8B",           # Base model (pre-training only)
    "chat": "Qwen/Qwen1.5-1.8B-Chat"       # Chat model (post-training with RLHF/SFT)
}

# Important: Requires transformers >= 4.37.0 for Qwen1.5
# No trust_remote_code needed for Qwen1.5

# Target layers for activation extraction
TARGET_LAYERS = [6, 12, 18]  # early, middle, late

# Dataset configurations
DATASETS = {
    "updesh_beta": {
        "path": DATA_ROOT / "updesh_beta.json",
        "size": 30000,
        "distribution": {
            "festivals": 0.30,
            "names": 0.20,
            "food": 0.20,
            "regional": 0.20,
            "honorifics": 0.10
        }
    },
    "dosa": {
        "path": DATA_ROOT / "dosa_dataset.json",
        "size": 615
    },
    "control": {
        "path": DATA_ROOT / "snli_control.json",
        "size": 5000
    }
}

# SAE Training configurations
SAE_CONFIG = {
    "reconstruction_loss_threshold": 0.05,
    "sparsity_ratio_target": 10,
    "dictionary_size": 16384,  # Will tune this
    "batch_size": 256,
    "learning_rate": 1e-4,
    "num_epochs": 100
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "reconstruction_loss": 0.05,
    "l0_sparsity_ratio": 10,
    "saebench_score": 0.70,
    "cohens_d": 0.8,
    "algorithmic_coherence": 0.60,
    "human_agreement": 0.70,
    "cross_validation_stability": 0.80
}

# Hardware configuration
DEVICE = "cuda"
NUM_GPUS = 4
BATCH_SIZE_PER_GPU = 64
NUM_WORKERS = 8

# Hugging Face cache - set environment variables
os.environ['HF_HOME'] = str(MODEL_CACHE)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE / "transformers")
os.environ['HF_HUB_CACHE'] = str(MODEL_CACHE / "hub")

# Minimum transformers version required
MIN_TRANSFORMERS_VERSION = "4.37.0"
