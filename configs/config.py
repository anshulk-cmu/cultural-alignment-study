"""
Configuration file for RQ1 Cultural Features Discovery
"""
import os
import logging
from pathlib import Path
from datetime import datetime
import torch

# ============================================================================
# BASE PATHS
# ============================================================================

PROJECT_ROOT = Path("/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features")
DATA_ROOT = Path("/datadrive/anshulk/data")
ACTIVATION_ROOT = Path("/datadrive/anshulk/activations")
MODEL_CACHE = Path("/datadrive/anshulk/models")

for path in [PROJECT_ROOT, DATA_ROOT, ACTIVATION_ROOT, MODEL_CACHE]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODELS = {
    "base": "Qwen/Qwen1.5-1.8B",
    "chat": "Qwen/Qwen1.5-1.8B-Chat"
}

TARGET_LAYERS = [6, 12, 18]

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASETS = {
    "updesh_beta": {
        "path": DATA_ROOT / "updesh_beta.json",
        "size": 30000
    },
    "snli_control": {
        "path": DATA_ROOT / "snli_control.json",
        "size": 5000
    },
    "hindi_control": {
        "path": DATA_ROOT / "hindi_control.json",
        "size": 5000
    }
}

# ============================================================================
# PHASE 1: ACTIVATION EXTRACTION
# ============================================================================

DEVICE = "cuda"
NUM_GPUS = torch.cuda.device_count()

BATCH_SIZE_PER_GPU = 32
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS if NUM_GPUS > 1 else BATCH_SIZE_PER_GPU
NUM_WORKERS = 4

MAX_LENGTH = 256
SAVE_EVERY_N_BATCHES = 50

USE_FP16 = True
EMPTY_CACHE_EVERY_N = 10
GRADIENT_CHECKPOINTING = False

# ============================================================================
# PHASE 2: SAE TRAINING
# ============================================================================

SAE_HIDDEN_DIM = 2048
SAE_DICT_SIZE = 16384
SAE_SPARSITY_COEF = 1e-3

SAE_BATCH_SIZE = 256
SAE_LEARNING_RATE = 1e-4
SAE_NUM_EPOCHS = 100
SAE_WARMUP_STEPS = 1000
SAE_WEIGHT_DECAY = 0.01

SAE_GPUS = [0, 1, 2]
SAE_NUM_WORKERS = 8

SAE_CHECKPOINT_EVERY = 10
SAE_EVAL_EVERY = 5

SAE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sae_models"
SAE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

VALIDATION_THRESHOLDS = {
    "reconstruction_loss": 0.05,
    "l0_sparsity_ratio": 10,
    "saebench_score": 0.70,
    "cohens_d": 0.8,
    "algorithmic_coherence": 0.60,
    "human_agreement": 0.70,
    "cross_validation_stability": 0.80
}

# ============================================================================
# ENVIRONMENT
# ============================================================================

os.environ['HF_HOME'] = str(MODEL_CACHE)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE / "transformers")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============================================================================
# LOGGING
# ============================================================================

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
