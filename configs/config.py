"""
Configuration file for RQ1 Cultural Features Discovery
"""
import os
import logging
from pathlib import Path
from datetime import datetime
import torch

# Base paths
PROJECT_ROOT = Path("/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features")
DATA_ROOT = Path("/datadrive/anshulk/data")
ACTIVATION_ROOT = Path("/datadrive/anshulk/activations")
MODEL_CACHE = Path("/datadrive/anshulk/models")

# Ensure directories exist
for path in [PROJECT_ROOT, DATA_ROOT, ACTIVATION_ROOT, MODEL_CACHE]:
    path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = {
    "base": "Qwen/Qwen1.5-1.8B",
    "chat": "Qwen/Qwen1.5-1.8B-Chat"
}

# Target layers for activation extraction
TARGET_LAYERS = [6, 12, 18]  # early (25%), middle (50%), late (75%)

# Dataset configurations
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

# SAE Training configurations
SAE_CONFIG = {
    "reconstruction_loss_threshold": 0.05,
    "sparsity_ratio_target": 10,
    "dictionary_size": 16384,
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

# Hardware configuration - OPTIMIZED FOR MEMORY
DEVICE = "cuda"
NUM_GPUS = torch.cuda.device_count()

# Memory-optimized batch sizes
BATCH_SIZE_PER_GPU = 32  # Conservative for safety
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS if NUM_GPUS > 1 else BATCH_SIZE_PER_GPU
NUM_WORKERS = 4

# Activation extraction settings
MAX_LENGTH = 256  # Most sentences are shorter
SAVE_EVERY_N_BATCHES = 50  # Save intermediate results

# Memory management
USE_FP16 = True  # Mixed precision for memory efficiency
EMPTY_CACHE_EVERY_N = 10  # Clear GPU cache every N batches
GRADIENT_CHECKPOINTING = False  # Not needed for inference

# Hugging Face cache
os.environ['HF_HOME'] = str(MODEL_CACHE)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE / "transformers")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional log file name (will be placed in LOG_DIR)
        level: Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
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
