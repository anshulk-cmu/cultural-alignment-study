import os
from pathlib import Path
import torch
import logging

PROJECT_ROOT = Path("/home/anshulk/cultural-alignment-study")
DATA_ROOT = Path("/data/user_data/anshulk/cultural-alignment-study/data")
MODEL_CACHE = Path("/data/models/huggingface")
ACTIVATION_ROOT = DATA_ROOT / "activations"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)
ACTIVATION_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_NAME_BASE = "/data/models/huggingface/qwen/Qwen1.5-1.8B"
MODEL_NAME_CHAT = "/data/models/huggingface/qwen/Qwen1.5-1.8B-Chat"

MODELS = {
    "base": MODEL_NAME_BASE,
    "chat": MODEL_NAME_CHAT
}

TARGET_LAYERS = [6, 12, 18]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
BATCH_SIZE_PER_GPU = 64
USE_FP16 = True
EMPTY_CACHE_EVERY_N = 10
SAVE_EVERY_N_BATCHES = 50

DATASETS = {
    "train": {
        "path": DATA_ROOT / "train" / "combined_data.json",
        "max_samples": 40000
    },
    "test": {
        "path": DATA_ROOT / "test" / "combined_data.json",
        "max_samples": 10000
    }
}

BATCH_SIZE = 64
MAX_LENGTH = 512
NUM_WORKERS = 4

SAE_HIDDEN_DIM = 2048
SAE_DICT_SIZE = 8192
SAE_SPARSITY_K = 256
SAE_AUX_K = 512
SAE_AUX_COEF = 0.03
SAE_DEAD_NEURON_MONITOR_STEPS = 1000
SAE_DEAD_NEURON_CHECK_EVERY = 3000
SAE_DEAD_NEURON_THRESHOLD = 0.001
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

VALIDATION_THRESHOLDS = {
    "reconstruction_loss": 0.05,
    "l0_sparsity_ratio": 10,
    "saebench_score": 0.70,
    "cohens_d": 0.8,
    "algorithmic_coherence": 0.60,
    "human_agreement": 0.70,
    "cross_validation_stability": 0.80
}

os.environ['HF_HOME'] = str(MODEL_CACHE)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE / "transformers")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
