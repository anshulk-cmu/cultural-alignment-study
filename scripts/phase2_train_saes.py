import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import torch
from pathlib import Path
from datetime import datetime
import logging
from torch.utils.data import ConcatDataset, DataLoader

from configs.config import (
    ACTIVATION_ROOT,
    SAE_OUTPUT_ROOT,
    SAE_HIDDEN_DIM,
    SAE_DICT_SIZE,
    SAE_SPARSITY_K,
    SAE_AUX_K,
    SAE_AUX_COEF,
    SAE_BATCH_SIZE,
    SAE_NUM_EPOCHS,
    SAE_GPUS,
    SAE_NUM_WORKERS,
    TARGET_LAYERS,
    setup_logger,
    SAE_DEAD_NEURON_CHECK_EVERY,
    SAE_DEAD_NEURON_MONITOR_STEPS,
    SAE_DEAD_NEURON_THRESHOLD
)
from utils.sae_model import SparseAutoencoder
from utils.activation_dataset import create_activation_dataloaders, collate_activations
from utils.sae_trainer import SAETrainer

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(
    'phase2_sae_training',
    f'phase2_sae_training_{timestamp}.log'
)


def find_latest_run(activation_root: Path) -> Path:
    run_dirs = sorted(activation_root.glob("run_*"))
    if not run_dirs:
        raise ValueError(f"No run directories found in {activation_root}")
    return run_dirs[-1]


def train_sae_for_model(
    model_type: str,
    layer_idx: int,
    run_dir: Path,
    save_dir: Path
):
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_type.upper()} SAE - Layer {layer_idx}")
    logger.info(f"{'='*80}")
    
    logger.info("\nLoading activation datasets...")
    
    dataset_names = ['updesh_beta', 'snli_control', 'hindi_control']
    
    train_loaders = create_activation_dataloaders(
        run_dir=run_dir,
        dataset_names=dataset_names,
        layer_idx=layer_idx,
        model_type=model_type,
        batch_size=SAE_BATCH_SIZE,
        num_workers=SAE_NUM_WORKERS,
        shuffle=True
    )
    
    val_loaders = create_activation_dataloaders(
        run_dir=run_dir,
        dataset_names=dataset_names,
        layer_idx=layer_idx,
        model_type=model_type,
        batch_size=SAE_BATCH_SIZE,
        num_workers=SAE_NUM_WORKERS,
        shuffle=False
    )
    
    train_dataset = ConcatDataset([loader.dataset for loader in train_loaders.values()])
    val_dataset = ConcatDataset([loader.dataset for loader in val_loaders.values()])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=SAE_BATCH_SIZE,
        shuffle=True,
        num_workers=SAE_NUM_WORKERS,
        collate_fn=collate_activations,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=SAE_BATCH_SIZE,
        shuffle=False,
        num_workers=SAE_NUM_WORKERS,
        collate_fn=collate_activations,
        pin_memory=True
    )
    
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    
    logger.info("\nInitializing SAE model...")
    model = SparseAutoencoder(
        input_dim=SAE_HIDDEN_DIM,
        dict_size=SAE_DICT_SIZE,
        sparsity_k=SAE_SPARSITY_K,
        aux_k=SAE_AUX_K,
        aux_coef=SAE_AUX_COEF
    )
    
    logger.info("\nInitializing trainer...")
    trainer = SAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device_ids=SAE_GPUS,
        save_dir=save_dir,
        dead_neuron_check_every=SAE_DEAD_NEURON_CHECK_EVERY,
        dead_neuron_monitor_steps=SAE_DEAD_NEURON_MONITOR_STEPS,
        dead_neuron_threshold=SAE_DEAD_NEURON_THRESHOLD
    )
    
    logger.info(f"\nStarting training...")
    trainer.train(num_epochs=SAE_NUM_EPOCHS)
    
    logger.info(f"\nCompleted {model_type} SAE training for layer {layer_idx}")
    logger.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
    logger.info(f"  Saved to: {save_dir}")


def main():
    
    try:
        logger.info("="*80)
        logger.info("PHASE 2: TRIPLE SAE TRAINING WITH AUXILIARY LOSS")
        logger.info("="*80)
        
        logger.info(f"\nSystem Information:")
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        logger.info(f"  GPUs to use: {SAE_GPUS}")
        
        for gpu_id in SAE_GPUS:
            logger.info(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        
        logger.info(f"\nConfiguration:")
        logger.info(f"  Input dimension: {SAE_HIDDEN_DIM}")
        logger.info(f"  Dictionary size: {SAE_DICT_SIZE}")
        logger.info(f"  Sparsity K: {SAE_SPARSITY_K}")
        logger.info(f"  Aux K: {SAE_AUX_K}")
        logger.info(f"  Aux coefficient: {SAE_AUX_COEF}")
        logger.info(f"  Batch size: {SAE_BATCH_SIZE}")
        logger.info(f"  Epochs: {SAE_NUM_EPOCHS}")
        logger.info(f"  Dead neuron check: every {SAE_DEAD_NEURON_CHECK_EVERY} steps")
        
        run_dir = find_latest_run(ACTIVATION_ROOT)
        logger.info(f"\nUsing activation data from: {run_dir}")
        
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SAE_OUTPUT_ROOT / f"triple_sae_k{SAE_SPARSITY_K}_dict{SAE_DICT_SIZE}_aux_{output_timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        model_types = ['base', 'chat', 'delta']
        total_saes = len(model_types) * len(TARGET_LAYERS)
        current_sae = 0
        
        for model_type in model_types:
            for layer_idx in TARGET_LAYERS:
                current_sae += 1
                logger.info(f"\n{'#'*80}")
                logger.info(f"SAE {current_sae}/{total_saes}")
                logger.info(f"{'#'*80}")
                
                save_dir = output_dir / f"{model_type}_layer{layer_idx}"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                train_sae_for_model(
                    model_type=model_type,
                    layer_idx=layer_idx,
                    run_dir=run_dir,
                    save_dir=save_dir
                )
        
        logger.info(f"\n{'='*80}")
        logger.info("ALL SAE TRAINING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Trained {total_saes} SAEs")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
