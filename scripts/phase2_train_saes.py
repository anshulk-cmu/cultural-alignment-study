"""
Phase 2: Train Triple SAE (Base, Chat, Delta)
Uses 3 GPUs for parallel training
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import torch
from pathlib import Path
from datetime import datetime
import logging
from utils.activation_dataset import create_activation_dataloaders, collate_activations
from torch.utils.data import ConcatDataset, DataLoader

# Updated config imports
from configs.config import (
    ACTIVATION_ROOT,
    SAE_OUTPUT_ROOT,
    SAE_HIDDEN_DIM,
    SAE_DICT_SIZE,
    SAE_SPARSITY_K,  # <-- NEW: Using K for sparsity
    SAE_BATCH_SIZE,
    SAE_NUM_EPOCHS,
    SAE_GPUS,
    SAE_NUM_WORKERS,
    TARGET_LAYERS,
    setup_logger,
    # NEW: Dead neuron params to pass to trainer
    SAE_DEAD_NEURON_CHECK_EVERY,
    SAE_DEAD_NEURON_MONITOR_STEPS,
    SAE_DEAD_NEURON_THRESHOLD
)
from utils.sae_model import SparseAutoencoder
from utils.activation_dataset import create_activation_dataloaders
from utils.sae_trainer import SAETrainer

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(
    'phase2_sae_training',
    f'phase2_sae_training_{timestamp}.log'
)


def find_latest_run(activation_root: Path) -> Path:
    """Find the most recent Phase 1 run directory"""
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
    """
    Train SAE for specific model type and layer
    
    Args:
        model_type: 'base', 'chat', or 'delta'
        layer_idx: Layer index (6, 12, or 18)
        run_dir: Phase 1 run directory
        save_dir: Directory to save trained model
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {model_type.upper()} SAE - LAYER {layer_idx}")
    logger.info(f"{'='*80}")
    
    # Create dataloaders
    logger.info("\nLoading activation data...")
    
    # Use updesh_beta for training, controls for validation
    train_loaders = create_activation_dataloaders(
        run_dir=run_dir,
        dataset_names=['updesh_beta'],
        layer_idx=layer_idx,
        model_type=model_type,
        batch_size=SAE_BATCH_SIZE,
        num_workers=SAE_NUM_WORKERS,
        shuffle=True
    )
    
    val_loaders = create_activation_dataloaders(
        run_dir=run_dir,
        dataset_names=['snli_control', 'hindi_control'],
        layer_idx=layer_idx,
        model_type=model_type,
        batch_size=SAE_BATCH_SIZE,
        num_workers=SAE_NUM_WORKERS,
        shuffle=False
    )
    
    if not train_loaders:
        raise ValueError(f"No training data found for {model_type}")
    
    # Combine training loaders
    train_loader = train_loaders['updesh_beta']
    
    # Combine validation loaders (concatenate)
    from torch.utils.data import ConcatDataset, DataLoader
    if val_loaders:
        val_datasets = [loader.dataset for loader in val_loaders.values()]
        val_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            val_dataset,
            batch_size=SAE_BATCH_SIZE,
            shuffle=False,
            num_workers=SAE_NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_activations
        )
    else:
        val_loader = None
    
    # Create model
    logger.info("\nInitializing SAE model...")
    # --- UPDATED MODEL CREATION ---
    # We now pass sparsity_k instead of sparsity_coef
    model = SparseAutoencoder(
        input_dim=SAE_HIDDEN_DIM,
        dict_size=SAE_DICT_SIZE,
        sparsity_k=SAE_SPARSITY_K  # <-- NEW
    )
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    # --- UPDATED TRAINER CREATION ---
    # We pass the new dead neuron parameters
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
    
    # Train
    logger.info(f"\nStarting training...")
    trainer.train(num_epochs=SAE_NUM_EPOCHS)
    
    logger.info(f"\n✓ Completed {model_type} SAE training for layer {layer_idx}")
    logger.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
    logger.info(f"  Saved to: {save_dir}")


def main():
    """Main execution"""
    
    try:
        logger.info("="*80)
        logger.info("PHASE 2: TRIPLE SAE TRAINING (TOP-K)")
        logger.info("="*80)
        
        # System info
        logger.info(f"\nSystem Information:")
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        logger.info(f"  GPUs to use: {SAE_GPUS}")
        
        for gpu_id in SAE_GPUS:
            logger.info(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        
        # Configuration
        logger.info(f"\nConfiguration:")
        logger.info(f"  Hidden dim: {SAE_HIDDEN_DIM}")
        logger.info(f"  Dictionary size: {SAE_DICT_SIZE}")
        # --- UPDATED LOGGING ---
        logger.info(f"  Sparsity K: {SAE_SPARSITY_K}")
        logger.info(f"  Batch size: {SAE_BATCH_SIZE}")
        logger.info(f"  Epochs: {SAE_NUM_EPOCHS}")
        logger.info(f"  Target layers: {TARGET_LAYERS}")
        
        # Find latest Phase 1 run
        logger.info(f"\nFinding Phase 1 activations...")
        run_dir = find_latest_run(ACTIVATION_ROOT)
        logger.info(f"  Using run: {run_dir.name}")
        
        # Create output directory
        output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SAE_OUTPUT_ROOT / f"triple_sae_k{SAE_SPARSITY_K}_dict{SAE_DICT_SIZE}_{output_timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Output directory: {output_dir}")
        
        # Train SAEs for each model type and layer
        model_types = ['base', 'chat', 'delta']
        
        for model_type in model_types:
            for layer_idx in TARGET_LAYERS:
                save_dir = output_dir / f"{model_type}_layer{layer_idx}"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                train_sae_for_model(
                    model_type=model_type,
                    layer_idx=layer_idx,
                    run_dir=run_dir,
                    save_dir=save_dir
                )
                
                # Clear GPU memory between models
                torch.cuda.empty_cache()
        
        logger.info("\n" + "="*80)
        logger.info("✓ PHASE 2 COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nAll SAE models saved to: {output_dir}")
        logger.info(f"\nNext steps:")
        logger.info(f"  - Analyze learned features")
        logger.info(f"  - Compute validation metrics")
        logger.info(f"  - Proceed to Phase 2.5: LLM Feature Labeling")
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        raise


if __name__ == "__main__":
    main()
