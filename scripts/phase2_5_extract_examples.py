# scripts/phase2_5_extract_examples.py
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')
import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from configs.config import ACTIVATION_ROOT, SAE_OUTPUT_ROOT, TARGET_LAYERS, setup_logger

logger = setup_logger('extract_examples', 'phase2_5_extract.log')


def validate_phase1_5_complete():
    """Validate that Phase 1.5 has been run and necessary files exist."""
    runs = sorted(ACTIVATION_ROOT.glob("run_*"))
    if not runs:
        logger.error("No activation runs found. Please run Phase 1 first.")
        return False

    latest_run = runs[-1]
    logger.info(f"Checking Phase 1.5 outputs in: {latest_run}")

    missing_files = []
    for layer_idx in TARGET_LAYERS:
        text_file = latest_run / f"layer{layer_idx}_texts.json"
        if not text_file.exists():
            missing_files.append(str(text_file))

        for model_type in ['base', 'chat', 'delta']:
            act_file = latest_run / f"layer{layer_idx}_{model_type}_activations.pt"
            if not act_file.exists():
                missing_files.append(str(act_file))

    if missing_files:
        logger.error("Phase 1.5 files not found. Please run phase1_5_prepare_texts.py first.")
        logger.error("Missing files:")
        for f in missing_files[:10]:  # Show first 10
            logger.error(f"  - {f}")
        if len(missing_files) > 10:
            logger.error(f"  ... and {len(missing_files) - 10} more files")
        return False

    logger.info("✓ All Phase 1.5 files found")
    return True

def load_activation_data(sae_dir, layer_idx, model_type):
    """Load activation data and corresponding texts from the latest run.

    Args:
        sae_dir: SAE directory (e.g., outputs/sae_models/triple_sae_.../base_layer6)
        layer_idx: Layer index (6, 12, or 18)
        model_type: Model type ('base', 'chat', or 'delta')

    Returns:
        Tuple of (activations, texts) where activations is [N, hidden_dim] tensor
    """
    # Navigate from SAE dir to activation root
    # SAE dir structure: outputs/sae_models/triple_sae_*/model_type_layerX
    # Need to get to: /datadrive/anshulk/activations/run_*/

    # Find activation root using config
    runs = sorted(ACTIVATION_ROOT.glob("run_*"))
    if not runs:
        raise ValueError(f"No activation runs found in {ACTIVATION_ROOT}")

    latest_run = runs[-1]

    # Check for consolidated files from phase1_5
    act_file = latest_run / f"layer{layer_idx}_{model_type}_activations.pt"
    text_file = latest_run / f"layer{layer_idx}_texts.json"

    if not act_file.exists():
        raise FileNotFoundError(
            f"Activation file not found: {act_file}\n"
            f"Please run phase1_5_prepare_texts.py first to create consolidated files."
        )

    if not text_file.exists():
        raise FileNotFoundError(
            f"Text file not found: {text_file}\n"
            f"Please run phase1_5_prepare_texts.py first to create consolidated files."
        )

    logger.info(f"Loading activations from {act_file}")
    activations = torch.load(act_file, map_location='cpu')

    logger.info(f"Loading texts from {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)

    # Validate dimensions match
    if activations.shape[0] != len(texts):
        logger.warning(
            f"Dimension mismatch: {activations.shape[0]} activations vs {len(texts)} texts. "
            f"Using minimum of both."
        )
        min_len = min(activations.shape[0], len(texts))
        activations = activations[:min_len]
        texts = texts[:min_len]

    return activations, texts

def extract_top_examples(sae_dir, output_dir, top_k=20):
    """Extract top-k activating examples for each SAE feature."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae_name = sae_dir.name
    parts = sae_name.split('_')
    model_type = parts[0]
    layer = int(parts[1].replace('layer', ''))

    logger.info(f"Processing {sae_name} (model: {model_type}, layer: {layer})")

    # Load SAE weights
    checkpoint_path = sae_dir / "best_model.pt"
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sae_weights = checkpoint['model_state_dict']['encoder.weight']
    dict_size = sae_weights.shape[0]
    logger.info(f"SAE dictionary size: {dict_size}")

    # Load activation data (pass model_type)
    activations, texts = load_activation_data(sae_dir, layer, model_type)
    logger.info(f"Loaded {activations.shape[0]} activation samples")
    
    # Encode activations through SAE
    with torch.no_grad():
        # Handle both 2D and 3D activations
        if activations.dim() == 3:
            # Average over sequence dimension for 3D: (batch, seq, hidden)
            activations = activations.mean(dim=1)
        elif activations.dim() == 2:
            # Already 2D: (batch, hidden)
            pass
        else:
            raise ValueError(f"Unexpected activation dimensionality: {activations.dim()}")
        
        # Encode: (batch, hidden) @ (dict_size, hidden).T = (batch, dict_size)
        encoded = torch.nn.functional.relu(torch.matmul(activations, sae_weights.T))
    
    logger.info(f"Encoded shape: {encoded.shape}")
    
    results = []
    
    for feat_idx in tqdm(range(dict_size), desc=f"{sae_name} features"):
        feat_acts = encoded[:, feat_idx].numpy()
        
        # Filter features with too few activations
        nonzero_mask = feat_acts > 0
        if nonzero_mask.sum() < 10:
            continue
        
        # Get top-k activating examples
        top_indices = np.argsort(feat_acts)[-top_k:][::-1]
        
        examples = []
        for idx in top_indices:
            if feat_acts[idx] > 0:
                examples.append({
                    'text': texts[int(idx)],
                    'activation': float(feat_acts[idx]),
                    'sample_idx': int(idx)
                })
        
        # Only save features with sufficient examples
        if len(examples) >= 10:
            results.append({
                'feature_id': f"{sae_name}_feat{feat_idx}",
                'sae_name': sae_name,
                'feature_idx': feat_idx,
                'num_examples': len(examples),
                'max_activation': float(feat_acts[top_indices[0]]),
                'mean_activation': float(feat_acts[nonzero_mask].mean()),
                'sparsity': float(nonzero_mask.sum() / len(feat_acts)),
                'examples': examples
            })
    
    # Save results
    output_file = output_dir / f"{sae_name}_examples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(results)} features with >=10 examples to {output_file}")
    return len(results)

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PHASE 2.5: EXTRACT TOP-ACTIVATING EXAMPLES")
    logger.info("="*80)

    # Validate Phase 1.5 is complete
    if not validate_phase1_5_complete():
        logger.error("\n" + "="*80)
        logger.error("PREREQUISITE CHECK FAILED")
        logger.error("="*80)
        logger.error("Please run: python scripts/phase1_5_prepare_texts.py")
        sys.exit(1)

    # Find latest SAE run
    runs = sorted(SAE_OUTPUT_ROOT.glob("triple_sae_*"))
    if not runs:
        logger.error(f"No SAE runs found in {SAE_OUTPUT_ROOT}")
        logger.error("Please run Phase 2 (SAE training) first.")
        sys.exit(1)

    latest_run = runs[-1]
    output_dir = SAE_OUTPUT_ROOT / "feature_examples"

    logger.info(f"\nExtracting examples from: {latest_run}")
    logger.info(f"Output directory: {output_dir}")

    total_features = 0
    sae_dirs = sorted([d for d in latest_run.iterdir() if d.is_dir()])

    logger.info(f"\nProcessing {len(sae_dirs)} SAE models...")

    for sae_dir in sae_dirs:
        try:
            total_features += extract_top_examples(sae_dir, output_dir)
        except Exception as e:
            logger.error(f"Error processing {sae_dir.name}: {str(e)}")
            continue

    logger.info(f"\n" + "="*80)
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"="*80)
    logger.info(f"Total features with >=10 examples: {total_features}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nNext step: python scripts/phase2_5_prioritize_cultural.py")
