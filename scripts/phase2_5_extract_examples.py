# scripts/phase2_5_extract_examples.py
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')
import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from configs.config import ACTIVATION_ROOT, SAE_OUTPUT_ROOT, setup_logger

logger = setup_logger('extract_examples', 'phase2_5_extract.log')

def load_activation_data(sae_dir, layer_idx):
    """Load activation data and corresponding texts from the latest run."""
    act_dir = Path(sae_dir).parent.parent.parent / "activations"
    runs = sorted(act_dir.glob("run_*"))
    if not runs:
        raise ValueError(f"No activation runs found in {act_dir}")
    
    latest_run = runs[-1]
    act_file = latest_run / f"layer{layer_idx}_activations.pt"
    text_file = latest_run / f"layer{layer_idx}_texts.json"
    
    logger.info(f"Loading activations from {act_file}")
    activations = torch.load(act_file, map_location='cpu')
    
    logger.info(f"Loading texts from {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    
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
    
    # Load activation data
    activations, texts = load_activation_data(sae_dir, layer)
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
    # Find latest SAE run
    runs = sorted(SAE_OUTPUT_ROOT.glob("triple_sae_*"))
    if not runs:
        logger.error(f"No SAE runs found in {SAE_OUTPUT_ROOT}")
        sys.exit(1)
    
    latest_run = runs[-1]
    output_dir = SAE_OUTPUT_ROOT / "feature_examples"
    
    logger.info(f"Extracting examples from: {latest_run}")
    logger.info(f"Output directory: {output_dir}")
    
    total_features = 0
    sae_dirs = sorted([d for d in latest_run.iterdir() if d.is_dir()])
    
    for sae_dir in sae_dirs:
        try:
            total_features += extract_top_examples(sae_dir, output_dir)
        except Exception as e:
            logger.error(f"Error processing {sae_dir.name}: {str(e)}")
            continue
    
    logger.info(f"=" * 80)
    logger.info(f"COMPLETE: Total features with >=10 examples: {total_features}")
    logger.info(f"Results saved to: {output_dir}")
