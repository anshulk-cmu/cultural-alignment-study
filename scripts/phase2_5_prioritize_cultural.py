# scripts/phase2_5_prioritize_cultural.py
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')
import json
import torch
import numpy as np
from pathlib import Path
from configs.config import SAE_OUTPUT_ROOT, setup_logger

logger = setup_logger('prioritize', 'phase2_5_prioritize.log')

def compute_cohens_d(base_acts, chat_acts):
    """Compute Cohen's d effect size between base and chat activations."""
    mean_diff = np.mean(chat_acts) - np.mean(base_acts)
    pooled_std = np.sqrt((np.std(base_acts)**2 + np.std(chat_acts)**2) / 2)
    return mean_diff / (pooled_std + 1e-8)

def load_delta_features(sae_run_dir):
    """Load all delta SAE feature IDs from the latest run."""
    delta_features = []
    
    for sae_dir in sae_run_dir.iterdir():
        if sae_dir.is_dir() and 'delta' in sae_dir.name:
            checkpoint_path = sae_dir / "best_model.pt"
            if not checkpoint_path.exists():
                logger.warning(f"No checkpoint found in {sae_dir.name}")
                continue
                
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            dict_size = checkpoint['model_state_dict']['encoder.weight'].shape[0]
            
            for i in range(dict_size):
                delta_features.append(f"{sae_dir.name}_feat{i}")
            
            logger.info(f"Loaded {dict_size} delta features from {sae_dir.name}")
    
    return set(delta_features)

def prioritize_cultural(validated_file, output_file):
    """Prioritize features based on cultural relevance and delta activation."""
    
    # Load validated features
    logger.info(f"Loading validated features from {validated_file}")
    with open(validated_file, 'r', encoding='utf-8') as f:
        validated = json.load(f)
    
    logger.info(f"Loaded {len(validated)} validated features")
    
    # Find latest SAE run and load delta features
    sae_runs = sorted(SAE_OUTPUT_ROOT.glob("triple_sae_*"))
    if not sae_runs:
        logger.error("No SAE runs found")
        return []
    
    sae_run = sae_runs[-1]
    logger.info(f"Using SAE run: {sae_run.name}")
    
    delta_feats = load_delta_features(sae_run)
    logger.info(f"Found {len(delta_feats)} delta features")
    
    # Cultural keywords for Indian English/Hindi context
    cultural_keywords = [
        # Language/linguistic patterns
        'hindi', 'indian', 'multilingual', 'code-switch', 'code-mixing', 'vernacular',
        
        # Cultural markers
        'cultural', 'festival', 'celebration', 'ritual', 'tradition',
        'diwali', 'holi', 'eid', 'navratri',
        
        # Social/communication patterns
        'honorific', 'respect', 'politeness', 'formal', 'informal',
        'dialogue', 'conversation', 'question', 'greeting',
        
        # Indian cultural concepts
        'family', 'kinship', 'relation', 'food', 'cuisine',
        'bollywood', 'cricket', 'regional', 'local',
        
        # Emotional/social
        'emotion', 'sentiment', 'affection', 'blessing',
        
        # Names/places
        'name', 'place', 'location', 'city', 'state'
    ]
    
    prioritized = []
    
    for feat in validated:
        # Skip invalidated features or those without final labels
        if feat.get('validation_action') == 'INVALIDATE' or not feat.get('final_label'):
            continue
        
        score = 0
        label_lower = feat['final_label'].lower()
        
        # +10 points: Delta feature (base-chat difference)
        if feat['feature_id'] in delta_feats:
            score += 10
            logger.debug(f"{feat['feature_id']}: +10 (delta feature)")
        
        # +5 points per cultural keyword match (max once)
        keyword_matches = []
        for keyword in cultural_keywords:
            if keyword in label_lower:
                keyword_matches.append(keyword)
        
        if keyword_matches:
            score += 5
            logger.debug(f"{feat['feature_id']}: +5 (keywords: {', '.join(keyword_matches)})")
        
        # +3 points: Feature activates on Updesh dataset (cultural content)
        # Check if any examples are from updesh dataset
        examples_str = str(feat.get('examples', [])).lower() if 'examples' in feat else ""
        if 'updesh' in examples_str or any(kw in label_lower for kw in ['indian', 'hindi', 'cultural']):
            score += 3
            logger.debug(f"{feat['feature_id']}: +3 (cultural content)")
        
        # +2 points: High activation strength (if available)
        if feat.get('max_activation', 0) > 10.0:
            score += 2
            logger.debug(f"{feat['feature_id']}: +2 (high activation)")
        
        # +2 points: Moderate sparsity (not too sparse, not too dense)
        sparsity = feat.get('sparsity', 0)
        if 0.01 < sparsity < 0.3:
            score += 2
            logger.debug(f"{feat['feature_id']}: +2 (good sparsity)")
        
        feat['priority_score'] = score
        feat['is_cultural_candidate'] = score >= 5
        feat['matched_keywords'] = keyword_matches if keyword_matches else []
        
        prioritized.append(feat)
    
    # Sort by priority score
    prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Filter cultural candidates
    cultural_features = [f for f in prioritized if f['is_cultural_candidate']]
    
    # Save all prioritized features
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prioritized, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(prioritized)} prioritized features to {output_file}")
    
    # Save cultural candidates separately
    cultural_file = SAE_OUTPUT_ROOT / "cultural_features_priority.json"
    with open(cultural_file, 'w', encoding='utf-8') as f:
        json.dump(cultural_features, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(cultural_features)} cultural candidates to {cultural_file}")
    
    # Statistics
    logger.info("=" * 80)
    logger.info("PRIORITIZATION COMPLETE")
    logger.info(f"Total validated features: {len(prioritized)}")
    logger.info(f"Cultural candidates (score ≥5): {len(cultural_features)}")
    logger.info(f"High-priority candidates (score ≥15): {sum(1 for f in cultural_features if f['priority_score'] >= 15)}")
    logger.info("=" * 80)
    
    # Score distribution
    score_distribution = {}
    for feat in prioritized:
        score = feat['priority_score']
        score_distribution[score] = score_distribution.get(score, 0) + 1
    
    logger.info("Score distribution:")
    for score in sorted(score_distribution.keys(), reverse=True)[:10]:
        logger.info(f"  Score {score}: {score_distribution[score]} features")
    
    return cultural_features

if __name__ == "__main__":
    validated_file = SAE_OUTPUT_ROOT / "labels_qwen_validated.json"
    output_file = SAE_OUTPUT_ROOT / "features_prioritized.json"
    
    logger.info("=" * 80)
    logger.info("Starting cultural feature prioritization")
    logger.info(f"Input file: {validated_file}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 80)
    
    cultural = prioritize_cultural(validated_file, output_file)
    
    if cultural:
        logger.info("\n" + "=" * 80)
        logger.info("TOP 20 CULTURAL FEATURES:")
        logger.info("=" * 80)
        
        for i, feat in enumerate(cultural[:20], 1):
            keywords = ', '.join(feat.get('matched_keywords', [])[:3])
            is_delta = "✓ DELTA" if feat['feature_id'] in cultural else ""
            
            logger.info(f"\n{i}. {feat['feature_id']}")
            logger.info(f"   Score: {feat['priority_score']} | {is_delta}")
            logger.info(f"   Label: {feat['final_label']}")
            if keywords:
                logger.info(f"   Keywords: {keywords}")
            if 'max_activation' in feat:
                logger.info(f"   Max activation: {feat['max_activation']:.3f}")
        
        logger.info("\n" + "=" * 80)
    else:
        logger.warning("No cultural features found")
