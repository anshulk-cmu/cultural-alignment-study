# scripts/phase2_5_create_dictionary.py
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')
import json
from pathlib import Path
from configs.config import SAE_OUTPUT_ROOT, setup_logger

logger = setup_logger('dictionary', 'phase2_5_dictionary.log')

def compute_confidence_score(feature):
    """Compute confidence score based on validation action and other factors."""
    score = 0.0
    
    # Base score from validation action
    if feature.get('validation_action') == 'KEEP':
        score = 5.0  # Highest confidence - label kept as-is
    elif feature.get('validation_action') == 'REVISE':
        score = 4.0  # High confidence - label improved
    else:
        score = 2.0  # Low confidence
    
    # Adjust based on coherence
    if not feature.get('is_coherent', False):
        score = max(1.0, score - 2.0)
    
    # Bonus for high priority
    if feature.get('priority_score', 0) >= 15:
        score = min(5.0, score + 0.5)
    
    return round(score, 1)

def determine_cultural_category(feature):
    """Determine cultural category based on label and keywords."""
    label_lower = feature.get('final_label', '').lower()
    
    # Check for specific cultural categories
    if any(kw in label_lower for kw in ['hindi', 'multilingual', 'code-switch', 'code-mixing']):
        return 'Linguistic'
    elif any(kw in label_lower for kw in ['festival', 'celebration', 'ritual', 'diwali', 'holi']):
        return 'Festivals'
    elif any(kw in label_lower for kw in ['honorific', 'respect', 'polite', 'formal']):
        return 'Social Norms'
    elif any(kw in label_lower for kw in ['food', 'cuisine', 'dish']):
        return 'Food'
    elif any(kw in label_lower for kw in ['name', 'naming']):
        return 'Names'
    elif any(kw in label_lower for kw in ['family', 'kinship', 'relation']):
        return 'Family'
    elif any(kw in label_lower for kw in ['emotion', 'sentiment', 'feeling']):
        return 'Emotional Expression'
    elif any(kw in label_lower for kw in ['dialogue', 'conversation', 'question']):
        return 'Discourse Patterns'
    else:
        return 'General Cultural'

def create_final_dictionary(prioritized_file, examples_dir, output_file, min_priority=5):
    """Create final feature dictionary from prioritized and validated features."""
    
    # Load prioritized features
    logger.info(f"Loading prioritized features from {prioritized_file}")
    with open(prioritized_file, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    logger.info(f"Loaded {len(features)} features")
    
    # Load examples map
    examples_dir = Path(examples_dir)
    examples_map = {}
    
    for file_path in examples_dir.glob("*_examples.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            feat_examples = json.load(f)
            for feat in feat_examples:
                examples_map[feat['feature_id']] = feat['examples']
    
    logger.info(f"Loaded examples for {len(examples_map)} features")
    
    dictionary = []
    
    for feat in features:
        # Filter: minimum priority score
        if feat.get('priority_score', 0) < min_priority:
            continue
        
        # Filter: must have final label
        if not feat.get('final_label'):
            continue
        
        # Get examples
        examples = examples_map.get(feat['feature_id'], [])
        
        # Compute confidence score
        confidence = compute_confidence_score(feat)
        
        # Determine cultural category
        cultural_category = determine_cultural_category(feat)
        
        # Create dictionary entry
        entry = {
            'feature_id': feat['feature_id'],
            'sae_name': feat['sae_name'],
            'feature_idx': feat['feature_idx'],
            
            # Labels
            'label_final': feat['final_label'],
            'label_initial': feat.get('label_qwen'),
            'label_validated': feat['final_label'],
            
            # Validation info
            'validation_action': feat.get('validation_action', 'UNKNOWN'),
            'validation_reason': feat.get('validation_reason', ''),
            
            # Confidence and quality
            'confidence_score': confidence,
            'is_coherent': feat.get('is_coherent', False),
            
            # Cultural relevance
            'is_cultural_candidate': feat.get('is_cultural_candidate', False),
            'cultural_category': cultural_category,
            'priority_score': feat.get('priority_score', 0),
            'matched_keywords': feat.get('matched_keywords', []),
            
            # Activation statistics
            'max_activation': feat.get('max_activation'),
            'mean_activation': feat.get('mean_activation'),
            'sparsity': feat.get('sparsity'),
            
            # Examples
            'top_examples': [
                {
                    'text': ex['text'][:500],  # Truncate long texts
                    'activation': ex['activation'],
                    'sample_idx': ex.get('sample_idx')
                }
                for ex in examples[:10]
            ],
            'num_total_examples': len(examples),
            
            # Research readiness
            'ready_for_dosa': confidence >= 4.0 and feat.get('is_cultural_candidate', False)
        }
        
        dictionary.append(entry)
    
    # Sort by cultural relevance and confidence
    dictionary.sort(
        key=lambda x: (
            x['is_cultural_candidate'],
            x['confidence_score'],
            x['priority_score']
        ),
        reverse=True
    )
    
    # Save dictionary
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved feature dictionary to {output_file}")
    
    # Compute statistics
    stats = {
        'total_features': len(dictionary),
        'cultural_candidates': sum(1 for f in dictionary if f['is_cultural_candidate']),
        'high_confidence': sum(1 for f in dictionary if f['confidence_score'] >= 4.0),
        'dosa_ready': sum(1 for f in dictionary if f['ready_for_dosa']),
        'validation_kept': sum(1 for f in dictionary if f['validation_action'] == 'KEEP'),
        'validation_revised': sum(1 for f in dictionary if f['validation_action'] == 'REVISE'),
    }
    
    # Category distribution
    category_dist = {}
    for feat in dictionary:
        if feat['is_cultural_candidate']:
            cat = feat['cultural_category']
            category_dist[cat] = category_dist.get(cat, 0) + 1
    
    # Create summary file
    summary_file = SAE_OUTPUT_ROOT / "PHASE2_5_SUMMARY.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 2.5: FEATURE LABELING SUMMARY (Qwen2.5-72B)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 80 + "\n")
        for key, val in stats.items():
            f.write(f"{key.replace('_', ' ').title()}: {val}\n")
        
        f.write("\n\nCULTURAL CATEGORY DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        for cat, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{cat}: {count}\n")
        
        f.write("\n\nTOP 30 CULTURAL FEATURES:\n")
        f.write("=" * 80 + "\n\n")
        
        cultural_features = [f for f in dictionary if f['is_cultural_candidate']]
        for i, feat in enumerate(cultural_features[:30], 1):
            f.write(f"{i}. {feat['label_final']}\n")
            f.write(f"   ID: {feat['feature_id']}\n")
            f.write(f"   Category: {feat['cultural_category']}\n")
            f.write(f"   Confidence: {feat['confidence_score']}/5.0 | Priority: {feat['priority_score']}\n")
            f.write(f"   Validation: {feat['validation_action']}\n")
            if feat.get('matched_keywords'):
                f.write(f"   Keywords: {', '.join(feat['matched_keywords'][:5])}\n")
            f.write(f"   Examples: {feat['num_total_examples']}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("NEXT STEPS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. Review {stats['dosa_ready']} features ready for DOSA validation\n")
        f.write(f"2. Proceed to Phase 3: Feature validation and analysis\n")
        f.write(f"3. Target: ≥20 RLHF-shifted cultural features for RQ1\n")
    
    logger.info("=" * 80)
    logger.info("FEATURE DICTIONARY CREATED")
    logger.info("=" * 80)
    for key, val in stats.items():
        logger.info(f"{key.replace('_', ' ').title()}: {val}")
    
    logger.info("\nCultural category distribution:")
    for cat, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat}: {count}")
    
    logger.info(f"\nSummary written to: {summary_file}")
    logger.info("=" * 80)
    
    return dictionary

if __name__ == "__main__":
    prioritized_file = SAE_OUTPUT_ROOT / "features_prioritized.json"
    examples_dir = SAE_OUTPUT_ROOT / "feature_examples"
    output_file = SAE_OUTPUT_ROOT / "FEATURE_DICTIONARY_FINAL.json"
    
    logger.info("=" * 80)
    logger.info("Creating final feature dictionary")
    logger.info(f"Input: {prioritized_file}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 80)
    
    # Create dictionary with minimum priority score of 5
    dictionary = create_final_dictionary(
        prioritized_file,
        examples_dir,
        output_file,
        min_priority=5
    )
    
    logger.info(f"\nDictionary created with {len(dictionary)} features")
    
    # Show top 10
    cultural = [f for f in dictionary if f['is_cultural_candidate']]
    if cultural:
        logger.info("\n" + "=" * 80)
        logger.info("TOP 10 CULTURAL FEATURES:")
        logger.info("=" * 80)
        
        for i, feat in enumerate(cultural[:10], 1):
            logger.info(f"\n{i}. {feat['label_final']}")
            logger.info(f"   {feat['feature_id']} | Category: {feat['cultural_category']}")
            logger.info(f"   Confidence: {feat['confidence_score']}/5.0 | Priority: {feat['priority_score']}")
