"""
analyze_label_corpus.py

Phase 1, Step 1: Extract and analyze label corpus from validated features
- Load all 3,600 validated labels
- Separate by SAE type (base/chat/delta)
- Compute TF-IDF to identify dominant terms
- Extract n-grams (bigrams, trigrams)
- Generate frequency distributions
- Output empirical taxonomy foundation

Usage:
    python analyze_label_corpus.py
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# Add project to path and import config
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT

# For TF-IDF and n-gram extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def setup_paths():
    """Setup project paths"""
    sae_output = SAE_OUTPUT_ROOT
    
    validated_file = sae_output / "labels_qwen3_validated.json"
    output_dir = sae_output / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    return validated_file, output_dir


def clean_label_text(label):
    """Clean label text for analysis"""
    # Remove quotes
    label = label.strip('"\'')
    # Lowercase for consistency
    label = label.lower()
    # Remove extra whitespace
    label = re.sub(r'\s+', ' ', label).strip()
    return label


def classify_sae_type(sae_name):
    """Classify SAE as base, chat, or delta"""
    sae_name_lower = sae_name.lower()
    if 'delta' in sae_name_lower:
        return 'delta'
    elif 'chat' in sae_name_lower:
        return 'chat'
    elif 'base' in sae_name_lower:
        return 'base'
    else:
        return 'unknown'


def load_and_parse_labels(validated_file):
    """Load validated labels and organize by SAE type"""
    print(f"Loading validated labels from: {validated_file}")
    
    with open(validated_file, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    print(f"✓ Loaded {len(features)} validated features\n")
    
    # Organize by SAE type
    labels_by_type = {
        'base': [],
        'chat': [],
        'delta': [],
        'unknown': []
    }
    
    # Track statistics
    stats = {
        'total_features': len(features),
        'kept': 0,
        'revised': 0,
        'invalidated': 0,
        'by_sae_type': defaultdict(int)
    }
    
    for feat in features:
        # Skip invalidated features
        validation_action = feat.get('validation_action', '').upper()
        
        if validation_action == 'INVALIDATE':
            stats['invalidated'] += 1
            continue
        elif validation_action == 'KEEP':
            stats['kept'] += 1
        elif validation_action == 'REVISE':
            stats['revised'] += 1
        
        # Get final label
        final_label = feat.get('final_label', '')
        if not final_label:
            continue
        
        # Clean and classify
        cleaned_label = clean_label_text(final_label)
        sae_type = classify_sae_type(feat.get('sae_name', ''))
        
        labels_by_type[sae_type].append({
            'feature_id': feat.get('feature_id'),
            'sae_name': feat.get('sae_name'),
            'label': cleaned_label,
            'validation_action': validation_action,
            'max_activation': feat.get('max_activation', 0),
            'sparsity': feat.get('sparsity', 0)
        })
        
        stats['by_sae_type'][sae_type] += 1
    
    return labels_by_type, stats


def compute_tfidf_analysis(labels_by_type, top_n=50):
    """Compute TF-IDF scores to identify important terms"""
    print("\n" + "="*80)
    print("TF-IDF ANALYSIS: Identifying Dominant Terms")
    print("="*80)
    
    results = {}
    
    for sae_type, features in labels_by_type.items():
        if not features:
            continue
        
        print(f"\n[{sae_type.upper()}] Analyzing {len(features)} labels...")
        
        # Extract label texts
        labels = [f['label'] for f in features]
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 1),  # Single words only
            stop_words='english',
            token_pattern=r'\b[a-z]{3,}\b'  # At least 3 chars
        )
        
        tfidf_matrix = vectorizer.fit_transform(labels)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF score for each term
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        term_scores = list(zip(feature_names, mean_tfidf))
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        results[sae_type] = {
            'top_terms': term_scores[:top_n],
            'total_unique_terms': len(feature_names),
            'labels': labels
        }
        
        # Print top 20
        print(f"\nTop 20 Terms by TF-IDF:")
        for i, (term, score) in enumerate(term_scores[:20], 1):
            print(f"  {i:2d}. {term:20s} {score:.4f}")
    
    return results


def extract_ngrams(labels_by_type, n=2, top_n=30):
    """Extract most common n-grams (phrases)"""
    print("\n" + "="*80)
    print(f"N-GRAM ANALYSIS: Extracting Common {n}-word Phrases")
    print("="*80)
    
    results = {}
    
    for sae_type, features in labels_by_type.items():
        if not features:
            continue
        
        print(f"\n[{sae_type.upper()}]")
        
        labels = [f['label'] for f in features]
        
        # Extract n-grams
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            max_features=500,
            token_pattern=r'\b[a-z]{3,}\b'
        )
        
        ngram_matrix = vectorizer.fit_transform(labels)
        ngrams = vectorizer.get_feature_names_out()
        
        # Count frequencies
        ngram_counts = ngram_matrix.sum(axis=0).A1
        ngram_freq = list(zip(ngrams, ngram_counts))
        ngram_freq.sort(key=lambda x: x[1], reverse=True)
        
        results[sae_type] = ngram_freq[:top_n]
        
        # Print top items
        print(f"Top {min(20, top_n)} {n}-grams:")
        for i, (ngram, count) in enumerate(ngram_freq[:20], 1):
            print(f"  {i:2d}. {ngram:40s} ({count:3.0f} occurrences)")
    
    return results





def generate_summary_report(stats, tfidf_results, bigrams, trigrams, output_dir):
    """Generate comprehensive text report"""
    report_file = output_dir / "LABEL_CORPUS_ANALYSIS.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LABEL CORPUS ANALYSIS - EMPIRICAL TAXONOMY DISCOVERY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("DATASET STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total features analyzed: {stats['total_features']}\n")
        f.write(f"  Validation KEEP: {stats['kept']}\n")
        f.write(f"  Validation REVISE: {stats['revised']}\n")
        f.write(f"  Validation INVALIDATE: {stats['invalidated']}\n\n")
        
        f.write("Distribution by SAE Type:\n")
        for sae_type, count in stats['by_sae_type'].items():
            f.write(f"  {sae_type.upper():10s}: {count:4d} features\n")
        
        # TF-IDF results
        f.write("\n\n" + "="*80 + "\n")
        f.write("TOP TERMS BY TF-IDF SCORE\n")
        f.write("="*80 + "\n")
        
        for sae_type, result in tfidf_results.items():
            if not result:
                continue
            f.write(f"\n[{sae_type.upper()}] Top 30 Terms:\n")
            f.write("-"*80 + "\n")
            for i, (term, score) in enumerate(result['top_terms'][:30], 1):
                f.write(f"{i:3d}. {term:25s} {score:.4f}\n")
        
        # N-grams
        f.write("\n\n" + "="*80 + "\n")
        f.write("MOST COMMON BIGRAMS (2-word phrases)\n")
        f.write("="*80 + "\n")
        
        for sae_type, ngram_list in bigrams.items():
            f.write(f"\n[{sae_type.upper()}]:\n")
            f.write("-"*80 + "\n")
            for i, (ngram, count) in enumerate(ngram_list[:20], 1):
                f.write(f"{i:3d}. {ngram:45s} ({count:3.0f})\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("MOST COMMON TRIGRAMS (3-word phrases)\n")
        f.write("="*80 + "\n")
        
        for sae_type, ngram_list in trigrams.items():
            f.write(f"\n[{sae_type.upper()}]:\n")
            f.write("-"*80 + "\n")
            for i, (ngram, count) in enumerate(ngram_list[:20], 1):
                f.write(f"{i:3d}. {ngram:50s} ({count:3.0f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("="*80 + "\n")
        f.write("\nNext Steps:\n")
        f.write("1. Review top terms and phrases across base/chat/delta SAEs\n")
        f.write("2. Run cluster_semantic_themes.py for unsupervised category discovery\n")
        f.write("3. Build empirical taxonomy in build_empirical_taxonomy.py\n")
    
    print(f"\n✓ Report saved to: {report_file}")


def save_structured_data(tfidf_results, bigrams, trigrams, stats, output_dir):
    """Save structured JSON data for downstream analysis"""
    output_file = output_dir / "label_corpus_data.json"
    
    # Prepare data (convert numpy types to native Python)
    data = {
        'statistics': stats,
        'tfidf_terms': {
            sae_type: {
                'top_50_terms': [(term, float(score)) for term, score in result['top_terms']],
                'total_unique_terms': result['total_unique_terms']
            }
            for sae_type, result in tfidf_results.items() if result
        },
        'bigrams': {
            sae_type: [(ngram, int(count)) for ngram, count in ngram_list]
            for sae_type, ngram_list in bigrams.items()
        },
        'trigrams': {
            sae_type: [(ngram, int(count)) for ngram, count in ngram_list]
            for sae_type, ngram_list in trigrams.items()
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Structured data saved to: {output_file}")


def main():
    print("="*80)
    print("LABEL CORPUS ANALYSIS - Empirical Taxonomy Discovery")
    print("="*80)
    print("\nThis script analyzes all 3,600 validated feature labels to:")
    print("  1. Identify dominant terms via TF-IDF")
    print("  2. Extract common phrases (bigrams, trigrams)")
    print("  3. Compare term distributions across base/chat/delta SAEs")
    print("  4. Prepare data for unsupervised clustering (next script)")
    print()
    
    # Setup
    validated_file, output_dir = setup_paths()
    
    if not validated_file.exists():
        print(f"ERROR: Validated labels file not found: {validated_file}")
        print("\nPlease ensure Phase 2.5b validation is complete.")
        return 1
    
    # Load and parse
    labels_by_type, stats = load_and_parse_labels(validated_file)
    
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION BY SAE TYPE")
    print("="*80)
    for sae_type, count in stats['by_sae_type'].items():
        print(f"  {sae_type.upper():10s}: {count:4d} features")
    
    # TF-IDF analysis
    tfidf_results = compute_tfidf_analysis(labels_by_type, top_n=50)
    
    # N-gram extraction
    bigrams = extract_ngrams(labels_by_type, n=2, top_n=30)
    trigrams = extract_ngrams(labels_by_type, n=3, top_n=30)
    
    # Generate outputs
    generate_summary_report(stats, tfidf_results, bigrams, trigrams, output_dir)
    
    save_structured_data(tfidf_results, bigrams, trigrams, stats, output_dir)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nKey findings:")
    print(f"  - Analyzed {stats['total_features']} features")
    print(f"  - Base SAEs: {stats['by_sae_type']['base']} features")
    print(f"  - Chat SAEs: {stats['by_sae_type']['chat']} features")
    print(f"  - Delta SAEs: {stats['by_sae_type']['delta']} features")
    print("\nNext Steps:")
    print("  1. Review LABEL_CORPUS_ANALYSIS.txt for dominant terms/phrases")
    print("  2. Run cluster_semantic_themes.py for unsupervised category discovery")
    print("  3. Categories will emerge from data, not predefined assumptions")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
