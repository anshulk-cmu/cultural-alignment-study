"""
analyze_label_corpus.py

Extract and analyze label corpus from validated features with advanced clustering
- Load validated labels
- Separate by SAE type (base/chat/delta)
- Compute TF-IDF for dominant terms
- Extract n-grams (bigrams, trigrams)
- Generate frequency distributions
- Apply UMAP + HDBSCAN clustering
- Extract topics using c-TF-IDF

Usage:
    python analyze_label_corpus.py
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import umap
import hdbscan


def setup_paths():
    sae_output = SAE_OUTPUT_ROOT
    validated_file = sae_output / "labels_qwen3_validated.json"
    output_dir = sae_output / "analysis"
    output_dir.mkdir(exist_ok=True)
    return validated_file, output_dir


def clean_label_text(label):
    label = label.strip('"\'').lower()
    label = re.sub(r'\s+', ' ', label).strip()
    return label


def classify_sae_type(sae_name):
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
    print(f"Loading labels from: {validated_file}")
    
    with open(validated_file, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    print(f"Loaded {len(features)} features\n")
    
    labels_by_type = {'base': [], 'chat': [], 'delta': [], 'unknown': []}
    
    stats = {
        'total_features': len(features),
        'kept': 0,
        'revised': 0,
        'invalidated': 0,
        'by_sae_type': defaultdict(int)
    }
    
    for feat in features:
        validation_action = feat.get('validation_action', '').upper()
        
        if validation_action == 'INVALIDATE':
            stats['invalidated'] += 1
            continue
        elif validation_action == 'KEEP':
            stats['kept'] += 1
        elif validation_action == 'REVISE':
            stats['revised'] += 1
        
        final_label = feat.get('final_label', '')
        if not final_label:
            continue
        
        cleaned_label = clean_label_text(final_label)
        sae_type = classify_sae_type(feat.get('sae_name', ''))
        
        labels_by_type[sae_type].append({
            'feature_id': feat.get('feature_id'),
            'sae_name': feat.get('sae_name'),
            'sae_type': sae_type,  # ✅ FIX: Added this line
            'label': cleaned_label,
            'validation_action': validation_action,
            'max_activation': feat.get('max_activation', 0),
            'sparsity': feat.get('sparsity', 0)
        })
        
        stats['by_sae_type'][sae_type] += 1
    
    return labels_by_type, stats


def compute_tfidf_analysis(labels_by_type, top_n=50):
    print("\n" + "="*80)
    print("TF-IDF ANALYSIS")
    print("="*80)
    
    results = {}
    
    for sae_type, features in labels_by_type.items():
        if not features:
            continue
        
        print(f"\n[{sae_type.upper()}] {len(features)} labels")
        
        labels = [f['label'] for f in features]
        
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 1),
            stop_words='english',
            token_pattern=r'\b[a-z]{3,}\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(labels)
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        term_scores = list(zip(feature_names, mean_tfidf))
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        results[sae_type] = {
            'top_terms': term_scores[:top_n],
            'total_unique_terms': len(feature_names),
            'labels': labels
        }
        
        print(f"Top 10 terms:")
        for i, (term, score) in enumerate(term_scores[:10], 1):
            print(f"  {i:2d}. {term:20s} {score:.4f}")
    
    return results


def extract_ngrams(labels_by_type, n=2, top_n=30):
    print("\n" + "="*80)
    print(f"{n}-GRAM ANALYSIS")
    print("="*80)
    
    results = {}
    
    for sae_type, features in labels_by_type.items():
        if not features:
            continue
        
        print(f"\n[{sae_type.upper()}]")
        
        labels = [f['label'] for f in features]
        
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            max_features=500,
            token_pattern=r'\b[a-z]{3,}\b'
        )
        
        ngram_matrix = vectorizer.fit_transform(labels)
        ngrams = vectorizer.get_feature_names_out()
        ngram_counts = ngram_matrix.sum(axis=0).A1
        ngram_freq = list(zip(ngrams, ngram_counts))
        ngram_freq.sort(key=lambda x: x[1], reverse=True)
        
        results[sae_type] = ngram_freq[:top_n]
        
        print(f"Top 10 {n}-grams:")
        for i, (ngram, count) in enumerate(ngram_freq[:10], 1):
            print(f"  {i:2d}. {ngram:40s} ({count:3.0f})")
    
    return results


def embed_labels(labels_data, model_name='all-MiniLM-L6-v2'):
    print("\n" + "="*80)
    print("EMBEDDING LABELS")
    print("="*80)
    
    import os
    cache_dir = Path("/data/user_data/anshulk/data/models/sentence_transformers")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir.parent)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir.parent / 'transformers')
    
    print(f"Model: {model_name}")
    print(f"Labels: {len(labels_data)}")
    
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    labels = [item['label'] for item in labels_data]
    
    print("\nEncoding...")
    embeddings = model.encode(labels, show_progress_bar=True, batch_size=32)
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def reduce_dimensions(embeddings, n_components=5, n_neighbors=15):
    print("\n" + "="*80)
    print("UMAP DIMENSIONALITY REDUCTION")
    print("="*80)
    
    print(f"n_components: {n_components}, n_neighbors: {n_neighbors}")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced shape: {reduced_embeddings.shape}")
    
    return reduced_embeddings, reducer


def cluster_with_hdbscan(reduced_embeddings, min_cluster_size=50, min_samples=10):
    print("\n" + "="*80)
    print("HDBSCAN CLUSTERING")
    print("="*80)
    
    print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nClusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        if cluster_id == -1:
            print(f"  Noise: {count:4d} ({100*count/len(cluster_labels):5.1f}%)")
        else:
            print(f"  Cluster {cluster_id:2d}: {count:4d} ({100*count/len(cluster_labels):5.1f}%)")
    
    return cluster_labels, clusterer


def extract_topics_ctfidf(labels_data, cluster_labels):
    print("\n" + "="*80)
    print("c-TF-IDF TOPIC EXTRACTION")
    print("="*80)
    
    unique_clusters = sorted(set(cluster_labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    documents_per_cluster = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:
            documents_per_cluster[cluster_id].append(labels_data[i]['label'])
    
    cluster_docs = []
    cluster_ids = []
    for cluster_id in sorted(documents_per_cluster.keys()):
        cluster_docs.append(' '.join(documents_per_cluster[cluster_id]))
        cluster_ids.append(cluster_id)
    
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        token_pattern=r'\b[a-z]{3,}\b'
    ).fit(cluster_docs)
    
    count_matrix = count_vectorizer.transform(cluster_docs)
    words = count_vectorizer.get_feature_names_out()
    
    tf = count_matrix.toarray()
    idf = np.log(tf.shape[0] / (np.sum(tf > 0, axis=0) + 1))
    ctfidf = tf * idf
    
    topics = {}
    for idx, cluster_id in enumerate(cluster_ids):
        top_indices = ctfidf[idx].argsort()[-10:][::-1]
        top_words = [(words[i], ctfidf[idx][i]) for i in top_indices]
        
        topics[cluster_id] = {
            'cluster_id': int(cluster_id),
            'size': len(documents_per_cluster[cluster_id]),
            'top_words': top_words,
            'sample_labels': documents_per_cluster[cluster_id][:5]
        }
        
        print(f"\nCluster {cluster_id} (n={topics[cluster_id]['size']})")
        words_str = ', '.join([w for w, _ in top_words[:6]])
        print(f"  Top words: {words_str}")
        print(f"  Sample: {topics[cluster_id]['sample_labels'][0][:60]}...")
    
    return topics


def analyze_cluster_distribution(labels_data, cluster_labels):
    print("\n" + "="*80)
    print("CLUSTER DISTRIBUTION BY SAE TYPE")
    print("="*80)
    
    distribution = defaultdict(lambda: defaultdict(int))
    
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:
            sae_type = labels_data[i]['sae_type']
            distribution[cluster_id][sae_type] += 1
    
    print("\n           BASE    CHAT   DELTA   TOTAL")
    print("-" * 80)
    
    for cluster_id in sorted(distribution.keys()):
        base = distribution[cluster_id]['base']
        chat = distribution[cluster_id]['chat']
        delta = distribution[cluster_id]['delta']
        total = base + chat + delta
        
        base_pct = 100 * base / total if total > 0 else 0
        chat_pct = 100 * chat / total if total > 0 else 0
        delta_pct = 100 * delta / total if total > 0 else 0
        
        print(f"Cluster {cluster_id:2d}: {base:4d} ({base_pct:4.1f}%)  "
              f"{chat:4d} ({chat_pct:4.1f}%)  {delta:4d} ({delta_pct:4.1f}%)  "
              f"{total:4d}")
    
    total_features = sum(len([d for d in labels_data if d['sae_type'] == st]) 
                        for st in ['base', 'chat', 'delta'])
    total_delta = len([d for d in labels_data if d['sae_type'] == 'delta'])
    expected_delta_prop = total_delta / total_features
    
    enrichment_scores = []
    for cluster_id in sorted(distribution.keys()):
        total = sum(distribution[cluster_id].values())
        delta = distribution[cluster_id]['delta']
        observed_delta_prop = delta / total if total > 0 else 0
        enrichment = observed_delta_prop / expected_delta_prop if expected_delta_prop > 0 else 0
        
        enrichment_scores.append({
            'cluster_id': int(cluster_id),
            'enrichment': enrichment,
            'delta_count': delta,
            'delta_pct': observed_delta_prop * 100
        })
    
    enrichment_scores.sort(key=lambda x: x['enrichment'], reverse=True)
    
    print("\n" + "="*80)
    print("DELTA SAE ENRICHMENT (RLHF-shifted themes)")
    print("="*80)
    
    for item in enrichment_scores:
        print(f"  Cluster {item['cluster_id']:2d}: {item['enrichment']:.2f}x "
              f"({item['delta_count']:3d} delta, {item['delta_pct']:.1f}%)")
    
    return distribution, enrichment_scores


def generate_report(stats, tfidf_results, bigrams, trigrams, topics, 
                   distribution, enrichment_scores, output_dir):
    report_file = output_dir / "CORPUS_ANALYSIS.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LABEL CORPUS ANALYSIS - ADVANCED CLUSTERING\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total features: {stats['total_features']}\n")
        f.write(f"  KEEP: {stats['kept']}\n")
        f.write(f"  REVISE: {stats['revised']}\n")
        f.write(f"  INVALIDATE: {stats['invalidated']}\n\n")
        
        f.write("Distribution by SAE Type:\n")
        for sae_type, count in stats['by_sae_type'].items():
            f.write(f"  {sae_type.upper():10s}: {count:4d}\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("TF-IDF TOP TERMS\n")
        f.write("="*80 + "\n")
        
        for sae_type, result in tfidf_results.items():
            if not result:
                continue
            f.write(f"\n[{sae_type.upper()}] Top 20:\n")
            f.write("-"*80 + "\n")
            for i, (term, score) in enumerate(result['top_terms'][:20], 1):
                f.write(f"{i:3d}. {term:25s} {score:.4f}\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("BIGRAMS\n")
        f.write("="*80 + "\n")
        
        for sae_type, ngram_list in bigrams.items():
            f.write(f"\n[{sae_type.upper()}]:\n")
            f.write("-"*80 + "\n")
            for i, (ngram, count) in enumerate(ngram_list[:15], 1):
                f.write(f"{i:3d}. {ngram:45s} ({count:3.0f})\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("TRIGRAMS\n")
        f.write("="*80 + "\n")
        
        for sae_type, ngram_list in trigrams.items():
            f.write(f"\n[{sae_type.upper()}]:\n")
            f.write("-"*80 + "\n")
            for i, (ngram, count) in enumerate(ngram_list[:15], 1):
                f.write(f"{i:3d}. {ngram:50s} ({count:3.0f})\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("DISCOVERED TOPICS (c-TF-IDF)\n")
        f.write("="*80 + "\n\n")
        
        for cluster_id, info in sorted(topics.items()):
            f.write(f"CLUSTER {cluster_id}\n")
            f.write("-"*80 + "\n")
            f.write(f"Size: {info['size']}\n\n")
            
            f.write("Top words:\n")
            for word, score in info['top_words']:
                f.write(f"  {word:30s} {score:.4f}\n")
            
            f.write("\nSample labels:\n")
            for label in info['sample_labels']:
                f.write(f"  {label}\n")
            
            base = distribution[cluster_id]['base']
            chat = distribution[cluster_id]['chat']
            delta = distribution[cluster_id]['delta']
            total = base + chat + delta
            
            f.write(f"\nDistribution:\n")
            f.write(f"  Base:  {base:4d} ({100*base/total:5.1f}%)\n")
            f.write(f"  Chat:  {chat:4d} ({100*chat/total:5.1f}%)\n")
            f.write(f"  Delta: {delta:4d} ({100*delta/total:5.1f}%)\n\n\n")
        
        f.write("="*80 + "\n")
        f.write("RLHF IMPACT (Delta Enrichment)\n")
        f.write("="*80 + "\n\n")
        
        for item in enrichment_scores[:10]:
            cluster_id = item['cluster_id']
            f.write(f"Cluster {cluster_id}: {item['enrichment']:.2f}x enrichment\n")
            f.write(f"  Delta: {item['delta_count']} ({item['delta_pct']:.1f}%)\n")
            f.write(f"  Top words: {', '.join([w for w, _ in topics[cluster_id]['top_words'][:5]])}\n\n")
    
    print(f"\nReport saved: {report_file}")


def save_results(tfidf_results, bigrams, trigrams, stats, embeddings, 
                reduced_embeddings, cluster_labels, topics, distribution, 
                enrichment_scores, labels_data, output_dir):
    
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
        },
        'clustering': {
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': int(list(cluster_labels).count(-1)),
            'topics': {
                str(k): {
                    'cluster_id': v['cluster_id'],
                    'size': v['size'],
                    'top_words': [(w, float(s)) for w, s in v['top_words']],
                    'sample_labels': v['sample_labels']
                }
                for k, v in topics.items()
            },
            'distribution': {str(k): dict(v) for k, v in distribution.items()},
            'enrichment': enrichment_scores
        },
        'feature_assignments': [
            {
                'feature_id': labels_data[i]['feature_id'],
                'label': labels_data[i]['label'],
                'sae_type': labels_data[i]['sae_type'],
                'cluster_id': cluster_labels[i].item()
            }
            for i in range(len(labels_data))
        ]
    }
    
    with open(output_dir / "corpus_analysis_data.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "reduced_embeddings.npy", reduced_embeddings)
    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    
    print("Saved: corpus_analysis_data.json")
    print("Saved: embeddings.npy, reduced_embeddings.npy, cluster_labels.npy")


def main():
    print("="*80)
    print("LABEL CORPUS ANALYSIS - UMAP + HDBSCAN + c-TF-IDF")
    print("="*80)
    
    validated_file, output_dir = setup_paths()
    
    if not validated_file.exists():
        print(f"ERROR: {validated_file} not found")
        return 1
    
    labels_by_type, stats = load_and_parse_labels(validated_file)
    
    print("\n" + "="*80)
    print("SAE TYPE DISTRIBUTION")
    print("="*80)
    for sae_type, count in stats['by_sae_type'].items():
        print(f"  {sae_type.upper():10s}: {count:4d}")
    
    tfidf_results = compute_tfidf_analysis(labels_by_type, top_n=50)
    bigrams = extract_ngrams(labels_by_type, n=2, top_n=30)
    trigrams = extract_ngrams(labels_by_type, n=3, top_n=30)
    
    all_labels = []
    for sae_type in ['base', 'chat', 'delta']:
        all_labels.extend(labels_by_type[sae_type])
    
    embeddings = embed_labels(all_labels)
    reduced_embeddings, reducer = reduce_dimensions(embeddings, n_components=5, n_neighbors=15)
    cluster_labels, clusterer = cluster_with_hdbscan(reduced_embeddings, min_cluster_size=50, min_samples=10)
    topics = extract_topics_ctfidf(all_labels, cluster_labels)
    distribution, enrichment_scores = analyze_cluster_distribution(all_labels, cluster_labels)
    
    generate_report(stats, tfidf_results, bigrams, trigrams, topics, 
                   distribution, enrichment_scores, output_dir)
    
    save_results(tfidf_results, bigrams, trigrams, stats, embeddings,
                reduced_embeddings, cluster_labels, topics, distribution,
                enrichment_scores, all_labels, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_pct = 100 * n_noise / len(cluster_labels)
    
    print(f"\nClusters discovered: {n_clusters}")
    print(f"Noise points: {n_noise} ({noise_pct:.1f}%)")
    print(f"Total features analyzed: {len(all_labels)}")
    
    top_enriched = enrichment_scores[:3]
    print("\nTop 3 RLHF-enriched clusters:")
    for item in top_enriched:
        print(f"  Cluster {item['cluster_id']}: {item['enrichment']:.2f}x enrichment")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
