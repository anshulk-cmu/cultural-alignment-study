"""
cluster_semantic_themes.py

Phase 1, Step 2: Unsupervised clustering to discover natural semantic themes
- Embed all 3,600 validated labels using sentence transformers
- Run K-means clustering for k=8, 9, 10, 11, 12
- Generate detailed reports for each k value
- Compare clusterings side-by-side for manual selection
- Analyze cluster distributions across base/chat/delta SAEs
- Output empirical taxonomy candidates for review

Usage:
    python cluster_semantic_themes.py
    
    Then manually review outputs in:
    outputs/sae_models/analysis/clustering/COMPARISON_REPORT.txt
    outputs/sae_models/analysis/clustering/k8/, k9/, k10/, k11/, k12/
    
Requirements:
    pip install sentence-transformers scikit-learn matplotlib seaborn scipy
"""

import sys
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# Add project to path
sys.path.append('/home/anshulk/cultural-alignment-study')
from configs.config import SAE_OUTPUT_ROOT

# Clustering and embedding
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine

# Suppress warnings
warnings.filterwarnings('ignore')

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for HPC
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, skipping visualizations")


def setup_paths():
    """Setup input/output paths"""
    analysis_dir = SAE_OUTPUT_ROOT / "analysis"
    
    validated_file = SAE_OUTPUT_ROOT / "labels_qwen3_validated.json"
    corpus_data_file = analysis_dir / "label_corpus_data.json"
    
    # Create clustering output directory
    clustering_dir = analysis_dir / "clustering"
    clustering_dir.mkdir(exist_ok=True)
    
    return validated_file, corpus_data_file, clustering_dir


def load_labels_data(validated_file):
    """Load validated labels with metadata"""
    print(f"Loading labels from: {validated_file}")
    
    with open(validated_file, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    print(f"✓ Loaded {len(features)} validated features\n")
    
    # Extract labels and metadata
    labels_data = []
    
    for feat in features:
        if feat.get('validation_action') == 'INVALIDATE':
            continue
        
        final_label = feat.get('final_label', '').strip('"\'').lower()
        if not final_label:
            continue
        
        sae_name = feat.get('sae_name', '')
        sae_type = 'delta' if 'delta' in sae_name.lower() else \
                   'chat' if 'chat' in sae_name.lower() else \
                   'base' if 'base' in sae_name.lower() else 'unknown'
        
        labels_data.append({
            'feature_id': feat.get('feature_id'),
            'label': final_label,
            'sae_type': sae_type,
            'sae_name': sae_name,
            'validation_action': feat.get('validation_action'),
            'max_activation': feat.get('max_activation', 0),
            'sparsity': feat.get('sparsity', 0)
        })
    
    return labels_data


def embed_labels(labels_data, model_name='all-MiniLM-L6-v2'):
    """Embed labels using sentence transformers"""
    print("="*80)
    print("EMBEDDING LABELS WITH SENTENCE TRANSFORMERS")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Labels to embed: {len(labels_data)}")
    
    # Set cache directory to user's writable location
    import os
    cache_dir = Path("/data/user_data/anshulk/data/models/sentence_transformers")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Override HF environment variables
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir.parent)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir.parent / 'transformers')
    
    print(f"\nCache directory: {cache_dir}")
    
    # Load model (will download if not cached)
    print("\nLoading sentence transformer model...")
    print("(First run will download ~90MB model - may take 1-2 minutes)")
    
    try:
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        print("\nIf on HPC compute node without internet:")
        print("1. Run this script on login node first to download model")
        print("2. Or pre-download with:")
        print(f"   python -c 'from sentence_transformers import SentenceTransformer; "
              f"SentenceTransformer(\"{model_name}\", cache_folder=\"{cache_dir}\")'")
        raise
    
    # Extract label texts
    labels = [item['label'] for item in labels_data]
    
    # Encode (with progress bar)
    print("\nEncoding labels (this may take 2-5 minutes)...")
    embeddings = model.encode(labels, 
                              show_progress_bar=True,
                              batch_size=32,
                              convert_to_numpy=True)
    
    print(f"\n✓ Embeddings shape: {embeddings.shape}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, model


def find_optimal_clusters(embeddings, min_k=5, max_k=20):
    """Find optimal number of clusters using silhouette analysis"""
    print("\n" + "="*80)
    print("FINDING OPTIMAL CLUSTER COUNT")
    print("="*80)
    print(f"\nTesting cluster counts from {min_k} to {max_k}...")
    
    silhouette_scores = []
    inertias = []
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)
        
        print(f"  k={k:2d}: silhouette={silhouette_avg:.4f}, inertia={kmeans.inertia_:.0f}")
    
    # Find best k (highest silhouette score)
    best_idx = np.argmax(silhouette_scores)
    best_k = min_k + best_idx
    best_score = silhouette_scores[best_idx]
    
    print(f"\n✓ Optimal cluster count: k={best_k} (silhouette={best_score:.4f})")
    
    return best_k, silhouette_scores, inertias, list(range(min_k, max_k + 1))


def perform_kmeans_clustering(embeddings, n_clusters):
    """Perform K-means clustering with optimal k"""
    print("\n" + "="*80)
    print(f"K-MEANS CLUSTERING (k={n_clusters})")
    print("="*80)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute per-sample silhouette scores
    silhouette_vals = silhouette_samples(embeddings, cluster_labels)
    
    print(f"\n✓ Clustering complete")
    print(f"  Cluster distribution:")
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = 100 * count / len(cluster_labels)
        avg_silhouette = silhouette_vals[cluster_labels == cluster_id].mean()
        print(f"    Cluster {cluster_id:2d}: {count:4d} features ({pct:5.1f}%) "
              f"| silhouette: {avg_silhouette:.3f}")
    
    return cluster_labels, silhouette_vals, kmeans


def perform_hierarchical_clustering(embeddings, n_clusters):
    """Perform hierarchical clustering for validation"""
    print("\n" + "="*80)
    print(f"HIERARCHICAL CLUSTERING (k={n_clusters})")
    print("="*80)
    
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hierarchical.fit_predict(embeddings)
    
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    
    print(f"\n✓ Hierarchical clustering complete")
    print(f"  Average silhouette score: {silhouette_avg:.4f}")
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = 100 * count / len(cluster_labels)
        print(f"    Cluster {cluster_id:2d}: {count:4d} features ({pct:5.1f}%)")
    
    return cluster_labels, silhouette_avg


def generate_cluster_labels(labels_data, cluster_assignments, n_top_terms=10):
    """Generate interpretable labels for each cluster"""
    print("\n" + "="*80)
    print("GENERATING CLUSTER LABELS")
    print("="*80)
    
    n_clusters = len(set(cluster_assignments))
    cluster_info = {}
    
    for cluster_id in range(n_clusters):
        # Get all labels in this cluster
        cluster_mask = cluster_assignments == cluster_id
        cluster_labels = [labels_data[i]['label'] for i in range(len(labels_data)) 
                         if cluster_mask[i]]
        
        # Extract most common terms (simple word frequency)
        all_words = []
        for label in cluster_labels:
            words = label.split()
            # Filter out very common words
            filtered = [w for w in words if len(w) > 3 and 
                       w not in {'and', 'the', 'with', 'from', 'that', 'this', 
                                'their', 'which', 'related', 'various'}]
            all_words.extend(filtered)
        
        word_freq = Counter(all_words)
        top_terms = word_freq.most_common(n_top_terms)
        
        # Generate descriptive label from top terms
        if top_terms:
            # Use top 3-5 most distinctive terms
            label_words = [term for term, _ in top_terms[:5]]
            auto_label = " & ".join(label_words[:3]).title()
        else:
            auto_label = f"Cluster {cluster_id}"
        
        # Get sample labels
        sample_labels = cluster_labels[:5]
        
        cluster_info[cluster_id] = {
            'cluster_id': cluster_id,
            'size': int(np.sum(cluster_mask)),
            'auto_label': auto_label,
            'top_terms': [(term, int(count)) for term, count in top_terms],
            'sample_labels': sample_labels
        }
        
        print(f"\nCluster {cluster_id}: {auto_label}")
        print(f"  Size: {cluster_info[cluster_id]['size']} features")
        print(f"  Top terms: {', '.join([t for t, _ in top_terms[:5]])}")
        print(f"  Sample: {sample_labels[0][:60]}...")
    
    return cluster_info


def analyze_cluster_distribution(labels_data, cluster_assignments):
    """Analyze how clusters distribute across base/chat/delta SAEs"""
    print("\n" + "="*80)
    print("CLUSTER DISTRIBUTION ACROSS SAE TYPES")
    print("="*80)
    
    n_clusters = len(set(cluster_assignments))
    
    # Count by cluster and SAE type
    distribution = defaultdict(lambda: defaultdict(int))
    
    for i, cluster_id in enumerate(cluster_assignments):
        sae_type = labels_data[i]['sae_type']
        distribution[cluster_id][sae_type] += 1
    
    # Print distribution table
    print("\n           BASE    CHAT   DELTA   TOTAL")
    print("-" * 80)
    
    for cluster_id in range(n_clusters):
        base = distribution[cluster_id]['base']
        chat = distribution[cluster_id]['chat']
        delta = distribution[cluster_id]['delta']
        total = base + chat + delta
        
        # Calculate percentages
        base_pct = 100 * base / total if total > 0 else 0
        chat_pct = 100 * chat / total if total > 0 else 0
        delta_pct = 100 * delta / total if total > 0 else 0
        
        print(f"Cluster {cluster_id:2d}: {base:4d} ({base_pct:4.1f}%)  "
              f"{chat:4d} ({chat_pct:4.1f}%)  {delta:4d} ({delta_pct:4.1f}%)  "
              f"{total:4d}")
    
    # Calculate enrichment (which clusters are overrepresented in delta?)
    print("\n" + "="*80)
    print("DELTA SAE ENRICHMENT ANALYSIS")
    print("="*80)
    print("\nClusters enriched in Delta SAEs (RLHF-shifted themes):\n")
    
    enrichment_scores = []
    
    for cluster_id in range(n_clusters):
        total = sum(distribution[cluster_id].values())
        delta = distribution[cluster_id]['delta']
        
        # Expected delta proportion (overall dataset)
        total_features = sum(len([d for d in labels_data if d['sae_type'] == st]) 
                           for st in ['base', 'chat', 'delta'])
        total_delta = len([d for d in labels_data if d['sae_type'] == 'delta'])
        expected_delta_prop = total_delta / total_features
        
        # Observed delta proportion in this cluster
        observed_delta_prop = delta / total if total > 0 else 0
        
        # Enrichment score (observed / expected)
        enrichment = observed_delta_prop / expected_delta_prop if expected_delta_prop > 0 else 0
        
        enrichment_scores.append({
            'cluster_id': cluster_id,
            'enrichment': enrichment,
            'delta_count': delta,
            'delta_pct': observed_delta_prop * 100
        })
    
    # Sort by enrichment
    enrichment_scores.sort(key=lambda x: x['enrichment'], reverse=True)
    
    for item in enrichment_scores[:10]:  # Top 10 enriched
        print(f"  Cluster {item['cluster_id']:2d}: {item['enrichment']:.2f}x enrichment "
              f"({item['delta_count']:3d} delta features, {item['delta_pct']:.1f}%)")
    
    return distribution, enrichment_scores


def save_clustering_results(labels_data, embeddings, cluster_assignments, 
                            cluster_info, distribution, enrichment_scores,
                            silhouette_vals, clustering_dir):
    """Save comprehensive clustering results"""
    output_file = clustering_dir / "clustering_results.json"
    
    # Prepare results
    results = {
        'n_clusters': len(cluster_info),
        'total_features': len(labels_data),
        'cluster_info': cluster_info,
        'enrichment_scores': enrichment_scores,
        'feature_assignments': [
            {
                'feature_id': labels_data[i]['feature_id'],
                'label': labels_data[i]['label'],
                'sae_type': labels_data[i]['sae_type'],
                'cluster_id': int(cluster_assignments[i]),
                'silhouette_score': float(silhouette_vals[i])
            }
            for i in range(len(labels_data))
        ],
        'distribution_by_cluster': {
            str(k): dict(v) for k, v in distribution.items()
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Clustering results saved to: {output_file}")
    
    # Save embeddings separately (large file)
    embeddings_file = clustering_dir / "label_embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"✓ Embeddings saved to: {embeddings_file}")


def generate_report(cluster_info, enrichment_scores, distribution, 
                   labels_data, cluster_assignments, clustering_dir):
    """Generate human-readable clustering report"""
    report_file = clustering_dir / "CLUSTERING_ANALYSIS.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SEMANTIC CLUSTERING ANALYSIS - EMPIRICAL TAXONOMY\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total features: {len(labels_data)}\n")
        f.write(f"Number of clusters: {len(cluster_info)}\n\n")
        
        f.write("DISCOVERED THEMES\n")
        f.write("="*80 + "\n\n")
        
        for cluster_id, info in cluster_info.items():
            f.write(f"CLUSTER {cluster_id}: {info['auto_label']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Size: {info['size']} features\n\n")
            
            f.write("Top terms:\n")
            for term, count in info['top_terms'][:10]:
                f.write(f"  • {term} ({count})\n")
            
            f.write("\nSample labels:\n")
            for label in info['sample_labels']:
                f.write(f"  • {label}\n")
            
            # Distribution across SAE types
            base = distribution[cluster_id]['base']
            chat = distribution[cluster_id]['chat']
            delta = distribution[cluster_id]['delta']
            total = base + chat + delta
            
            f.write(f"\nDistribution:\n")
            f.write(f"  Base:  {base:4d} ({100*base/total:5.1f}%)\n")
            f.write(f"  Chat:  {chat:4d} ({100*chat/total:5.1f}%)\n")
            f.write(f"  Delta: {delta:4d} ({100*delta/total:5.1f}%)\n")
            
            f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("RLHF IMPACT ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Clusters most enriched in Delta SAEs (RLHF-shifted themes):\n\n")
        
        for item in enrichment_scores[:10]:
            cluster_id = item['cluster_id']
            f.write(f"{cluster_id:2d}. {cluster_info[cluster_id]['auto_label']}\n")
            f.write(f"    Enrichment: {item['enrichment']:.2f}x\n")
            f.write(f"    Delta features: {item['delta_count']} ({item['delta_pct']:.1f}%)\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("="*80 + "\n")
        f.write("1. Review discovered themes and their distribution\n")
        f.write("2. Identify RLHF-shifted clusters for deeper analysis\n")
        f.write("3. Run build_empirical_taxonomy.py to formalize hierarchy\n")
        f.write("4. Proceed to Cohen's d calculation for priority scoring\n")
    
    print(f"✓ Report saved to: {report_file}")


def plot_silhouette_analysis(silhouette_scores, k_values, clustering_dir):
    """Plot silhouette scores for cluster selection"""
    if not PLOTTING_AVAILABLE:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Optimal Cluster Count Selection', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Mark best k
    best_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_idx]
    plt.axvline(best_k, color='r', linestyle='--', alpha=0.5, 
                label=f'Optimal k={best_k}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(clustering_dir / "silhouette_analysis.png", dpi=150)
    plt.close()
    
    print(f"✓ Silhouette plot saved")


def plot_cluster_distribution(distribution, cluster_info, clustering_dir):
    """Plot cluster distribution across SAE types"""
    if not PLOTTING_AVAILABLE:
        return
    
    n_clusters = len(distribution)
    
    # Prepare data
    clusters = list(range(n_clusters))
    base_counts = [distribution[i]['base'] for i in clusters]
    chat_counts = [distribution[i]['chat'] for i in clusters]
    delta_counts = [distribution[i]['delta'] for i in clusters]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(n_clusters)
    width = 0.6
    
    p1 = ax.bar(x, base_counts, width, label='Base', color='#3498db')
    p2 = ax.bar(x, chat_counts, width, bottom=base_counts, label='Chat', color='#2ecc71')
    p3 = ax.bar(x, delta_counts, width, 
                bottom=np.array(base_counts) + np.array(chat_counts),
                label='Delta', color='#e74c3c')
    
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Cluster Distribution Across SAE Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(clustering_dir / "cluster_distribution.png", dpi=150)
    plt.close()
    
    print(f"✓ Distribution plot saved")


def generate_comparison_report(all_results, clustering_dir):
    """Generate comparison report across different k values"""
    report_file = clustering_dir / "COMPARISON_REPORT.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CLUSTERING COMPARISON: k=8 through k=12\n")
        f.write("="*80 + "\n\n")
        
        f.write("INSTRUCTIONS FOR MANUAL SELECTION\n")
        f.write("-"*80 + "\n")
        f.write("Review each clustering (k=8-12) and select based on:\n")
        f.write("1. Semantic interpretability - Do cluster labels make sense?\n")
        f.write("2. Size balance - Are clusters roughly equal or is one huge?\n")
        f.write("3. RLHF signal - Do delta-enriched clusters align with research?\n")
        f.write("4. Distinctiveness - Are clusters clearly different from each other?\n\n")
        
        f.write("="*80 + "\n")
        f.write("QUANTITATIVE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("   k  | Silhouette | Size Balance | Min Size | Max Size | Largest Cluster\n")
        f.write("-"*80 + "\n")
        
        for k in sorted(all_results.keys()):
            result = all_results[k]
            sizes = result['cluster_sizes']
            max_size_idx = np.argmax(sizes)
            
            f.write(f"  {k:2d}  |   {result['silhouette_avg']:.4f}   |    {result['size_balance']:.3f}    |"
                   f"  {min(sizes):4d}    |  {max(sizes):4d}    | "
                   f"Cluster {max_size_idx} ({100*max(sizes)/sum(sizes):.1f}%)\n")
        
        f.write("\nMetric Definitions:\n")
        f.write("  - Silhouette: Higher = better separated (but all are low for text)\n")
        f.write("  - Size Balance: Lower = more balanced cluster sizes (CV of sizes)\n")
        f.write("  - Largest Cluster: Should be <30% for good balance\n\n")
        
        # Detailed comparison
        f.write("="*80 + "\n")
        f.write("CLUSTER THEMES BY k VALUE\n")
        f.write("="*80 + "\n\n")
        
        for k in sorted(all_results.keys()):
            result = all_results[k]
            cluster_info = result['cluster_info']
            enrichment = result['enrichment_scores']
            
            f.write(f"\n{'='*80}\n")
            f.write(f"k={k} CLUSTERING\n")
            f.write(f"{'='*80}\n\n")
            
            # Sort clusters by size
            sorted_clusters = sorted(cluster_info.items(), 
                                    key=lambda x: x[1]['size'], reverse=True)
            
            for cluster_id, info in sorted_clusters:
                f.write(f"Cluster {cluster_id}: {info['auto_label']}\n")
                f.write(f"  Size: {info['size']:4d} features ({100*info['size']/sum(result['cluster_sizes']):.1f}%)\n")
                
                # Find enrichment for this cluster
                enrich_item = next((e for e in enrichment if e['cluster_id'] == cluster_id), None)
                if enrich_item:
                    f.write(f"  Delta enrichment: {enrich_item['enrichment']:.2f}x")
                    if enrich_item['enrichment'] > 1.2:
                        f.write(" ← RLHF-shifted theme\n")
                    else:
                        f.write("\n")
                
                f.write(f"  Top terms: {', '.join([t for t, _ in info['top_terms'][:8]])}\n")
                f.write(f"  Sample: {info['sample_labels'][0][:70]}...\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        # Analyze which k might be best
        f.write("Automated suggestions (review manually):\n\n")
        
        # Find k with best balance
        best_balance_k = min(all_results.keys(), 
                            key=lambda k: all_results[k]['size_balance'])
        f.write(f"• Most balanced sizes: k={best_balance_k} "
               f"(CV={all_results[best_balance_k]['size_balance']:.3f})\n")
        
        # Find k with highest silhouette
        best_sil_k = max(all_results.keys(), 
                        key=lambda k: all_results[k]['silhouette_avg'])
        f.write(f"• Highest silhouette: k={best_sil_k} "
               f"(score={all_results[best_sil_k]['silhouette_avg']:.4f})\n")
        
        # Find k with no huge clusters (all <25%)
        good_max_size = []
        for k, result in all_results.items():
            max_pct = 100 * max(result['cluster_sizes']) / sum(result['cluster_sizes'])
            if max_pct < 25:
                good_max_size.append((k, max_pct))
        
        if good_max_size:
            f.write(f"\n• No dominant clusters (all <25%): ")
            f.write(", ".join([f"k={k} (max {pct:.1f}%)" for k, pct in good_max_size]))
            f.write("\n")
        
        f.write("\n\nNEXT STEPS:\n")
        f.write("-"*80 + "\n")
        f.write("1. Review clustering_dir/k8/, k9/, k10/, k11/, k12/ subdirectories\n")
        f.write("2. Read CLUSTERING_ANALYSIS.txt in each subdirectory\n")
        f.write("3. Check cluster_distribution.png for visual comparison\n")
        f.write("4. Select the k value that makes most research sense\n")
        f.write("5. Note your choice and reasoning in research log\n")
        f.write("6. Proceed with that k value for taxonomy building\n")
    
    print(f"\n✓ Comparison report saved to: {report_file}")


def main():
    print("="*80)
    print("SEMANTIC CLUSTERING ANALYSIS")
    print("="*80)
    print("\nThis script performs unsupervised clustering to discover themes:")
    print("  1. Embed labels with sentence transformers")
    print("  2. Run K-means clustering for k=8, 9, 10, 11, 12")
    print("  3. Generate detailed reports for each k value")
    print("  4. Create side-by-side comparison for manual selection")
    print("  5. Analyze RLHF impact via cluster distribution\n")
    print("Output: Multiple clusterings for you to manually review and select")
    print("        the most interpretable and balanced option.\n")
    
    print("="*80)
    print("DEPENDENCY CHECK")
    print("="*80)
    
    # Check if sentence-transformers is installed
    try:
        import sentence_transformers
        print("✓ sentence-transformers installed")
    except ImportError:
        print("✗ sentence-transformers not installed")
        print("\nPlease install:")
        print("  pip install sentence-transformers")
        return 1
    
    # Set cache directory to writable user space
    import os
    cache_dir = Path("/data/user_data/anshulk/data/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir / "sentence_transformers")
    
    print(f"✓ Model cache: {cache_dir / 'sentence_transformers'}")
    print("\nNote: First run will download ~90MB model (all-MiniLM-L6-v2)")
    print("      Subsequent runs will use cached model\n")
    
    # Setup
    validated_file, corpus_data_file, clustering_dir = setup_paths()
    
    if not validated_file.exists():
        print(f"ERROR: Validated labels not found: {validated_file}")
        return 1
    
    # Load data
    labels_data = load_labels_data(validated_file)
    
    print(f"Loaded {len(labels_data)} features:")
    sae_counts = Counter([d['sae_type'] for d in labels_data])
    for sae_type, count in sae_counts.items():
        print(f"  {sae_type.upper():10s}: {count:4d}")
    
    # Embed labels
    embeddings, model = embed_labels(labels_data)
    
    # Find silhouette scores for reference
    print("\n" + "="*80)
    print("SILHOUETTE ANALYSIS FOR REFERENCE")
    print("="*80)
    best_k, silhouette_scores, inertias, k_values = find_optimal_clusters(
        embeddings, min_k=5, max_k=20
    )
    
    if PLOTTING_AVAILABLE:
        print("\nGenerating silhouette plot...")
        plot_silhouette_analysis(silhouette_scores, k_values, clustering_dir)
    
    # Now run clustering for k=8-12 for manual inspection
    print("\n" + "="*80)
    print("GENERATING CLUSTERINGS FOR MANUAL INSPECTION (k=8-12)")
    print("="*80)
    
    k_range = [8, 9, 10, 11, 12]
    all_results = {}
    
    for k in k_range:
        print(f"\n{'='*80}")
        print(f"CLUSTERING WITH k={k}")
        print(f"{'='*80}")
        
        # Create subdirectory for this k
        k_dir = clustering_dir / f"k{k}"
        k_dir.mkdir(exist_ok=True)
        
        # Perform K-means clustering
        cluster_assignments, silhouette_vals, kmeans = perform_kmeans_clustering(
            embeddings, k
        )
        
        # Generate cluster labels
        cluster_info = generate_cluster_labels(labels_data, cluster_assignments)
        
        # Analyze distribution
        distribution, enrichment_scores = analyze_cluster_distribution(
            labels_data, cluster_assignments
        )
        
        # Calculate additional metrics for comparison
        cluster_sizes = [info['size'] for info in cluster_info.values()]
        size_balance = np.std(cluster_sizes) / np.mean(cluster_sizes)  # CV
        
        # Save results
        save_clustering_results(labels_data, embeddings, cluster_assignments,
                               cluster_info, distribution, enrichment_scores,
                               silhouette_vals, k_dir)
        
        # Generate detailed report
        generate_report(cluster_info, enrichment_scores, distribution,
                       labels_data, cluster_assignments, k_dir)
        
        # Generate plots
        if PLOTTING_AVAILABLE:
            plot_cluster_distribution(distribution, cluster_info, k_dir)
        
        # Store for comparison
        all_results[k] = {
            'cluster_info': cluster_info,
            'silhouette_avg': np.mean(silhouette_vals),
            'size_balance': size_balance,
            'enrichment_scores': enrichment_scores,
            'distribution': distribution,
            'cluster_sizes': cluster_sizes
        }
        
        print(f"\n✓ k={k} results saved to: {k_dir}")
    
    # Generate comparison report
    generate_comparison_report(all_results, clustering_dir)
    
    print("\n" + "="*80)
    print("✓ CLUSTERING ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated clusterings for k=8, 9, 10, 11, 12")
    print(f"\nOutputs in: {clustering_dir}")
    print(f"  - COMPARISON_REPORT.txt (side-by-side comparison)")
    print(f"  - silhouette_analysis.png (for reference)")
    print(f"  - k8/, k9/, k10/, k11/, k12/ (detailed results for each k)")
    print(f"\nEach k-directory contains:")
    print(f"  - CLUSTERING_ANALYSIS.txt (full report)")
    print(f"  - clustering_results.json (structured data)")
    print(f"  - cluster_distribution.png (visual)")
    
    print("\n" + "="*80)
    print("MANUAL INSPECTION WORKFLOW")
    print("="*80)
    print("\n1. Read COMPARISON_REPORT.txt for overview")
    print("2. Review each k-directory's CLUSTERING_ANALYSIS.txt")
    print("3. Consider:")
    print("   • Are cluster themes interpretable and distinct?")
    print("   • Is size distribution reasonable (no 40% cluster)?")
    print("   • Do RLHF-enriched clusters make research sense?")
    print("4. Select optimal k based on research interpretability")
    print("5. Document your choice and reasoning")
    print("\nNext: Once k is selected, run build_empirical_taxonomy.py --k <chosen_k>")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
