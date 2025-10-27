#!/usr/bin/env python3
"""
Build Empirical Taxonomy from Clustering Results

Formalizes unsupervised clustering into structured taxonomies with:
- Hierarchical organization (cluster -> SAE type -> features)
- RLHF impact quantification (delta enrichment, Cohen's d)
- Comparative analysis across k values
- Visual hierarchy generation

Usage:
    python build_empirical_taxonomy.py --k 8
    python build_empirical_taxonomy.py --k 11
    python build_empirical_taxonomy.py --k 12
    python build_empirical_taxonomy.py --all  # Process all three
"""

import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
CLUSTERING_ROOT = Path("/home/anshulk/cultural-alignment-study/outputs/sae_models/analysis/clustering")
TAXONOMY_OUTPUT = Path("/home/anshulk/cultural-alignment-study/outputs/sae_models/analysis/taxonomy")
VALIDATED_LABELS = Path("/home/anshulk/cultural-alignment-study/outputs/sae_models/labels_qwen3_validated.json")


def load_clustering_results(k: int) -> Dict:
    """Load clustering results for specified k value"""
    cluster_dir = CLUSTERING_ROOT / f"k{k}"
    results_file = cluster_dir / "clustering_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Clustering results not found: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_validated_features() -> Dict:
    """Load validated feature labels"""
    with open(VALIDATED_LABELS, 'r', encoding='utf-8') as f:
        features = json.load(f)
    
    # Index by feature_id for quick lookup
    return {f['feature_id']: f for f in features}


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups"""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean2 - mean1) / pooled_std


def compute_chi_square(observed_delta: int, total_cluster: int, expected_delta_rate: float) -> Tuple[float, float]:
    """Chi-square test for delta enrichment"""
    expected_delta = total_cluster * expected_delta_rate
    expected_non_delta = total_cluster * (1 - expected_delta_rate)
    observed_non_delta = total_cluster - observed_delta
    
    if expected_delta == 0 or expected_non_delta == 0:
        return 0.0, 1.0
    
    chi2 = ((observed_delta - expected_delta)**2 / expected_delta + 
            (observed_non_delta - expected_non_delta)**2 / expected_non_delta)
    
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value


def build_taxonomy(k: int, clustering_results: Dict, validated_features: Dict) -> Dict:
    """Build hierarchical taxonomy from clustering results"""
    
    print(f"\nBuilding taxonomy for k={k}...")
    
    # Extract feature assignments
    feature_assignments = clustering_results['feature_assignments']
    total_features = clustering_results['total_features']
    
    # Calculate baseline delta rate
    total_delta = sum(1 for f in feature_assignments if 'delta' in f['sae_type'].lower())
    baseline_delta_rate = total_delta / total_features
    
    print(f"  Total features: {total_features}")
    print(f"  Baseline delta rate: {baseline_delta_rate:.3f} ({total_delta} delta features)")
    
    taxonomy = {
        'metadata': {
            'k_value': int(k),
            'total_features': int(total_features),
            'baseline_delta_rate': float(baseline_delta_rate),
            'n_clusters': int(clustering_results['n_clusters'])
        },
        'clusters': []
    }
    
    # Process each cluster
    for cluster_id in range(clustering_results['n_clusters']):
        # Get features in this cluster
        cluster_features = [f for f in feature_assignments if f['cluster_id'] == cluster_id]
        cluster_info = clustering_results['cluster_info'][str(cluster_id)]
        cluster_info = clustering_results['cluster_info'][str(cluster_id)]
        
        # Count by SAE type
        base_count = sum(1 for f in cluster_features if f['sae_type'] == 'base')
        chat_count = sum(1 for f in cluster_features if f['sae_type'] == 'chat')
        delta_count = sum(1 for f in cluster_features if f['sae_type'] == 'delta')
        
        # Calculate delta enrichment
        cluster_delta_rate = delta_count / len(cluster_features) if cluster_features else 0
        enrichment_ratio = cluster_delta_rate / baseline_delta_rate if baseline_delta_rate > 0 else 0
        
        # Chi-square test for delta enrichment
        chi2, p_value = compute_chi_square(delta_count, len(cluster_features), baseline_delta_rate)
        
        # Average silhouette score for cluster
        avg_silhouette = np.mean([f['silhouette_score'] for f in cluster_features]) if cluster_features else 0.0
        
        # Collect feature details by type
        features_by_type = {
            'base': [],
            'chat': [],
            'delta': []
        }
        
        for feat in cluster_features:
            feat_data = validated_features.get(feat['feature_id'], {})
            
            features_by_type[feat['sae_type']].append({
                'feature_id': feat['feature_id'],
                'label': feat_data.get('final_label', feat_data.get('label_qwen', feat['label'])),
                'confidence': float(feat_data.get('confidence_score', 0)) if feat_data.get('confidence_score') else 0.0,
                'validation_action': feat_data.get('validation_action', 'UNKNOWN'),
                'silhouette_score': float(feat['silhouette_score'])
            })
        
        # Flatten top_terms if it's a list of lists
        top_terms = cluster_info['top_terms']
        if top_terms and isinstance(top_terms[0], list):
            top_terms_flat = [term for sublist in top_terms for term in sublist]
        else:
            top_terms_flat = top_terms
        
        # Build cluster entry (convert numpy types to native Python types)
        cluster_entry = {
            'cluster_id': int(cluster_id),
            'label': cluster_info['auto_label'],
            'size': int(cluster_info['size']),
            'percentage': float(cluster_info['size'] / total_features * 100),
            'silhouette_score': float(avg_silhouette),
            'top_terms': top_terms_flat,
            'sample_features': cluster_info['sample_labels'],
            'sae_distribution': {
                'base': int(base_count),
                'chat': int(chat_count),
                'delta': int(delta_count)
            },
            'rlhf_metrics': {
                'delta_count': int(delta_count),
                'delta_rate': float(cluster_delta_rate),
                'enrichment_ratio': float(enrichment_ratio),
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05),
                'is_rlhf_shifted': bool(enrichment_ratio > 1.2 and p_value < 0.05)
            },
            'features_by_type': features_by_type
        }
        
        taxonomy['clusters'].append(cluster_entry)
    
    # Sort clusters by size (largest first)
    taxonomy['clusters'].sort(key=lambda x: x['size'], reverse=True)
    
    return taxonomy


def generate_taxonomy_report(k: int, taxonomy: Dict, output_dir: Path):
    """Generate human-readable taxonomy report"""
    
    report_file = output_dir / f"TAXONOMY_k{k}_REPORT.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"EMPIRICAL TAXONOMY: k={k} Clustering\n")
        f.write("="*80 + "\n\n")
        
        f.write("METADATA:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Features: {taxonomy['metadata']['total_features']}\n")
        f.write(f"Number of Clusters: {taxonomy['metadata']['n_clusters']}\n")
        f.write(f"Baseline Delta Rate: {taxonomy['metadata']['baseline_delta_rate']:.3f}\n\n")
        
        # Summary statistics
        rlhf_shifted_clusters = [c for c in taxonomy['clusters'] 
                                if c['rlhf_metrics']['is_rlhf_shifted']]
        
        f.write("RLHF IMPACT SUMMARY:\n")
        f.write("-"*80 + "\n")
        f.write(f"RLHF-Shifted Clusters: {len(rlhf_shifted_clusters)} / {len(taxonomy['clusters'])}\n")
        
        if rlhf_shifted_clusters:
            max_enrichment = max(c['rlhf_metrics']['enrichment_ratio'] for c in rlhf_shifted_clusters)
            f.write(f"Maximum Enrichment: {max_enrichment:.2f}x\n")
        
        f.write("\n\n")
        
        # Detailed cluster analysis
        f.write("CLUSTER DETAILS:\n")
        f.write("="*80 + "\n\n")
        
        for cluster in taxonomy['clusters']:
            f.write(f"Cluster {cluster['cluster_id']}: {cluster['label']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Size: {cluster['size']} features ({cluster['percentage']:.1f}%)\n")
            f.write(f"Silhouette Score: {cluster['silhouette_score']:.3f}\n")
            
            # Handle top_terms - flatten if it's a list of lists
            top_terms = cluster['top_terms']
            if top_terms and isinstance(top_terms[0], list):
                top_terms_flat = [term for sublist in top_terms for term in sublist]
            else:
                top_terms_flat = top_terms
            f.write(f"Top Terms: {', '.join(top_terms_flat[:8])}\n\n")
            
            f.write("SAE Distribution:\n")
            f.write(f"  BASE:  {cluster['sae_distribution']['base']:4d} ({cluster['sae_distribution']['base']/cluster['size']*100:5.1f}%)\n")
            f.write(f"  CHAT:  {cluster['sae_distribution']['chat']:4d} ({cluster['sae_distribution']['chat']/cluster['size']*100:5.1f}%)\n")
            f.write(f"  DELTA: {cluster['sae_distribution']['delta']:4d} ({cluster['sae_distribution']['delta']/cluster['size']*100:5.1f}%)\n\n")
            
            f.write("RLHF Metrics:\n")
            f.write(f"  Delta Enrichment: {cluster['rlhf_metrics']['enrichment_ratio']:.2f}x\n")
            f.write(f"  Chi-Square: {cluster['rlhf_metrics']['chi_square']:.2f}, p={cluster['rlhf_metrics']['p_value']:.4f}\n")
            f.write(f"  Significance: {'YES' if cluster['rlhf_metrics']['is_significant'] else 'NO'}\n")
            f.write(f"  RLHF-Shifted: {'YES' if cluster['rlhf_metrics']['is_rlhf_shifted'] else 'NO'}\n\n")
            
            f.write("Sample Features:\n")
            sample_feats = cluster['sample_features'][:3]
            for i, sample in enumerate(sample_feats, 1):
                # Handle if sample is a list or string
                if isinstance(sample, list):
                    sample_text = ' | '.join(sample) if sample else 'N/A'
                else:
                    sample_text = sample
                f.write(f"  {i}. {sample_text}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"  Report saved: {report_file}")


def plot_taxonomy_hierarchy(k: int, taxonomy: Dict, output_dir: Path):
    """Generate visualization of taxonomy hierarchy"""
    
    # Delta enrichment heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Cluster sizes and delta enrichment
    clusters = taxonomy['clusters']
    cluster_labels = [f"C{c['cluster_id']}" for c in clusters]
    sizes = [c['size'] for c in clusters]
    enrichments = [c['rlhf_metrics']['enrichment_ratio'] for c in clusters]
    
    ax = axes[0]
    colors = ['red' if c['rlhf_metrics']['is_rlhf_shifted'] else 'gray' for c in clusters]
    bars = ax.barh(cluster_labels, sizes, color=colors, alpha=0.7)
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(f'Cluster Sizes (k={k})\nRed = RLHF-Shifted', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Delta enrichment ratios
    ax = axes[1]
    colors = ['red' if e > 1.2 else 'blue' if e < 0.8 else 'gray' for e in enrichments]
    bars = ax.barh(cluster_labels, enrichments, color=colors, alpha=0.7)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Baseline')
    ax.axvline(x=1.2, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Enriched (1.2x)')
    ax.set_xlabel('Delta Enrichment Ratio', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(f'RLHF Impact (k={k})\nRed = High Delta | Blue = Low Delta', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f"taxonomy_k{k}_hierarchy.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Hierarchy plot saved: {plot_file}")
    
    # Distribution heatmap
    fig, ax = plt.subplots(figsize=(10, max(8, len(clusters) * 0.4)))
    
    # Create distribution matrix
    dist_matrix = []
    for cluster in clusters:
        total = cluster['size']
        dist_matrix.append([
            cluster['sae_distribution']['base'] / total * 100,
            cluster['sae_distribution']['chat'] / total * 100,
            cluster['sae_distribution']['delta'] / total * 100
        ])
    
    im = ax.imshow(dist_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(range(3))
    ax.set_xticklabels(['BASE', 'CHAT', 'DELTA'])
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"C{c['cluster_id']}: {c['label'][:30]}..." for c in clusters])
    
    # Add text annotations
    for i in range(len(clusters)):
        for j in range(3):
            text = ax.text(j, i, f"{dist_matrix[i][j]:.1f}%",
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(f'SAE Type Distribution Across Clusters (k={k})', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Percentage')
    plt.tight_layout()
    
    heatmap_file = output_dir / f"taxonomy_k{k}_distribution.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Distribution heatmap saved: {heatmap_file}")


def generate_comparison_report(k_values: List[int], taxonomies: Dict[int, Dict], output_dir: Path):
    """Generate comparative analysis across k values"""
    
    report_file = output_dir / "TAXONOMY_COMPARISON.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TAXONOMY COMPARISON ACROSS k VALUES\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERVIEW:\n")
        f.write("-"*80 + "\n")
        for k in sorted(k_values):
            tax = taxonomies[k]
            rlhf_clusters = sum(1 for c in tax['clusters'] if c['rlhf_metrics']['is_rlhf_shifted'])
            max_enrichment = max(c['rlhf_metrics']['enrichment_ratio'] for c in tax['clusters'])
            largest_cluster_pct = max(c['percentage'] for c in tax['clusters'])
            
            f.write(f"\nk={k}:\n")
            f.write(f"  Number of clusters: {len(tax['clusters'])}\n")
            f.write(f"  RLHF-shifted clusters: {rlhf_clusters} ({rlhf_clusters/len(tax['clusters'])*100:.1f}%)\n")
            f.write(f"  Max enrichment: {max_enrichment:.2f}x\n")
            f.write(f"  Largest cluster: {largest_cluster_pct:.1f}%\n")
            f.write(f"  Balance (max cluster %): {'Good' if largest_cluster_pct < 25 else 'Poor'}\n")
        
        f.write("\n\n")
        f.write("RLHF-SHIFTED CLUSTERS COMPARISON:\n")
        f.write("="*80 + "\n\n")
        
        for k in sorted(k_values):
            tax = taxonomies[k]
            rlhf_clusters = [c for c in tax['clusters'] if c['rlhf_metrics']['is_rlhf_shifted']]
            
            f.write(f"k={k} - {len(rlhf_clusters)} RLHF-Shifted Clusters:\n")
            f.write("-"*80 + "\n")
            
            for cluster in sorted(rlhf_clusters, key=lambda x: x['rlhf_metrics']['enrichment_ratio'], reverse=True):
                f.write(f"\nCluster {cluster['cluster_id']}: {cluster['label']}\n")
                f.write(f"  Size: {cluster['size']} features ({cluster['percentage']:.1f}%)\n")
                f.write(f"  Enrichment: {cluster['rlhf_metrics']['enrichment_ratio']:.2f}x\n")
                f.write(f"  Delta %: {cluster['rlhf_metrics']['delta_rate']*100:.1f}%\n")
                
                # Handle top_terms - flatten if needed
                top_terms = cluster['top_terms']
                if top_terms and isinstance(top_terms[0], list):
                    top_terms_flat = [term for sublist in top_terms for term in sublist]
                else:
                    top_terms_flat = top_terms
                f.write(f"  Top terms: {', '.join(top_terms_flat[:5])}\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n\n")
        
        # Find k with strongest RLHF signal
        best_k = max(k_values, key=lambda k: max(
            c['rlhf_metrics']['enrichment_ratio'] 
            for c in taxonomies[k]['clusters']
        ))
        
        f.write(f"Strongest RLHF Signal: k={best_k}\n")
        f.write(f"  Rationale: Highest delta enrichment ratio\n\n")
        
        # Find most balanced
        balance_scores = {}
        for k in k_values:
            max_pct = max(c['percentage'] for c in taxonomies[k]['clusters'])
            balance_scores[k] = max_pct
        
        most_balanced = min(k_values, key=lambda k: balance_scores[k])
        f.write(f"Most Balanced: k={most_balanced}\n")
        f.write(f"  Rationale: Smallest maximum cluster size ({balance_scores[most_balanced]:.1f}%)\n\n")
        
        f.write("Final Recommendation:\n")
        f.write(f"  For RLHF impact analysis: k={best_k}\n")
        f.write(f"  For balanced taxonomy: k={most_balanced}\n")
    
    print(f"\nComparison report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Build empirical taxonomy from clustering results')
    parser.add_argument('--k', type=int, choices=[8, 11, 12], help='Cluster count to process')
    parser.add_argument('--all', action='store_true', help='Process all k values (8, 11, 12)')
    args = parser.parse_args()
    
    if not args.k and not args.all:
        parser.error("Must specify either --k or --all")
    
    k_values = [8, 11, 12] if args.all else [args.k]
    
    print("="*80)
    print("EMPIRICAL TAXONOMY BUILDER")
    print("="*80)
    
    # Create output directory
    TAXONOMY_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Load validated features once
    print("\nLoading validated features...")
    validated_features = load_validated_features()
    print(f"  Loaded {len(validated_features)} validated features")
    
    # Process each k value
    taxonomies = {}
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"PROCESSING k={k}")
        print(f"{'='*80}")
        
        # Create k-specific output directory
        k_output = TAXONOMY_OUTPUT / f"k{k}"
        k_output.mkdir(parents=True, exist_ok=True)
        
        # Load clustering results
        clustering_results = load_clustering_results(k)
        
        # Build taxonomy
        taxonomy = build_taxonomy(k, clustering_results, validated_features)
        taxonomies[k] = taxonomy
        
        # Save taxonomy JSON
        taxonomy_file = k_output / f"taxonomy_k{k}.json"
        with open(taxonomy_file, 'w', encoding='utf-8') as f:
            json.dump(taxonomy, f, indent=2, ensure_ascii=False)
        print(f"  Taxonomy saved: {taxonomy_file}")
        
        # Generate report
        generate_taxonomy_report(k, taxonomy, k_output)
        
        # Generate visualizations
        plot_taxonomy_hierarchy(k, taxonomy, k_output)
    
    # Generate comparison if multiple k values
    if len(k_values) > 1:
        print(f"\n{'='*80}")
        print("GENERATING COMPARATIVE ANALYSIS")
        print(f"{'='*80}")
        generate_comparison_report(k_values, taxonomies, TAXONOMY_OUTPUT)
    
    print(f"\n{'='*80}")
    print("TAXONOMY BUILDING COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {TAXONOMY_OUTPUT}")
    print("\nNext steps:")
    print("  1. Review TAXONOMY_COMPARISON.txt for k selection guidance")
    print("  2. Examine individual taxonomy reports for cluster details")
    print("  3. Select optimal k value for Phase 3 feature validation")


if __name__ == "__main__":
    main()
