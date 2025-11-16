#!/usr/bin/env python3
"""
Comprehensive EDA for Cultural Alignment Activation Dataset
============================================================

Analyzes 33,522 culturally-grounded sentences with their activations from
Qwen2-1.5B base and instruct models across layers 6, 12, 18.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# UMAP and HDBSCAN for clustering
import umap
import hdbscan

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input paths
    BASE_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data")
    ACTIVATION_DIR = BASE_DIR / "Activations"
    INDEX_FILE = ACTIVATION_DIR / "activation_index.csv"
    
    # Output paths
    OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/EDA_results")
    HEAVY_DATA_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/EDA_data")
    
    # Analysis parameters
    LAYERS = [6, 12, 18]
    HIDDEN_SIZE = 1536
    N_SAMPLES_MANUAL = 200  # For manual validation sampling
    SIMILARITY_THRESHOLD = 0.95  # For near-duplicate detection
    
    # Visualization parameters
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    UMAP_RANDOM_STATE = 42
    HDBSCAN_MIN_CLUSTER_SIZE = 50
    
    def __init__(self):
        # Create output directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.HEAVY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.OUTPUT_DIR / "plots").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "reports").mkdir(exist_ok=True)
        (self.OUTPUT_DIR / "tables").mkdir(exist_ok=True)

config = Config()

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal_width = 80
        
    def section(self, title):
        msg = f"\n{'='*self.terminal_width}\n{title.upper()}\n{'='*self.terminal_width}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
    
    def subsection(self, title):
        msg = f"\n{'-'*60}\n{title}\n{'-'*60}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
    
    def result(self, key, value):
        msg = f"  • {key}: {value}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

log = Logger(config.OUTPUT_DIR / "eda_log.txt")

# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================

def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, pd.DataFrame):
        # Flatten MultiIndex columns if present
        df = obj.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
        return df.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# ============================================================================
# PHASE 0: DATA LOADING
# ============================================================================

def load_data():
    """Load all data: index CSV and activation arrays."""
    log.section("Phase 0: Data Loading")
    
    # Load index
    log.log("Loading activation index...")
    df = pd.read_csv(config.INDEX_FILE)
    log.result("Total sentences", len(df))
    log.result("Columns", list(df.columns))
    
    # Load activations
    activations = {}
    for model in ['base', 'instruct']:
        activations[model] = {}
        for layer in config.LAYERS:
            file_path = config.ACTIVATION_DIR / f"{model}_layer{layer}_activations.npy"
            log.log(f"Loading {model} layer {layer}...")
            activations[model][layer] = np.load(file_path)
            log.result(f"  Shape", activations[model][layer].shape)
    
    return df, activations

# ============================================================================
# PHASE 1: TEXT QUALITY & SANITY
# ============================================================================

def analyze_text_quality(df):
    """Comprehensive text-level quality analysis."""
    log.section("Phase 1: Text Quality & Sanity")
    
    results = {}
    
    # 1.1 Sentence length distributions
    log.subsection("1.1 Sentence Length Analysis")
    df['char_length'] = df['sentence'].str.len()
    df['word_count'] = df['sentence'].str.split().str.len()
    
    # Statistics by group
    length_stats = df.groupby('group_type')[['char_length', 'word_count']].describe()
    log.log("Length statistics by group:")
    print(length_stats)
    # Convert to JSON-serializable format (flatten MultiIndex columns)
    length_stats_flat = length_stats.copy()
    length_stats_flat.columns = ['_'.join(col).strip() for col in length_stats_flat.columns.values]
    results['length_stats'] = length_stats_flat.to_dict()
    
    # Plot length distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Character length by group
    for group in df['group_type'].unique():
        data = df[df['group_type'] == group]['char_length']
        axes[0, 0].hist(data, bins=50, alpha=0.5, label=group)
    axes[0, 0].set_xlabel('Character Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Character Length Distribution by Group')
    axes[0, 0].legend()
    
    # Word count by group
    for group in df['group_type'].unique():
        data = df[df['group_type'] == group]['word_count']
        axes[0, 1].hist(data, bins=50, alpha=0.5, label=group)
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Word Count Distribution by Group')
    axes[0, 1].legend()
    
    # Box plots
    df.boxplot(column='char_length', by='group_type', ax=axes[1, 0])
    axes[1, 0].set_title('Character Length by Group (Boxplot)')
    axes[1, 0].set_xlabel('Group Type')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)
    
    df.boxplot(column='word_count', by='group_type', ax=axes[1, 1])
    axes[1, 1].set_title('Word Count by Group (Boxplot)')
    axes[1, 1].set_xlabel('Group Type')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "01_length_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved length distribution plots")
    
    # 1.2 Lexical diversity
    log.subsection("1.2 Lexical Diversity Analysis")
    
    def compute_lexical_diversity(text):
        """Type-token ratio and hapax legomena."""
        words = text.lower().split()
        if len(words) == 0:
            return 0, 0
        word_counts = Counter(words)
        types = len(word_counts)
        tokens = len(words)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        return types / tokens if tokens > 0 else 0, hapax / types if types > 0 else 0
    
    diversity_data = []
    for group in df['group_type'].unique():
        group_text = ' '.join(df[df['group_type'] == group]['sentence'].tolist())
        ttr, hapax_ratio = compute_lexical_diversity(group_text)
        diversity_data.append({
            'group': group,
            'type_token_ratio': ttr,
            'hapax_ratio': hapax_ratio,
            'total_tokens': len(group_text.split())
        })
    
    diversity_df = pd.DataFrame(diversity_data)
    log.log("Lexical diversity by group:")
    print(diversity_df)
    results['lexical_diversity'] = diversity_df
    
    # 1.3 N-gram analysis
    log.subsection("1.3 N-gram Frequency Analysis")
    
    def get_top_ngrams(texts, n=2, top_k=20):
        """Extract top k n-grams."""
        vectorizer = TfidfVectorizer(ngram_range=(n, n), max_features=top_k, stop_words='english')
        vectorizer.fit(texts)
        return vectorizer.get_feature_names_out()
    
    ngram_results = {}
    for group in df['group_type'].unique():
        texts = df[df['group_type'] == group]['sentence'].tolist()
        bigrams = get_top_ngrams(texts, n=2, top_k=15)
        trigrams = get_top_ngrams(texts, n=3, top_k=10)
        ngram_results[group] = {
            'top_bigrams': bigrams.tolist(),
            'top_trigrams': trigrams.tolist()
        }
        log.log(f"\n{group.upper()} - Top bigrams:")
        log.log(f"  {', '.join(bigrams[:10])}")
    
    results['ngrams'] = ngram_results
    
    # 1.4 Near-duplicate detection - FULL DATASET
    log.subsection("1.4 Near-Duplicate Detection (FULL DATASET)")
    log.log("Computing TF-IDF similarity matrix for FULL dataset...")
    log.log("This will take several minutes - please be patient...")
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['sentence'])
    
    log.log(f"Computing pairwise similarities for all {len(df):,} sentences...")
    log.log("Creating 33,522 × 33,522 similarity matrix...")
    
    # Compute full similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find near-duplicates (upper triangle only to avoid duplicates)
    log.log("Identifying near-duplicate pairs (checking upper triangle only)...")
    high_sim_pairs = []
    
    n = len(df)
    for i in range(n):
        if i % 1000 == 0:
            log.log(f"  Progress: {i:,}/{n:,} rows ({i/n*100:.1f}%)")
        
        for j in range(i+1, n):  # Only upper triangle
            if similarity_matrix[i, j] > config.SIMILARITY_THRESHOLD:
                high_sim_pairs.append((i, j, similarity_matrix[i, j]))
    
    log.log(f"  Progress: {n:,}/{n:,} rows (100.0%)")
    log.result("Near-duplicate pairs found (FULL dataset)", len(high_sim_pairs))
    
    # Track unique sentences with duplicates
    dup_indices = set()
    
    # Create detailed dataframe of duplicates
    if len(high_sim_pairs) > 0:
        log.log("Creating detailed duplicate analysis...")
        dup_records = []
        
        for idx1, idx2, sim in high_sim_pairs:
            dup_indices.add(idx1)
            dup_indices.add(idx2)
            
            dup_records.append({
                'sentence1_idx': idx1,
                'sentence2_idx': idx2,
                'row_id_1': df.iloc[idx1]['row_id'],
                'row_id_2': df.iloc[idx2]['row_id'],
                'sentence_num_1': df.iloc[idx1]['sentence_num'],
                'sentence_num_2': df.iloc[idx2]['sentence_num'],
                'group_type_1': df.iloc[idx1]['group_type'],
                'group_type_2': df.iloc[idx2]['group_type'],
                'state_1': df.iloc[idx1]['state'],
                'state_2': df.iloc[idx2]['state'],
                'attribute_1': df.iloc[idx1]['attribute'],
                'attribute_2': df.iloc[idx2]['attribute'],
                'similarity': sim,
                'sentence_1': df.iloc[idx1]['sentence'],
                'sentence_2': df.iloc[idx2]['sentence']
            })
        
        dup_df = pd.DataFrame(dup_records)
        dup_df = dup_df.sort_values('similarity', ascending=False)
        
        # Save to CSV
        dup_file = config.OUTPUT_DIR / "tables" / "near_duplicates_full.csv"
        dup_df.to_csv(dup_file, index=False)
        log.log(f"✓ Saved near-duplicates to {dup_file}")
        
        # Analyze duplicate patterns
        log.log("\nDuplicate pattern analysis:")
        
        # Same vs different groups
        same_group = (dup_df['group_type_1'] == dup_df['group_type_2']).sum()
        log.result("  Same group type", f"{same_group}/{len(dup_df)} ({same_group/len(dup_df)*100:.1f}%)")
        
        # Same vs different states
        same_state = (dup_df['state_1'] == dup_df['state_2']).sum()
        log.result("  Same state", f"{same_state}/{len(dup_df)} ({same_state/len(dup_df)*100:.1f}%)")
        
        # Same vs different attributes
        same_attr = (dup_df['attribute_1'] == dup_df['attribute_2']).sum()
        log.result("  Same attribute", f"{same_attr}/{len(dup_df)} ({same_attr/len(dup_df)*100:.1f}%)")
        
        # Show top examples
        log.log("\nTop 5 near-duplicate pairs (highest similarity):")
        for i, row in dup_df.head(5).iterrows():
            log.log(f"\n  Pair {i+1} (similarity: {row['similarity']:.4f}):")
            log.log(f"    [{row['group_type_1']}] {row['state_1']} / {row['attribute_1']} - Row {row['row_id_1']}.{row['sentence_num_1']}")
            log.log(f"    → {row['sentence_1'][:120]}...")
            log.log(f"    [{row['group_type_2']}] {row['state_2']} / {row['attribute_2']} - Row {row['row_id_2']}.{row['sentence_num_2']}")
            log.log(f"    → {row['sentence_2'][:120]}...")
        
        # Add flag to main dataframe
        df['has_near_duplicate'] = df.index.isin(dup_indices)
        log.result("Unique sentences with duplicates", len(dup_indices))
        log.result("Percentage of dataset", f"{len(dup_indices)/len(df)*100:.2f}%")
        
        results['near_duplicates'] = {
            'total_sentences': len(df),
            'pairs_found': len(high_sim_pairs),
            'unique_sentences_with_duplicates': len(dup_indices),
            'percentage_with_duplicates': float(len(dup_indices)/len(df)*100),
            'same_group_pairs': int(same_group),
            'same_state_pairs': int(same_state),
            'same_attribute_pairs': int(same_attr),
            'threshold': config.SIMILARITY_THRESHOLD
        }
    else:
        df['has_near_duplicate'] = False
        log.log("✓ No near-duplicates found at threshold {config.SIMILARITY_THRESHOLD}!")
        
        results['near_duplicates'] = {
            'total_sentences': len(df),
            'pairs_found': 0,
            'unique_sentences_with_duplicates': 0,
            'percentage_with_duplicates': 0.0,
            'threshold': config.SIMILARITY_THRESHOLD
        }
    
    # Save results
    with open(config.OUTPUT_DIR / "reports" / "01_text_quality.json", 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    
    return df, results

# ============================================================================
# PHASE 2: COVERAGE & BALANCE
# ============================================================================

def analyze_coverage_balance(df):
    """Analyze distributional coverage and balance."""
    log.section("Phase 2: Coverage & Balance Audit")
    
    results = {}
    
    # 2.1 Group type distribution
    log.subsection("2.1 Group Type Distribution")
    group_counts = df['group_type'].value_counts()
    log.log("Sentence counts by group:")
    for group, count in group_counts.items():
        log.result(group, f"{count:,} ({count/len(df)*100:.1f}%)")
    results['group_counts'] = group_counts.to_dict()
    
    # 2.2 Cross-tabulations
    log.subsection("2.2 Cross-Tabulation Analysis")
    
    # Group × State
    group_state = pd.crosstab(df['group_type'], df['state'])
    log.log(f"\nGroup × State cross-tab shape: {group_state.shape}")
    log.log(f"Cells with <30 sentences: {(group_state < 30).sum().sum()}")
    
    # Save full crosstab
    group_state.to_csv(config.OUTPUT_DIR / "tables" / "crosstab_group_state.csv")
    
    # Group × Attribute
    group_attr = pd.crosstab(df['group_type'], df['attribute'])
    log.log(f"\nGroup × Attribute cross-tab shape: {group_attr.shape}")
    log.log(f"Cells with <30 sentences: {(group_attr < 30).sum().sum()}")
    
    # Save full crosstab
    group_attr.to_csv(config.OUTPUT_DIR / "tables" / "crosstab_group_attribute.csv")
    
    results['low_count_cells'] = {
        'group_state': int((group_state < 30).sum().sum()),
        'group_attribute': int((group_attr < 30).sum().sum())
    }
    
    # 2.3 State distribution analysis
    log.subsection("2.3 State Coverage")
    state_counts = df.groupby(['state', 'group_type']).size().unstack(fill_value=0)
    
    # Top and bottom states by total count
    state_totals = state_counts.sum(axis=1).sort_values(ascending=False)
    log.log("\nTop 10 states by sentence count:")
    for state, count in state_totals.head(10).items():
        log.result(state, f"{count:,}")
    
    log.log("\nBottom 10 states by sentence count:")
    for state, count in state_totals.tail(10).items():
        log.result(state, f"{count:,}")
    
    # Visualize state distribution
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top 20 states
    top_states = state_totals.head(20)
    state_counts.loc[top_states.index].plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title('Top 20 States by Sentence Count (Stacked by Group)')
    axes[0].set_xlabel('State')
    axes[0].set_ylabel('Sentence Count')
    axes[0].legend(title='Group Type')
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')
    
    # Bottom 20 states
    bottom_states = state_totals.tail(20)
    state_counts.loc[bottom_states.index].plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Bottom 20 States by Sentence Count (Stacked by Group)')
    axes[1].set_xlabel('State')
    axes[1].set_ylabel('Sentence Count')
    axes[1].legend(title='Group Type')
    plt.sca(axes[1])
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "02_state_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved state distribution plots")
    
    # 2.4 Attribute distribution
    log.subsection("2.4 Attribute Coverage")
    attr_counts = df.groupby(['attribute', 'group_type']).size().unstack(fill_value=0)
    
    log.log("\nAttribute distribution:")
    attr_totals = attr_counts.sum(axis=1).sort_values(ascending=False)
    for attr, count in attr_totals.items():
        log.result(attr, f"{count:,}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(14, 8))
    attr_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Attribute Distribution (Stacked by Group)')
    ax.set_xlabel('Attribute')
    ax.set_ylabel('Sentence Count')
    ax.legend(title='Group Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "02_attribute_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved attribute distribution plot")
    
    # 2.5 Correctness analysis
    log.subsection("2.5 Model Correctness by Group")
    
    correctness = df.groupby('group_type')[['base_correct', 'instruct_correct']].mean()
    log.log("\nCorrectness rates by group:")
    print(correctness)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    correctness.plot(kind='bar', ax=ax)
    ax.set_title('Model Correctness by Group Type')
    ax.set_xlabel('Group Type')
    ax.set_ylabel('Correctness Rate')
    ax.legend(['Base Model', 'Instruct Model'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "02_correctness_by_group.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved correctness plot")
    
    results['correctness'] = correctness.to_dict()
    
    # Save results
    with open(config.OUTPUT_DIR / "reports" / "02_coverage_balance.json", 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    
    return results

# ============================================================================
# PHASE 3: SEMANTIC STRUCTURE EXPLORATION
# ============================================================================

def analyze_semantic_structure(df):
    """Semantic analysis using sentence embeddings."""
    log.section("Phase 3: Semantic Structure Exploration")
    
    results = {}
    
    # 3.1 Generate sentence embeddings
    log.subsection("3.1 Generating Sentence Embeddings")
    log.log("Loading Sentence-BERT model...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model
    
    log.log(f"Encoding {len(df)} sentences (this will take a few minutes)...")
    embeddings = model.encode(df['sentence'].tolist(), 
                             show_progress_bar=True,
                             batch_size=256)
    
    log.result("Embedding shape", embeddings.shape)
    
    # Save embeddings for future use
    embedding_file = config.HEAVY_DATA_DIR / "sentence_embeddings.npy"
    np.save(embedding_file, embeddings)
    log.log(f"✓ Saved embeddings to {embedding_file}")
    
    # 3.2 UMAP dimensionality reduction
    log.subsection("3.2 UMAP Dimensionality Reduction")
    log.log("Running UMAP...")
    
    reducer = umap.UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        n_components=2,
        random_state=config.UMAP_RANDOM_STATE,
        metric='cosine',
        verbose=True
    )
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Save 2D embeddings
    umap_file = config.HEAVY_DATA_DIR / "umap_2d.npy"
    np.save(umap_file, embedding_2d)
    log.log(f"✓ Saved UMAP embeddings to {umap_file}")
    
    # Add to dataframe
    df['umap_x'] = embedding_2d[:, 0]
    df['umap_y'] = embedding_2d[:, 1]
    
    # 3.3 HDBSCAN clustering
    log.subsection("3.3 HDBSCAN Clustering")
    log.log("Running HDBSCAN...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=10,
        metric='euclidean'
    )
    
    cluster_labels = clusterer.fit_predict(embedding_2d)
    df['cluster'] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    log.result("Clusters found", n_clusters)
    log.result("Noise points", f"{n_noise} ({n_noise/len(df)*100:.1f}%)")
    
    results['clustering'] = {
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise),
        'noise_percentage': float(n_noise/len(df)*100)
    }
    
    # 3.4 Visualize UMAP projections
    log.subsection("3.4 Creating Visualizations")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Colored by group_type
    ax1 = plt.subplot(2, 3, 1)
    for group in df['group_type'].unique():
        mask = df['group_type'] == group
        ax1.scatter(df[mask]['umap_x'], df[mask]['umap_y'], 
                   alpha=0.3, s=10, label=group)
    ax1.set_title('UMAP: Colored by Group Type')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend()
    
    # Plot 2: Colored by cluster
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(df['umap_x'], df['umap_y'], 
                         c=df['cluster'], cmap='tab20', alpha=0.3, s=10)
    ax2.set_title('UMAP: HDBSCAN Clusters')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax2)
    
    # Plot 3: Colored by state (top 10 states only)
    ax3 = plt.subplot(2, 3, 3)
    top_states = df['state'].value_counts().head(10).index
    for i, state in enumerate(top_states):
        mask = df['state'] == state
        ax3.scatter(df[mask]['umap_x'], df[mask]['umap_y'], 
                   alpha=0.3, s=10, label=state[:20])
    ax3.set_title('UMAP: Top 10 States')
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.legend(fontsize=8)
    
    # Plot 4: Colored by attribute
    ax4 = plt.subplot(2, 3, 4)
    attributes = df['attribute'].unique()
    for attr in attributes:
        mask = df['attribute'] == attr
        ax4.scatter(df[mask]['umap_x'], df[mask]['umap_y'], 
                   alpha=0.3, s=10, label=attr)
    ax4.set_title('UMAP: Colored by Attribute')
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')
    ax4.legend(fontsize=8)
    
    # Plot 5: Colored by base_correct
    ax5 = plt.subplot(2, 3, 5)
    for correct in [True, False]:
        mask = df['base_correct'] == correct
        ax5.scatter(df[mask]['umap_x'], df[mask]['umap_y'], 
                   alpha=0.3, s=10, label=f"Base: {correct}")
    ax5.set_title('UMAP: Base Model Correctness')
    ax5.set_xlabel('UMAP 1')
    ax5.set_ylabel('UMAP 2')
    ax5.legend()
    
    # Plot 6: Colored by instruct_correct
    ax6 = plt.subplot(2, 3, 6)
    for correct in [True, False]:
        mask = df['instruct_correct'] == correct
        ax6.scatter(df[mask]['umap_x'], df[mask]['umap_y'], 
                   alpha=0.3, s=10, label=f"Instruct: {correct}")
    ax6.set_title('UMAP: Instruct Model Correctness')
    ax6.set_xlabel('UMAP 1')
    ax6.set_ylabel('UMAP 2')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "03_umap_all.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved UMAP visualizations")
    
    # 3.5 Cluster analysis
    log.subsection("3.5 Cluster Composition Analysis")
    
    cluster_composition = []
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            continue  # Skip noise
        
        cluster_data = df[df['cluster'] == cluster_id]
        composition = {
            'cluster_id': int(cluster_id),
            'size': len(cluster_data),
            'suppression_pct': (cluster_data['group_type'] == 'suppression').mean() * 100,
            'enhancement_pct': (cluster_data['group_type'] == 'enhancement').mean() * 100,
            'control_pct': (cluster_data['group_type'] == 'control').mean() * 100,
            'top_states': cluster_data['state'].value_counts().head(3).to_dict(),
            'top_attributes': cluster_data['attribute'].value_counts().head(3).to_dict(),
            'base_accuracy': cluster_data['base_correct'].mean(),
            'instruct_accuracy': cluster_data['instruct_correct'].mean()
        }
        cluster_composition.append(composition)
    
    cluster_df = pd.DataFrame(cluster_composition)
    cluster_df.to_csv(config.OUTPUT_DIR / "tables" / "cluster_composition.csv", index=False)
    log.log(f"✓ Analyzed {len(cluster_df)} clusters")
    
    # 3.6 Simple probing baselines
    log.subsection("3.6 Baseline Probing on Embeddings")
    
    # Prepare labels
    group_labels = df['group_type'].map({'suppression': 0, 'enhancement': 1, 'control': 2}).values
    attribute_labels = pd.Categorical(df['attribute']).codes
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Probe 1: Group type classification
    log.log("Training group_type probe...")
    probe_group = LogisticRegression(max_iter=1000, random_state=42)
    scores_group = cross_val_score(probe_group, embeddings_scaled, group_labels, 
                                   cv=5, scoring='accuracy')
    
    log.result("Group type probe accuracy", f"{scores_group.mean():.3f} ± {scores_group.std():.3f}")
    
    # Probe 2: Attribute classification
    log.log("Training attribute probe...")
    probe_attr = LogisticRegression(max_iter=1000, random_state=42)
    scores_attr = cross_val_score(probe_attr, embeddings_scaled, attribute_labels, 
                                  cv=5, scoring='accuracy')
    
    log.result("Attribute probe accuracy", f"{scores_attr.mean():.3f} ± {scores_attr.std():.3f}")
    
    results['probing'] = {
        'group_type_accuracy': float(scores_group.mean()),
        'group_type_std': float(scores_group.std()),
        'attribute_accuracy': float(scores_attr.mean()),
        'attribute_std': float(scores_attr.std())
    }
    
    # Save results
    with open(config.OUTPUT_DIR / "reports" / "03_semantic_structure.json", 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    
    return df, embeddings, results

# ============================================================================
# PHASE 4: ACTIVATION-LEVEL ANALYSIS
# ============================================================================

def analyze_activations(df, activations):
    """Deep dive into activation space."""
    log.section("Phase 4: Activation-Level Deep Dive")
    
    results = {}
    
    # 4.1 Distributional diagnostics
    log.subsection("4.1 Activation Distribution Statistics")
    
    stats = {}
    for model in ['base', 'instruct']:
        stats[model] = {}
        for layer in config.LAYERS:
            acts = activations[model][layer]
            stats[model][layer] = {
                'mean': float(acts.mean()),
                'std': float(acts.std()),
                'min': float(acts.min()),
                'max': float(acts.max()),
                'median': float(np.median(acts)),
                'per_dim_std_mean': float(acts.std(axis=0).mean()),
                'per_dim_std_std': float(acts.std(axis=0).std())
            }
            
            log.log(f"{model.upper()} Layer {layer}:")
            log.result("  Mean", f"{stats[model][layer]['mean']:.4f}")
            log.result("  Std", f"{stats[model][layer]['std']:.4f}")
            log.result("  Per-dim std (mean)", f"{stats[model][layer]['per_dim_std_mean']:.4f}")
    
    results['activation_stats'] = stats
    
    # Plot per-dimension statistics
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    for layer_idx, layer in enumerate(config.LAYERS):
        # Base model
        base_acts = activations['base'][layer]
        base_stds = base_acts.std(axis=0)
        axes[layer_idx, 0].hist(base_stds, bins=50, alpha=0.7, color='blue')
        axes[layer_idx, 0].set_title(f'Base Layer {layer}: Per-Dimension Std Dev')
        axes[layer_idx, 0].set_xlabel('Std Dev')
        axes[layer_idx, 0].set_ylabel('Frequency')
        
        # Instruct model
        inst_acts = activations['instruct'][layer]
        inst_stds = inst_acts.std(axis=0)
        axes[layer_idx, 1].hist(inst_stds, bins=50, alpha=0.7, color='red')
        axes[layer_idx, 1].set_title(f'Instruct Layer {layer}: Per-Dimension Std Dev')
        axes[layer_idx, 1].set_xlabel('Std Dev')
        axes[layer_idx, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "04_activation_stds.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved activation std plots")
    
    # 4.2 Group geometry analysis
    log.subsection("4.2 Group Centroid Analysis")
    
    group_geometry = {}
    
    for model in ['base', 'instruct']:
        group_geometry[model] = {}
        
        for layer in config.LAYERS:
            acts = activations[model][layer]
            
            # Compute centroids
            centroids = {}
            for group in df['group_type'].unique():
                mask = df['group_type'] == group
                centroids[group] = acts[mask].mean(axis=0)
            
            # Compute pairwise distances
            distances = {}
            groups = list(centroids.keys())
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    cos_sim = np.dot(centroids[g1], centroids[g2]) / \
                             (np.linalg.norm(centroids[g1]) * np.linalg.norm(centroids[g2]))
                    euclidean = np.linalg.norm(centroids[g1] - centroids[g2])
                    distances[f"{g1}_vs_{g2}"] = {
                        'cosine_similarity': float(cos_sim),
                        'euclidean_distance': float(euclidean)
                    }
            
            group_geometry[model][layer] = {
                'centroid_norms': {g: float(np.linalg.norm(c)) for g, c in centroids.items()},
                'pairwise_distances': distances
            }
            
            log.log(f"\n{model.upper()} Layer {layer} - Group Distances:")
            for pair, dists in distances.items():
                log.result(f"  {pair}", 
                          f"cos={dists['cosine_similarity']:.4f}, "
                          f"eucl={dists['euclidean_distance']:.2f}")
    
    results['group_geometry'] = group_geometry
    
    # Visualize centroid distances
    fig, axes = plt.subplots(len(config.LAYERS), 2, figsize=(15, 12))
    
    for layer_idx, layer in enumerate(config.LAYERS):
        for model_idx, model in enumerate(['base', 'instruct']):
            ax = axes[layer_idx, model_idx]
            
            distances = group_geometry[model][layer]['pairwise_distances']
            pairs = list(distances.keys())
            cos_sims = [distances[p]['cosine_similarity'] for p in pairs]
            
            ax.barh(pairs, cos_sims)
            ax.set_xlabel('Cosine Similarity')
            ax.set_title(f'{model.upper()} Layer {layer}')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "04_group_distances.png", dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved group distance plots")
    
    # 4.3 UMAP on activations
    log.subsection("4.3 UMAP on Activation Space")
    
    for model in ['base', 'instruct']:
        for layer in config.LAYERS:
            log.log(f"Running UMAP on {model} layer {layer}...")
            
            acts = activations[model][layer]
            
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=42,
                metric='cosine'
            )
            
            acts_2d = reducer.fit_transform(acts)
            
            # Save
            umap_file = config.HEAVY_DATA_DIR / f"umap_activations_{model}_layer{layer}.npy"
            np.save(umap_file, acts_2d)
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # By group
            for group in df['group_type'].unique():
                mask = df['group_type'] == group
                axes[0].scatter(acts_2d[mask, 0], acts_2d[mask, 1], 
                              alpha=0.3, s=10, label=group)
            axes[0].set_title(f'{model.upper()} Layer {layer}: By Group')
            axes[0].legend()
            
            # By correctness
            for correct in [True, False]:
                mask = df[f'{model}_correct'] == correct
                axes[1].scatter(acts_2d[mask, 0], acts_2d[mask, 1], 
                              alpha=0.3, s=10, label=f"Correct: {correct}")
            axes[1].set_title(f'{model.upper()} Layer {layer}: By Correctness')
            axes[1].legend()
            
            # By state (top 8)
            top_states = df['state'].value_counts().head(8).index
            for state in top_states:
                mask = df['state'] == state
                axes[2].scatter(acts_2d[mask, 0], acts_2d[mask, 1], 
                              alpha=0.3, s=10, label=state[:15])
            axes[2].set_title(f'{model.upper()} Layer {layer}: Top States')
            axes[2].legend(fontsize=8)
            
            plt.tight_layout()
            plt.savefig(config.OUTPUT_DIR / "plots" / f"04_umap_activations_{model}_layer{layer}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            log.log(f"✓ Saved UMAP visualization for {model} layer {layer}")
    
    # 4.3.5 HDBSCAN Clustering on Activations (Layer 18)
    log.subsection("4.3.5 HDBSCAN Clustering on Activation Space")
    
    clustering_results = {}
    
    for model in ['base', 'instruct']:
        log.log(f"Running HDBSCAN on {model} layer 18 activations...")
        
        acts = activations[model][18]
        
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            metric='euclidean'
        )
        
        cluster_labels = clusterer.fit_predict(acts_2d)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        log.log(f"{model.upper()} Layer 18:")
        log.result("  Clusters found", n_clusters)
        log.result("  Noise points", f"{n_noise} ({n_noise/len(df)*100:.1f}%)")
        
        # Analyze cluster composition
        cluster_composition = []
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                continue  # Skip noise
            
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df[cluster_mask]
            
            composition = {
                'model': model,
                'layer': 18,
                'cluster_id': int(cluster_id),
                'size': int(cluster_mask.sum()),
                'suppression_count': int((cluster_data['group_type'] == 'suppression').sum()),
                'enhancement_count': int((cluster_data['group_type'] == 'enhancement').sum()),
                'control_count': int((cluster_data['group_type'] == 'control').sum()),
                'suppression_pct': float((cluster_data['group_type'] == 'suppression').mean() * 100),
                'enhancement_pct': float((cluster_data['group_type'] == 'enhancement').mean() * 100),
                'control_pct': float((cluster_data['group_type'] == 'control').mean() * 100),
                'base_accuracy': float(cluster_data['base_correct'].mean()),
                'instruct_accuracy': float(cluster_data['instruct_correct'].mean()),
                'top_states': cluster_data['state'].value_counts().head(3).to_dict(),
                'top_attributes': cluster_data['attribute'].value_counts().head(3).to_dict()
            }
            cluster_composition.append(composition)
        
        # Save cluster composition
        comp_df = pd.DataFrame(cluster_composition)
        comp_df.to_csv(
            config.OUTPUT_DIR / "tables" / f"activation_clusters_{model}_layer18.csv",
            index=False
        )
        log.log(f"✓ Saved {model} cluster composition to CSV")
        
        # Calculate cluster purity metrics
        if len(comp_df) > 0:
            # Find clusters dominated by each group (>70% purity)
            supp_pure = comp_df[comp_df['suppression_pct'] > 70.0]
            enh_pure = comp_df[comp_df['enhancement_pct'] > 70.0]
            ctrl_pure = comp_df[comp_df['control_pct'] > 70.0]
            
            log.log(f"\n{model.upper()} Cluster purity analysis:")
            log.result("  Suppression-pure clusters (>70%)", 
                      f"{len(supp_pure)} clusters, {supp_pure['size'].sum():,} sentences")
            log.result("  Enhancement-pure clusters (>70%)", 
                      f"{len(enh_pure)} clusters, {enh_pure['size'].sum():,} sentences")
            log.result("  Control-pure clusters (>70%)", 
                      f"{len(ctrl_pure)} clusters, {ctrl_pure['size'].sum():,} sentences")
            
            # Average within-cluster purity
            avg_purity = comp_df[['suppression_pct', 'enhancement_pct', 'control_pct']].max(axis=1).mean()
            log.result("  Average cluster purity", f"{avg_purity:.1f}%")
        
        clustering_results[model] = {
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_percentage': float(n_noise/len(df)*100),
            'cluster_compositions': cluster_composition
        }
        
        # Add cluster labels to temporary array for visualization
        df[f'{model}_act_cluster'] = cluster_labels
    
    # Compare base vs instruct clustering
    log.log("\nCross-model cluster comparison:")
    
    # How many sentences stay in same vs different clusters?
    same_cluster_type = 0
    for i in range(len(df)):
        base_cluster = df.iloc[i]['base_act_cluster']
        inst_cluster = df.iloc[i]['instruct_act_cluster']
        
        # Both in actual clusters (not noise)
        if base_cluster != -1 and inst_cluster != -1:
            # Check if they're in same group-dominant cluster
            base_group = df.iloc[i]['group_type']
            # This is simplified - you could do more sophisticated comparison
            same_cluster_type += 1
    
    # Visualize clusters side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_idx, model in enumerate(['base', 'instruct']):
        # Use the UMAP coordinates from earlier
        umap_file = config.HEAVY_DATA_DIR / f"umap_activations_{model}_layer18.npy"
        acts_2d = np.load(umap_file)
        
        cluster_labels = df[f'{model}_act_cluster'].values
        
        scatter = axes[model_idx].scatter(
            acts_2d[:, 0], acts_2d[:, 1],
            c=cluster_labels, cmap='tab20',
            alpha=0.3, s=10
        )
        axes[model_idx].set_title(f'{model.upper()} Layer 18: HDBSCAN Clusters')
        axes[model_idx].set_xlabel('UMAP 1')
        axes[model_idx].set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=axes[model_idx])
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "04_activation_clusters_comparison.png",
               dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved activation cluster comparison plot")
    
    # Save clustering results
    results['activation_clustering'] = clustering_results
    
    # Clean up temporary columns
    df.drop(columns=['base_act_cluster', 'instruct_act_cluster'], inplace=True)

    # 4.4 Cross-model comparison
    log.subsection("4.4 Base vs Instruct Comparison")
    
    comparison = {}
    for layer in config.LAYERS:
        base_acts = activations['base'][layer]
        inst_acts = activations['instruct'][layer]
        
        # Compute sentence-level similarities
        cos_sims = []
        for i in range(len(base_acts)):
            cos_sim = np.dot(base_acts[i], inst_acts[i]) / \
                     (np.linalg.norm(base_acts[i]) * np.linalg.norm(inst_acts[i]))
            cos_sims.append(cos_sim)
        
        cos_sims = np.array(cos_sims)
        
        # Analyze by group
        group_sims = {}
        for group in df['group_type'].unique():
            mask = df['group_type'] == group
            group_sims[group] = {
                'mean': float(cos_sims[mask].mean()),
                'std': float(cos_sims[mask].std()),
                'median': float(np.median(cos_sims[mask]))
            }
        
        comparison[layer] = {
            'overall_mean': float(cos_sims.mean()),
            'overall_std': float(cos_sims.std()),
            'by_group': group_sims
        }
        
        log.log(f"\nLayer {layer} Base-Instruct Similarity:")
        log.result("  Overall", f"{cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
        for group, stats in group_sims.items():
            log.result(f"  {group}", f"{stats['mean']:.4f} ± {stats['std']:.4f}")
    
    results['base_instruct_comparison'] = comparison
    
    # Visualize similarities
    fig, axes = plt.subplots(1, len(config.LAYERS), figsize=(15, 5))
    
    for layer_idx, layer in enumerate(config.LAYERS):
        base_acts = activations['base'][layer]
        inst_acts = activations['instruct'][layer]
        
        cos_sims = []
        for i in range(len(base_acts)):
            cos_sim = np.dot(base_acts[i], inst_acts[i]) / \
                     (np.linalg.norm(base_acts[i]) * np.linalg.norm(inst_acts[i]))
            cos_sims.append(cos_sim)
        
        cos_sims = np.array(cos_sims)
        
        # Plot by group
        for group in df['group_type'].unique():
            mask = df['group_type'] == group
            axes[layer_idx].hist(cos_sims[mask], bins=50, alpha=0.5, label=group)
        
        axes[layer_idx].set_xlabel('Cosine Similarity')
        axes[layer_idx].set_ylabel('Frequency')
        axes[layer_idx].set_title(f'Layer {layer}')
        axes[layer_idx].legend()
        axes[layer_idx].axvline(x=cos_sims.mean(), color='black', 
                               linestyle='--', label='Overall Mean')
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "plots" / "04_base_instruct_similarity.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    log.log("✓ Saved base-instruct similarity plots")
    
    # Save results
    with open(config.OUTPUT_DIR / "reports" / "04_activation_analysis.json", 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    
    return results

# ============================================================================
# PHASE 5: MANUAL VALIDATION PREPARATION
# ============================================================================

def prepare_manual_validation(df):
    """Prepare stratified sample for manual inspection."""
    log.section("Phase 5: Manual Validation Sampling")
    
    # Stratified sampling
    log.log(f"Generating stratified sample of {config.N_SAMPLES_MANUAL} sentences...")
    
    # Sample proportional to group and spread across states/attributes
    samples_per_group = config.N_SAMPLES_MANUAL // 3
    
    sampled_dfs = []
    for group in df['group_type'].unique():
        group_df = df[df['group_type'] == group]
        
        # Further stratify by state
        states = group_df['state'].value_counts().head(10).index  # Top 10 states
        state_samples = []
        
        for state in states:
            state_df = group_df[group_df['state'] == state]
            n_samples = min(len(state_df), max(1, samples_per_group // len(states)))
            state_samples.append(state_df.sample(n=n_samples, random_state=42))
        
        sampled_dfs.append(pd.concat(state_samples))
    
    sample_df = pd.concat(sampled_dfs)
    
    # Add columns for manual annotation
    sample_df['manual_factual_correct'] = ''
    sample_df['manual_cultural_specific'] = ''
    sample_df['manual_style_issues'] = ''
    sample_df['manual_notes'] = ''
    
    # Save for manual review
    output_file = config.OUTPUT_DIR / "tables" / "manual_validation_sample.csv"
    sample_df[['row_id', 'sentence_num', 'sentence', 'group_type', 'state', 
               'attribute', 'base_correct', 'instruct_correct',
               'manual_factual_correct', 'manual_cultural_specific',
               'manual_style_issues', 'manual_notes']].to_csv(output_file, index=False)
    
    log.result("Sample size", len(sample_df))
    log.result("Distribution", sample_df['group_type'].value_counts().to_dict())
    log.log(f"✓ Saved to {output_file}")
    log.log("\nInstructions for manual validation:")
    log.log("1. Open manual_validation_sample.csv")
    log.log("2. For each sentence, fill in:")
    log.log("   - manual_factual_correct: yes/no/unsure")
    log.log("   - manual_cultural_specific: yes/no (is it truly about the culture?)")
    log.log("   - manual_style_issues: list any repetitive/generic phrasing")
    log.log("   - manual_notes: any other observations")
    
    return sample_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_summary_report(df, all_results):
    """Generate comprehensive summary report."""
    log.section("Generating Summary Report")
    
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("CULTURAL ALIGNMENT EDA - EXECUTIVE SUMMARY")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\nDataset: {len(df):,} sentences")
    report_lines.append(f"Models: Qwen2-1.5B Base & Instruct")
    report_lines.append(f"Layers analyzed: {config.LAYERS}")
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("1. DATA DISTRIBUTION")
    report_lines.append("-"*80)
    report_lines.append(f"\nGroup counts:")
    for group, count in df['group_type'].value_counts().items():
        report_lines.append(f"  {group:15s}: {count:6,} ({count/len(df)*100:5.2f}%)")
    
    report_lines.append(f"\nUnique states: {df['state'].nunique()}")
    report_lines.append(f"Unique attributes: {df['attribute'].nunique()}")
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("2. TEXT QUALITY")
    report_lines.append("-"*80)
    report_lines.append(f"\nSentence length (words):")
    report_lines.append(f"  Mean: {df['word_count'].mean():.1f}")
    report_lines.append(f"  Median: {df['word_count'].median():.1f}")
    report_lines.append(f"  Std: {df['word_count'].std():.1f}")
    
    # Add duplicate info if available
    if 'has_near_duplicate' in df.columns:
        n_dup = df['has_near_duplicate'].sum()
        report_lines.append(f"\nNear-duplicates:")
        report_lines.append(f"  Sentences with duplicates: {n_dup:,} ({n_dup/len(df)*100:.2f}%)")
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("3. MODEL CORRECTNESS")
    report_lines.append("-"*80)
    correctness = df.groupby('group_type')[['base_correct', 'instruct_correct']].mean()
    report_lines.append(f"\nAccuracy by group:")
    for group in correctness.index:
        base_acc = correctness.loc[group, 'base_correct']
        inst_acc = correctness.loc[group, 'instruct_correct']
        report_lines.append(f"  {group:15s}: Base={base_acc:.3f}, Instruct={inst_acc:.3f}, "
                          f"Delta={inst_acc-base_acc:+.3f}")
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("4. SEMANTIC CLUSTERING")
    report_lines.append("-"*80)
    if 'clustering' in all_results.get('semantic', {}):
        clust = all_results['semantic']['clustering']
        report_lines.append(f"\nHDBSCAN results:")
        report_lines.append(f"  Clusters found: {clust['n_clusters']}")
        report_lines.append(f"  Noise points: {clust['n_noise']} ({clust['noise_percentage']:.1f}%)")
    
    if 'probing' in all_results.get('semantic', {}):
        probe = all_results['semantic']['probing']
        report_lines.append(f"\nBaseline probing (on embeddings):")
        report_lines.append(f"  Group type accuracy: {probe['group_type_accuracy']:.3f} "
                          f"± {probe['group_type_std']:.3f}")
        report_lines.append(f"  Attribute accuracy: {probe['attribute_accuracy']:.3f} "
                          f"± {probe['attribute_std']:.3f}")
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("5. ACTIVATION GEOMETRY")
    report_lines.append("-"*80)
    if 'base_instruct_comparison' in all_results.get('activations', {}):
        comp = all_results['activations']['base_instruct_comparison']
        report_lines.append(f"\nBase-Instruct similarity (cosine):")
        for layer in config.LAYERS:
            overall = comp[layer]['overall_mean']
            report_lines.append(f"  Layer {layer}: {overall:.4f}")
            for group, stats in comp[layer]['by_group'].items():
                report_lines.append(f"    {group:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("6. KEY FINDINGS")
    report_lines.append("-"*80)
    
    # Calculate suppression effect
    supp_base = df[df['group_type'] == 'suppression']['base_correct'].mean()
    supp_inst = df[df['group_type'] == 'suppression']['instruct_correct'].mean()
    supp_delta = supp_inst - supp_base
    
    enh_base = df[df['group_type'] == 'enhancement']['base_correct'].mean()
    enh_inst = df[df['group_type'] == 'enhancement']['instruct_correct'].mean()
    enh_delta = enh_inst - enh_base
    
    report_lines.append(f"\n• Suppression effect: {supp_delta:+.3f} "
                       f"({supp_base:.3f} → {supp_inst:.3f})")
    report_lines.append(f"• Enhancement effect: {enh_delta:+.3f} "
                       f"({enh_base:.3f} → {enh_inst:.3f})")
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("7. OUTPUT FILES")
    report_lines.append("-"*80)
    report_lines.append(f"\nPlots: {config.OUTPUT_DIR / 'plots'}")
    report_lines.append(f"Reports: {config.OUTPUT_DIR / 'reports'}")
    report_lines.append(f"Tables: {config.OUTPUT_DIR / 'tables'}")
    report_lines.append(f"Heavy data: {config.HEAVY_DATA_DIR}")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write report
    report_text = '\n'.join(report_lines)
    with open(config.OUTPUT_DIR / "SUMMARY_REPORT.txt", 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    log.log("✓ Summary report saved")

def main():
    """Main execution pipeline."""
    start_time = datetime.now()
    
    log.section("Cultural Alignment EDA Pipeline")
    log.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log("MODE: FULL DATASET ANALYSIS (NO SAMPLING)")
    
    all_results = {}
    
    try:
        # Phase 0: Load data
        df, activations = load_data()
        
        # Phase 1: Text quality
        df, text_results = analyze_text_quality(df)
        all_results['text'] = text_results
        
        # Phase 2: Coverage and balance
        coverage_results = analyze_coverage_balance(df)
        all_results['coverage'] = coverage_results
        
        # Phase 3: Semantic structure
        df, embeddings, semantic_results = analyze_semantic_structure(df)
        all_results['semantic'] = semantic_results
        
        # Phase 4: Activation analysis
        activation_results = analyze_activations(df, activations)
        all_results['activations'] = activation_results
        
        # Phase 5: Manual validation prep
        sample_df = prepare_manual_validation(df)
        
        # Generate summary
        generate_summary_report(df, all_results)
        
        # Save enhanced dataframe
        df_output = config.OUTPUT_DIR / "tables" / "enhanced_dataset.csv"
        df.to_csv(df_output, index=False)
        log.log(f"✓ Saved enhanced dataset to {df_output}")
        
    except Exception as e:
        log.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        log.log(traceback.format_exc())
        raise
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log.section("Pipeline Complete")
    log.log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"Total duration: {duration/60:.1f} minutes")
    log.log(f"\nAll outputs saved to:")
    log.log(f"  {config.OUTPUT_DIR}")
    log.log(f"  {config.HEAVY_DATA_DIR}")

if __name__ == "__main__":
    main()