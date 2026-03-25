#!/usr/bin/env python3
"""
Sanskriti Dataset — Full EDA Pipeline
======================================
Runs all 8 EDA sections and saves:
  - Plots  → PLOTS_DIR  (home, lightweight PNGs)
  - CSVs   → ANALYSIS_DIR (data volume, heavier tables)

Usage:
    python scripts/eda_pipeline.py
    python scripts/eda_pipeline.py --section 5   # run only section 5
"""

import argparse
import os
import re
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────
PLOTS_DIR = "/home/anshulk/cultural-mi/plots"
ANALYSIS_DIR = "/data/user_data/anshulk/cultural-mi/analysis"
DATASET_CACHE = "/data/user_data/anshulk/cultural-mi/dataset"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its",
    "you", "your", "we", "our", "they", "their", "he", "she", "him", "her",
    "following", "according", "given", "among", "also", "one", "would", "could",
}
LETTERS = ["A", "B", "C", "D"]
OPT_KEYS = ["option1", "option2", "option3", "option4"]


def tokenize(text):
    return [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", str(text))]


def tokenize_no_stop(text):
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


def get_ground_truth(row):
    ans = str(row["answer"]).strip().lower()
    for opt, letter in zip(OPT_KEYS, LETTERS):
        if str(row[opt]).strip().lower() == ans:
            return letter
    return None


def save_csv(df, name):
    path = os.path.join(ANALYSIS_DIR, name)
    df.to_csv(path, index=False)
    print(f"    → {name}")


def save_csv_idx(df, name):
    path = os.path.join(ANALYSIS_DIR, name)
    df.to_csv(path)
    print(f"    → {name}")


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {name}")


def timer(msg):
    class _T:
        def __enter__(self):
            self.t = time.time()
            print(f"\n{'='*60}\n  {msg}\n{'='*60}")
            return self
        def __exit__(self, *a):
            print(f"  ✓ done in {time.time()-self.t:.1f}s")
    return _T()


# ── entity extraction ────────────────────────────────────────────────
def _extract_entity_regex(q):
    """Extract cultural entity from templated questions via regex."""
    q = str(q)
    patterns = [
        r"famous for (.+?)\?",
        r"home to the (.+?)\?",
        r"home to (.+?)\?",
        r"Where is the (.+?) famous",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    m = re.match(r"^(.+?)\s+is associated", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _build_entity_key(row):
    """Combined entity key: regex-extracted if available, else (state|attr|answer) fallback."""
    regex = _extract_entity_regex(row["question"])
    if regex is not None:
        return regex
    return f"{row['state']}|{row['attribute']}|{str(row['answer']).strip()}"


# ── load dataset ─────────────────────────────────────────────────────
def load_data():
    from datasets import load_dataset
    ds = load_dataset("13ari/Sanskriti", cache_dir=DATASET_CACHE)["train"]
    df = pd.DataFrame(ds)
    df["ground_truth_letter"] = df.apply(get_ground_truth, axis=1)
    df["question_id"] = range(len(df))
    usable = df[df["ground_truth_letter"].notna()].copy().reset_index(drop=True)
    # Add entity keys (regex + fallback) for entity-level analysis in Step 1
    usable["entity_key"] = usable.apply(_build_entity_key, axis=1)
    n_regex = usable["question"].apply(_extract_entity_regex).notna().sum()
    n_fallback = len(usable) - n_regex
    print(f"  Entity keys: {n_regex} regex + {n_fallback} fallback = {usable['entity_key'].nunique()} unique")
    save_csv(usable, "sanskriti_usable.csv")
    print(f"  Total: {len(df)}, Usable: {len(usable)}, Excluded: {len(df)-len(usable)}")
    return usable


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Distribution & Coverage
# ══════════════════════════════════════════════════════════════════════
def section_1(df):
    with timer("Section 1: Distribution & Coverage"):
        n = len(df)

        # ── CSVs ──
        state_dist = df["state"].value_counts().reset_index()
        state_dist.columns = ["state", "count"]
        state_dist["pct"] = (state_dist["count"] / n * 100).round(2)
        save_csv(state_dist, "distribution_states.csv")

        attr_dist = df["attribute"].value_counts().reset_index()
        attr_dist.columns = ["attribute", "count"]
        attr_dist["pct"] = (attr_dist["count"] / n * 100).round(2)
        attr_dist["sparse"] = attr_dist["count"] < 200
        save_csv(attr_dist, "distribution_attributes.csv")

        qtype_dist = df["question_type"].value_counts().reset_index()
        qtype_dist.columns = ["question_type", "count"]
        qtype_dist["pct"] = (qtype_dist["count"] / n * 100).round(2)
        save_csv(qtype_dist, "distribution_qtypes.csv")

        # cross-tabs
        cross_sa = pd.crosstab(df["state"], df["attribute"])
        cross_sq = pd.crosstab(df["state"], df["question_type"])
        save_csv_idx(cross_sa, "cross_tab_state_attribute.csv")
        save_csv_idx(cross_sq, "cross_tab_state_qtype.csv")

        # coverage
        rows = []
        for state in cross_sa.index:
            for attr in cross_sa.columns:
                c = int(cross_sa.loc[state, attr])
                rows.append({"state": state, "attribute": attr, "count": c,
                             "empty": c == 0, "below_threshold": 0 < c < 125, "reliable": c >= 125})
        cov_df = pd.DataFrame(rows)
        save_csv(cov_df, "coverage_state_attribute.csv")

        # state summary
        state_summary = pd.DataFrame({"total": df["state"].value_counts()})
        for attr in df["attribute"].unique():
            state_summary[f"attr_{attr}"] = df[df["attribute"] == attr]["state"].value_counts()
        for qt in df["question_type"].unique():
            state_summary[f"qtype_{qt}"] = df[df["question_type"] == qt]["state"].value_counts()
        state_summary = state_summary.fillna(0).astype(int)
        save_csv_idx(state_summary, "state_summary.csv")

        # ── plots ──
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle("Sanskriti Dataset: Distribution Analysis", fontsize=16, fontweight="bold")

        sc = df["state"].value_counts()
        ax = axes[0, 0]
        sc.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(sc)))
        ax.set_title(f"Questions per State (n={len(sc)})")
        ax.set_xlabel("Count")
        ax.axvline(x=125, color="red", linestyle="--", alpha=0.7, label="Min reliable (125)")
        ax.legend(); ax.invert_yaxis()

        ac = df["attribute"].value_counts()
        ax = axes[0, 1]
        colors = ["#e74c3c" if c < 200 else "#3498db" for c in ac.values]
        ac.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Questions per Attribute (n={len(ac)})")
        ax.axvline(x=200, color="red", linestyle="--", alpha=0.7, label="Sparse (200)")
        ax.legend(); ax.invert_yaxis()

        qc = df["question_type"].value_counts()
        ax = axes[1, 0]
        qc.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", 4))
        ax.set_title("Questions per Question Type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        for i, (_, val) in enumerate(qc.items()):
            ax.text(i, val + 50, str(val), ha="center", fontweight="bold")

        ax = axes[1, 1]; ax.axis("off")
        empty = int(cov_df["empty"].sum())
        reliable = int(cov_df["reliable"].sum())
        total_cells = len(cov_df)
        txt = (f"Coverage Summary\n{'='*35}\n"
               f"Total cells (state×attr): {total_cells}\n"
               f"Empty cells:              {empty} ({empty/total_cells*100:.1f}%)\n"
               f"Reliable (≥125):          {reliable} ({reliable/total_cells*100:.1f}%)\n")
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=12,
                va="top", fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        save_plot(fig, "eda_01_distributions.png")

        # heatmaps
        fig, ax = plt.subplots(figsize=(20, 14))
        sns.heatmap(cross_sa, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title("State × Attribute Distribution", fontsize=14)
        save_plot(fig, "eda_02_state_attr_heatmap.png")

        fig, ax = plt.subplots(figsize=(10, 14))
        sns.heatmap(cross_sq, annot=True, fmt="d", cmap="YlGnBu", ax=ax, linewidths=0.5)
        ax.set_title("State × Question Type Distribution", fontsize=14)
        save_plot(fig, "eda_03_state_qtype_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Answer Position Bias
# ══════════════════════════════════════════════════════════════════════
def section_2(df):
    with timer("Section 2: Answer Position Bias"):
        n = len(df)

        # overall
        gt_dist = df["ground_truth_letter"].value_counts().sort_index()
        gt_csv = gt_dist.reset_index()
        gt_csv.columns = ["letter", "count"]
        gt_csv["pct"] = (gt_csv["count"] / n * 100).round(2)
        gt_csv["expected"] = int(n / 4)
        save_csv(gt_csv, "position_bias_overall.csv")

        chi2_all, p_all = stats.chisquare(gt_dist.values, f_exp=[n/4]*4)

        # per question type
        rows = []
        chi2_qt_rows = []
        for qt in df["question_type"].unique():
            sub = df[df["question_type"] == qt]
            dist = sub["ground_truth_letter"].value_counts().reindex(LETTERS, fill_value=0)
            chi2_qt, p_qt = stats.chisquare(dist.values, f_exp=[len(sub)/4]*4)
            chi2_qt_rows.append({"question_type": qt, "chi2": round(chi2_qt, 2), "p_value": p_qt, "n": len(sub)})
            for letter in LETTERS:
                c = int(dist.get(letter, 0))
                rows.append({"question_type": qt, "letter": letter, "count": c,
                             "pct": round(c / len(sub) * 100, 2)})
        save_csv(pd.DataFrame(rows), "position_bias_by_qtype.csv")

        # per attribute
        rows = []
        for attr in df["attribute"].unique():
            sub = df[df["attribute"] == attr]
            dist = sub["ground_truth_letter"].value_counts().reindex(LETTERS, fill_value=0)
            chi2_a, p_a = stats.chisquare(dist.values, f_exp=[len(sub)/4]*4)
            for letter in LETTERS:
                c = int(dist.get(letter, 0))
                rows.append({"attribute": attr, "letter": letter, "count": c,
                             "pct": round(c / len(sub) * 100, 2), "n_total": len(sub),
                             "chi2": round(chi2_a, 2), "p_value": p_a})
        save_csv(pd.DataFrame(rows), "position_bias_by_attribute.csv")

        # per state
        rows = []
        for state in df["state"].unique():
            sub = df[df["state"] == state]
            dist = sub["ground_truth_letter"].value_counts().reindex(LETTERS, fill_value=0)
            chi2_s, p_s = stats.chisquare(dist.values, f_exp=[len(sub)/4]*4)
            for letter in LETTERS:
                c = int(dist.get(letter, 0))
                rows.append({"state": state, "letter": letter, "count": c,
                             "pct": round(c / len(sub) * 100, 2), "n_total": len(sub),
                             "chi2": round(chi2_s, 2), "p_value": p_s})
        save_csv(pd.DataFrame(rows), "position_bias_by_state.csv")

        # ── plot ──
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle("Answer Position Bias Analysis", fontsize=16, fontweight="bold")

        ax = axes[0, 0]
        bars = ax.bar(gt_dist.index, gt_dist.values, color=["#e74c3c","#3498db","#2ecc71","#f39c12"])
        ax.axhline(y=n/4, color="gray", linestyle="--", alpha=0.7, label="Uniform (25%)")
        ax.set_title("Overall Ground Truth Distribution")
        for bar, val in zip(bars, gt_dist.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    f"{val}\n({val/n*100:.1f}%)", ha="center", fontsize=10)
        ax.legend()
        ax.text(0.98, 0.98, f"χ²={chi2_all:.1f}, p={p_all:.2e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow"))

        ax = axes[0, 1]
        pd.crosstab(df["question_type"], df["ground_truth_letter"], normalize="index")[LETTERS].multiply(100).plot(
            kind="bar", ax=ax, width=0.8)
        ax.set_title("GT Distribution per Question Type (%)")
        ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

        top_attrs = df["attribute"].value_counts().head(12).index
        ax = axes[0, 2]
        pd.crosstab(df[df["attribute"].isin(top_attrs)]["attribute"],
                     df[df["attribute"].isin(top_attrs)]["ground_truth_letter"],
                     normalize="index")[LETTERS].multiply(100).plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("GT Distribution per Attribute (top 12, %)")
        ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        ax = axes[1, 0]
        chi2_df = pd.DataFrame(chi2_qt_rows)
        colors_chi = ["red" if p < 0.01 else "orange" if p < 0.05 else "green" for p in chi2_df["p_value"]]
        ax.barh(chi2_df["question_type"], chi2_df["chi2"], color=colors_chi)
        ax.set_title("Position Uniformity χ² per Question Type")
        for i, row in chi2_df.iterrows():
            ax.text(row["chi2"]+1, i, f"p={row['p_value']:.2e}", va="center", fontsize=9)

        ax = axes[1, 1]
        top_states = df["state"].value_counts().head(10).index
        pd.crosstab(df[df["state"].isin(top_states)]["state"],
                     df[df["state"].isin(top_states)]["ground_truth_letter"],
                     normalize="index")[LETTERS].multiply(100).plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("GT Distribution (top 10 states, %)")
        ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        # India position in Country Prediction
        cp = df[df["question_type"] == "Country Prediction"]
        india_pos = []
        for _, row in cp.iterrows():
            for opt, letter in zip(OPT_KEYS, LETTERS):
                if "india" in str(row[opt]).strip().lower():
                    india_pos.append(letter); break
        ipc = pd.Series(india_pos).value_counts().sort_index()
        ax = axes[1, 2]
        bars = ax.bar(ipc.index, ipc.values, color=["#e74c3c","#3498db","#2ecc71","#f39c12"])
        ax.set_title(f'Position of "India" in Country Prediction (n={len(cp)})')
        for bar, val in zip(bars, ipc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                    f"{val}\n({val/len(cp)*100:.1f}%)", ha="center", fontsize=10)
        ax.axhline(y=len(cp)/4, color="gray", linestyle="--", alpha=0.5)

        save_plot(fig, "eda_04_position_bias.png")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Country Prediction Audit
# ══════════════════════════════════════════════════════════════════════
def section_3(df):
    with timer("Section 3: Country Prediction Audit"):
        n = len(df)
        cp = df[df["question_type"] == "Country Prediction"].copy()

        # answer distribution
        answer_dist = cp["answer"].str.strip().value_counts().reset_index()
        answer_dist.columns = ["answer", "count"]
        save_csv(answer_dist, "country_prediction_answers.csv")

        # distractors
        all_dist = []
        for _, row in cp.iterrows():
            ans_lower = str(row["answer"]).strip().lower()
            for opt in OPT_KEYS:
                val = str(row[opt]).strip()
                if val.lower() != ans_lower:
                    all_dist.append(val)
        dist_counts = Counter(all_dist)
        dist_df = pd.DataFrame(dist_counts.most_common(), columns=["distractor", "count"])
        dist_df["pct"] = (dist_df["count"] / len(all_dist) * 100).round(2)
        save_csv(dist_df, "country_prediction_distractors.csv")

        # per attribute
        cp_attr = cp["attribute"].value_counts().reset_index()
        cp_attr.columns = ["attribute", "count_in_cp"]
        total_attr = df["attribute"].value_counts().reset_index()
        total_attr.columns = ["attribute", "count_total"]
        cp_attr = cp_attr.merge(total_attr, on="attribute")
        cp_attr["pct_of_attr"] = (cp_attr["count_in_cp"] / cp_attr["count_total"] * 100).round(1)
        save_csv(cp_attr, "country_prediction_by_attribute.csv")

        # india position
        india_pos = []
        for _, row in cp.iterrows():
            for opt, letter in zip(OPT_KEYS, LETTERS):
                if "india" in str(row[opt]).strip().lower():
                    india_pos.append(letter); break
        ipc = pd.Series(india_pos).value_counts().sort_index()
        audit = pd.DataFrame({
            "metric": ["total_cp_questions", "pct_of_dataset", "answer_is_india_pct",
                       "india_pos_A", "india_pos_B", "india_pos_C", "india_pos_D",
                       "unique_distractors"],
            "value": [len(cp), f"{len(cp)/n*100:.1f}%",
                      f"{(answer_dist[answer_dist['answer']=='India']['count'].sum())/len(cp)*100:.1f}%",
                      int(ipc.get("A",0)), int(ipc.get("B",0)), int(ipc.get("C",0)), int(ipc.get("D",0)),
                      len(dist_counts)]
        })
        save_csv(audit, "country_prediction_audit.csv")

        print(f"  Country Prediction: {len(cp)} questions, 100% answer=India")
        print(f"  Unique distractors: {len(dist_counts)}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Text & Lexical Analysis
# ══════════════════════════════════════════════════════════════════════
def section_4(df):
    with timer("Section 4: Text & Lexical Analysis"):

        # 4a. question length
        df["q_word_count"] = df["question"].apply(lambda x: len(tokenize(x)))

        qlen_by_type = df.groupby("question_type")["q_word_count"].agg(
            ["mean","median","std","min","max","count"]).round(2)
        save_csv_idx(qlen_by_type, "question_length_by_qtype.csv")

        qlen_by_attr = df.groupby("attribute")["q_word_count"].agg(
            ["mean","median","std","min","max","count"]).round(2)
        save_csv_idx(qlen_by_attr, "question_length_by_attribute.csv")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ax = axes[0]
        df["q_word_count"].hist(bins=50, ax=ax, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_title(f'Question Length (words)\nμ={df["q_word_count"].mean():.1f}')
        ax.axvline(df["q_word_count"].median(), color="red", linestyle="--",
                   label=f'Median={df["q_word_count"].median():.0f}')
        ax.legend()
        ax = axes[1]
        for qt in df["question_type"].unique():
            sub = df[df["question_type"] == qt]
            ax.hist(sub["q_word_count"], bins=30, alpha=0.5, label=f'{qt} (μ={sub["q_word_count"].mean():.1f})')
        ax.set_title("Question Length by Type"); ax.legend(fontsize=8)
        ax = axes[2]
        top_attrs = df["attribute"].value_counts().head(12).index
        df[df["attribute"].isin(top_attrs)].boxplot(column="q_word_count", by="attribute", ax=ax, vert=False)
        ax.set_title("Question Length by Attribute (top 12)"); plt.suptitle("")
        save_plot(fig, "eda_05_question_length.png")

        # 4b. option length correct vs incorrect
        opt_rows = []
        for _, row in df.iterrows():
            gt = row["ground_truth_letter"]
            for opt, letter in zip(OPT_KEYS, LETTERS):
                wc = len(tokenize(str(row[opt])))
                opt_rows.append({"letter": letter, "word_count": wc,
                                 "is_correct": letter == gt, "question_type": row["question_type"]})
        opt_df = pd.DataFrame(opt_rows)
        correct_lens = opt_df[opt_df["is_correct"]]["word_count"]
        incorrect_lens = opt_df[~opt_df["is_correct"]]["word_count"]

        t_stat, p_val = stats.ttest_ind(correct_lens, incorrect_lens)
        pooled_std = np.sqrt((correct_lens.std()**2 + incorrect_lens.std()**2) / 2)
        cohens_d = (correct_lens.mean() - incorrect_lens.mean()) / pooled_std

        opt_summary = opt_df.groupby("is_correct")["word_count"].agg(["mean","median","std"]).round(3)
        opt_summary.index = ["incorrect", "correct"]
        save_csv_idx(opt_summary, "option_length_correct_vs_incorrect.csv")
        save_csv_idx(opt_df.groupby("letter")["word_count"].agg(["mean","median","std"]).round(3),
                     "option_length_by_position.csv")
        save_csv_idx(opt_df.groupby(["question_type","is_correct"])["word_count"].agg(
            ["mean","median","std"]).round(3), "option_length_by_type_correctness.csv")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ax = axes[0]
        ax.hist(correct_lens, bins=40, alpha=0.6, label=f"Correct (μ={correct_lens.mean():.2f})",
                color="green", density=True)
        ax.hist(incorrect_lens, bins=40, alpha=0.6, label=f"Incorrect (μ={incorrect_lens.mean():.2f})",
                color="red", density=True)
        ax.set_title(f"Option Length: Correct vs Incorrect\nCohen's d={cohens_d:.3f}")
        ax.legend()
        ax = axes[1]
        pos_lens = opt_df.groupby("letter")["word_count"].mean()
        bars = ax.bar(pos_lens.index, pos_lens.values, color=["#e74c3c","#3498db","#2ecc71","#f39c12"])
        ax.set_title("Mean Option Length by Position")
        for bar, val in zip(bars, pos_lens.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{val:.2f}", ha="center")
        ax = axes[2]
        opt_df.groupby(["question_type","is_correct"])["word_count"].mean().unstack().plot(
            kind="bar", ax=ax, color=["red","green"], alpha=0.7)
        ax.set_title("Option Length by Type: Correct vs Incorrect")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        save_plot(fig, "eda_06_option_length_bias.png")

        # 4c. word frequency
        all_words = []
        attr_words = {}
        for _, row in df.iterrows():
            words = tokenize_no_stop(row["question"])
            all_words.extend(words)
            attr = row["attribute"]
            attr_words.setdefault(attr, []).extend(words)

        word_freq = Counter(all_words)
        all_bigrams = []
        all_trigrams = []
        for q in df["question"]:
            tokens = tokenize_no_stop(q)
            all_bigrams.extend(zip(tokens, tokens[1:]))
            all_trigrams.extend(zip(tokens, tokens[1:], tokens[2:]))
        bigram_freq = Counter(all_bigrams)
        trigram_freq = Counter(all_trigrams)

        save_csv(pd.DataFrame(word_freq.most_common(200), columns=["word","count"]),
                 "word_freq_unigrams_top200.csv")
        save_csv(pd.DataFrame([(" ".join(b), c) for b, c in bigram_freq.most_common(100)],
                               columns=["bigram","count"]), "word_freq_bigrams_top100.csv")
        save_csv(pd.DataFrame([(" ".join(t), c) for t, c in trigram_freq.most_common(100)],
                               columns=["trigram","count"]), "word_freq_trigrams_top100.csv")

        attr_rows = []
        for attr, words in attr_words.items():
            for word, count in Counter(words).most_common(20):
                attr_rows.append({"attribute": attr, "word": word, "count": count})
        save_csv(pd.DataFrame(attr_rows), "word_freq_by_attribute_top20.csv")

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, data, title, color in [
            (axes[0], word_freq.most_common(30), "Top 30 Unigrams", "steelblue"),
            (axes[1], bigram_freq.most_common(20), "Top 20 Bigrams", "coral"),
            (axes[2], trigram_freq.most_common(20), "Top 20 Trigrams", "mediumpurple"),
        ]:
            labels = [" ".join(w) if isinstance(w, tuple) else w for w, _ in data]
            vals = [c for _, c in data]
            ax.barh(range(len(labels)), vals, color=color)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis(); ax.set_title(title)
        save_plot(fig, "eda_07_word_frequency.png")

        # 4d. question templates
        def extract_template(q):
            q = str(q).strip()
            if q.startswith("According to you, which of the following is closely associated to"):
                return "According to you, which ... closely associated to {region}?"
            if q.startswith("Which state is famous for"):
                return "Which state is famous for {X}?"
            if q.startswith("Which country is the home to"):
                return "Which country is the home to {X}?"
            if "associated to which country" in q.lower():
                return "The {X} is associated to which country?"
            if q.startswith("Where is the") and "famous within" in q:
                return "Where is the {X} famous within {state}?"
            if q.startswith("Which of the given regions is home to"):
                return "Which of the given regions is home to the {X}?"
            if "which country" in q.lower():
                return 'Other "which country" pattern'
            if "which state" in q.lower():
                return 'Other "which state" pattern'
            if "closely associated" in q.lower():
                return 'Other "closely associated" pattern'
            return "_other_"

        df["template"] = df["question"].apply(extract_template)
        tmpl_counts = df["template"].value_counts().reset_index()
        tmpl_counts.columns = ["template", "count"]
        tmpl_counts["pct"] = (tmpl_counts["count"] / len(df) * 100).round(2)
        save_csv(tmpl_counts, "question_templates.csv")
        save_csv_idx(pd.crosstab(df["template"], df["question_type"]), "templates_by_qtype.csv")

        # 4e. lexical diversity (MTLD)
        from lexicalrichness import LexicalRichness

        def compute_mtld(texts, label):
            text = " ".join(texts.astype(str).tolist())
            if len(text.split()) < 50:
                return {"group": label, "mtld": np.nan, "n_words": len(text.split())}
            try:
                lr = LexicalRichness(text)
                return {"group": label, "mtld": round(lr.mtld(threshold=0.72), 2),
                        "ttr": round(lr.ttr, 4), "n_words": lr.words, "n_unique": lr.terms}
            except Exception:
                return {"group": label, "mtld": np.nan, "n_words": len(text.split())}

        attr_div = [dict(compute_mtld(df[df["attribute"]==a]["question"], a), type="attribute",
                         n_questions=len(df[df["attribute"]==a])) for a in df["attribute"].unique()]
        state_div = [dict(compute_mtld(df[df["state"]==s]["question"], s), type="state",
                          n_questions=len(df[df["state"]==s])) for s in df["state"].unique()]
        qtype_div = [dict(compute_mtld(df[df["question_type"]==q]["question"], q), type="question_type",
                          n_questions=len(df[df["question_type"]==q])) for q in df["question_type"].unique()]

        save_csv(pd.DataFrame(attr_div).sort_values("mtld", ascending=False), "lexical_diversity_by_attribute.csv")
        save_csv(pd.DataFrame(state_div).sort_values("mtld", ascending=False), "lexical_diversity_by_state.csv")
        save_csv(pd.DataFrame(qtype_div).sort_values("mtld", ascending=False), "lexical_diversity_by_qtype.csv")

        # ngram diversity
        def ngram_diversity(texts):
            tokens = []
            for t in texts:
                tokens.extend(re.findall(r"\b[a-zA-Z]+\b", str(t).lower()))
            scores = {}
            for n in range(1, 5):
                grams = list(zip(*[tokens[i:] for i in range(n)]))
                scores[f"ngd_{n}"] = round(len(set(grams)) / max(len(grams), 1), 4)
            scores["ngd_avg"] = round(np.mean([scores[f"ngd_{n}"] for n in range(1, 5)]), 4)
            return scores

        ngd_rows = []
        for attr in df["attribute"].unique():
            scores = ngram_diversity(df[df["attribute"]==attr]["question"])
            scores["attribute"] = attr
            scores["n_questions"] = len(df[df["attribute"]==attr])
            ngd_rows.append(scores)
        save_csv(pd.DataFrame(ngd_rows).sort_values("ngd_avg", ascending=False),
                 "ngram_diversity_by_attribute.csv")

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle("Lexical Diversity Analysis", fontsize=16, fontweight="bold")
        for ax, data, title in [
            (axes[0,0], pd.DataFrame(attr_div).dropna(subset=["mtld"]).sort_values("mtld"),
             "MTLD by Attribute"),
            (axes[0,1], pd.DataFrame(state_div).dropna(subset=["mtld"]).sort_values("mtld"),
             "MTLD by State"),
            (axes[1,0], pd.DataFrame(qtype_div).dropna(subset=["mtld"]).sort_values("mtld"),
             "MTLD by Question Type"),
        ]:
            if len(data) > 20:
                data = pd.concat([data.head(10), data.tail(10)])
            colors = ["#e74c3c" if n < 200 else "#3498db" for n in data.get("n_questions", [999]*len(data))]
            ax.barh(data["group"], data["mtld"], color=colors)
            ax.set_title(title); ax.set_xlabel("MTLD")

        ngd_sorted = pd.DataFrame(ngd_rows).sort_values("ngd_avg")
        axes[1,1].barh(ngd_sorted["attribute"], ngd_sorted["ngd_avg"], color="mediumpurple")
        axes[1,1].set_title("N-gram Diversity by Attribute"); axes[1,1].set_xlabel("NGD Average")
        save_plot(fig, "eda_08_lexical_diversity.png")

        print(f"  Cohen's d (option length bias): {cohens_d:.4f}")
        print(f"  Templates: {len(tmpl_counts)} unique, top covers {tmpl_counts.iloc[0]['pct']}%")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Semantic Analysis
# ══════════════════════════════════════════════════════════════════════
def section_5(df):
    with timer("Section 5: Semantic Analysis"):
        from sentence_transformers import SentenceTransformer
        import umap as umap_lib
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # 5a. compute / load embeddings
        emb_path = os.path.join(ANALYSIS_DIR, "question_embeddings.npy")
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
            print(f"  Loaded cached embeddings: {embeddings.shape}")
        else:
            print("  Computing question embeddings...")
            embeddings = model.encode(df["question"].tolist(), batch_size=256,
                                      show_progress_bar=True, normalize_embeddings=True)
            np.save(emb_path, embeddings)

        opt_path = os.path.join(ANALYSIS_DIR, "option_embeddings.npz")
        if os.path.exists(opt_path):
            od = np.load(opt_path)
            opt_embs = [od["opt1"], od["opt2"], od["opt3"], od["opt4"]]
            print("  Loaded cached option embeddings")
        else:
            print("  Computing option embeddings...")
            opt_embs = [model.encode(df[k].astype(str).tolist(), batch_size=256,
                                     normalize_embeddings=True) for k in OPT_KEYS]
            np.savez(opt_path, opt1=opt_embs[0], opt2=opt_embs[1], opt3=opt_embs[2], opt4=opt_embs[3])

        states = df["state"].unique().tolist()
        state_embs = model.encode(states, normalize_embeddings=True)
        state_emb_dict = dict(zip(states, state_embs))

        # 5b. UMAP
        print("  Running UMAP...")
        reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                                metric="cosine", random_state=42)
        coords = reducer.fit_transform(embeddings)
        save_csv(pd.DataFrame({"question_id": df["question_id"], "umap_x": coords[:,0], "umap_y": coords[:,1]}),
                 "umap_coordinates.csv")

        fig, axes = plt.subplots(1, 3, figsize=(27, 8))
        for ax, col, title, top_n in [
            (axes[0], "question_type", "by Question Type", None),
            (axes[1], "attribute", "by Attribute (top 8)", 8),
            (axes[2], "state", "by State (top 8)", 8),
        ]:
            groups = df[col].unique() if top_n is None else df[col].value_counts().head(top_n).index
            for g in groups:
                mask = df[col] == g
                ax.scatter(coords[mask,0], coords[mask,1], s=1, alpha=0.3, label=g)
            if top_n:
                other = ~df[col].isin(groups)
                ax.scatter(coords[other,0], coords[other,1], s=1, alpha=0.1, color="gray", label="Other")
            ax.set_title(f"UMAP {title}"); ax.legend(markerscale=5, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        save_plot(fig, "eda_09_umap.png")

        # 5c. near-duplicate detection
        print("  Finding near-duplicates (sim > 0.85)...")
        dups = []
        chunk = 2000
        n = len(embeddings)
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            sims = cosine_similarity(embeddings[i:end], embeddings)
            for li in range(end - i):
                gi = i + li
                high = np.where(sims[li, gi+1:] > 0.85)[0] + gi + 1
                for j in high:
                    dups.append({
                        "idx_a": gi, "idx_b": int(j), "sim": round(float(sims[li, j]), 4),
                        "question_a": df.iloc[gi]["question"][:100],
                        "question_b": df.iloc[int(j)]["question"][:100],
                        "same_state": df.iloc[gi]["state"] == df.iloc[int(j)]["state"],
                        "same_attr": df.iloc[gi]["attribute"] == df.iloc[int(j)]["attribute"],
                    })
        dup_df = pd.DataFrame(dups)
        save_csv(dup_df, "near_duplicates.csv")
        print(f"  Near-duplicate pairs: {len(dup_df)}")

        # 5d. no-question baseline
        nq_preds = []
        for i in range(len(df)):
            se = state_emb_dict[df.iloc[i]["state"]]
            sims = [float(np.dot(se, opt_embs[j][i])) for j in range(4)]
            nq_preds.append(LETTERS[np.argmax(sims)])
        nq_correct = np.mean([p == g for p, g in zip(nq_preds, df["ground_truth_letter"])])

        nq_rows = [{"slice": "overall", "accuracy_pct": round(nq_correct*100, 2), "n": len(df)}]
        for qt in df["question_type"].unique():
            mask = df["question_type"] == qt
            acc = np.mean([p == g for p, g, m in zip(nq_preds, df["ground_truth_letter"], mask) if m])
            nq_rows.append({"slice": qt, "accuracy_pct": round(acc*100, 2),
                            "n": int(mask.sum())})
        save_csv(pd.DataFrame(nq_rows), "no_question_baseline.csv")

        nq_attr_rows = []
        for attr in df["attribute"].unique():
            mask = df["attribute"] == attr
            acc = np.mean([p == g for p, g, m in zip(nq_preds, df["ground_truth_letter"], mask) if m])
            nq_attr_rows.append({"attribute": attr, "nq_accuracy": round(acc*100, 2),
                                 "n": int(mask.sum())})
        save_csv(pd.DataFrame(nq_attr_rows).sort_values("nq_accuracy", ascending=False),
                 "no_question_baseline_by_attribute.csv")

        # 5e. question-answer overlap
        qa_sims = []
        for i in range(len(df)):
            gt_idx = LETTERS.index(df.iloc[i]["ground_truth_letter"])
            qa_sims.append(float(np.dot(embeddings[i], opt_embs[gt_idx][i])))

        save_csv_idx(pd.DataFrame({"question_type": df["question_type"], "qa_overlap": qa_sims}).groupby(
            "question_type")["qa_overlap"].agg(["mean","median","std"]).round(4), "qa_overlap_by_qtype.csv")
        save_csv_idx(pd.DataFrame({"attribute": df["attribute"], "qa_overlap": qa_sims}).groupby(
            "attribute")["qa_overlap"].agg(["mean","median","std"]).round(4), "qa_overlap_by_attribute.csv")

        # 5f. TF-IDF per state
        from sklearn.feature_extraction.text import TfidfVectorizer
        state_docs = {s: " ".join(df[df["state"]==s]["question"].astype(str)) for s in states}
        sl = list(state_docs.keys())
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
        tfidf_mat = tfidf.fit_transform([state_docs[s] for s in sl])
        fnames = tfidf.get_feature_names_out()
        tfidf_rows = []
        for i, s in enumerate(sl):
            scores = tfidf_mat[i].toarray().flatten()
            for rank, idx in enumerate(scores.argsort()[-15:][::-1]):
                tfidf_rows.append({"state": s, "rank": rank+1, "term": fnames[idx],
                                   "tfidf_score": round(float(scores[idx]), 4)})
        save_csv(pd.DataFrame(tfidf_rows), "tfidf_terms_per_state.csv")

        # combined plot
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("Semantic Analysis", fontsize=16, fontweight="bold")

        ax = axes[0, 0]
        for qt in df["question_type"].unique():
            mask = df["question_type"] == qt
            vals = [s for s, m in zip(qa_sims, mask) if m]
            ax.hist(vals, bins=50, alpha=0.5, label=f"{qt} (μ={np.mean(vals):.3f})", density=True)
        ax.set_title("Question-Answer Cosine Similarity"); ax.legend(fontsize=8)

        ax = axes[0, 1]
        nq_df = pd.DataFrame(nq_rows)
        nq_qt = nq_df[nq_df["slice"] != "overall"]
        bars = ax.bar(nq_qt["slice"], nq_qt["accuracy_pct"], color=["#e74c3c","#3498db","#2ecc71","#f39c12"])
        ax.axhline(y=25, color="gray", linestyle="--", label="Random (25%)")
        ax.axhline(y=nq_correct*100, color="black", linestyle="--", label=f"Overall ({nq_correct*100:.1f}%)")
        ax.set_title("No-Question Baseline Accuracy"); ax.legend()
        ax.set_xticklabels(nq_qt["slice"], rotation=20, ha="right")
        for bar, val in zip(bars, nq_qt["accuracy_pct"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{val:.1f}%", ha="center", fontsize=9)

        ax = axes[1, 0]
        thresholds = [0.85, 0.88, 0.90, 0.93, 0.95, 0.97, 0.99]
        counts = [len(dup_df[dup_df["sim"] >= t]) for t in thresholds]
        ax.plot(thresholds, counts, "o-", color="steelblue", linewidth=2)
        ax.set_title(f"Near-Duplicate Pairs by Threshold (total: {len(dup_df):,})")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        for t, c in zip(thresholds, counts):
            ax.annotate(f"{c:,}", (t,c), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8)

        ax = axes[1, 1]
        nqa = pd.DataFrame(nq_attr_rows).sort_values("nq_accuracy")
        colors = ["#e74c3c" if a > 30 else "#3498db" for a in nqa["nq_accuracy"]]
        ax.barh(nqa["attribute"], nqa["nq_accuracy"], color=colors)
        ax.axvline(x=25, color="gray", linestyle="--", label="Random (25%)")
        ax.set_title("No-Question Baseline by Attribute"); ax.legend()
        save_plot(fig, "eda_10_semantic_analysis.png")

        print(f"  No-question baseline: {nq_correct*100:.2f}%")

        # 5g. BERTopic
        print("  Running BERTopic...")
        from bertopic import BERTopic
        topic_model = BERTopic(embedding_model=None, nr_topics=20, min_topic_size=100, verbose=False)
        topics, _ = topic_model.fit_transform(df["question"].tolist(), embeddings)

        topic_info = topic_model.get_topic_info()
        save_csv(topic_info, "bertopic_topics.csv")
        save_csv(pd.DataFrame({"question_id": df["question_id"], "bertopic_id": topics}),
                 "bertopic_assignments.csv")
        bt_vs_attr = pd.crosstab(pd.Series(topics, name="topic"), df["attribute"], normalize="index")
        save_csv_idx(bt_vs_attr, "bertopic_vs_attribute.csv")
        save_csv_idx(pd.crosstab(pd.Series(topics, name="topic"), df["question_type"], normalize="index"),
                     "bertopic_vs_qtype.csv")

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        non_out = topic_info[topic_info["Topic"] != -1].sort_values("Count", ascending=True)
        axes[0].barh(non_out["Name"].str[:50], non_out["Count"], color="steelblue")
        axes[0].set_title(f'BERTopic Topics (outlier -1: {topic_info[topic_info["Topic"]==-1]["Count"].values[0]})')

        top_t = topic_info[topic_info["Topic"] != -1].nlargest(10, "Count")["Topic"].tolist()
        hd = bt_vs_attr.loc[bt_vs_attr.index.isin(top_t)]
        tl = [topic_info[topic_info["Topic"]==t]["Name"].values[0][:40] for t in hd.index]
        sns.heatmap(hd, cmap="YlOrRd", ax=axes[1], annot=True, fmt=".2f", linewidths=0.5, yticklabels=tl)
        axes[1].set_title("Top 10 Topics vs Attributes (row-normalized)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        save_plot(fig, "eda_11_bertopic.png")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: Distractor Quality
# ══════════════════════════════════════════════════════════════════════
def section_6(df):
    with timer("Section 6: Distractor Quality"):
        opt_data = np.load(os.path.join(ANALYSIS_DIR, "option_embeddings.npz"))
        opt_embs = [opt_data["opt1"], opt_data["opt2"], opt_data["opt3"], opt_data["opt4"]]

        # 6a. correct-distractor similarity
        rows = []
        for i in range(len(df)):
            gt_idx = LETTERS.index(df.iloc[i]["ground_truth_letter"])
            correct_emb = opt_embs[gt_idx][i]
            sims = [float(np.dot(correct_emb, opt_embs[j][i])) for j in range(4) if j != gt_idx]
            rows.append({"question_id": df.iloc[i]["question_id"],
                         "question_type": df.iloc[i]["question_type"],
                         "attribute": df.iloc[i]["attribute"],
                         "mean_distractor_sim": np.mean(sims),
                         "max_distractor_sim": np.max(sims),
                         "min_distractor_sim": np.min(sims)})
        dist_df = pd.DataFrame(rows)
        save_csv(dist_df, "distractor_similarity.csv")
        save_csv_idx(dist_df.groupby("question_type")["mean_distractor_sim"].agg(
            ["mean","median","std","count"]).round(4), "distractor_similarity_by_qtype.csv")
        save_csv_idx(dist_df.groupby("attribute")["mean_distractor_sim"].agg(
            ["mean","median","std","count"]).round(4).sort_values("mean", ascending=False),
            "distractor_similarity_by_attribute.csv")

        # 6b. answer-in-question leakage
        leaks = []
        for _, row in df.iterrows():
            q = str(row["question"]).lower()
            ans = str(row["answer"]).strip().lower()
            if len(ans) > 3 and ans in q:
                leaks.append({"question_id": row["question_id"], "question": row["question"][:100],
                              "answer": row["answer"], "question_type": row["question_type"]})
        save_csv(pd.DataFrame(leaks), "answer_in_question_leakage.csv")

        # 6c. state prediction distractor domain
        sp = df[df["question_type"] == "State Prediction"]
        all_states = {s.lower().replace("_"," ") for s in df["state"].unique()}
        sp_state_dist = 0; sp_total = 0
        for _, row in sp.iterrows():
            gt = row["ground_truth_letter"]
            for opt, letter in zip(OPT_KEYS, LETTERS):
                if letter != gt:
                    sp_total += 1
                    val = str(row[opt]).strip().lower().replace("_"," ")
                    if val in all_states:
                        sp_state_dist += 1
        save_csv(pd.DataFrame([{
            "state_pred_distractors_total": sp_total,
            "distractors_that_are_states": sp_state_dist,
            "pct": round(sp_state_dist/max(sp_total,1)*100, 1),
            "answer_in_question_count": len(leaks),
            "answer_in_question_pct": round(len(leaks)/len(df)*100, 2),
        }]), "distractor_quality_summary.csv")

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        ax = axes[0]
        for qt in dist_df["question_type"].unique():
            sub = dist_df[dist_df["question_type"] == qt]
            ax.hist(sub["mean_distractor_sim"], bins=50, alpha=0.5,
                    label=f'{qt} (μ={sub["mean_distractor_sim"].mean():.3f})', density=True)
        ax.set_title("Distractor Plausibility (cosine sim)"); ax.legend(fontsize=8)

        ax = axes[1]
        da = dist_df.groupby("attribute")["mean_distractor_sim"].mean().sort_values()
        ax.barh(da.index, da.values, color="coral", alpha=0.7)
        ax.set_title("Distractor Plausibility by Attribute")
        save_plot(fig, "eda_12_distractor_quality.png")

        print(f"  Answer-in-question leakage: {len(leaks)} ({len(leaks)/len(df)*100:.1f}%)")
        print(f"  State Pred distractors that are states: {sp_state_dist}/{sp_total} ({sp_state_dist/max(sp_total,1)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: Cultural Specificity
# ══════════════════════════════════════════════════════════════════════
def section_7(df):
    with timer("Section 7: Cultural Specificity"):
        df = df.copy()
        # Regex-only extraction (for per-template analysis)
        df["regex_entity"] = df["question"].apply(_extract_entity_regex)
        found = df["regex_entity"].notna().sum()
        unique_regex = df["regex_entity"].dropna().nunique()
        print(f"  Regex-extracted: {found}/{len(df)} ({found/len(df)*100:.1f}%)")

        # Regex-only entity CSV (backwards-compatible)
        entity_counts = df["regex_entity"].dropna().value_counts().reset_index()
        entity_counts.columns = ["entity", "question_count"]
        ent_states = df.dropna(subset=["regex_entity"]).groupby("regex_entity")["state"].nunique()
        ent_qtypes = df.dropna(subset=["regex_entity"]).groupby("regex_entity")["question_type"].nunique()
        entity_counts["n_states"] = [int(ent_states.get(e, 0)) for e in entity_counts["entity"]]
        entity_counts["n_qtypes"] = [int(ent_qtypes.get(e, 0)) for e in entity_counts["entity"]]
        save_csv(entity_counts, "cultural_entities.csv")

        entity_detail = df.dropna(subset=["regex_entity"]).groupby("regex_entity").agg(
            question_count=("question_id", "count"),
            n_states=("state", "nunique"),
            states=("state", lambda x: ",".join(sorted(x.unique()))),
            attributes=("attribute", lambda x: ",".join(sorted(x.unique()))),
            qtypes=("question_type", lambda x: ",".join(sorted(x.unique())))
        ).sort_values("question_count", ascending=False)
        save_csv_idx(entity_detail, "cultural_entities_detail.csv")

        # Combined entity keys (regex + fallback) — uses entity_key from load_data()
        combined = df.groupby("entity_key").agg(
            question_count=("question_id", "count"),
            n_states=("state", "nunique"),
            states=("state", lambda x: ",".join(sorted(x.unique()))),
            attributes=("attribute", lambda x: ",".join(sorted(x.unique()))),
            qtypes=("question_type", lambda x: ",".join(sorted(x.unique()))),
            extraction_method=("regex_entity", lambda x: "regex" if x.notna().all()
                               else ("fallback" if x.isna().all() else "mixed"))
        ).sort_values("question_count", ascending=False)
        save_csv_idx(combined, "cultural_entities_combined.csv")

        unique_combined = df["entity_key"].nunique()
        n_fallback = df["regex_entity"].isna().sum()
        print(f"  Combined entity keys: {unique_combined} unique "
              f"({unique_regex} regex + {df['regex_entity'].isna().apply(lambda x: x).sum()} fallback questions)")

        # Extraction rate by question type
        by_qt = df.groupby("question_type").agg(
            extracted=("regex_entity", lambda x: x.notna().sum()),
            total=("question_id", "count"))
        by_qt["pct"] = (by_qt["extracted"] / by_qt["total"] * 100).round(1)
        by_qt["missing"] = by_qt["total"] - by_qt["extracted"]
        save_csv_idx(by_qt, "entity_extraction_by_qtype.csv")

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        top_ents = entity_counts.head(20)
        axes[0].barh(range(len(top_ents)), top_ents["question_count"], color="mediumpurple")
        axes[0].set_yticks(range(len(top_ents)))
        axes[0].set_yticklabels([e[:40] for e in top_ents["entity"]], fontsize=8)
        axes[0].set_title(f"Top 20 Regex-Extracted Entities ({unique_regex} regex, {unique_combined} combined)")
        axes[0].invert_yaxis()

        uniq = ent_states.value_counts().sort_index()
        axes[1].bar(uniq.index, uniq.values, color="steelblue")
        axes[1].set_title(f"Entity State Uniqueness ({(ent_states==1).mean()*100:.1f}% unique to 1 state)")
        axes[1].set_xlabel("# States"); axes[1].set_ylabel("# Entities")
        save_plot(fig, "eda_13_cultural_specificity.png")


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: Data Quality Final Checks
# ══════════════════════════════════════════════════════════════════════
def section_8(df):
    with timer("Section 8: Data Quality Final Checks"):
        n = len(df)

        # exact duplicates
        dup_q = df[df.duplicated(subset=["question"], keep=False)].sort_values("question")
        n_dup_groups = dup_q.groupby("question").ngroups
        n_dup_rows = len(dup_q)

        conflicting = []
        for q, grp in dup_q.groupby("question"):
            if grp["ground_truth_letter"].nunique() > 1:
                conflicting.append({"question": q[:100], "n_rows": len(grp),
                                    "answers": ",".join(grp["ground_truth_letter"].unique())})
        if conflicting:
            save_csv(pd.DataFrame(conflicting), "conflicting_duplicates.csv")

        dup_summary = dup_q.groupby("question").agg(
            count=("question_id", "count"),
            states=("state", lambda x: ",".join(sorted(x.unique()))),
            same_answer=("ground_truth_letter", lambda x: x.nunique() == 1)
        ).sort_values("count", ascending=False)
        save_csv_idx(dup_summary, "exact_duplicates.csv")

        # source column
        col = "short explaination / source link"
        answer_in_source = 0
        if col in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row.get(col)) and str(row["answer"]).lower() in str(row[col]).lower():
                    answer_in_source += 1

        # near-dup count
        nd_path = os.path.join(ANALYSIS_DIR, "near_duplicates.csv")
        nd_questions = 0
        if os.path.exists(nd_path):
            nd = pd.read_csv(nd_path)
            nd_questions = len(set(nd["idx_a"].tolist() + nd["idx_b"].tolist()))

        quality = pd.DataFrame([{
            "total_rows": n,
            "unique_questions": df["question"].nunique(),
            "unique_pct": round(df["question"].nunique()/n*100, 1),
            "exact_dup_rows": n_dup_rows,
            "exact_dup_groups": n_dup_groups,
            "conflicting_dup_groups": len(conflicting),
            "near_dup_questions_sim85": nd_questions,
            "near_dup_pct": round(nd_questions/n*100, 1),
            "answer_in_source_col": answer_in_source,
        }])
        save_csv(quality, "data_quality_summary.csv")

        print(f"  Unique questions: {df['question'].nunique()}/{n} ({df['question'].nunique()/n*100:.1f}%)")
        print(f"  Exact dup groups: {n_dup_groups}, rows: {n_dup_rows}")
        print(f"  Conflicting dups: {len(conflicting)}")
        print(f"  Near-dup questions (sim>0.85): {nd_questions}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Sanskriti EDA Pipeline")
    parser.add_argument("--section", type=int, default=0,
                        help="Run only this section (1-8). 0 = all.")
    args = parser.parse_args()

    t0 = time.time()
    df = load_data()

    sections = {
        1: section_1, 2: section_2, 3: section_3, 4: section_4,
        5: section_5, 6: section_6, 7: section_7, 8: section_8,
    }

    if args.section == 0:
        for i in sorted(sections):
            sections[i](df)
    elif args.section in sections:
        # sections 6-8 depend on section 5 outputs (embeddings)
        sections[args.section](df)
    else:
        print(f"Invalid section: {args.section}. Use 1-8.")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  EDA COMPLETE — {elapsed/60:.1f} minutes")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  CSVs:  {ANALYSIS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
