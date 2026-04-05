"""
Step 1: Merge & Analyze
=======================
Merges base and instruct results, assigns behavioral labels,
computes three-tier stats, entity-level analysis, and generates plots.

Run after both models have completed eval_step1.py.

Usage:
    python merge_step1.py
"""

import json
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ============================================================
# Config & Logging
# ============================================================

def load_config():
    with open("/home/anshulk/cultural-mi/configs/config.yaml") as f:
        return yaml.safe_load(f)


def setup_logger():
    log_dir = "/home/anshulk/cultural-mi/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"step1_merge_{timestamp}.log")

    logger = logging.getLogger("step1_merge")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================
# Load & Merge
# ============================================================

def load_and_merge(config, log):
    """Load base + instruct results and original dataset, merge into one DataFrame."""
    results_dir = config["results"]["step1"]

    base_path = os.path.join(results_dir, "base_results.csv")
    inst_path = os.path.join(results_dir, "instruct_results.csv")

    log.info(f"Loading base results from {base_path}")
    base = pd.read_csv(base_path)
    log.info(f"  {len(base)} rows")

    log.info(f"Loading instruct results from {inst_path}")
    inst = pd.read_csv(inst_path)
    log.info(f"  {len(inst)} rows")

    # Rename columns to avoid collisions
    base = base.rename(columns={
        "predicted_letter": "predicted_base",
        "correct": "base_correct",
        "forced_choice": "base_forced",
        "top1_token_id": "base_top1_token_id",
    })
    for col in ["logprob_A", "logprob_B", "logprob_C", "logprob_D"]:
        base = base.rename(columns={col: f"base_{col}"})

    inst = inst.rename(columns={
        "predicted_letter": "predicted_instruct",
        "correct": "instruct_correct",
        "forced_choice": "instruct_forced",
        "top1_token_id": "instruct_top1_token_id",
    })
    for col in ["logprob_A", "logprob_B", "logprob_C", "logprob_D"]:
        inst = inst.rename(columns={col: f"instruct_{col}"})

    # Merge on question_id
    merged = base.merge(
        inst.drop(columns=["ground_truth_letter"]),
        on="question_id",
        how="inner",
    )
    log.info(f"Merged: {len(merged)} rows (should be {len(base)})")
    assert len(merged) == len(base), f"Merge mismatch: {len(merged)} != {len(base)}"

    # Load prepared dataset (saved by eval_step1.py with entity_key attached)
    prep_path = os.path.join(results_dir, "sanskriti_prepared.csv")
    log.info(f"Loading prepared dataset from {prep_path}")
    orig = pd.read_csv(prep_path)

    # Merge metadata columns
    meta_cols = ["question_id", "state", "attribute", "question", "option1", "option2",
                 "option3", "option4", "answer", "question_type"]
    if "entity_key" in orig.columns:
        meta_cols.append("entity_key")
    merged = merged.merge(orig[meta_cols], on="question_id", how="left")

    # Validate question types
    qtypes = sorted(merged["question_type"].unique())
    log.info(f"Question types found: {qtypes}")
    assert len(merged[merged["question_type"] == "Country Prediction"]) > 0, "No Country Prediction found"

    return merged


# ============================================================
# Behavioral Labeling
# ============================================================

def assign_labels(df, log):
    """Assign behavioral labels based on base_correct and instruct_correct."""
    conditions = [
        (df["base_correct"] == 1) & (df["instruct_correct"] == 0),
        (df["base_correct"] == 0) & (df["instruct_correct"] == 1),
        (df["base_correct"] == 1) & (df["instruct_correct"] == 1),
        (df["base_correct"] == 0) & (df["instruct_correct"] == 0),
    ]
    labels = ["suppression", "enhancement", "control_both_correct", "control_both_wrong"]
    df["behavioral_label"] = np.select(conditions, labels, default="unknown")

    log.info("Behavioral label distribution:")
    for label in labels:
        count = (df["behavioral_label"] == label).sum()
        pct = count / len(df) * 100
        log.info(f"  {label}: {count} ({pct:.1f}%)")

    return df


# ============================================================
# Three-Tier Stats
# ============================================================

def compute_tier_stats(df, tier_name, mask, log):
    """Compute stats for a single tier."""
    subset = df[mask]
    n = len(subset)
    if n == 0:
        return {}

    stats = {
        "tier": tier_name,
        "n_questions": n,
        "base_accuracy": float(subset["base_correct"].mean()),
        "instruct_accuracy": float(subset["instruct_correct"].mean()),
        "base_forced_rate": float(subset["base_forced"].mean()),
        "instruct_forced_rate": float(subset["instruct_forced"].mean()),
    }

    for label in ["suppression", "enhancement", "control_both_correct", "control_both_wrong"]:
        count = (subset["behavioral_label"] == label).sum()
        stats[f"{label}_count"] = int(count)
        stats[f"{label}_pct"] = float(count / n * 100)

    log.info(f"--- {tier_name} (n={n}) ---")
    log.info(f"  Base accuracy:    {stats['base_accuracy']*100:.2f}%")
    log.info(f"  Instruct accuracy:{stats['instruct_accuracy']*100:.2f}%")
    log.info(f"  Suppression:      {stats['suppression_count']} ({stats['suppression_pct']:.1f}%)")
    log.info(f"  Enhancement:      {stats['enhancement_count']} ({stats['enhancement_pct']:.1f}%)")
    log.info(f"  Both correct:     {stats['control_both_correct_count']} ({stats['control_both_correct_pct']:.1f}%)")
    log.info(f"  Both wrong:       {stats['control_both_wrong_count']} ({stats['control_both_wrong_pct']:.1f}%)")
    log.info(f"  Base forced:      {stats['base_forced_rate']*100:.2f}%")
    log.info(f"  Instruct forced:  {stats['instruct_forced_rate']*100:.2f}%")

    return stats


def compute_all_tiers(df, log):
    """Compute stats for all three tiers."""
    tiers = {}

    # Tier 1: Full dataset
    tiers["tier1_full"] = compute_tier_stats(
        df, "Tier 1: Full (21,726)", pd.Series(True, index=df.index), log
    )

    # Tier 2: Without Country Prediction
    mask_t2 = df["question_type"] != "Country Prediction"
    n_t2 = mask_t2.sum()
    log.debug(f"Tier 2 count: {n_t2} (expected ~16163)")
    tiers["tier2_no_cp"] = compute_tier_stats(
        df, "Tier 2: No Country Prediction", mask_t2, log
    )

    # Tier 3: Hard subset (Association + General Awareness)
    mask_t3 = df["question_type"].isin(["Association", "General Awareness"])
    n_t3 = mask_t3.sum()
    log.debug(f"Tier 3 count: {n_t3} (expected ~10903)")
    tiers["tier3_hard"] = compute_tier_stats(
        df, "Tier 3: Hard (Association + General Awareness)", mask_t3, log
    )

    return tiers


# ============================================================
# Per-Dimension Breakdowns
# ============================================================

def per_question_type_stats(df, log):
    """Accuracy and suppression by question type."""
    rows = []
    for qt in sorted(df["question_type"].unique()):
        sub = df[df["question_type"] == qt]
        rows.append({
            "question_type": qt,
            "n": len(sub),
            "base_accuracy": sub["base_correct"].mean(),
            "instruct_accuracy": sub["instruct_correct"].mean(),
            "suppression_rate": (sub["behavioral_label"] == "suppression").mean(),
            "enhancement_rate": (sub["behavioral_label"] == "enhancement").mean(),
        })
    result = pd.DataFrame(rows)
    log.info("Per question type:")
    for _, row in result.iterrows():
        log.info(f"  {row['question_type']:25s} n={row['n']:5d} "
                 f"base={row['base_accuracy']*100:5.1f}% "
                 f"inst={row['instruct_accuracy']*100:5.1f}% "
                 f"supp={row['suppression_rate']*100:5.1f}%")
    return result


def per_attribute_stats(df, log):
    """Suppression rate by attribute."""
    rows = []
    for attr in sorted(df["attribute"].unique()):
        sub = df[df["attribute"] == attr]
        rows.append({
            "attribute": attr,
            "n": len(sub),
            "base_accuracy": sub["base_correct"].mean(),
            "instruct_accuracy": sub["instruct_correct"].mean(),
            "suppression_rate": (sub["behavioral_label"] == "suppression").mean(),
            "suppression_count": (sub["behavioral_label"] == "suppression").sum(),
            "low_confidence": len(sub) < 200,
        })
    return pd.DataFrame(rows).sort_values("suppression_rate", ascending=False)


def per_state_stats(df, log):
    """Suppression rate by state."""
    rows = []
    for state in sorted(df["state"].unique()):
        sub = df[df["state"] == state]
        rows.append({
            "state": state,
            "n": len(sub),
            "base_accuracy": sub["base_correct"].mean(),
            "instruct_accuracy": sub["instruct_correct"].mean(),
            "suppression_rate": (sub["behavioral_label"] == "suppression").mean(),
            "suppression_count": (sub["behavioral_label"] == "suppression").sum(),
        })
    return pd.DataFrame(rows).sort_values("suppression_rate", ascending=False)


def position_distribution(df, log):
    """A/B/C/D prediction distribution per model vs ground truth."""
    rows = []
    for letter in "ABCD":
        rows.append({
            "letter": letter,
            "ground_truth_count": (df["ground_truth_letter"] == letter).sum(),
            "ground_truth_pct": (df["ground_truth_letter"] == letter).mean() * 100,
            "base_pred_count": (df["predicted_base"] == letter).sum(),
            "base_pred_pct": (df["predicted_base"] == letter).mean() * 100,
            "instruct_pred_count": (df["predicted_instruct"] == letter).sum(),
            "instruct_pred_pct": (df["predicted_instruct"] == letter).mean() * 100,
        })
    result = pd.DataFrame(rows)
    log.info("Position distribution:")
    for _, row in result.iterrows():
        log.info(f"  {row['letter']}: GT={row['ground_truth_pct']:.1f}% "
                 f"Base={row['base_pred_pct']:.1f}% "
                 f"Inst={row['instruct_pred_pct']:.1f}%")
    return result


# ============================================================
# Entity-Level Analysis
# ============================================================

def entity_analysis(df, log):
    """Entity-level behavioral labels using entity_key already in the DataFrame."""
    if "entity_key" not in df.columns:
        log.warning("entity_key column not found in DataFrame — skipping entity analysis")
        return pd.DataFrame()

    n_missing = df["entity_key"].isna().sum()
    if n_missing > 0:
        log.warning(f"{n_missing} questions have no entity_key — excluded from entity analysis")
        df = df[df["entity_key"].notna()]

    # Per-entity stats
    entity_groups = df.groupby("entity_key")
    rows = []
    for ek, group in entity_groups:
        n = len(group)
        n_supp = (group["behavioral_label"] == "suppression").sum()
        n_enh = (group["behavioral_label"] == "enhancement").sum()
        n_bc = (group["behavioral_label"] == "control_both_correct").sum()
        n_bw = (group["behavioral_label"] == "control_both_wrong").sum()

        if n_supp == n:
            entity_label = "suppressed"
        elif n_enh == n:
            entity_label = "enhanced"
        elif n_bc == n:
            entity_label = "both_correct"
        elif n_bw == n:
            entity_label = "both_wrong"
        else:
            entity_label = "mixed"

        rows.append({
            "entity_key": ek,
            "n_questions": n,
            "entity_label": entity_label,
            "suppression_count": n_supp,
            "suppression_rate": n_supp / n if n > 0 else 0,
            "enhancement_count": n_enh,
            "base_accuracy": group["base_correct"].mean(),
            "instruct_accuracy": group["instruct_correct"].mean(),
        })

    result = pd.DataFrame(rows)

    log.info("Entity-level analysis:")
    for label in ["suppressed", "enhanced", "both_correct", "both_wrong", "mixed"]:
        count = (result["entity_label"] == label).sum()
        log.info(f"  {label}: {count} entities")

    q_level_supp = (df["behavioral_label"] == "suppression").mean() * 100
    e_level_supp = (result["entity_label"] == "suppressed").mean() * 100
    log.info(f"  Question-level suppression: {q_level_supp:.1f}%")
    log.info(f"  Entity-level suppression:   {e_level_supp:.1f}%")

    # Top 20 most suppressed entities
    top20 = result.nlargest(20, "suppression_count")
    log.info("Top 20 most suppressed entities (by suppression question count):")
    for _, row in top20.iterrows():
        log.info(f"  {row['entity_key'][:60]:60s} supp={row['suppression_count']}/{row['n_questions']}")

    return result


# ============================================================
# Plots
# ============================================================

PLOT_DIR = "/home/anshulk/cultural-mi/plots/step1"


def plot_accuracy_by_qtype(qtype_df, log):
    """Plot 1: Grouped bar — accuracy by question type (base vs instruct)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(qtype_df))
    w = 0.35
    ax.bar(x - w/2, qtype_df["base_accuracy"] * 100, w, label="Base", color="#4C72B0")
    ax.bar(x + w/2, qtype_df["instruct_accuracy"] * 100, w, label="Instruct", color="#DD8452")
    ax.set_xlabel("Question Type")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Question Type")
    ax.set_xticks(x)
    ax.set_xticklabels(qtype_df["question_type"], rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    for i, (b, inst) in enumerate(zip(qtype_df["base_accuracy"], qtype_df["instruct_accuracy"])):
        ax.text(i - w/2, b * 100 + 1, f"{b*100:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, inst * 100 + 1, f"{inst*100:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "accuracy_by_question_type.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Plot saved: {path}")


def plot_behavioral_labels(tier_stats, log):
    """Plot 2: Stacked bar — behavioral label distribution across tiers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = list(tier_stats.keys())
    tier_labels = ["Full\n(21,726)", "No CP\n(16,163)", "Hard\n(10,903)"]
    labels = ["suppression", "enhancement", "control_both_correct", "control_both_wrong"]
    colors = ["#C44E52", "#55A868", "#4C72B0", "#8C8C8C"]
    display_names = ["Suppression", "Enhancement", "Both Correct", "Both Wrong"]

    x = np.arange(len(tiers))
    bottom = np.zeros(len(tiers))

    for label, color, dname in zip(labels, colors, display_names):
        values = [tier_stats[t].get(f"{label}_pct", 0) for t in tiers]
        ax.bar(x, values, bottom=bottom, label=dname, color=color, width=0.5)
        for i, v in enumerate(values):
            if v > 3:
                ax.text(i, bottom[i] + v/2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += values

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Behavioral Label Distribution by Tier")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "behavioral_label_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Plot saved: {path}")


def plot_suppression_by_attribute(attr_df, log):
    """Plot 3: Horizontal bar — suppression rate by attribute."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_df = attr_df.sort_values("suppression_rate")
    colors = ["#8C8C8C" if lc else "#C44E52" for lc in sorted_df["low_confidence"]]
    bars = ax.barh(sorted_df["attribute"], sorted_df["suppression_rate"] * 100, color=colors)
    ax.set_xlabel("Suppression Rate (%)")
    ax.set_title("Suppression Rate by Attribute")
    for bar, n in zip(bars, sorted_df["n"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"n={n}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "suppression_by_attribute.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Plot saved: {path}")


def plot_suppression_by_state(state_df, log):
    """Plot 4: Horizontal bar — suppression rate by state (top 20)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    top20 = state_df.nlargest(20, "suppression_rate")
    sorted_df = top20.sort_values("suppression_rate")
    ax.barh(sorted_df["state"], sorted_df["suppression_rate"] * 100, color="#C44E52")
    ax.set_xlabel("Suppression Rate (%)")
    ax.set_title("Suppression Rate by State (Top 20)")
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row["suppression_rate"] * 100 + 0.3, i,
                f"n={row['n']}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "suppression_by_state.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Plot saved: {path}")


def plot_position_distribution(pos_df, log):
    """Plot 5: Grouped bar — A/B/C/D prediction distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(4)
    w = 0.25
    ax.bar(x - w, pos_df["ground_truth_pct"], w, label="Ground Truth", color="#4C72B0")
    ax.bar(x, pos_df["base_pred_pct"], w, label="Base", color="#DD8452")
    ax.bar(x + w, pos_df["instruct_pred_pct"], w, label="Instruct", color="#55A868")
    ax.set_xlabel("Answer Position")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Prediction Position Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.legend()
    ax.set_ylim(0, 50)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "position_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Plot saved: {path}")


def plot_confidence_distribution(df, log):
    """Plot 6: Histogram of max A/B/C/D logprob (base vs instruct)."""
    base_lp_cols = ["base_logprob_A", "base_logprob_B", "base_logprob_C", "base_logprob_D"]
    inst_lp_cols = ["instruct_logprob_A", "instruct_logprob_B", "instruct_logprob_C", "instruct_logprob_D"]

    base_max_lp = df[base_lp_cols].max(axis=1)
    inst_max_lp = df[inst_lp_cols].max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-12, 0, 50)
    ax.hist(base_max_lp, bins=bins, alpha=0.5, label="Base", color="#4C72B0")
    ax.hist(inst_max_lp, bins=bins, alpha=0.5, label="Instruct", color="#DD8452")
    ax.set_xlabel("Max Log-Probability (A/B/C/D)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confidence_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Backing CSV
    conf_df = pd.DataFrame({
        "base_max_logprob_mean": [base_max_lp.mean()],
        "base_max_logprob_median": [base_max_lp.median()],
        "instruct_max_logprob_mean": [inst_max_lp.mean()],
        "instruct_max_logprob_median": [inst_max_lp.median()],
    })
    log.info(f"Plot saved: {path}")
    return conf_df


# ============================================================
# Sanity Checks
# ============================================================

def run_sanity_checks(df, tier_stats, config, log):
    """Run all sanity checks from Section 12 of the plan."""
    log.info("=" * 60)
    log.info("SANITY CHECKS")
    log.info("=" * 60)
    all_pass = True

    # F1: Base accuracy (Tier 2)
    t2 = tier_stats["tier2_no_cp"]
    base_acc = t2["base_accuracy"] * 100
    if 40 <= base_acc <= 70:
        log.info(f"[PASS] F1: Base accuracy (Tier 2) = {base_acc:.1f}% (expected 40-70%)")
    else:
        log.warning(f"[WARN] F1: Base accuracy (Tier 2) = {base_acc:.1f}% (expected 40-70%)")
        all_pass = False

    # F2: Instruct > Base
    t1 = tier_stats["tier1_full"]
    if t1["instruct_accuracy"] > t1["base_accuracy"]:
        log.info(f"[PASS] F2: Instruct ({t1['instruct_accuracy']*100:.1f}%) > Base ({t1['base_accuracy']*100:.1f}%)")
    else:
        log.error(f"[FAIL] F2: Instruct ({t1['instruct_accuracy']*100:.1f}%) <= Base ({t1['base_accuracy']*100:.1f}%)")
        all_pass = False

    # F3: Country Prediction accuracy
    cp = df[df["question_type"] == "Country Prediction"]
    base_cp = cp["base_correct"].mean() * 100
    inst_cp = cp["instruct_correct"].mean() * 100
    if base_cp >= 95 and inst_cp >= 95:
        log.info(f"[PASS] F3: Country Prediction: base={base_cp:.1f}%, inst={inst_cp:.1f}% (expected >=95%)")
    else:
        log.warning(f"[WARN] F3: Country Prediction: base={base_cp:.1f}%, inst={inst_cp:.1f}% (expected >=95%)")
        all_pass = False

    # F4: Forced-choice rate
    base_fc = df["base_forced"].mean() * 100
    inst_fc = df["instruct_forced"].mean() * 100
    if inst_fc < 5:
        log.info(f"[PASS] F4: Forced choice: base={base_fc:.1f}%, inst={inst_fc:.1f}%")
    else:
        log.warning(f"[WARN] F4: Forced choice: base={base_fc:.1f}%, inst={inst_fc:.1f}% (inst expected <5%)")
        all_pass = False

    # F5: Position distribution
    for model, col in [("base", "predicted_base"), ("instruct", "predicted_instruct")]:
        for letter in "ABCD":
            pct = (df[col] == letter).mean() * 100
            if pct > 40:
                log.warning(f"[WARN] F5: {model} predicts '{letter}' at {pct:.1f}% (>40% = severe bias)")
                all_pass = False

    # F6/F7: Suppression/Enhancement rates
    supp_rate = t1["suppression_pct"]
    enh_rate = t1["enhancement_pct"]
    log.info(f"[INFO] F6: Suppression rate = {supp_rate:.1f}% (expected 5-15%)")
    log.info(f"[INFO] F7: Enhancement rate = {enh_rate:.1f}% (expected 4-12%)")

    # F8: Per-state suppression concentration
    supp_df = df[df["behavioral_label"] == "suppression"]
    if len(supp_df) > 0:
        state_supp = supp_df["state"].value_counts(normalize=True)
        if state_supp.iloc[0] > 0.15:
            log.warning(f"[WARN] F8: {state_supp.index[0]} has {state_supp.iloc[0]*100:.1f}% of all suppression")
            all_pass = False
        else:
            log.info(f"[PASS] F8: No state >15% of suppression (max: {state_supp.index[0]} at {state_supp.iloc[0]*100:.1f}%)")

    # F10/F11: Activation checks
    for model_type in ["base", "instruct"]:
        act_dir = os.path.join(config["activations"], model_type)
        for hname in ["embed", "layer_04", "layer_08", "layer_14", "layer_20", "layer_26", "layer_30", "layer_31"]:
            for ptype in ["mean_pool", "last_token"]:
                path = os.path.join(act_dir, ptype, f"{hname}.npy")
                if os.path.exists(path):
                    arr = np.load(path, mmap_mode="r")
                    if arr.shape != (len(df), 4096):
                        log.error(f"[FAIL] F10: {path} shape={arr.shape}, expected ({len(df)}, 4096)")
                        all_pass = False
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        log.error(f"[FAIL] F11: {path} has NaN/Inf")
                        all_pass = False
                else:
                    log.error(f"[FAIL] F10: Missing {path}")
                    all_pass = False

    if all_pass:
        log.info("ALL SANITY CHECKS PASSED")
    else:
        log.warning("SOME CHECKS FAILED — review warnings/errors above")

    return all_pass


# ============================================================
# Main
# ============================================================

def main():
    config = load_config()
    log = setup_logger()

    log.info("=" * 60)
    log.info("Step 1: Merge & Analysis")
    log.info("=" * 60)

    results_dir = config["results"]["step1"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Load and merge
    df = load_and_merge(config, log)

    # 2. Assign behavioral labels
    df = assign_labels(df, log)

    # 3. Save master CSV
    master_path = os.path.join(results_dir, "sanskriti_behavioral_labels.csv")
    df.to_csv(master_path, index=False)
    log.info(f"Master CSV saved: {master_path} ({len(df)} rows)")

    # 4. Three-tier stats
    tier_stats = compute_all_tiers(df, log)

    # 5. Per-dimension breakdowns
    qtype_df = per_question_type_stats(df, log)
    qtype_df.to_csv(os.path.join(results_dir, "accuracy_by_question_type.csv"), index=False)

    attr_df = per_attribute_stats(df, log)
    attr_df.to_csv(os.path.join(results_dir, "suppression_by_attribute.csv"), index=False)

    state_df = per_state_stats(df, log)
    state_df.to_csv(os.path.join(results_dir, "suppression_by_state.csv"), index=False)

    pos_df = position_distribution(df, log)
    pos_df.to_csv(os.path.join(results_dir, "position_distribution.csv"), index=False)

    # 6. Behavioral label counts for all tiers
    label_rows = []
    for tier_name, tier_data in tier_stats.items():
        for label in ["suppression", "enhancement", "control_both_correct", "control_both_wrong"]:
            label_rows.append({
                "tier": tier_name,
                "label": label,
                "count": tier_data.get(f"{label}_count", 0),
                "pct": tier_data.get(f"{label}_pct", 0),
            })
    pd.DataFrame(label_rows).to_csv(
        os.path.join(results_dir, "behavioral_label_counts.csv"), index=False
    )

    # 7. Entity-level analysis
    entity_df = entity_analysis(df, log)
    if len(entity_df) > 0:
        entity_df.to_csv(os.path.join(results_dir, "entity_behavioral_labels.csv"), index=False)

    # 8. Forced choice audit
    forced_df = df[(df["base_forced"] == True) | (df["instruct_forced"] == True)]
    forced_df.to_csv(os.path.join(results_dir, "forced_choice_audit.csv"), index=False)
    log.info(f"Forced choice audit: {len(forced_df)} questions")

    # 9. Save aggregate stats JSON
    agg = {
        "tiers": tier_stats,
        "n_questions": len(df),
        "n_forced_base": int(df["base_forced"].sum()),
        "n_forced_instruct": int(df["instruct_forced"].sum()),
    }
    with open(os.path.join(results_dir, "step1_aggregate_stats.json"), "w") as f:
        json.dump(agg, f, indent=2)
    log.info("Aggregate stats saved")

    # 10. Plots
    log.info("Generating plots...")
    plot_accuracy_by_qtype(qtype_df, log)
    plot_behavioral_labels(tier_stats, log)
    plot_suppression_by_attribute(attr_df, log)
    plot_suppression_by_state(state_df, log)
    plot_position_distribution(pos_df, log)
    conf_df = plot_confidence_distribution(df, log)
    conf_df.to_csv(os.path.join(results_dir, "confidence_distribution.csv"), index=False)

    # 11. Sanity checks
    run_sanity_checks(df, tier_stats, config, log)

    log.info("=" * 60)
    log.info("MERGE & ANALYSIS COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
