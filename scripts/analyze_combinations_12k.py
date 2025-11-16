#!/usr/bin/env python3
"""
Create 12K Targeted Dataset for Sentence Generation
Analyzes all state×attribute×question_type combinations and selects balanced samples
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# File paths
INPUT_FILE = "/home/anshulk/cultural-alignment-study/outputs/sanskriti_test_knowledge/comprehensive_results.csv"
OUTPUT_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data")
ANALYSIS_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/sanskriti_test_knowledge")

OUTPUT_FILE = OUTPUT_DIR / "sanskriti_12k_targeted.csv"
ANALYSIS_FILE = ANALYSIS_DIR / "sanskriti_12k_analysis.txt"
COMBINATION_STATS_FILE = ANALYSIS_DIR / "combination_statistics.csv"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Target sizes
TARGET_SUPPRESSION = 4000
TARGET_ENHANCEMENT = 4000
TARGET_CONTROL = 4000
TOTAL_TARGET = 12000


def load_data():
    """Load the comprehensive results CSV."""
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} questions")
    print(f"\nColumns: {list(df.columns)}")
    
    # Verify required columns exist
    required_cols = ['state', 'attribute', 'question_type', 'base_correct', 'instruct_correct']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def analyze_combinations(df):
    """Analyze all state×attribute×question_type combinations."""
    print("\n" + "="*80)
    print("ANALYZING ALL COMBINATIONS")
    print("="*80)
    
    combinations = []
    
    # Group by all three dimensions
    grouped = df.groupby(['state', 'attribute', 'question_type'])
    
    print(f"\nFound {len(grouped)} unique combinations")
    
    for (state, attribute, qtype), group in grouped:
        n = len(group)
        
        # Calculate accuracies
        base_correct = group['base_correct'].sum()
        instruct_correct = group['instruct_correct'].sum()
        
        base_acc = base_correct / n if n > 0 else 0
        instruct_acc = instruct_correct / n if n > 0 else 0
        gap = base_acc - instruct_acc
        
        # Count suppression and enhancement
        suppression_count = ((group['base_correct'] == True) & 
                            (group['instruct_correct'] == False)).sum()
        enhancement_count = ((group['base_correct'] == False) & 
                            (group['instruct_correct'] == True)).sum()
        
        both_correct = ((group['base_correct'] == True) & 
                       (group['instruct_correct'] == True)).sum()
        both_wrong = ((group['base_correct'] == False) & 
                     (group['instruct_correct'] == False)).sum()
        
        combinations.append({
            'state': state,
            'attribute': attribute,
            'question_type': qtype,
            'n_questions': n,
            'base_accuracy': base_acc,
            'instruct_accuracy': instruct_acc,
            'gap': gap,
            'suppression_count': suppression_count,
            'enhancement_count': enhancement_count,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'suppression_rate': suppression_count / n if n > 0 else 0,
            'enhancement_rate': enhancement_count / n if n > 0 else 0
        })
    
    combo_df = pd.DataFrame(combinations)
    
    # Save combination statistics
    combo_df.to_csv(COMBINATION_STATS_FILE, index=False)
    print(f"\nSaved combination statistics to {COMBINATION_STATS_FILE}")
    
    return combo_df


def select_suppression_questions(df, combo_df, target=4000):
    """Select questions from top suppression combinations."""
    print(f"\n{'='*80}")
    print(f"SELECTING SUPPRESSION GROUP (Target: {target})")
    print(f"{'='*80}")
    
    # Sort by gap (highest positive gaps first)
    suppression_combos = combo_df[combo_df['gap'] > 0].sort_values('gap', ascending=False)
    
    print(f"\nFound {len(suppression_combos)} combinations with positive gaps")
    print(f"\nTop 20 Suppression Combinations:")
    print(suppression_combos.head(20)[['state', 'attribute', 'question_type', 
                                        'n_questions', 'gap', 'suppression_count']])
    
    selected_questions = []
    selected_count = 0
    
    # First, get all actual suppression cases
    suppression_mask = (df['base_correct'] == True) & (df['instruct_correct'] == False)
    actual_suppressions = df[suppression_mask]
    selected_questions.append(actual_suppressions)
    selected_count += len(actual_suppressions)
    
    print(f"\nSelected {len(actual_suppressions)} actual suppression cases")
    
    # If we need more, add from top combinations
    if selected_count < target:
        needed = target - selected_count
        print(f"Need {needed} more questions from high-gap combinations")
        
        for idx, row in suppression_combos.iterrows():
            if selected_count >= target:
                break
            
            # Get questions from this combination that aren't already selected
            mask = ((df['state'] == row['state']) & 
                   (df['attribute'] == row['attribute']) & 
                   (df['question_type'] == row['question_type']))
            
            combo_questions = df[mask]
            
            # Exclude already selected suppressions
            combo_questions = combo_questions[~combo_questions.index.isin(actual_suppressions.index)]
            
            # Take what we need
            take_count = min(len(combo_questions), target - selected_count)
            if take_count > 0:
                selected_questions.append(combo_questions.head(take_count))
                selected_count += take_count
    
    result = pd.concat(selected_questions, ignore_index=True)
    print(f"\nFinal suppression group size: {len(result)}")
    
    return result


def select_enhancement_questions(df, combo_df, target=4000):
    """Select questions from top enhancement combinations."""
    print(f"\n{'='*80}")
    print(f"SELECTING ENHANCEMENT GROUP (Target: {target})")
    print(f"{'='*80}")
    
    # Sort by gap (highest negative gaps first)
    enhancement_combos = combo_df[combo_df['gap'] < 0].sort_values('gap', ascending=True)
    
    print(f"\nFound {len(enhancement_combos)} combinations with negative gaps")
    print(f"\nTop 20 Enhancement Combinations:")
    print(enhancement_combos.head(20)[['state', 'attribute', 'question_type', 
                                        'n_questions', 'gap', 'enhancement_count']])
    
    selected_questions = []
    selected_count = 0
    
    # First, get all actual enhancement cases
    enhancement_mask = (df['base_correct'] == False) & (df['instruct_correct'] == True)
    actual_enhancements = df[enhancement_mask]
    selected_questions.append(actual_enhancements)
    selected_count += len(actual_enhancements)
    
    print(f"\nSelected {len(actual_enhancements)} actual enhancement cases")
    
    # If we need more, add from top combinations
    if selected_count < target:
        needed = target - selected_count
        print(f"Need {needed} more questions from enhancement combinations")
        
        for idx, row in enhancement_combos.iterrows():
            if selected_count >= target:
                break
            
            # Get questions from this combination
            mask = ((df['state'] == row['state']) & 
                   (df['attribute'] == row['attribute']) & 
                   (df['question_type'] == row['question_type']))
            
            combo_questions = df[mask]
            
            # Exclude already selected enhancements
            combo_questions = combo_questions[~combo_questions.index.isin(actual_enhancements.index)]
            
            # Take what we need
            take_count = min(len(combo_questions), target - selected_count)
            if take_count > 0:
                selected_questions.append(combo_questions.head(take_count))
                selected_count += take_count
    
    result = pd.concat(selected_questions, ignore_index=True)
    print(f"\nFinal enhancement group size: {len(result)}")
    
    return result


def select_control_questions(df, combo_df, target=4000, suppression_df=None, enhancement_df=None):
    """Select questions from near-zero gap combinations."""
    print(f"\n{'='*80}")
    print(f"SELECTING CONTROL GROUP (Target: {target})")
    print(f"{'='*80}")
    
    # Find combinations with gaps between -0.02 and +0.02
    neutral_combos = combo_df[(combo_df['gap'] >= -0.02) & (combo_df['gap'] <= 0.02)]
    
    print(f"\nFound {len(neutral_combos)} near-zero gap combinations")
    print(f"\nSample Neutral Combinations:")
    print(neutral_combos.head(20)[['state', 'attribute', 'question_type', 
                                    'n_questions', 'gap']])
    
    # Get all questions from neutral combinations
    neutral_questions = []
    
    for idx, row in neutral_combos.iterrows():
        mask = ((df['state'] == row['state']) & 
               (df['attribute'] == row['attribute']) & 
               (df['question_type'] == row['question_type']))
        
        neutral_questions.append(df[mask])
    
    if neutral_questions:
        all_neutral = pd.concat(neutral_questions, ignore_index=True)
        
        # Exclude questions already in suppression or enhancement groups
        if suppression_df is not None:
            all_neutral = all_neutral[~all_neutral.index.isin(suppression_df.index)]
        if enhancement_df is not None:
            all_neutral = all_neutral[~all_neutral.index.isin(enhancement_df.index)]
        
        print(f"\nAvailable neutral questions: {len(all_neutral)}")
        
        # Random sample
        if len(all_neutral) >= target:
            result = all_neutral.sample(n=target, random_state=42)
        else:
            print(f"WARNING: Only {len(all_neutral)} neutral questions available, need {target}")
            result = all_neutral
    else:
        print("WARNING: No neutral combinations found")
        result = pd.DataFrame()
    
    print(f"\nFinal control group size: {len(result)}")
    
    return result


def create_analysis_report(df, suppression_df, enhancement_df, control_df, combo_df):
    """Create detailed analysis report of the 12K dataset."""
    
    lines = []
    
    lines.append("="*80)
    lines.append("SANSKRITI 12K TARGETED DATASET - ANALYSIS REPORT")
    lines.append("="*80)
    lines.append("")
    
    # Overall statistics
    lines.append("DATASET COMPOSITION:")
    lines.append(f"  Total Questions:        {len(df)}")
    lines.append(f"  Suppression Group:      {len(suppression_df)} ({len(suppression_df)/len(df)*100:.1f}%)")
    lines.append(f"  Enhancement Group:      {len(enhancement_df)} ({len(enhancement_df)/len(df)*100:.1f}%)")
    lines.append(f"  Control Group:          {len(control_df)} ({len(control_df)/len(df)*100:.1f}%)")
    lines.append("")
    
    # Performance metrics
    lines.append("OVERALL PERFORMANCE:")
    base_acc = df['base_correct'].mean()
    instruct_acc = df['instruct_correct'].mean()
    gap = base_acc - instruct_acc
    
    lines.append(f"  Base Model Accuracy:       {base_acc:.4f} ({df['base_correct'].sum()}/{len(df)})")
    lines.append(f"  Instruct Model Accuracy:   {instruct_acc:.4f} ({df['instruct_correct'].sum()}/{len(df)})")
    lines.append(f"  Knowledge Gap:             {gap:+.4f} ({gap*100:+.2f}%)")
    lines.append("")
    
    # Group-specific performance
    lines.append("PERFORMANCE BY GROUP:")
    lines.append("")
    
    for name, group_df in [("Suppression", suppression_df), 
                           ("Enhancement", enhancement_df), 
                           ("Control", control_df)]:
        if len(group_df) > 0:
            base_acc = group_df['base_correct'].mean()
            instruct_acc = group_df['instruct_correct'].mean()
            gap = base_acc - instruct_acc
            
            lines.append(f"{name} Group ({len(group_df)} questions):")
            lines.append(f"  Base Accuracy:     {base_acc:.4f}")
            lines.append(f"  Instruct Accuracy: {instruct_acc:.4f}")
            lines.append(f"  Gap:               {gap:+.4f}")
            lines.append("")
    
    # Distribution by dimensions
    lines.append("="*80)
    lines.append("DISTRIBUTION BY DIMENSIONS")
    lines.append("="*80)
    lines.append("")
    
    # Question types
    lines.append("QUESTION TYPE DISTRIBUTION:")
    qtype_dist = df['question_type'].value_counts().sort_index()
    for qtype, count in qtype_dist.items():
        lines.append(f"  {qtype:<30} {count:>5} ({count/len(df)*100:>5.1f}%)")
    lines.append("")
    
    # Attributes
    lines.append("ATTRIBUTE DISTRIBUTION (Top 16):")
    attr_dist = df['attribute'].value_counts().head(16)
    for attr, count in attr_dist.items():
        lines.append(f"  {attr:<30} {count:>5} ({count/len(df)*100:>5.1f}%)")
    lines.append("")
    
    # States
    lines.append("STATE DISTRIBUTION (Top 20):")
    state_dist = df['state'].value_counts().head(20)
    for state, count in state_dist.items():
        lines.append(f"  {state:<35} {count:>5} ({count/len(df)*100:>5.1f}%)")
    lines.append("")
    
    # Top combinations in each group
    lines.append("="*80)
    lines.append("TOP COMBINATIONS IN EACH GROUP")
    lines.append("="*80)
    lines.append("")
    
    # Suppression group combinations
    lines.append("SUPPRESSION GROUP - Top 10 Combinations:")
    supp_combos = suppression_df.groupby(['state', 'attribute', 'question_type']).size()
    supp_combos = supp_combos.sort_values(ascending=False).head(10)
    for (state, attr, qtype), count in supp_combos.items():
        lines.append(f"  {state} × {attr} × {qtype}: {count} questions")
    lines.append("")
    
    # Enhancement group combinations
    lines.append("ENHANCEMENT GROUP - Top 10 Combinations:")
    enh_combos = enhancement_df.groupby(['state', 'attribute', 'question_type']).size()
    enh_combos = enh_combos.sort_values(ascending=False).head(10)
    for (state, attr, qtype), count in enh_combos.items():
        lines.append(f"  {state} × {attr} × {qtype}: {count} questions")
    lines.append("")
    
    # Control group combinations
    lines.append("CONTROL GROUP - Top 10 Combinations:")
    ctrl_combos = control_df.groupby(['state', 'attribute', 'question_type']).size()
    ctrl_combos = ctrl_combos.sort_values(ascending=False).head(10)
    for (state, attr, qtype), count in ctrl_combos.items():
        lines.append(f"  {state} × {attr} × {qtype}: {count} questions")
    lines.append("")
    
    # Highest gap combinations
    lines.append("="*80)
    lines.append("HIGHEST GAP COMBINATIONS IN DATASET")
    lines.append("="*80)
    lines.append("")
    
    lines.append("Top 15 Suppression Hotspots (Highest Positive Gaps):")
    lines.append(f"{'State':<25} {'Attribute':<25} {'Q-Type':<20} {'N':<5} {'Gap':<8}")
    lines.append("-"*80)
    
    top_supp = combo_df[combo_df['gap'] > 0].nlargest(15, 'gap')
    for idx, row in top_supp.iterrows():
        lines.append(f"{row['state']:<25} {row['attribute']:<25} {row['question_type']:<20} "
                    f"{row['n_questions']:<5} {row['gap']:>+.4f}")
    lines.append("")
    
    lines.append("Top 15 Enhancement Hotspots (Highest Negative Gaps):")
    lines.append(f"{'State':<25} {'Attribute':<25} {'Q-Type':<20} {'N':<5} {'Gap':<8}")
    lines.append("-"*80)
    
    top_enh = combo_df[combo_df['gap'] < 0].nsmallest(15, 'gap')
    for idx, row in top_enh.iterrows():
        lines.append(f"{row['state']:<25} {row['attribute']:<25} {row['question_type']:<20} "
                    f"{row['n_questions']:<5} {row['gap']:>+.4f}")
    lines.append("")
    
    # Sample questions
    lines.append("="*80)
    lines.append("SAMPLE QUESTIONS FROM EACH GROUP")
    lines.append("="*80)
    lines.append("")
    
    for name, group_df in [("Suppression", suppression_df.head(3)), 
                           ("Enhancement", enhancement_df.head(3)), 
                           ("Control", control_df.head(3))]:
        lines.append(f"{name} Group Samples:")
        for idx, row in group_df.iterrows():
            lines.append(f"\nQuestion: {row['question'][:100]}...")
            lines.append(f"  State: {row['state']} | Attribute: {row['attribute']} | Type: {row['question_type']}")
            lines.append(f"  Answer: {row['answer']}")
            lines.append(f"  Base: {'✓' if row['base_correct'] else '✗'} | Instruct: {'✓' if row['instruct_correct'] else '✗'}")
        lines.append("")
    
    lines.append("="*80)
    lines.append("END OF ANALYSIS")
    lines.append("="*80)
    
    return "\n".join(lines)


def main():
    """Main execution."""
    
    print("="*80)
    print("CREATING 12K TARGETED DATASET FOR SENTENCE GENERATION")
    print("="*80)
    print(f"\nInput: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Analysis: {ANALYSIS_FILE}")
    
    # Load data
    df = load_data()
    
    # Analyze all combinations
    combo_df = analyze_combinations(df)
    
    # Select questions for each group
    suppression_df = select_suppression_questions(df, combo_df, TARGET_SUPPRESSION)
    enhancement_df = select_enhancement_questions(df, combo_df, TARGET_ENHANCEMENT)
    control_df = select_control_questions(df, combo_df, TARGET_CONTROL, 
                                         suppression_df, enhancement_df)
    
    # Add group labels
    suppression_df = suppression_df.copy()
    enhancement_df = enhancement_df.copy()
    control_df = control_df.copy()
    
    suppression_df['group_type'] = 'suppression'
    enhancement_df['group_type'] = 'enhancement'
    control_df['group_type'] = 'control'
    
    # Combine all groups
    final_df = pd.concat([suppression_df, enhancement_df, control_df], ignore_index=True)
    
    print(f"\n{'='*80}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions: {len(final_df)}")
    print(f"  Suppression: {len(suppression_df)}")
    print(f"  Enhancement: {len(enhancement_df)}")
    print(f"  Control:     {len(control_df)}")
    
    # Save final dataset
    print(f"\nSaving final dataset to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Dataset saved successfully!")
    
    # Create and save analysis report
    print(f"\nGenerating analysis report...")
    report = create_analysis_report(final_df, suppression_df, enhancement_df, 
                                   control_df, combo_df)
    
    with open(ANALYSIS_FILE, 'w') as f:
        f.write(report)
    
    print(f"Analysis report saved to {ANALYSIS_FILE}")
    
    print(f"\n{'='*80}")
    print("DATASET CREATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nFiles created:")
    print(f"  1. Dataset:         {OUTPUT_FILE}")
    print(f"  2. Analysis Report: {ANALYSIS_FILE}")
    print(f"  3. Combo Stats:     {COMBINATION_STATS_FILE}")
    print(f"\nReady for sentence generation (3 sentences per question = {len(final_df) * 3:,} total sentences)")


if __name__ == "__main__":
    main()