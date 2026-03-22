#!/usr/bin/env python3
"""
Comprehensive Cultural Knowledge Evaluation: Sanskriti Dataset
Zero-shot evaluation of Qwen2-1.5B Base and Instruct models
Full 21,853 question analysis across all dimensions
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time
from typing import Tuple, List, Dict
import gc
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/data/user_data/anshulk/cultural-alignment-study")
DATA_FILE = BASE_DIR / "sanskriti_data" / "sanskriti_qa_master_hf.csv"
MODELS_DIR = BASE_DIR / "qwen_models"

# Output directory
OUTPUT_DIR = Path("/home/anshulk/cultural-alignment-study/outputs/sanskriti_test_knowledge")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "comprehensive_results.csv"
ANALYSIS_FILE = OUTPUT_DIR / "comprehensive_analysis.txt"

# Model paths
BASE_MODEL_PATH = MODELS_DIR / "Qwen2-1.5B"
INSTRUCT_MODEL_PATH = MODELS_DIR / "Qwen2-1.5B-Instruct"

# Settings
BATCH_SIZE = 128
TEMPERATURE = 0.0
MAX_TOKENS_BASE = 30
MAX_TOKENS_INSTRUCT = 50
GPU_MEMORY_FRACTION = 0.75  # 75% utilization

# Set GPU memory fraction
torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 1)

def load_dataset() -> pd.DataFrame:
    """Load complete Sanskriti dataset."""
    
    print("="*80)
    print("LOADING SANSKRITI DATASET")
    print("="*80)
    
    df = pd.read_csv(DATA_FILE)
    
    print(f"\nTotal questions: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    print(f"\nQuestion Type Distribution:")
    print(df['question_type'].value_counts())
    
    print(f"\nAttribute Distribution:")
    print(df['attribute'].value_counts())
    
    print(f"\nState Distribution:")
    print(df['state'].value_counts())
    
    return df


def load_model_and_tokenizer(model_path: Path, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with multi-GPU support."""
    
    print(f"\n{'='*80}")
    print(f"LOADING {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\nAvailable GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Using: {props.total_memory * GPU_MEMORY_FRACTION / 1024**3:.2f} GB (75%)")
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )
    
    tokenizer.padding_side = 'left'

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",  # Automatic multi-GPU distribution
        trust_remote_code=True
    )
    
    model.eval()
    
    print(f"\n{model_name} loaded successfully")
    print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    return model, tokenizer


def create_base_prompt(question: str, options: List[str]) -> str:
    """Create prompt for base model."""
    
    prompt = f"""Question: {question}

Options:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Answer: The correct option is ("""
    
    return prompt


def create_instruct_prompt(question: str, options: List[str], tokenizer: AutoTokenizer) -> str:
    """Create prompt for instruct model using chat template."""
    
    options_text = f"""A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
    
    messages = [
        {
            "role": "user", 
            "content": f"""Answer this multiple choice question. You MUST respond with ONLY a single letter: A, B, C, or D.

Question: {question}

{options_text}

Respond with ONLY the letter (A, B, C, or D), nothing else:"""
        }
    ]
    
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback
        return f"""User: Answer this multiple choice question. You MUST respond with ONLY a single letter: A, B, C, or D.

Question: {question}

{options_text}

Respond with ONLY the letter (A, B, C, or D), nothing else:
Assistant:"""


def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 30
) -> List[str]:
    """Generate responses in batches."""
    
    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    input_lengths = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            temperature=None,
            top_p=None
        )
    
    # Decode only the generated part
    responses = []
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output[input_lengths:], skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses


def extract_answer_letter(response: str) -> str:
    """Extract A/B/C/D from model response using multiple strategies."""
    
    response = response.strip().upper()
    
    if not response:
        return ""
    
    # Strategy 1: First character if it's A/B/C/D followed by space or punctuation
    if response and response[0] in ['A', 'B', 'C', 'D']:
        if len(response) == 1 or response[1] in [' ', '.', ')', ']', ',', '\n']:
            return response[0]
    
    # Strategy 2: Pattern matching with brackets/parens
    match = re.search(r'[(\[]?([ABCD])[)\].,\s]', response)
    if match:
        return match.group(1)
    
    # Strategy 3: "ANSWER IS" or "OPTION" patterns
    match = re.search(r'(?:ANSWER|OPTION)[\s:]*(?:IS)?[\s:]*([ABCD])', response)
    if match:
        return match.group(1)
    
    # Strategy 4: First A/B/C/D in first 10 characters
    for char in response[:10]:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    # Strategy 5: Any A/B/C/D in response
    match = re.search(r'([ABCD])', response)
    if match:
        return match.group(1)
    
    return ""


def check_correctness(model_response: str, gold_answer: str, options: List[str]) -> bool:
    """Check if model answer matches gold answer."""
    
    selected_letter = extract_answer_letter(model_response)
    
    if not selected_letter or selected_letter not in ['A', 'B', 'C', 'D']:
        return False
    
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    selected_option = options[letter_to_idx[selected_letter]]
    
    # Normalize and compare
    selected_normalized = selected_option.lower().strip()
    gold_normalized = gold_answer.lower().strip()
    
    return selected_normalized == gold_normalized


def test_model(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_type: str,
    prompt_fn,
    max_new_tokens: int,
    batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Test model on all questions with batched inference."""
    
    print(f"\n{'='*80}")
    print(f"TESTING {model_type.upper()} MODEL")
    print(f"Total questions: {len(df)}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"{'='*80}\n")
    
    all_responses = []
    all_letters = []
    all_correctness = []
    
    # Process in batches
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Processing {model_type} batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        # Create prompts for batch
        prompts = []
        for _, row in batch_df.iterrows():
            question = row['question']
            options = [row['option1'], row['option2'], row['option3'], row['option4']]
            
            if model_type == "base":
                prompt = prompt_fn(question, options)
            else:
                prompt = prompt_fn(question, options, tokenizer)
            
            prompts.append(prompt)
        
        # Generate responses
        responses = batch_generate(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens
        )
        
        # Process responses
        for i, (_, row) in enumerate(batch_df.iterrows()):
            response = responses[i]
            options = [row['option1'], row['option2'], row['option3'], row['option4']]
            gold_answer = row['answer']
            
            answer_letter = extract_answer_letter(response)
            is_correct = check_correctness(response, gold_answer, options)
            
            all_responses.append(response)
            all_letters.append(answer_letter)
            all_correctness.append(is_correct)
        
        # Periodic garbage collection
        if (batch_idx + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        # Progress update
        if (batch_idx + 1) % 20 == 0:
            current_acc = sum(all_correctness) / len(all_correctness)
            print(f"  Batch {batch_idx+1}/{num_batches} | Current accuracy: {current_acc:.2%}")
    
    # Calculate final accuracy
    accuracy = sum(all_correctness) / len(all_correctness)
    
    print(f"\n{'='*80}")
    print(f"{model_type.upper()} MODEL RESULTS:")
    print(f"Accuracy: {accuracy:.2%} ({sum(all_correctness)}/{len(all_correctness)})")
    print(f"{'='*80}\n")
    
    # Add columns to dataframe
    df[f'{model_type}_response'] = all_responses
    df[f'{model_type}_letter'] = all_letters
    df[f'{model_type}_correct'] = all_correctness
    
    return df


def comprehensive_analysis(df: pd.DataFrame) -> str:
    """Generate comprehensive analysis report."""
    
    lines = []
    
    lines.append("="*100)
    lines.append("COMPREHENSIVE CULTURAL KNOWLEDGE ANALYSIS")
    lines.append("SANSKRITI BENCHMARK: Complete 21,853 Question Evaluation")
    lines.append("="*100)
    
    # Overall Performance
    lines.append("\n" + "="*100)
    lines.append("1. OVERALL PERFORMANCE")
    lines.append("="*100)
    
    base_acc = df['base_correct'].mean()
    instruct_acc = df['instruct_correct'].mean()
    knowledge_gap = base_acc - instruct_acc
    
    lines.append(f"\nTotal Questions:           {len(df):,}")
    lines.append(f"Base Model Accuracy:       {base_acc:.4f} ({df['base_correct'].sum():,}/{len(df):,})")
    lines.append(f"Instruct Model Accuracy:   {instruct_acc:.4f} ({df['instruct_correct'].sum():,}/{len(df):,})")
    lines.append(f"Knowledge Gap:             {knowledge_gap:+.4f} ({knowledge_gap*100:+.2f}%)")
    
    # Response Quality
    base_valid = df['base_letter'].apply(lambda x: x in ['A', 'B', 'C', 'D']).sum()
    instruct_valid = df['instruct_letter'].apply(lambda x: x in ['A', 'B', 'C', 'D']).sum()
    
    lines.append(f"\nResponse Quality:")
    lines.append(f"  Base valid responses:     {base_valid:,}/{len(df):,} ({base_valid/len(df):.2%})")
    lines.append(f"  Instruct valid responses: {instruct_valid:,}/{len(df):,} ({instruct_valid/len(df):.2%})")
    
    # Suppression vs Enhancement
    suppression_mask = df['base_correct'] & ~df['instruct_correct']
    enhancement_mask = ~df['base_correct'] & df['instruct_correct']
    both_wrong = ~df['base_correct'] & ~df['instruct_correct']
    both_correct = df['base_correct'] & df['instruct_correct']
    
    lines.append(f"\nPattern Analysis:")
    lines.append(f"  Suppression (Base✓, Instruct✗):  {suppression_mask.sum():,} ({suppression_mask.sum()/len(df):.2%})")
    lines.append(f"  Enhancement (Base✗, Instruct✓):  {enhancement_mask.sum():,} ({enhancement_mask.sum()/len(df):.2%})")
    lines.append(f"  Both Correct:                     {both_correct.sum():,} ({both_correct.sum()/len(df):.2%})")
    lines.append(f"  Both Wrong:                       {both_wrong.sum():,} ({both_wrong.sum()/len(df):.2%})")
    lines.append(f"  Net Suppression Rate:             {(suppression_mask.sum() - enhancement_mask.sum())/len(df):.2%}")
    
    # Question Type Analysis
    lines.append("\n" + "="*100)
    lines.append("2. QUESTION TYPE ANALYSIS (4 Types)")
    lines.append("="*100)
    
    lines.append(f"\n{'Question Type':<30} {'Count':<8} {'Base Acc':<12} {'Instruct Acc':<12} {'Gap':<10} {'Suppression':<12}")
    lines.append("-"*100)
    
    for qtype in sorted(df['question_type'].unique()):
        qtype_df = df[df['question_type'] == qtype]
        count = len(qtype_df)
        base_acc = qtype_df['base_correct'].mean()
        inst_acc = qtype_df['instruct_correct'].mean()
        gap = base_acc - inst_acc
        supp = (qtype_df['base_correct'] & ~qtype_df['instruct_correct']).sum()
        
        lines.append(f"{qtype:<30} {count:<8} {base_acc:<12.4f} {inst_acc:<12.4f} {gap:+<10.4f} {supp:<12}")
    
    # Attribute Analysis
    lines.append("\n" + "="*100)
    lines.append("3. CULTURAL ATTRIBUTE ANALYSIS (16 Attributes)")
    lines.append("="*100)
    
    lines.append(f"\n{'Attribute':<30} {'Count':<8} {'Base Acc':<12} {'Instruct Acc':<12} {'Gap':<10} {'Suppression':<12}")
    lines.append("-"*100)
    
    attr_results = []
    for attr in sorted(df['attribute'].unique()):
        attr_df = df[df['attribute'] == attr]
        count = len(attr_df)
        base_acc = attr_df['base_correct'].mean()
        inst_acc = attr_df['instruct_correct'].mean()
        gap = base_acc - inst_acc
        supp = (attr_df['base_correct'] & ~attr_df['instruct_correct']).sum()
        
        attr_results.append((attr, count, base_acc, inst_acc, gap, supp))
    
    # Sort by gap (descending)
    attr_results.sort(key=lambda x: x[4], reverse=True)
    
    for attr, count, base_acc, inst_acc, gap, supp in attr_results:
        lines.append(f"{attr:<30} {count:<8} {base_acc:<12.4f} {inst_acc:<12.4f} {gap:+<10.4f} {supp:<12}")
    
    # State Analysis
    lines.append("\n" + "="*100)
    lines.append("4. GEOGRAPHIC ANALYSIS (All 36 Regions)")
    lines.append("="*100)
    
    lines.append(f"\n{'State/UT':<35} {'Count':<8} {'Base Acc':<12} {'Instruct Acc':<12} {'Gap':<10} {'Suppression':<12}")
    lines.append("-"*100)
    
    state_results = []
    for state in sorted(df['state'].unique()):
        state_df = df[df['state'] == state]
        count = len(state_df)
        base_acc = state_df['base_correct'].mean()
        inst_acc = state_df['instruct_correct'].mean()
        gap = base_acc - inst_acc
        supp = (state_df['base_correct'] & ~state_df['instruct_correct']).sum()
        
        state_results.append((state, count, base_acc, inst_acc, gap, supp))
    
    # Sort by gap (descending)
    state_results.sort(key=lambda x: x[4], reverse=True)
    
    for state, count, base_acc, inst_acc, gap, supp in state_results:
        lines.append(f"{state:<35} {count:<8} {base_acc:<12.4f} {inst_acc:<12.4f} {gap:+<10.4f} {supp:<12}")
    
    # Top Suppression Examples
    lines.append("\n" + "="*100)
    lines.append("5. TOP SUPPRESSION EXAMPLES")
    lines.append("="*100)
    
    suppression_df = df[suppression_mask].head(10)
    for idx, row in suppression_df.iterrows():
        lines.append(f"\nExample {idx}:")
        lines.append(f"  Question: {row['question'][:100]}...")
        lines.append(f"  State: {row['state']} | Attribute: {row['attribute']} | Type: {row['question_type']}")
        lines.append(f"  Gold Answer: {row['answer']}")
        lines.append(f"  Base Response: {row['base_response'][:80]}... (Letter: {row['base_letter']}) ✓")
        lines.append(f"  Instruct Response: {row['instruct_response'][:80]}... (Letter: {row['instruct_letter']}) ✗")
    
    # Cross-dimensional insights
    lines.append("\n" + "="*100)
    lines.append("6. CROSS-DIMENSIONAL INSIGHTS")
    lines.append("="*100)
    
    # Question Type × Attribute
    lines.append("\nWorst Performing Combinations (Question Type × Attribute):")
    worst_combos = []
    for qtype in df['question_type'].unique():
        for attr in df['attribute'].unique():
            combo_df = df[(df['question_type'] == qtype) & (df['attribute'] == attr)]
            if len(combo_df) >= 10:  # Minimum sample size
                gap = combo_df['base_correct'].mean() - combo_df['instruct_correct'].mean()
                worst_combos.append((qtype, attr, len(combo_df), gap))
    
    worst_combos.sort(key=lambda x: x[3], reverse=True)
    for qtype, attr, count, gap in worst_combos[:15]:
        lines.append(f"  {qtype} × {attr}: {count} questions, Gap: {gap:+.4f}")
    
    # State × Attribute
    lines.append("\nWorst Performing State-Attribute Combinations:")
    worst_state_combos = []
    for state in df['state'].unique():
        for attr in df['attribute'].unique():
            combo_df = df[(df['state'] == state) & (df['attribute'] == attr)]
            if len(combo_df) >= 5:  # Lower threshold for state combinations
                gap = combo_df['base_correct'].mean() - combo_df['instruct_correct'].mean()
                worst_state_combos.append((state, attr, len(combo_df), gap))
    
    worst_state_combos.sort(key=lambda x: x[3], reverse=True)
    for state, attr, count, gap in worst_state_combos[:20]:
        lines.append(f"  {state} × {attr}: {count} questions, Gap: {gap:+.4f}")
    
    lines.append("\n" + "="*100)
    lines.append("END OF COMPREHENSIVE ANALYSIS")
    lines.append("="*100)
    
    return "\n".join(lines)


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations."""
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Overall Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Overall', 'Valid Responses', 'Suppression', 'Enhancement']
    base_metrics = [
        df['base_correct'].mean(),
        (df['base_letter'].apply(lambda x: x in ['A', 'B', 'C', 'D'])).mean(),
        (df['base_correct'] & ~df['instruct_correct']).mean(),
        (~df['base_correct'] & df['instruct_correct']).mean()
    ]
    instruct_metrics = [
        df['instruct_correct'].mean(),
        (df['instruct_letter'].apply(lambda x: x in ['A', 'B', 'C', 'D'])).mean(),
        (df['base_correct'] & ~df['instruct_correct']).mean(),
        (~df['base_correct'] & df['instruct_correct']).mean()
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, base_metrics, width, label='Base', alpha=0.8)
    ax.bar(x + width/2, instruct_metrics, width, label='Instruct', alpha=0.8)
    
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Question Type Performance
    fig, ax = plt.subplots(figsize=(12, 6))
    qtypes = sorted(df['question_type'].unique())
    base_accs = [df[df['question_type']==qt]['base_correct'].mean() for qt in qtypes]
    inst_accs = [df[df['question_type']==qt]['instruct_correct'].mean() for qt in qtypes]
    
    x = np.arange(len(qtypes))
    width = 0.35
    
    ax.bar(x - width/2, base_accs, width, label='Base', alpha=0.8)
    ax.bar(x + width/2, inst_accs, width, label='Instruct', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance by Question Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'question_type_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Attribute Performance (sorted by gap)
    fig, ax = plt.subplots(figsize=(14, 8))
    attrs = sorted(df['attribute'].unique())
    attr_data = []
    for attr in attrs:
        attr_df = df[df['attribute'] == attr]
        base_acc = attr_df['base_correct'].mean()
        inst_acc = attr_df['instruct_correct'].mean()
        gap = base_acc - inst_acc
        attr_data.append((attr, base_acc, inst_acc, gap))
    
    attr_data.sort(key=lambda x: x[3], reverse=True)
    attrs_sorted = [x[0] for x in attr_data]
    base_sorted = [x[1] for x in attr_data]
    inst_sorted = [x[2] for x in attr_data]
    
    x = np.arange(len(attrs_sorted))
    width = 0.35
    
    ax.bar(x - width/2, base_sorted, width, label='Base', alpha=0.8)
    ax.bar(x + width/2, inst_sorted, width, label='Instruct', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance by Cultural Attribute (Sorted by Knowledge Gap)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attrs_sorted, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attribute_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. State Performance (top 20 by gap)
    states = df['state'].unique()
    state_data = []
    for state in states:
        state_df = df[df['state'] == state]
        base_acc = state_df['base_correct'].mean()
        inst_acc = state_df['instruct_correct'].mean()
        gap = base_acc - inst_acc
        count = len(state_df)
        state_data.append((state, base_acc, inst_acc, gap, count))
    
    state_data.sort(key=lambda x: x[3], reverse=True)
    top_states = state_data[:20]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    states_sorted = [x[0] for x in top_states]
    base_sorted = [x[1] for x in top_states]
    inst_sorted = [x[2] for x in top_states]
    
    x = np.arange(len(states_sorted))
    width = 0.35
    
    ax.bar(x - width/2, base_sorted, width, label='Base', alpha=0.8)
    ax.bar(x + width/2, inst_sorted, width, label='Instruct', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Top 20 States by Knowledge Gap', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(states_sorted, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'state_performance_top20.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Suppression Heatmap (Question Type × Attribute)
    fig, ax = plt.subplots(figsize=(16, 10))
    
    qtypes = sorted(df['question_type'].unique())
    attrs = sorted(df['attribute'].unique())
    
    heatmap_data = np.zeros((len(qtypes), len(attrs)))
    
    for i, qtype in enumerate(qtypes):
        for j, attr in enumerate(attrs):
            combo_df = df[(df['question_type'] == qtype) & (df['attribute'] == attr)]
            if len(combo_df) > 0:
                suppression_rate = (combo_df['base_correct'] & ~combo_df['instruct_correct']).mean()
                heatmap_data[i, j] = suppression_rate
            else:
                heatmap_data[i, j] = np.nan
    
    sns.heatmap(heatmap_data, xticklabels=attrs, yticklabels=qtypes, 
                annot=True, fmt='.3f', cmap='RdYlGn_r', center=0.1,
                cbar_kws={'label': 'Suppression Rate'}, ax=ax)
    
    ax.set_title('Suppression Rate Heatmap (Question Type × Attribute)', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'suppression_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Knowledge Gap Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # By Question Type
    qtype_gaps = []
    for qtype in sorted(df['question_type'].unique()):
        qtype_df = df[df['question_type'] == qtype]
        gap = qtype_df['base_correct'].mean() - qtype_df['instruct_correct'].mean()
        qtype_gaps.append(gap)
    
    axes[0, 0].bar(range(len(qtypes)), qtype_gaps, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xticks(range(len(qtypes)))
    axes[0, 0].set_xticklabels(qtypes, rotation=15, ha='right')
    axes[0, 0].set_ylabel('Knowledge Gap')
    axes[0, 0].set_title('Knowledge Gap by Question Type')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # By Attribute (top 10)
    attr_gaps = [(attr, df[df['attribute']==attr]['base_correct'].mean() - 
                  df[df['attribute']==attr]['instruct_correct'].mean()) 
                 for attr in df['attribute'].unique()]
    attr_gaps.sort(key=lambda x: x[1], reverse=True)
    top_attrs = attr_gaps[:10]
    
    axes[0, 1].barh([x[0] for x in top_attrs], [x[1] for x in top_attrs], alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Knowledge Gap')
    axes[0, 1].set_title('Top 10 Attributes by Knowledge Gap')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # By State (top 10)
    state_gaps = [(state, df[df['state']==state]['base_correct'].mean() - 
                   df[df['state']==state]['instruct_correct'].mean()) 
                  for state in df['state'].unique()]
    state_gaps.sort(key=lambda x: x[1], reverse=True)
    top_states_gap = state_gaps[:10]
    
    axes[1, 0].barh([x[0] for x in top_states_gap], [x[1] for x in top_states_gap], alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Knowledge Gap')
    axes[1, 0].set_title('Top 10 States by Knowledge Gap')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Pattern distribution
    patterns = ['Both Correct', 'Suppression\n(Base✓, Inst✗)', 
                'Enhancement\n(Base✗, Inst✓)', 'Both Wrong']
    pattern_counts = [
        (df['base_correct'] & df['instruct_correct']).sum(),
        (df['base_correct'] & ~df['instruct_correct']).sum(),
        (~df['base_correct'] & df['instruct_correct']).sum(),
        (~df['base_correct'] & ~df['instruct_correct']).sum()
    ]
    
    colors = ['green', 'red', 'blue', 'gray']
    axes[1, 1].bar(patterns, pattern_counts, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Pattern Distribution')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'knowledge_gap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ All visualizations saved")


def save_detailed_breakdowns(df: pd.DataFrame, output_dir: Path):
    """Save detailed CSV breakdowns for each dimension."""
    
    print("\n" + "="*80)
    print("SAVING DETAILED BREAKDOWNS")
    print("="*80 + "\n")
    
    # Question Type breakdown
    qtype_breakdown = []
    for qtype in sorted(df['question_type'].unique()):
        qtype_df = df[df['question_type'] == qtype]
        qtype_breakdown.append({
            'question_type': qtype,
            'count': len(qtype_df),
            'base_accuracy': qtype_df['base_correct'].mean(),
            'instruct_accuracy': qtype_df['instruct_correct'].mean(),
            'knowledge_gap': qtype_df['base_correct'].mean() - qtype_df['instruct_correct'].mean(),
            'suppression_count': (qtype_df['base_correct'] & ~qtype_df['instruct_correct']).sum(),
            'enhancement_count': (~qtype_df['base_correct'] & qtype_df['instruct_correct']).sum(),
            'base_valid_responses': (qtype_df['base_letter'].apply(lambda x: x in ['A','B','C','D'])).sum(),
            'instruct_valid_responses': (qtype_df['instruct_letter'].apply(lambda x: x in ['A','B','C','D'])).sum()
        })
    
    pd.DataFrame(qtype_breakdown).to_csv(
        output_dir / 'breakdown_question_type.csv', 
        index=False
    )
    print("✓ Question type breakdown saved")
    
    # Attribute breakdown
    attr_breakdown = []
    for attr in sorted(df['attribute'].unique()):
        attr_df = df[df['attribute'] == attr]
        attr_breakdown.append({
            'attribute': attr,
            'count': len(attr_df),
            'base_accuracy': attr_df['base_correct'].mean(),
            'instruct_accuracy': attr_df['instruct_correct'].mean(),
            'knowledge_gap': attr_df['base_correct'].mean() - attr_df['instruct_correct'].mean(),
            'suppression_count': (attr_df['base_correct'] & ~attr_df['instruct_correct']).sum(),
            'enhancement_count': (~attr_df['base_correct'] & attr_df['instruct_correct']).sum()
        })
    
    pd.DataFrame(attr_breakdown).to_csv(
        output_dir / 'breakdown_attribute.csv', 
        index=False
    )
    print("✓ Attribute breakdown saved")
    
    # State breakdown
    state_breakdown = []
    for state in sorted(df['state'].unique()):
        state_df = df[df['state'] == state]
        state_breakdown.append({
            'state': state,
            'count': len(state_df),
            'base_accuracy': state_df['base_correct'].mean(),
            'instruct_accuracy': state_df['instruct_correct'].mean(),
            'knowledge_gap': state_df['base_correct'].mean() - state_df['instruct_correct'].mean(),
            'suppression_count': (state_df['base_correct'] & ~state_df['instruct_correct']).sum(),
            'enhancement_count': (~state_df['base_correct'] & state_df['instruct_correct']).sum()
        })
    
    pd.DataFrame(state_breakdown).to_csv(
        output_dir / 'breakdown_state.csv', 
        index=False
    )
    print("✓ State breakdown saved")
    
    # Cross-dimensional: Question Type × Attribute
    cross_qtype_attr = []
    for qtype in df['question_type'].unique():
        for attr in df['attribute'].unique():
            combo_df = df[(df['question_type'] == qtype) & (df['attribute'] == attr)]
            if len(combo_df) > 0:
                cross_qtype_attr.append({
                    'question_type': qtype,
                    'attribute': attr,
                    'count': len(combo_df),
                    'base_accuracy': combo_df['base_correct'].mean(),
                    'instruct_accuracy': combo_df['instruct_correct'].mean(),
                    'knowledge_gap': combo_df['base_correct'].mean() - combo_df['instruct_correct'].mean(),
                    'suppression_count': (combo_df['base_correct'] & ~combo_df['instruct_correct']).sum()
                })
    
    pd.DataFrame(cross_qtype_attr).to_csv(
        output_dir / 'breakdown_qtype_x_attribute.csv', 
        index=False
    )
    print("✓ Question Type × Attribute breakdown saved")
    
    # Cross-dimensional: State × Attribute
    cross_state_attr = []
    for state in df['state'].unique():
        for attr in df['attribute'].unique():
            combo_df = df[(df['state'] == state) & (df['attribute'] == attr)]
            if len(combo_df) > 0:
                cross_state_attr.append({
                    'state': state,
                    'attribute': attr,
                    'count': len(combo_df),
                    'base_accuracy': combo_df['base_correct'].mean(),
                    'instruct_accuracy': combo_df['instruct_correct'].mean(),
                    'knowledge_gap': combo_df['base_correct'].mean() - combo_df['instruct_correct'].mean(),
                    'suppression_count': (combo_df['base_correct'] & ~combo_df['instruct_correct']).sum()
                })
    
    pd.DataFrame(cross_state_attr).to_csv(
        output_dir / 'breakdown_state_x_attribute.csv', 
        index=False
    )
    print("✓ State × Attribute breakdown saved")
    
    # Suppression examples
    suppression_df = df[df['base_correct'] & ~df['instruct_correct']]
    suppression_df[['question', 'answer', 'attribute', 'state', 'question_type',
                    'base_response', 'base_letter', 'instruct_response', 'instruct_letter']].to_csv(
        output_dir / 'suppression_examples.csv',
        index=False
    )
    print(f"✓ Suppression examples saved ({len(suppression_df)} cases)")
    
    # Enhancement examples
    enhancement_df = df[~df['base_correct'] & df['instruct_correct']]
    enhancement_df[['question', 'answer', 'attribute', 'state', 'question_type',
                    'base_response', 'base_letter', 'instruct_response', 'instruct_letter']].to_csv(
        output_dir / 'enhancement_examples.csv',
        index=False
    )
    print(f"✓ Enhancement examples saved ({len(enhancement_df)} cases)")


def main():
    """Main execution pipeline."""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE CULTURAL KNOWLEDGE EVALUATION")
    print("SANSKRITI BENCHMARK: 21,853 Questions")
    print("Models: Qwen2-1.5B Base vs Instruct")
    print("="*100)
    
    overall_start = time.time()
    
    # Load dataset
    df = load_dataset()
    
    # Test Base Model
    print("\n" + "="*100)
    print("PHASE 1: BASE MODEL EVALUATION")
    print("="*100)
    
    base_model, base_tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH, "Qwen2-1.5B Base")
    
    df = test_model(
        df,
        base_model,
        base_tokenizer,
        "base",
        create_base_prompt,
        MAX_TOKENS_BASE,
        BATCH_SIZE
    )
    
    # Clean up
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✓ Base model evaluation complete, memory cleared")
    
    # Test Instruct Model
    print("\n" + "="*100)
    print("PHASE 2: INSTRUCT MODEL EVALUATION")
    print("="*100)
    
    instruct_model, instruct_tokenizer = load_model_and_tokenizer(
        INSTRUCT_MODEL_PATH,
        "Qwen2-1.5B Instruct"
    )
    
    df = test_model(
        df,
        instruct_model,
        instruct_tokenizer,
        "instruct",
        create_instruct_prompt,
        MAX_TOKENS_INSTRUCT,
        BATCH_SIZE
    )
    
    # Clean up
    del instruct_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✓ Instruct model evaluation complete, memory cleared")
    
    # Save complete results
    print("\n" + "="*100)
    print("PHASE 3: SAVING RESULTS")
    print("="*100)
    
    print(f"\nSaving complete results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("✓ Complete results CSV saved")
    
    # Generate comprehensive analysis
    print("\n" + "="*100)
    print("PHASE 4: COMPREHENSIVE ANALYSIS")
    print("="*100)
    
    analysis_text = comprehensive_analysis(df)
    
    print(f"\nSaving analysis to {ANALYSIS_FILE}...")
    with open(ANALYSIS_FILE, 'w', encoding='utf-8') as f:
        f.write(analysis_text)
    print("✓ Analysis report saved")
    
    # Print analysis to console
    print("\n" + analysis_text)
    
    # Create visualizations
    print("\n" + "="*100)
    print("PHASE 5: GENERATING VISUALIZATIONS")
    print("="*100)
    
    create_visualizations(df, OUTPUT_DIR)
    
    # Save detailed breakdowns
    print("\n" + "="*100)
    print("PHASE 6: DETAILED BREAKDOWNS")
    print("="*100)
    
    save_detailed_breakdowns(df, OUTPUT_DIR)
    
    # Summary
    overall_time = time.time() - overall_start
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100)
    
    print(f"\nTotal execution time: {overall_time/60:.1f} minutes ({overall_time/3600:.2f} hours)")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. comprehensive_results.csv - Complete predictions for all 21,853 questions")
    print("  2. comprehensive_analysis.txt - Detailed analysis report")
    print("  3. breakdown_*.csv - Dimensional breakdowns (5 files)")
    print("  4. suppression_examples.csv - Cases where base correct but instruct wrong")
    print("  5. enhancement_examples.csv - Cases where base wrong but instruct correct")
    print("  6. *.png - Visualizations (6 charts)")
    
    print("\n" + "="*100)
    print("SUCCESS: Comprehensive evaluation completed")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()