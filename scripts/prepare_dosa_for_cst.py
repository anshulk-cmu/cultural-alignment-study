#!/usr/bin/env python3
"""
prepare_dosa_for_cst.py

Download DOSA artifacts from GitHub, convert clue-based descriptions into
natural language statements using Llama-3.2-3B-Instruct, and store for
Causal Sufficiency Testing (CST).

Output: /data/user_data/anshulk/cultural-alignment-study/data/dosa_cst_samples.csv
"""

import os
import sys
import pandas as pd
import requests
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

# Configuration
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/microsoft/DOSA/main/data"
OUTPUT_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/data")
OUTPUT_FILE = OUTPUT_DIR / "dosa_cst_samples.csv"
MODEL_PATH = "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"

# 18 states from DOSA (excluding Madhya Pradesh which wasn't collected)
STATES = [
    "andhra_pradesh", "assam", "bihar", "chhattisgarh", "delhi",
    "gujarat", "haryana", "jharkhand", "karnataka", "kerala",
    "maharashtra", "odisha", "punjab", "rajasthan", "tamil_nadu",
    "telangana", "uttar_pradesh", "west_bengal"
]

ARTIFACT_TYPES = ["original_artifacts", "expanded_artifacts"]


def download_csv_from_github(state, artifact_type):
    """Download CSV file from DOSA GitHub repository."""
    url = f"{GITHUB_RAW_BASE}/{state}/{artifact_type}.csv"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save temporarily
        temp_file = f"/tmp/dosa_{state}_{artifact_type}.csv"
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        df = pd.read_csv(temp_file)
        os.remove(temp_file)
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"   Warning: Could not download {state}/{artifact_type}.csv: {e}")
        return None


def load_llm():
    """Load Llama-3.2-3B-Instruct model for text generation."""
    print(f"\nLoading model from {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    print(f"   Model loaded successfully on {model.device}")
    return tokenizer, model


def convert_clues_to_statement(artifact_name, clues, state, tokenizer, model):
    """Convert clue-based description into natural language statement."""
    
    # Prepare prompt
    system_prompt = """You are a helpful assistant that converts bullet-point cultural clues into natural, flowing paragraphs. Your task is to create a single coherent paragraph (2-4 sentences) that naturally describes the cultural artifact based on the given clues."""
    
    # Clean and format clues
    clue_list = []
    for col in clues.index:
        if col.startswith('clue') or col == 'clues':
            clue = str(clues[col])
            if clue and clue.lower() not in ['nan', 'none', '']:
                clue_list.append(clue.strip())
    
    if not clue_list:
        return None
    
    clues_text = "\n".join([f"- {clue}" for clue in clue_list])
    
    user_prompt = f"""State: {state.replace('_', ' ').title()}
Artifact: {artifact_name}

Clues:
{clues_text}

Convert these clues into a natural paragraph (2-4 sentences) that describes this cultural artifact. Write in a flowing, descriptive style. Do not include the artifact name in your description."""
    
    # Format for Llama-3.2 chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract response
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "assistant" in generated:
        response = generated.split("assistant")[-1].strip()
    else:
        response = generated[len(input_text):].strip()
    
    return response


def process_all_artifacts():
    """Main processing loop."""
    
    print("="*80)
    print("DOSA CST SAMPLE PREPARATION")
    print("="*80)
    
    # Load model
    tokenizer, model = load_llm()
    
    # Collect all artifacts
    all_samples = []
    total_artifacts = 0
    
    print("\nDownloading and processing DOSA artifacts...")
    
    for state in tqdm(STATES, desc="States"):
        for artifact_type in ARTIFACT_TYPES:
            df = download_csv_from_github(state, artifact_type)
            
            if df is None:
                continue
            
            # Process each artifact
            for idx, row in df.iterrows():
                # Get artifact name (first column is usually the artifact name)
                artifact_name = row.iloc[0] if len(row) > 0 else None
                
                if not artifact_name or pd.isna(artifact_name):
                    continue
                
                # Convert clues to statement
                statement = convert_clues_to_statement(
                    artifact_name, row, state, tokenizer, model
                )
                
                if statement:
                    all_samples.append({
                        'state': state,
                        'artifact_name': artifact_name,
                        'artifact_type': artifact_type.replace('_artifacts', ''),
                        'converted_text': statement,
                        'original_clues': ' | '.join([
                            str(row[col]) for col in row.index 
                            if (col.startswith('clue') or col == 'clues') 
                            and not pd.isna(row[col])
                        ]),
                        'category': 'cultural_validation',
                        'source': 'dosa_github'
                    })
                    total_artifacts += 1
    
    print(f"\n   Processed {total_artifacts} artifacts from {len(STATES)} states")
    
    # Save to CSV
    df_output = pd.DataFrame(all_samples)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n   Saved to: {OUTPUT_FILE}")
    print(f"   Total samples: {len(df_output)}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total artifacts: {len(df_output)}")
    print(f"\nBy state:")
    print(df_output['state'].value_counts().to_string())
    print(f"\nBy type:")
    print(df_output['artifact_type'].value_counts().to_string())
    
    # Create summary JSON
    summary = {
        'total_samples': len(df_output),
        'by_state': df_output['state'].value_counts().to_dict(),
        'by_type': df_output['artifact_type'].value_counts().to_dict(),
        'output_file': str(OUTPUT_FILE)
    }
    
    summary_file = OUTPUT_DIR / "dosa_cst_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n   Summary saved to: {summary_file}")
    
    return df_output


if __name__ == "__main__":
    try:
        df = process_all_artifacts()
        print("\n" + "="*80)
        print("SUCCESS: DOSA CST samples prepared!")
        print("="*80)
        print(f"\nNext steps:")
        print(f"1. Use {OUTPUT_FILE} for CST validation")
        print(f"2. Run samples through Qwen models to extract activations")
        print(f"3. Perform feature ablation and measure KL-divergence")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
