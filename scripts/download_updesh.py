"""
Download and process Updesh dataset from Hugging Face
Extracts 15k English + 15k Hindi samples (30k total)
Only extracts the assistant's final response from each conversation
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from configs.config import DATA_ROOT

def extract_assistant_response(messages):
    """
    Extract only the assistant's response (last message with role='assistant')
    from the messages list
    """
    # Loop through messages in reverse to find the last assistant message
    for message in reversed(messages):
        if message.get('role') == 'assistant':
            return message.get('content', '')
    return None

def download_and_process_updesh(english_samples=15000, hindi_samples=15000, seed=42):
    """Download Updesh dataset and extract assistant responses"""
    print("="*80)
    print("DOWNLOADING UPDESH DATASET FROM HUGGING FACE")
    print("="*80)
    
    random.seed(seed)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Download English dataset
    print("\n[1/4] Loading English dataset...")
    try:
        english_dataset = load_dataset(
            "microsoft/Updesh_beta",
            "cultural_multihop_reasoning",
            split="eng_Latn"
        )
        print(f"   Loaded {len(english_dataset)} English samples")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Download Hindi dataset
    print("\n[2/4] Loading Hindi dataset...")
    try:
        hindi_dataset = load_dataset(
            "microsoft/Updesh_beta",
            "cultural_multihop_reasoning",
            split="hin_Deva"
        )
        print(f"   Loaded {len(hindi_dataset)} Hindi samples")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Process English
    print(f"\n[3/4] Processing English samples...")
    english_data = []
    total_english = len(english_dataset)
    
    # Sample indices
    num_samples = min(english_samples, total_english)
    selected_indices = random.sample(range(total_english), num_samples)
    
    for idx in tqdm(selected_indices, desc="   Extracting English"):
        sample = english_dataset[idx]
        messages = sample.get('messages', [])  # Changed from 'conversations' to 'messages'
        assistant_response = extract_assistant_response(messages)
        
        if assistant_response:
            english_data.append({
                'text': assistant_response,
                'language': 'english',
                'category': 'cultural_reasoning',
                'source': 'updesh_eng',
                'id': sample.get('id', '')
            })
    
    print(f"   Extracted {len(english_data)} English responses")
    
    # Process Hindi
    print(f"\n[4/4] Processing Hindi samples...")
    hindi_data = []
    total_hindi = len(hindi_dataset)
    
    # Sample indices
    num_samples = min(hindi_samples, total_hindi)
    selected_indices = random.sample(range(total_hindi), num_samples)
    
    for idx in tqdm(selected_indices, desc="   Extracting Hindi"):
        sample = hindi_dataset[idx]
        messages = sample.get('messages', [])  # Changed from 'conversations' to 'messages'
        assistant_response = extract_assistant_response(messages)
        
        if assistant_response:
            hindi_data.append({
                'text': assistant_response,
                'language': 'hindi',
                'category': 'cultural_reasoning',
                'source': 'updesh_hin',
                'id': sample.get('id', '')
            })
    
    print(f"   Extracted {len(hindi_data)} Hindi responses")
    
    # Combine and save
    print(f"\n[5/5] Combining and saving...")
    combined_data = english_data + hindi_data
    random.shuffle(combined_data)
    
    output_file = DATA_ROOT / "updesh_beta.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved to: {output_file}")
    print(f"  Total: {len(combined_data)} samples")
    print(f"  English: {len(english_data)} | Hindi: {len(hindi_data)}")
    
    # Calculate statistics
    avg_length = sum(len(item['text']) for item in combined_data) / len(combined_data)
    print(f"  Average text length: {avg_length:.0f} characters")
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE TEXTS")
    print("="*80)
    if english_data:
        print("\n[English Sample]")
        print(english_data[0]['text'][:300] + "...")
    if hindi_data:
        print("\n[Hindi Sample]")
        print(hindi_data[0]['text'][:300] + "...")
    
    print("\n" + "="*80)
    print("✓ UPDESH DATASET DOWNLOAD COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    download_and_process_updesh()
