"""
Download and combine datasets: Updesh (38k), SNLI control (6k), Hindi control (6k)
Creates 80/20 train/test split maintaining dataset ratios
"""

import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from configs.config import DATA_ROOT

SEED = 42
UPDESH_ENG_SAMPLES = 19000
UPDESH_HIN_SAMPLES = 19000
SNLI_SAMPLES = 6000
HINDI_CONTROL_SAMPLES = 6000
TRAIN_SPLIT = 0.8

def count_words(text):
    return len(text.strip().split())

def extract_assistant_response(messages):
    for message in reversed(messages):
        if message.get('role') == 'assistant':
            return message.get('content', '')
    return None

def split_dataset(data, train_ratio):
    """Split single dataset into train/test maintaining ratio"""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def download_updesh(english_samples, hindi_samples):
    print("\n[1/3] Downloading Updesh dataset...")
    
    english_dataset = load_dataset(
        "microsoft/Updesh_beta",
        "cultural_multihop_reasoning",
        split="eng_Latn"
    )
    hindi_dataset = load_dataset(
        "microsoft/Updesh_beta",
        "cultural_multihop_reasoning",
        split="hin_Deva"
    )
    
    print(f"   Loaded {len(english_dataset)} English, {len(hindi_dataset)} Hindi samples")
    
    # Process English
    english_data = []
    selected_indices = random.sample(range(len(english_dataset)), min(english_samples, len(english_dataset)))
    for idx in tqdm(selected_indices, desc="   Extracting English"):
        sample = english_dataset[idx]
        response = extract_assistant_response(sample.get('messages', []))
        if response:
            english_data.append({
                'text': response,
                'language': 'english',
                'category': 'cultural_reasoning',
                'source': 'updesh_eng'
            })
    
    # Process Hindi
    hindi_data = []
    selected_indices = random.sample(range(len(hindi_dataset)), min(hindi_samples, len(hindi_dataset)))
    for idx in tqdm(selected_indices, desc="   Extracting Hindi"):
        sample = hindi_dataset[idx]
        response = extract_assistant_response(sample.get('messages', []))
        if response:
            hindi_data.append({
                'text': response,
                'language': 'hindi',
                'category': 'cultural_reasoning',
                'source': 'updesh_hin'
            })
    
    print(f"   Extracted {len(english_data)} English, {len(hindi_data)} Hindi responses")
    return english_data + hindi_data

def download_snli_control(num_samples):
    print("\n[2/3] Downloading SNLI control...")
    
    dataset = load_dataset('stanfordnlp/snli', split='train')
    print(f"   Loaded {len(dataset)} samples")
    
    valid_premises = [
        item['premise']
        for item in dataset
        if item['label'] != -1 and len(item['premise'].strip()) > 10
    ]
    
    sampled = random.sample(valid_premises, min(num_samples, len(valid_premises)))
    
    data = [
        {'text': text, 'language': 'english', 'category': 'control', 'source': 'snli'}
        for text in sampled
    ]
    
    print(f"   Extracted {len(data)} samples")
    return data

def download_hindi_control(num_samples, min_words=10, max_words=50):
    print("\n[3/3] Downloading Hindi control...")
    
    dataset = load_dataset('cfilt/iitb-english-hindi', split='train')
    print(f"   Loaded {len(dataset)} translation pairs")
    
    valid_sentences = []
    for item in dataset:
        hindi_text = item['translation']['hi']
        word_count = count_words(hindi_text)
        if min_words <= word_count <= max_words:
            valid_sentences.append(hindi_text.strip())
    
    sampled = random.sample(valid_sentences, min(num_samples, len(valid_sentences)))
    
    data = [
        {'text': text, 'language': 'hindi', 'category': 'control', 'source': 'iitb'}
        for text in sampled
    ]
    
    print(f"   Extracted {len(data)} samples")
    return data

def main():
    print("="*80)
    print("DOWNLOADING AND COMBINING DATASETS")
    print("="*80)
    
    random.seed(SEED)
    
    # Download all datasets
    updesh_data = download_updesh(UPDESH_ENG_SAMPLES, UPDESH_HIN_SAMPLES)
    snli_data = download_snli_control(SNLI_SAMPLES)
    hindi_control_data = download_hindi_control(HINDI_CONTROL_SAMPLES)
    
    print("\n[4/4] Splitting each dataset 80/20...")
    
    # Split each dataset individually to maintain ratios
    updesh_train, updesh_test = split_dataset(updesh_data, TRAIN_SPLIT)
    snli_train, snli_test = split_dataset(snli_data, TRAIN_SPLIT)
    hindi_train, hindi_test = split_dataset(hindi_control_data, TRAIN_SPLIT)
    
    # Combine train and test separately
    train_data = updesh_train + snli_train + hindi_train
    test_data = updesh_test + snli_test + hindi_test
    
    # Shuffle within train and test
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f"   Train samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    # Create data directory and subdirectories
    data_dir = DATA_ROOT / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train
    train_file = train_dir / "combined_data.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # Save test
    test_file = test_dir / "combined_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Statistics
    print(f"\n✓ Train: {len(train_data)} samples → {train_file}")
    print(f"✓ Test: {len(test_data)} samples → {test_file}")
    
    # Dataset breakdown
    def count_sources(data):
        updesh = sum(1 for x in data if x['source'].startswith('updesh'))
        snli = sum(1 for x in data if x['source'] == 'snli')
        iitb = sum(1 for x in data if x['source'] == 'iitb')
        return updesh, snli, iitb
    
    train_updesh, train_snli, train_iitb = count_sources(train_data)
    test_updesh, test_snli, test_iitb = count_sources(test_data)
    
    print(f"\nTrain - Updesh: {train_updesh}, SNLI: {train_snli}, Hindi Control: {train_iitb}")
    print(f"Test - Updesh: {test_updesh}, SNLI: {test_snli}, Hindi Control: {test_iitb}")
    
    print("\n" + "="*80)
    print("✓ DOWNLOAD COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
