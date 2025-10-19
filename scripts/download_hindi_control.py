"""
Download Hindi control sentences from IITB corpus
Extract 5000 generic Hindi sentences (10-50 words each)
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import json
import random
from datasets import load_dataset
from pathlib import Path
from configs.config import DATA_ROOT

def count_words(text):
    """Count words in text"""
    return len(text.strip().split())

def download_hindi_control(min_words=10, max_words=50, num_samples=5000, seed=42):
    """Extract Hindi control sentences"""
    print("="*80)
    print("DOWNLOADING HINDI CONTROL SET")
    print("="*80)
    
    # Load dataset
    print("\n[1/4] Loading IITB corpus...")
    dataset = load_dataset('cfilt/iitb-english-hindi', split='train')
    print(f"   Loaded {len(dataset)} translation pairs")
    
    # Filter
    print(f"\n[2/4] Filtering ({min_words}-{max_words} words)...")
    valid_sentences = []
    
    for item in dataset:
        hindi_text = item['translation']['hi']
        word_count = count_words(hindi_text)
        
        if min_words <= word_count <= max_words:
            valid_sentences.append({
                'text': hindi_text.strip(),
                'word_count': word_count
            })
    
    print(f"   Valid sentences: {len(valid_sentences)}")
    
    # Sample
    print(f"\n[3/4] Sampling {num_samples}...")
    random.seed(seed)
    sampled = random.sample(valid_sentences, min(num_samples, len(valid_sentences)))
    
    # Format
    output_data = [
        {
            'text': item['text'],
            'language': 'hindi',
            'category': 'control',
            'source': 'iitb',
            'word_count': item['word_count']
        }
        for item in sampled
    ]
    
    # Save
    print("\n[4/4] Saving...")
    output_file = DATA_ROOT / "hindi_control.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Stats
    word_counts = [item['word_count'] for item in output_data]
    print(f"\n✓ Saved to: {output_file}")
    print(f"  Total: {len(output_data)} samples")
    print(f"  Avg length: {sum(len(item['text']) for item in output_data)/len(output_data):.0f} chars")
    print(f"  Avg words: {sum(word_counts)/len(word_counts):.1f}")
    
    # Samples
    print("\n" + "="*80)
    print("SAMPLE TEXTS")
    print("="*80)
    for i in range(min(2, len(output_data))):
        print(f"\n[Sample {i+1}] {output_data[i]['text'][:150]}...")
    
    print("\n" + "="*80)
    print("✓ HINDI CONTROL SET COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    download_hindi_control()
