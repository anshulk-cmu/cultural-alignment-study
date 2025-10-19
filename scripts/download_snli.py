"""
Download SNLI control set - 5000 generic non-cultural sentences
"""
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

from datasets import load_dataset
import json
import random
from pathlib import Path
from configs.config import DATA_ROOT

def download_snli_control(num_samples=5000, seed=42):
    print("="*80)
    print("DOWNLOADING SNLI CONTROL SET")
    print("="*80)
    
    print("\n[1/3] Loading SNLI dataset...")
    dataset = load_dataset('stanfordnlp/snli', split='train')
    print(f"   Loaded {len(dataset)} total samples")
    
    print("\n[2/3] Filtering and sampling...")
    # Filter valid premises (label != -1)
    valid_premises = [
        item['premise']
        for item in dataset
        if item['label'] != -1 and len(item['premise'].strip()) > 10
    ]
    print(f"   Valid premises: {len(valid_premises)}")
    
    # Sample
    random.seed(seed)
    sampled = random.sample(valid_premises, num_samples)
    
    # Format
    output = [
        {'text': text, 'language': 'english', 'category': 'control', 'source': 'snli'}
        for text in sampled
    ]
    
    print("\n[3/3] Saving...")
    output_file = DATA_ROOT / "snli_control.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Stats
    avg_length = sum(len(item['text']) for item in output) / len(output)
    
    print(f"\n✓ Saved to: {output_file}")
    print(f"  Total samples: {len(output)}")
    print(f"  Average length: {avg_length:.0f} characters")
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE TEXTS")
    print("="*80)
    print(f"\n[Sample 1] {output[0]['text']}")
    print(f"\n[Sample 2] {output[1]['text']}")
    
    print("\n" + "="*80)
    print("✓ SNLI CONTROL SET DOWNLOAD COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    download_snli_control()
