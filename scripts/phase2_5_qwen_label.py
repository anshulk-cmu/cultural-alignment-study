# scripts/phase2_5_qwen_label.py
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import SAE_OUTPUT_ROOT, setup_logger

logger = setup_logger('qwen_labeling', 'phase2_5_qwen.log')

def load_qwen():
    """Load Qwen2.5-72B-Instruct model with 8-bit quantization."""
    model_path = "/mnt/nfs-shared-centralus/anshulk/models/qwen2.5-72b-instruct"
    
    logger.info(f"Loading Qwen2.5-72B from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Qwen2.5 optimized for bfloat16
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    
    logger.info("Qwen2.5-72B loaded successfully")
    return tokenizer, model

def create_labeling_prompt(feature_data):
    """Create a prompt for feature labeling with top examples."""
    examples_text = ""
    num_examples = min(20, len(feature_data['examples']))
    
    for i, ex in enumerate(feature_data['examples'][:num_examples], 1):
        # Truncate long texts but preserve key content
        text = ex['text'][:300]
        if len(ex['text']) > 300:
            text += "..."
        examples_text += f"{i}. {text} (activation: {ex['activation']:.3f})\n\n"
    
    prompt = f"""You are analyzing a sparse autoencoder feature from a language model trained on Indian English and Hindi text. Below are the top {num_examples} text examples where this feature activated most strongly.

EXAMPLES:
{examples_text}

TASK:
Analyze these examples and determine:
1. Do these examples form a COHERENT group with a shared pattern?
2. If YES: Provide a concise 1-2 sentence label describing the linguistic, cultural, or semantic pattern.
3. If NO: Output exactly "INCOHERENT"

Focus on:
- Cultural markers (Indian festivals, names, food, regional terms, honorifics)
- Linguistic patterns (Hindi words, code-switching, grammatical structures)
- Semantic themes (topics, sentiment, discourse patterns)

OUTPUT FORMAT:
If coherent: A clear, specific label (e.g., "Hindi honorifics and respectful address forms" or "References to Indian festivals and celebrations")
If incoherent: "INCOHERENT"

LABEL:"""
    return prompt

def generate_label(tokenizer, model, prompt, max_new_tokens=150):
    """Generate a label using Qwen2.5 model."""
    # Qwen2.5 uses a specific chat format
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in linguistic analysis and cultural understanding."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated tokens
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()

def label_all_features(examples_dir, output_file):
    """Label all features from extracted examples."""
    tokenizer, model = load_qwen()
    
    examples_dir = Path(examples_dir)
    examples_files = sorted(examples_dir.glob("*_examples.json"))
    
    if not examples_files:
        logger.error(f"No example files found in {examples_dir}")
        return []
    
    logger.info(f"Found {len(examples_files)} example files to process")
    
    all_results = []
    
    for file_path in examples_files:
        logger.info(f"Processing {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        logger.info(f"Loaded {len(features)} features from {file_path.name}")
        
        for feature in tqdm(features, desc=f"Labeling {file_path.stem}"):
            try:
                prompt = create_labeling_prompt(feature)
                label = generate_label(tokenizer, model, prompt)
                
                # Check coherence
                is_coherent = 'INCOHERENT' not in label.upper()
                
                result = {
                    'feature_id': feature['feature_id'],
                    'sae_name': feature['sae_name'],
                    'feature_idx': feature['feature_idx'],
                    'num_examples': feature['num_examples'],
                    'label_qwen': label,
                    'is_coherent': is_coherent
                }
                
                # Copy over additional statistics if present
                for key in ['max_activation', 'mean_activation', 'sparsity']:
                    if key in feature:
                        result[key] = feature[key]
                
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error labeling feature {feature['feature_id']}: {str(e)}")
                continue
        
        # Save intermediate results after each file
        intermediate_file = output_file.parent / f"{output_file.stem}_partial.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved intermediate results to {intermediate_file}")
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(all_results)} labels to {output_file}")
    return all_results

if __name__ == "__main__":
    examples_dir = SAE_OUTPUT_ROOT / "feature_examples"
    output_file = SAE_OUTPUT_ROOT / "labels_qwen_initial.json"
    
    logger.info("=" * 80)
    logger.info("Starting Qwen2.5-72B feature labeling")
    logger.info(f"Input directory: {examples_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 80)
    
    # Run labeling
    results = label_all_features(examples_dir, output_file)
    
    # Summary statistics
    if results:
        coherent = sum(1 for r in results if r['is_coherent'])
        coherent_pct = (coherent / len(results)) * 100
        
        logger.info("=" * 80)
        logger.info("LABELING COMPLETE")
        logger.info(f"Total features labeled: {len(results)}")
        logger.info(f"Coherent features: {coherent} ({coherent_pct:.1f}%)")
        logger.info(f"Incoherent features: {len(results) - coherent} ({100 - coherent_pct:.1f}%)")
        logger.info("=" * 80)
    else:
        logger.error("No features were successfully labeled")
