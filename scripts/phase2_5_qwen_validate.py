# scripts/phase2_5_qwen_validate.py
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs.config import SAE_OUTPUT_ROOT, setup_logger

logger = setup_logger('qwen_validation', 'phase2_5_validate.log')

def load_qwen():
    """Load Qwen2.5-72B-Instruct model for validation."""
    model_path = "/mnt/nfs-shared-centralus/anshulk/models/qwen2.5-72b-instruct"
    
    logger.info(f"Loading Qwen2.5-72B from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    
    logger.info("Qwen2.5-72B loaded successfully for validation")
    return tokenizer, model

def create_validation_prompt(feature_data, initial_label, examples):
    """Create a validation prompt for label quality assessment."""
    examples_text = ""
    num_examples = min(20, len(examples))
    
    for i, ex in enumerate(examples[:num_examples], 1):
        text = ex['text'][:300]
        if len(ex['text']) > 300:
            text += "..."
        examples_text += f"{i}. {text}\n\n"
    
    prompt = f"""You are validating a feature label from sparse autoencoder analysis of Indian English and Hindi text.

INITIAL LABEL: "{initial_label}"

EXAMPLES ({num_examples} total):
{examples_text}

TASK:
Evaluate whether the label accurately describes the pattern in these examples.

EVALUATION CRITERIA:
- KEEP: Label accurately describes ≥15 out of {num_examples} examples
- REVISE: Label is partially correct but needs refinement (provide improved label)
- INVALIDATE: Label doesn't match examples OR no coherent pattern exists (< 10 examples match)

OUTPUT FORMAT (be explicit):
ACTION: [KEEP|REVISE|INVALIDATE]
REVISED_LABEL: [Only if ACTION is REVISE - provide the improved label here]
REASON: [Brief 1-2 sentence explanation of your decision]

Your response:"""
    return prompt

def generate_validation(tokenizer, model, prompt, max_new_tokens=200):
    """Generate validation response using Qwen2.5."""
    messages = [
        {"role": "system", "content": "You are a rigorous linguistic analyst validating feature labels for interpretability research."},
        {"role": "user", "content": prompt}
    ]
    
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
            temperature=0.2,  # Lower temperature for more consistent validation
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()

def parse_validation_response(response):
    """Parse the validation response to extract action, revised label, and reason."""
    action = 'KEEP'  # Default
    revised_label = None
    reason = ""
    
    # Extract ACTION
    if 'INVALIDATE' in response.upper():
        action = 'INVALIDATE'
    elif 'REVISE' in response.upper():
        action = 'REVISE'
    elif 'KEEP' in response.upper():
        action = 'KEEP'
    
    # Extract REVISED_LABEL if action is REVISE
    if action == 'REVISE' and 'REVISED_LABEL:' in response:
        try:
            revised_part = response.split('REVISED_LABEL:')[1]
            if 'REASON:' in revised_part:
                revised_label = revised_part.split('REASON:')[0].strip()
            else:
                revised_label = revised_part.strip()
        except:
            logger.warning("Failed to parse REVISED_LABEL from response")
    
    # Extract REASON
    if 'REASON:' in response:
        try:
            reason = response.split('REASON:')[1].strip()
        except:
            pass
    
    return action, revised_label, reason

def validate_labels(initial_labels_file, examples_dir, output_file):
    """Validate all initial labels using Qwen2.5-72B."""
    # Load model
    tokenizer, model = load_qwen()
    
    # Load initial labels
    logger.info(f"Loading initial labels from {initial_labels_file}")
    with open(initial_labels_file, 'r', encoding='utf-8') as f:
        initial_labels = json.load(f)
    
    logger.info(f"Loaded {len(initial_labels)} initial labels")
    
    # Load all examples into a map
    examples_dir = Path(examples_dir)
    examples_map = {}
    
    for file_path in examples_dir.glob("*_examples.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
            for feat in features:
                examples_map[feat['feature_id']] = feat['examples']
    
    logger.info(f"Loaded examples for {len(examples_map)} features")
    
    validated = []
    
    for label_data in tqdm(initial_labels, desc="Validating labels"):
        try:
            # Skip incoherent features
            if not label_data.get('is_coherent', False):
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'Initial label was INCOHERENT'
                label_data['final_label'] = None
                validated.append(label_data)
                continue
            
            # Get examples for this feature
            examples = examples_map.get(label_data['feature_id'], [])
            if not examples:
                logger.warning(f"No examples found for {label_data['feature_id']}")
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'No examples available'
                label_data['final_label'] = None
                validated.append(label_data)
                continue
            
            # Create validation prompt
            prompt = create_validation_prompt(
                label_data, 
                label_data['label_qwen'], 
                examples
            )
            
            # Generate validation
            response = generate_validation(tokenizer, model, prompt)
            
            # Parse response
            action, revised_label, reason = parse_validation_response(response)
            
            # Store results
            label_data['validation_action'] = action
            label_data['validation_response'] = response
            label_data['validation_reason'] = reason
            
            # Determine final label
            if action == 'KEEP':
                label_data['final_label'] = label_data['label_qwen']
            elif action == 'REVISE':
                label_data['final_label'] = revised_label if revised_label else label_data['label_qwen']
            else:  # INVALIDATE
                label_data['final_label'] = None
            
            validated.append(label_data)
            
        except Exception as e:
            logger.error(f"Error validating feature {label_data.get('feature_id', 'unknown')}: {str(e)}")
            label_data['validation_action'] = 'ERROR'
            label_data['validation_reason'] = str(e)
            label_data['final_label'] = None
            validated.append(label_data)
            continue
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)
    
    # Compute statistics
    stats = {
        'KEEP': sum(1 for v in validated if v.get('validation_action') == 'KEEP'),
        'REVISE': sum(1 for v in validated if v.get('validation_action') == 'REVISE'),
        'INVALIDATE': sum(1 for v in validated if v.get('validation_action') == 'INVALIDATE'),
        'ERROR': sum(1 for v in validated if v.get('validation_action') == 'ERROR')
    }
    
    total = len(validated)
    final_valid = stats['KEEP'] + stats['REVISE']
    
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info(f"Total features validated: {total}")
    logger.info(f"KEEP: {stats['KEEP']} ({stats['KEEP']/total*100:.1f}%)")
    logger.info(f"REVISE: {stats['REVISE']} ({stats['REVISE']/total*100:.1f}%)")
    logger.info(f"INVALIDATE: {stats['INVALIDATE']} ({stats['INVALIDATE']/total*100:.1f}%)")
    logger.info(f"ERROR: {stats['ERROR']} ({stats['ERROR']/total*100:.1f}%)")
    logger.info(f"Final valid features: {final_valid} ({final_valid/total*100:.1f}%)")
    logger.info("=" * 80)
    
    return validated

if __name__ == "__main__":
    initial_file = SAE_OUTPUT_ROOT / "labels_qwen_initial.json"
    examples_dir = SAE_OUTPUT_ROOT / "feature_examples"
    output_file = SAE_OUTPUT_ROOT / "labels_qwen_validated.json"
    
    logger.info("=" * 80)
    logger.info("Starting Qwen2.5-72B label validation")
    logger.info(f"Input file: {initial_file}")
    logger.info(f"Examples directory: {examples_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 80)
    
    validate_labels(initial_file, examples_dir, output_file)
