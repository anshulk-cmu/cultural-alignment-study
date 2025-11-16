#!/usr/bin/env python3
"""
Batch sentence generation for cultural alignment study.
Uses Anthropic's Message Batches API with few-shot prompting.
"""

import anthropic
import pandas as pd
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# Set random seed for reproducibility
random.seed(42)

# Configuration
DATA_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/sanskriti_data")
INPUT_FILE = DATA_DIR / "sanskriti_12k_targeted.csv"
OUTPUT_FILE = DATA_DIR / "generated_sentences_12k_batch.csv"
BATCH_METADATA_FILE = DATA_DIR / "batch_metadata_12k.json"

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1024

# System prompt
SYSTEM_PROMPT = """You are generating training data for a cultural knowledge model.

Given a single multiple-choice QA row about Indian culture, write exactly THREE natural sentences (40–60 words each) that encode the same cultural fact.

You will receive:
- state
- attribute
- question
- answer

From these, FIRST infer:
- the core cultural concept (e.g., "Jarawa body painting", "Hodi craft", "pandanus in cuisine")
- the correct region or territory from the answer (e.g., "South Andaman and Middle Andaman Islands", "Nicobar district").

GENERAL RULES
1. Each sentence MUST:
   - mention the cultural concept and the correct region
   - be factually consistent with the question–answer pair
   - be self-contained and understandable without the question.

2. Length: 40–60 words per sentence. Do NOT exceed 60 words or go below 40.

3. Do NOT:
   - restate or quote the question or answer text verbatim.
   - mention "question", "answer", "multiple choice", "options", "dataset", "row", "model", or "probes".
   - invent extra specific facts (dates, numbers, named festivals, legal acts, etc.) that are not logically implied.
   - use value judgements like "iconic", "famous", "important" unless the question clearly implies it.

4. Lexical variety / anti-boilerplate:
   - Avoid using the **same opening pattern** more than once across the three sentences. Do NOT start every sentence with "In the…", "While many…", "Jarawa body painting is…", "Pandanus is…", "Hodi craft is…", etc.
   - Avoid generic scaffolding like:
     - "While many parts of India…"
     - "While mainland India…"
     - "While many regions…"
     - "This reflects how island regions…"
   - Use concrete details (who uses it, where, in what situations) instead of vague phrases like "way of life", "connection to the environment" in every sentence.
   - Vary subject and structure: sometimes start with people ("Local communities use…"), sometimes with place ("In Nicobar district, …"), sometimes with the object ("These canoes…").

5. Style:
   - Neutral, descriptive tone, like a good encyclopedia or travel-guide paragraph.
   - No bullet points, no lists, no headings.
   - Third person only; no "I", "we", or "you".

SENTENCE TYPES
Generate three **different** perspectives on the same fact:

- Sentence 1: DEFINITONAL / DESCRIPTIVE  
  Explain what the concept is and clearly tie it to the correct region.

- Sentence 2: USAGE / CONTEXTUAL  
  Describe how people in that region actually use or experience this thing (daily life, ceremonies, travel, food, housing, etc.).

- Sentence 3: CONTRAST / POSITIONING  
  Optionally compare it to broader India or other regions, BUT:
  - do NOT use boilerplate templates like "While many parts of India…" or "Unlike mainland India…".
  - Use more specific phrasings, only if it adds clarity.

OUTPUT FORMAT (STRICT)
Return ONLY three lines:

Sentence 1: <text here, 40–60 words>
Sentence 2: <text here, 40–60 words>
Sentence 3: <text here, 40–60 words>

No extra commentary."""


def create_user_prompt(state: str, attribute: str, question: str, answer: str) -> str:
    """Create the user prompt with examples and the actual task."""
    
    return f"""EXAMPLE 1
state: Andaman_and_Nicobar
attribute: Art
question: Which of the given regions is home to the Jarawa body painting?
answer: South Andaman and Middle Andaman Islands

Sentence 1: Jarawa body painting, practiced in the South Andaman and Middle Andaman Islands, consists of carefully placed patterns on the skin that signal identity and continuity with long-standing customs developed by communities living in the forests and coastal settlements of these remote islands.

Sentence 2: In South Andaman and Middle Andaman Islands, Jarawa body painting appears before group activities, shared gatherings, and key social moments, with designs applied by relatives or peers who learn techniques through observation and practice rather than through formal instruction or commercial workshops.

Sentence 3: Outside the South Andaman and Middle Andaman Islands, body decoration in India often takes the form of temporary festival cosmetics or salon-based designs, whereas Jarawa body painting remains a community-held practice rooted in everyday interactions, local materials, and the specific social structures of these island communities.

EXAMPLE 2
state: Andaman_and_Nicobar
attribute: Art
question: Where is the Hodi Craft famous within Andaman_and_Nicobar?
answer: Nicobar district

Sentence 1: Hodi craft in the Nicobar district refers to traditional hand-made items shaped from local materials, with techniques that reflect accumulated experience of island residents who adapt their objects to humid conditions, limited infrastructure, and the need for durable goods that can be repaired or reused across many years.

Sentence 2: Within villages of the Nicobar district, people practicing hodi craft create objects for storage, carrying food, or organizing household spaces, working individually or in small groups and sharing practical tips about materials, shapes, and repairs as part of everyday conversation rather than through large-scale organized workshops.

Sentence 3: In contrast to industrially produced household items found in many mainland markets, hodi craft in the Nicobar district remains closely tied to local surroundings, with makers depending on nearby plants and simple tools, and adjusting designs to the rhythms of island life instead of mass-production schedules.

NOW GENERATE FOR:
state: {state}
attribute: {attribute}
question: {question}
answer: {answer}

Return ONLY three lines in the format:
Sentence 1: <text here, 40–60 words>
Sentence 2: <text here, 40–60 words>
Sentence 3: <text here, 40–60 words>"""


def load_data(num_samples: int = None) -> pd.DataFrame:
    """Load the sanskriti 12K targeted dataset."""
    
    print(f"Loading data from {INPUT_FILE}...")
    
    # Load CSV
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Total rows in dataset: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Verify group_type column exists
    if 'group_type' not in df.columns:
        raise ValueError("Missing 'group_type' column in dataset!")
    
    # Show group distribution
    print(f"\nGroup Distribution:")
    group_counts = df['group_type'].value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nAttribute distribution (top 10):")
    for attr, count in df['attribute'].value_counts().head(10).items():
        print(f"  {attr}: {count}")
    
    print(f"\nState distribution (top 10):")
    for state, count in df['state'].value_counts().head(10).items():
        print(f"  {state}: {count}")
    
    print(f"\nProcessing all {len(df)} questions")
    print(f"This will generate {len(df) * 3} sentences total")
    
    return df

def generate_stratified_temperatures(n: int) -> List[float]:
    """Generate stratified temperature values across bins [0.80-0.85, 0.85-0.90, 0.90-0.95]."""
    
    bins = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95)]
    temperatures = []
    
    samples_per_bin = n // len(bins)
    remainder = n % len(bins)
    
    for i, (low, high) in enumerate(bins):
        num_samples = samples_per_bin + (1 if i < remainder else 0)
        
        for _ in range(num_samples):
            temperatures.append(random.uniform(low, high))
    
    random.shuffle(temperatures)
    
    return temperatures


def create_batch_requests(df: pd.DataFrame) -> Tuple[List[Request], List[float]]:
    """Create batch requests for all rows in the dataframe."""
    
    requests = []
    n_rows = len(df)
    
    temperatures = generate_stratified_temperatures(n_rows)
    
    print(f"\nTemperature distribution:")
    print(f"  0.80-0.85: {sum(1 for t in temperatures if 0.80 <= t < 0.85)} samples")
    print(f"  0.85-0.90: {sum(1 for t in temperatures if 0.85 <= t < 0.90)} samples")
    print(f"  0.90-0.95: {sum(1 for t in temperatures if 0.90 <= t <= 0.95)} samples")
    
    for idx, row in df.iterrows():
        custom_id = f"row_{idx}"
        
        state = str(row['state'])
        attribute = str(row['attribute'])
        question = str(row['question'])
        answer = str(row['answer'])
        
        user_prompt = create_user_prompt(state, attribute, question, answer)
        temperature = temperatures[len(requests)]
        
        request = Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }]
            )
        )
        
        requests.append(request)
    
    print(f"Created {len(requests)} batch requests")
    return requests, temperatures

def estimate_cost(num_requests: int) -> dict:
    """Estimate batch processing cost."""
    
    # Based on previous run: 1000 questions = $3.75
    cost_per_1000 = 3.75
    estimated_cost = (num_requests / 1000) * cost_per_1000
    
    # Token estimates
    avg_input_tokens = 800  # per question
    avg_output_tokens = 600  # 3 sentences × ~200 tokens each
    
    total_input_tokens = num_requests * avg_input_tokens
    total_output_tokens = num_requests * avg_output_tokens
    total_sentences = num_requests * 3
    
    return {
        "num_questions": num_requests,
        "num_sentences": total_sentences,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost_usd": estimated_cost,
        "cost_range": f"${estimated_cost * 0.9:.2f} - ${estimated_cost * 1.1:.2f}"
    }

def submit_batch(client: anthropic.Anthropic, requests: List[Request], 
                temperatures: List[float], df: pd.DataFrame) -> str:
    """Submit the batch for processing with cost confirmation."""
    
    # Show cost estimation
    cost_info = estimate_cost(len(requests))
    
    print("\n" + "="*80)
    print("BATCH SUBMISSION SUMMARY")
    print("="*80)
    print(f"Total Questions:        {cost_info['num_questions']:,}")
    print(f"Total Sentences:        {cost_info['num_sentences']:,}")
    print(f"Estimated Input Tokens: {cost_info['estimated_input_tokens']:,}")
    print(f"Estimated Output Tokens: {cost_info['estimated_output_tokens']:,}")
    print(f"Estimated Cost:         ${cost_info['estimated_cost_usd']:.2f}")
    print(f"Cost Range:             {cost_info['cost_range']}")
    
    # Group breakdown
    print(f"\nGroup Breakdown:")
    group_counts = df['group_type'].value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count} questions → {count * 3} sentences")
    
    print("="*80)
    
    # Confirmation prompt
    response = input("\nProceed with batch submission? (y/n): ")
    if response.lower() != 'y':
        print("Batch submission cancelled.")
        exit(0)
    
    print("\nSubmitting batch...")
    message_batch = client.messages.batches.create(requests=requests)
    
    batch_id = message_batch.id
    print(f"Batch submitted successfully!")
    print(f"Batch ID: {batch_id}")
    print(f"Status: {message_batch.processing_status}")
    print(f"Created at: {message_batch.created_at}")
    print(f"Expires at: {message_batch.expires_at}")
    
    # Enhanced metadata with group info
    metadata = {
        "batch_id": batch_id,
        "created_at": str(message_batch.created_at),
        "expires_at": str(message_batch.expires_at),
        "num_requests": len(requests),
        "model": MODEL,
        "random_seed": 42,
        "temperature_range": "0.80-0.95",
        "source_file": "sanskriti_12k_targeted.csv",
        "purpose": "Targeted sentence generation for probing and training",
        "group_distribution": group_counts.to_dict(),
        "cost_estimate": cost_info,
        "temperature_stats": {
            "min": min(temperatures),
            "max": max(temperatures),
            "mean": sum(temperatures) / len(temperatures),
            "count_0.80_0.85": sum(1 for t in temperatures if 0.80 <= t < 0.85),
            "count_0.85_0.90": sum(1 for t in temperatures if 0.85 <= t < 0.90),
            "count_0.90_0.95": sum(1 for t in temperatures if 0.90 <= t <= 0.95)
        }
    }
    
    with open(BATCH_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Batch metadata saved to {BATCH_METADATA_FILE}")
    
    return batch_id


def poll_batch_completion(client: anthropic.Anthropic, batch_id: str, 
                          poll_interval: int = 60) -> Dict:
    """Poll the batch until completion."""
    
    print(f"\nPolling batch {batch_id} for completion...")
    print(f"This may take up to 1 hour for most batches...")
    
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        
        status = message_batch.processing_status
        counts = message_batch.request_counts
        
        print(f"\nStatus: {status}")
        print(f"Processing: {counts.processing}")
        print(f"Succeeded: {counts.succeeded}")
        print(f"Errored: {counts.errored}")
        print(f"Canceled: {counts.canceled}")
        print(f"Expired: {counts.expired}")
        
        if status == "ended":
            print("\n✓ Batch processing completed!")
            return message_batch
        
        print(f"Waiting {poll_interval} seconds before next check...")
        time.sleep(poll_interval)


def parse_sentence_output(text: str) -> Dict[str, str]:
    """Parse the three sentences from the model output."""
    
    sentences = {
        "sentence_1": "",
        "sentence_2": "",
        "sentence_3": ""
    }
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith("Sentence 1:"):
            sentences["sentence_1"] = line.replace("Sentence 1:", "").strip()
        elif line.startswith("Sentence 2:"):
            sentences["sentence_2"] = line.replace("Sentence 2:", "").strip()
        elif line.startswith("Sentence 3:"):
            sentences["sentence_3"] = line.replace("Sentence 3:", "").strip()
    
    return sentences


def process_batch_results(client: anthropic.Anthropic, batch_id: str, 
                         original_df: pd.DataFrame, 
                         temperatures: List[float]) -> pd.DataFrame:
    """Process and parse batch results into a dataframe."""
    
    print(f"\nRetrieving results for batch {batch_id}...")
    
    results_data = []
    error_count = 0
    
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        row_idx = int(custom_id.split('_')[1])
        
        original_row = original_df.iloc[row_idx]
        
        result_entry = original_row.to_dict()
        result_entry["row_id"] = row_idx
        result_entry["temperature"] = temperatures[row_idx]
        result_entry["result_type"] = result.result.type
        
        if result.result.type == "succeeded":
            message = result.result.message
            text_content = message.content[0].text
            
            sentences = parse_sentence_output(text_content)
            
            result_entry.update(sentences)
            result_entry["raw_output"] = text_content
            result_entry["input_tokens"] = message.usage.input_tokens
            result_entry["output_tokens"] = message.usage.output_tokens
            
        elif result.result.type == "errored":
            error_count += 1
            result_entry["error_type"] = result.result.error.type
            result_entry["error_message"] = str(result.result.error)
            print(f"Error in {custom_id}: {result.result.error.type}")
            
        else:
            error_count += 1
            result_entry["error_type"] = result.result.type
        
        results_data.append(result_entry)
    
    results_df = pd.DataFrame(results_data)
    
    print(f"\nProcessed {len(results_data)} results")
    print(f"Succeeded: {len(results_df[results_df['result_type'] == 'succeeded'])}")
    print(f"Errors/Issues: {error_count}")
    
    # Error breakdown by group
    if 'group_type' in results_df.columns:
        print(f"\nError breakdown by group:")
        for group in ['suppression', 'enhancement', 'control']:
            group_df = results_df[results_df['group_type'] == group]
            group_errors = len(group_df[group_df['result_type'] != 'succeeded'])
            print(f"  {group}: {group_errors} errors / {len(group_df)} total")
    
    return results_df


def save_results(results_df: pd.DataFrame):
    """Save results to CSV."""
    
    print(f"\nSaving results to {OUTPUT_FILE}...")
    results_df.to_csv(OUTPUT_FILE, index=False)
    print("✓ Results saved successfully!")
    
    successful_rows = results_df[results_df['result_type'] == 'succeeded']
    
    # Group-wise success statistics
    if 'group_type' in results_df.columns and len(successful_rows) > 0:
        print("\nSuccess rate by group:")
        for group in ['suppression', 'enhancement', 'control']:
            group_df = results_df[results_df['group_type'] == group]
            group_success = len(group_df[group_df['result_type'] == 'succeeded'])
            success_rate = (group_success / len(group_df) * 100) if len(group_df) > 0 else 0
            print(f"  {group}: {group_success}/{len(group_df)} ({success_rate:.1f}%)")
    
    if len(successful_rows) > 0:
        print("\nSample of generated sentences:")
        print("=" * 80)
        
        sample_row = successful_rows.iloc[0]
        print(f"State: {sample_row['state']}")
        print(f"Attribute: {sample_row['attribute']}")
        print(f"Question: {sample_row['question']}")
        print(f"Answer: {sample_row['answer']}")
        print(f"Temperature: {sample_row['temperature']:.3f}")
        print(f"\nSentence 1: {sample_row['sentence_1']}")
        print(f"\nSentence 2: {sample_row['sentence_2']}")
        print(f"\nSentence 3: {sample_row['sentence_3']}")
        print("=" * 80)
    else:
        print("\n No successful results to display. Check error messages in the output CSV.")

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("Sanskriti Cultural Alignment Study - Batch Sentence Generation")
    print("Reproducible run with seed=42")
    print("=" * 80)
    
    print("\nInitializing Anthropic client...")
    client = anthropic.Anthropic()
    
    df = load_data()
    
    requests, temperatures = create_batch_requests(df)
    
    batch_id = submit_batch(client, requests, temperatures, df)
    
    batch_result = poll_batch_completion(client, batch_id)
    
    results_df = process_batch_results(client, batch_id, df, temperatures)
    
    save_results(results_df)
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"Batch ID: {batch_id}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Metadata file: {BATCH_METADATA_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()