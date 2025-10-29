#!/usr/bin/env python3
"""
dosa_complete_pipeline.py

Complete pipeline to:
1. Download ALL DOSA artifacts from GitHub (CSVs, TXT files, notes)
2. Create consolidated original DOSA dataset CSV
3. Convert clues to natural language statements using Llama-3.2-3B-Instruct
4. Save LLM-generated statements to separate CSV for CST

Output files:
- /data/user_data/anshulk/cultural-alignment-study/data/dosa_consolidated_original.csv
- /data/user_data/anshulk/cultural-alignment-study/data/dosa_cst_samples.csv
- /data/user_data/anshulk/cultural-alignment-study/data/dosa_processing_report.json
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
import time
import re
from collections import defaultdict

# Configuration
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/microsoft/DOSA/main/data"
GITHUB_API_BASE = "https://api.github.com/repos/microsoft/DOSA/contents/data"
OUTPUT_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/data")
CONSOLIDATED_FILE = OUTPUT_DIR / "dosa_consolidated_original.csv"
CST_FILE = OUTPUT_DIR / "dosa_cst_samples.csv"
REPORT_FILE = OUTPUT_DIR / "dosa_processing_report.json"
MODEL_PATH = "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"

# 18 states from DOSA (excluding Madhya Pradesh which wasn't collected)
STATES = [
    "andhra_pradesh", "assam", "bihar", "chhattisgarh", "delhi",
    "gujarat", "haryana", "jharkhand", "karnataka", "kerala",
    "maharashtra", "odisha", "punjab", "rajasthan", "tamil_nadu",
    "telangana", "uttar_pradesh", "west_bengal"
]


class DOSADownloader:
    """Download and parse all DOSA artifacts from GitHub"""
    
    def __init__(self):
        self.session = requests.Session()
        self.artifacts = []
        self.stats = defaultdict(int)
        
    def get_state_files(self, state):
        """Get list of all files in a state directory"""
        url = f"{GITHUB_API_BASE}/{state}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            files = response.json()
            return [f['name'] for f in files if isinstance(files, list)]
        except Exception as e:
            print(f"   Warning: Could not list files for {state}: {e}")
            return []
    
    def download_file(self, state, filename):
        """Download a single file from GitHub"""
        url = f"{GITHUB_RAW_BASE}/{state}/{filename}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"   Warning: Could not download {state}/{filename}: {e}")
            return None
    
    def parse_csv_file(self, state, filename, content):
        """Parse CSV file handling different column orders"""
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(content))
            
            # Handle empty dataframes
            if df.empty:
                return []
            
            # Detect column order
            columns = [c.lower().strip() for c in df.columns]
            
            # Determine artifact and clues columns
            if 'artifact' in columns[0]:
                artifact_col = 0
                clues_start = 1
            elif 'artifact' in columns[-1]:
                # Bihar case: clues,artifact
                artifact_col = len(columns) - 1
                clues_start = 0
            else:
                # Try first column as artifact
                artifact_col = 0
                clues_start = 1
            
            artifacts = []
            file_type = 'expanded' if 'expanded' in filename else 'original'
            
            for idx, row in df.iterrows():
                # Get artifact name
                artifact_name = row.iloc[artifact_col]
                
                # Skip if artifact name is empty
                if pd.isna(artifact_name) or str(artifact_name).strip() == '':
                    continue
                
                artifact_name = str(artifact_name).strip()
                
                # Collect all clues
                clues = []
                for col_idx, col_name in enumerate(df.columns):
                    if col_idx == artifact_col:
                        continue
                    
                    clue_value = row[col_name]
                    if pd.notna(clue_value) and str(clue_value).strip():
                        clues.append(str(clue_value).strip())
                
                if clues:  # Only add if we have clues
                    artifacts.append({
                        'state': state,
                        'artifact_name': artifact_name,
                        'source_file': filename,
                        'source_type': file_type,
                        'clues': ' | '.join(clues),
                        'num_clues': len(clues)
                    })
                    self.stats[f'{state}_csv_artifacts'] += 1
            
            return artifacts
            
        except Exception as e:
            print(f"   Error parsing CSV {state}/{filename}: {e}")
            return []
    
    def parse_txt_file(self, state, filename, content):
        """Parse individual .txt clue files"""
        try:
            # Extract artifact name from filename
            # Pattern: clue[N]_artifact_name.txt or clues[N]_artifact_name.txt
            match = re.match(r'clues?\d*_(.+)\.txt', filename, re.IGNORECASE)
            if not match:
                return None
            
            artifact_name = match.group(1).replace('_', ' ').strip()
            
            # Clean content
            clues = content.strip()
            if not clues:
                return None
            
            self.stats[f'{state}_txt_artifacts'] += 1
            
            return {
                'state': state,
                'artifact_name': artifact_name,
                'source_file': filename,
                'source_type': 'individual_txt',
                'clues': clues,
                'num_clues': 1
            }
            
        except Exception as e:
            print(f"   Error parsing TXT {state}/{filename}: {e}")
            return None
    
    def download_state_artifacts(self, state):
        """Download and parse all artifacts for a state"""
        print(f"\n{'='*60}")
        print(f"Processing: {state.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        # Get list of files
        files = self.get_state_files(state)
        if not files:
            print(f"   No files found for {state}")
            return
        
        print(f"   Found {len(files)} files")
        
        csv_files = [f for f in files if f.endswith('.csv')]
        txt_files = [f for f in files if f.endswith('.txt') and 
                     re.match(r'clues?\d*_.+\.txt', f, re.IGNORECASE)]
        
        print(f"   - CSV files: {len(csv_files)}")
        print(f"   - Individual TXT files: {len(txt_files)}")
        
        # Process CSV files
        for filename in csv_files:
            content = self.download_file(state, filename)
            if content:
                artifacts = self.parse_csv_file(state, filename, content)
                self.artifacts.extend(artifacts)
                print(f"   ✓ {filename}: {len(artifacts)} artifacts")
        
        # Process individual TXT files
        for filename in txt_files:
            content = self.download_file(state, filename)
            if content:
                artifact = self.parse_txt_file(state, filename, content)
                if artifact:
                    self.artifacts.append(artifact)
        
        if txt_files:
            print(f"   ✓ Individual TXT files: {len(txt_files)} processed")
        
        state_total = len([a for a in self.artifacts if a['state'] == state])
        print(f"   Total artifacts for {state}: {state_total}")
        self.stats[f'{state}_total'] = state_total
    
    def download_all(self):
        """Download artifacts from all states"""
        print("="*80)
        print("DOSA COMPLETE DOWNLOAD - ALL ARTIFACTS")
        print("="*80)
        
        for state in tqdm(STATES, desc="States"):
            self.download_state_artifacts(state)
            time.sleep(0.5)  # Be nice to GitHub
        
        print(f"\n{'='*80}")
        print("DOWNLOAD COMPLETE")
        print(f"{'='*80}")
        print(f"Total artifacts downloaded: {len(self.artifacts)}")
        
        return self.artifacts


class DOSAConsolidator:
    """Consolidate and deduplicate DOSA artifacts"""
    
    def __init__(self, artifacts):
        self.artifacts = artifacts
    
    def deduplicate(self):
        """Deduplicate artifacts by (state, artifact_name)"""
        print("\nDeduplicating artifacts...")
        
        # Group by state and artifact name
        grouped = defaultdict(list)
        for artifact in self.artifacts:
            key = (artifact['state'], artifact['artifact_name'].lower())
            grouped[key].append(artifact)
        
        deduplicated = []
        for key, group in grouped.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge multiple entries
                # Prefer expanded over original, prefer CSV over TXT
                priority_order = ['expanded', 'original', 'individual_txt']
                sorted_group = sorted(group, 
                    key=lambda x: priority_order.index(x['source_type']) 
                    if x['source_type'] in priority_order else 999)
                
                merged = sorted_group[0].copy()
                
                # Combine all unique clues
                all_clues = []
                for item in group:
                    clues = item['clues'].split(' | ')
                    all_clues.extend(clues)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_clues = []
                for clue in all_clues:
                    clue_lower = clue.lower().strip()
                    if clue_lower not in seen:
                        seen.add(clue_lower)
                        unique_clues.append(clue)
                
                merged['clues'] = ' | '.join(unique_clues)
                merged['num_clues'] = len(unique_clues)
                merged['source_file'] = '; '.join([g['source_file'] for g in group])
                merged['merged_from'] = len(group)
                
                deduplicated.append(merged)
        
        print(f"   Before deduplication: {len(self.artifacts)}")
        print(f"   After deduplication: {len(deduplicated)}")
        
        return deduplicated
    
    def save_consolidated(self, artifacts, output_file):
        """Save consolidated artifacts to CSV"""
        df = pd.DataFrame(artifacts)
        
        # Add metadata columns
        df['category'] = 'cultural_artifact'
        df['source_dataset'] = 'dosa_github'
        
        # Reorder columns
        columns = ['state', 'artifact_name', 'clues', 'num_clues', 
                   'source_file', 'source_type', 'category', 'source_dataset']
        if 'merged_from' in df.columns:
            columns.insert(4, 'merged_from')
        
        df = df[columns]
        df = df.sort_values(['state', 'artifact_name'])
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Consolidated file saved: {output_file}")
        print(f"   Total unique artifacts: {len(df)}")
        
        return df


class DOSAStatementGenerator:
    """Generate natural language statements using LLM"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load Llama-3.2-3B-Instruct model"""
        print(f"\n{'='*80}")
        print("LOADING LLM MODEL")
        print(f"{'='*80}")
        print(f"Model path: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"✓ Model loaded successfully on {self.model.device}")
    
    def generate_statement(self, artifact_name, clues, state):
        """Convert clues to natural language statement"""
        
        system_prompt = """You are a helpful assistant that converts bullet-point cultural clues into natural, flowing paragraphs. Your task is to create a single coherent paragraph (2-4 sentences) that naturally describes the cultural artifact based on the given clues."""
        
        # Parse clues
        clue_list = [c.strip() for c in clues.split('|') if c.strip()]
        
        if not clue_list:
            return None
        
        clues_text = "\n".join([f"- {clue}" for clue in clue_list])
        
        user_prompt = f"""State: {state.replace('_', ' ').title()}
Artifact: {artifact_name}

Clues:
{clues_text}

Convert these clues into a natural paragraph (2-4 sentences) that describes this cultural artifact. Write in a flowing, descriptive style. Do not include the artifact name in your description."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "assistant" in generated:
            response = generated.split("assistant")[-1].strip()
        else:
            response = generated[len(input_text):].strip()
        
        return response
    
    def generate_all_statements(self, consolidated_df, output_file):
        """Generate statements for all artifacts"""
        print(f"\n{'='*80}")
        print("GENERATING NATURAL LANGUAGE STATEMENTS")
        print(f"{'='*80}")
        print(f"Total artifacts to process: {len(consolidated_df)}")
        
        results = []
        failed = 0
        
        for idx, row in tqdm(consolidated_df.iterrows(), 
                            total=len(consolidated_df), 
                            desc="Generating statements"):
            try:
                statement = self.generate_statement(
                    row['artifact_name'],
                    row['clues'],
                    row['state']
                )
                
                if statement:
                    results.append({
                        'state': row['state'],
                        'artifact_name': row['artifact_name'],
                        'converted_text': statement,
                        'original_clues': row['clues'],
                        'num_clues': row['num_clues'],
                        'source_file': row['source_file'],
                        'source_type': row['source_type'],
                        'category': 'cultural_validation',
                        'source': 'dosa_github'
                    })
                else:
                    failed += 1
            
            except Exception as e:
                print(f"\n   Error processing {row['artifact_name']}: {e}")
                failed += 1
        
        print(f"\n✓ Successfully generated: {len(results)} statements")
        print(f"✗ Failed: {failed}")
        
        # Save to CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        
        print(f"✓ Statements saved to: {output_file}")
        
        return df_results


def generate_report(downloader, consolidated_df, statements_df, output_file):
    """Generate processing report"""
    report = {
        'total_artifacts_downloaded': len(downloader.artifacts),
        'total_unique_artifacts': len(consolidated_df),
        'total_statements_generated': len(statements_df),
        'by_state': {},
        'by_source_type': consolidated_df['source_type'].value_counts().to_dict(),
        'statistics': downloader.stats,
        'output_files': {
            'consolidated': str(CONSOLIDATED_FILE),
            'cst_samples': str(CST_FILE),
            'report': str(REPORT_FILE)
        }
    }
    
    # Per-state statistics
    for state in STATES:
        state_artifacts = consolidated_df[consolidated_df['state'] == state]
        state_statements = statements_df[statements_df['state'] == state]
        report['by_state'][state] = {
            'artifacts': len(state_artifacts),
            'statements': len(state_statements)
        }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {output_file}")
    
    return report


def main():
    """Main pipeline"""
    print("="*80)
    print("DOSA COMPLETE PROCESSING PIPELINE")
    print("="*80)
    print("\nPhase 1: Download all artifacts")
    print("Phase 2: Consolidate and deduplicate")
    print("Phase 3: Generate natural language statements")
    print("="*80)
    
    # Phase 1: Download
    downloader = DOSADownloader()
    artifacts = downloader.download_all()
    
    if not artifacts:
        print("\nERROR: No artifacts downloaded!")
        sys.exit(1)
    
    # Phase 2: Consolidate
    consolidator = DOSAConsolidator(artifacts)
    deduplicated = consolidator.deduplicate()
    consolidated_df = consolidator.save_consolidated(deduplicated, CONSOLIDATED_FILE)
    
    # Phase 3: Generate statements
    generator = DOSAStatementGenerator(MODEL_PATH)
    generator.load_model()
    statements_df = generator.generate_all_statements(consolidated_df, CST_FILE)
    
    # Generate report
    report = generate_report(downloader, consolidated_df, statements_df, REPORT_FILE)
    
    # Final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"   Total artifacts downloaded: {len(artifacts)}")
    print(f"   Unique artifacts consolidated: {len(consolidated_df)}")
    print(f"   Statements generated: {len(statements_df)}")
    print(f"\n📁 Output files:")
    print(f"   1. Consolidated original: {CONSOLIDATED_FILE}")
    print(f"   2. CST samples: {CST_FILE}")
    print(f"   3. Processing report: {REPORT_FILE}")
    print(f"\n✓ All artifacts by state:")
    for state in STATES:
        count = report['by_state'][state]['artifacts']
        print(f"   {state.replace('_', ' ').title():20} {count:3} artifacts")
    
    print(f"\n{'='*80}")
    
    return consolidated_df, statements_df, report


if __name__ == "__main__":
    try:
        consolidated_df, statements_df, report = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
