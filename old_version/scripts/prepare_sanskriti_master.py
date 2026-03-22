#!/usr/bin/env python3

import os
import glob
import pandas as pd

BASE_DIR = "/data/user_data/anshulk/cultural-alignment-study/sanskriti_data"
OUTPUT_CSV = os.path.join(BASE_DIR, "sanskriti_qa_master.csv")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "sanskriti_qa_master.parquet")

# Map any possible existing column names to a clean standard
COL_RENAME_MAP = {
    "state": "state",
    "State": "state",
    "STATE": "state",

    "attribute": "attribute",
    "Attribute": "attribute",
    "ATTRIBUTE": "attribute",

    "question": "question",
    "Question": "question",
    "QUESTION": "question",

    "option1": "option1",
    "Option1": "option1",
    "OPTION1": "option1",
    "option_1": "option1",

    "option2": "option2",
    "Option2": "option2",
    "OPTION2": "option2",
    "option_2": "option2",

    "option3": "option3",
    "Option3": "option3",
    "OPTION3": "option3",
    "option_3": "option3",

    "option4": "option4",
    "Option4": "option4",
    "OPTION4": "option4",
    "option_4": "option4",

    "answer": "answer",
    "Answer": "answer",
    "ANSWER": "answer",

    "Short Answer": "short_answer",
    "short_answer": "short_answer",
    "Short_Answer": "short_answer",
    "shortAnswer": "short_answer",
}

STANDARD_COLS = [
    "state",
    "attribute",
    "question",
    "option1",
    "option2",
    "option3",
    "option4",
    "answer",
    "short_answer",
]

def load_and_merge():
    csv_paths = sorted(glob.glob(os.path.join(BASE_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {BASE_DIR}")

    dfs = []
    for path in csv_paths:
        print(f"Reading: {path}")
        df = pd.read_csv(path)

        # Normalize column names first
        df = df.rename(columns={c: COL_RENAME_MAP.get(c, c) for c in df.columns})

        # Only keep columns we care about (if extras exist)
        missing = [c for c in STANDARD_COLS if c not in df.columns]
        if missing:
            print(f"Warning: {path} is missing columns: {missing}")

        # Add any missing standard columns as empty if necessary
        for c in STANDARD_COLS:
            if c not in df.columns:
                df[c] = pd.NA

        df = df[STANDARD_COLS]
        dfs.append(df)

    # Concatenate
    full = pd.concat(dfs, ignore_index=True)

    # Strip whitespace in important string columns
    for col in ["state", "attribute", "question",
                "option1", "option2", "option3", "option4",
                "answer", "short_answer"]:
        if col in full.columns and pd.api.types.is_string_dtype(full[col]):
            full[col] = full[col].astype(str).str.strip()

    # Drop exact duplicates
    before = len(full)
    full = full.drop_duplicates().reset_index(drop=True)
    after = len(full)
    print(f"Dropped {before - after} duplicate rows. Final rows: {after}")

    # Add QA ID
    full.insert(0, "qa_id", range(len(full)))

    return full

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    df = load_and_merge()

    print(f"Saving master CSV to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saving master Parquet to: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Preview:")
    print(df.head())

if __name__ == "__main__":
    main()
