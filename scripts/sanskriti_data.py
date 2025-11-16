#!/usr/bin/env python3
import os
import subprocess

# Target directory
TARGET_DIR = "/data/user_data/anshulk/cultural-alignment-study/sanskriti_data"

# Make sure directory exists
os.makedirs(TARGET_DIR, exist_ok=True)

# File IDs and desired output names
files = {
    "1AXez8mbPWVHqC3ukpkdAQA316y78W-7d": "sanskriti_part1.csv",
    "1dN5a-FtnJ9FsLOAjxyncCNE8NT0PFdvY": "sanskriti_part2.csv",
    "1hGtDN3teD7Q1MEVp7kkqeWwNE7Y3ELKo": "sanskriti_part3.csv",
    "1zpjLv6epsuwKe6H6BLexxCfuY7rBdSU3": "sanskriti_part4.csv",
}

def ensure_gdown():
    """Install gdown locally if not available."""
    try:
        subprocess.run(["gdown", "--version"], check=True, stdout=subprocess.DEVNULL)
        print("gdown already installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("gdown not found. Installing with pip...")
        subprocess.run(
            ["python", "-m", "pip", "install", "--user", "gdown"],
            check=True
        )
        print("gdown installed.")

def download_files():
    ensure_gdown()

    for file_id, filename in files.items():
        output_path = os.path.join(TARGET_DIR, filename)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"\nDownloading {filename} from {url} ...")
        subprocess.run(
            ["gdown", "--id", file_id, "-O", output_path],
            check=True
        )
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    download_files()
    print("\nAll downloads completed.")
