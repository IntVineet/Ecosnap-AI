#!/usr/bin/env python3
"""
Download and prepare the official TACO dataset for YOLO training
"""
import os
import requests
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'taco'
TACO_URL = 'https://github.com/pedropro/TACO/releases/download/v1.0.0/taco_dataset.zip'
TACO_ZIP = DATA_ROOT / 'taco_dataset.zip'
TACO_EXTRACTED = DATA_ROOT / 'taco_dataset'

# Download TACO dataset
print("📦 Downloading TACO dataset...")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

if not TACO_ZIP.exists():
    with requests.get(TACO_URL, stream=True) as r:
        r.raise_for_status()
        with open(TACO_ZIP, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Downloaded: {TACO_ZIP}")
else:
    print(f"✅ Already downloaded: {TACO_ZIP}")

# Extract TACO dataset
if not TACO_EXTRACTED.exists():
    print("📂 Extracting TACO dataset...")
    with zipfile.ZipFile(TACO_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_ROOT)
    print(f"✅ Extracted to: {TACO_EXTRACTED}")
else:
    print(f"✅ Already extracted: {TACO_EXTRACTED}")

print("🎉 TACO dataset is ready!")
