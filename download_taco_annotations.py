#!/usr/bin/env python3
"""
Download TACO annotations.json for image download
"""
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
ANNOTATIONS_PATH = DATA_DIR / 'annotations.json'

TACO_ANNOTATIONS_URL = 'https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json'

print("📦 Downloading TACO annotations.json...")

if not ANNOTATIONS_PATH.exists():
    r = requests.get(TACO_ANNOTATIONS_URL)
    with open(ANNOTATIONS_PATH, 'wb') as f:
        f.write(r.content)
    print(f"✅ Downloaded: {ANNOTATIONS_PATH}")
else:
    print(f"✅ Already exists: {ANNOTATIONS_PATH}")
