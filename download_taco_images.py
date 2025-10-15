#!/usr/bin/env python3
"""
Download TACO dataset images using official script
"""
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'taco'

print("📦 Downloading TACO dataset images using official script...")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Download the official script if not present
script_path = PROJECT_ROOT / 'download.py'
if not script_path.exists():
    import requests
    url = 'https://raw.githubusercontent.com/pedropro/TACO/master/download.py'
    print(f"Downloading official download.py script...")
    r = requests.get(url)
    with open(script_path, 'wb') as f:
        f.write(r.content)
    print(f"✅ Downloaded: {script_path}")
else:
    print(f"✅ Official download.py script already exists.")

# Run the official download script
try:
    subprocess.run(['python3', str(script_path)], cwd=str(PROJECT_ROOT), check=True)
    print("🎉 TACO images downloaded!")
except Exception as e:
    print(f"❌ Error running download.py: {e}")
