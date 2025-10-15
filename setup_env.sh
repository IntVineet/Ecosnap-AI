#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python3 not found on PATH." >&2
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "[INFO] Creating virtual environment..."
  "$PYTHON_BIN" -m venv venv
fi

source venv/bin/activate

python -m pip install --upgrade pip

echo "[INFO] Installing requirements..."
pip install -r requirements.txt

echo "[INFO] Environment setup complete. Activate with: source venv/bin/activate"