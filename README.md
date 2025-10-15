# GenAI_Jovac: Multi-Dataset Waste Detection with YOLO

This project provides an end-to-end workflow for preparing multiple waste-related datasets (TACO, COCO subset, RecycleNet placeholders), converting annotations to YOLO format, training YOLOv8 models, merging datasets, and performing live camera or image detection with enriched environmental metadata.

## Features
- Unified helper utilities in `main_utils.py`
- Modular Jupyter notebooks for each stage:
  1. `1_prepare_taco.ipynb`
  2. `2_prepare_coco.ipynb`
  3. `3_prepare_recyclenet.ipynb`
  4. `4_merge_datasets.ipynb`
  5. `5_final_camera_detection.ipynb`
- Placeholder dataset support so code executes even without full downloads
- YOLOv8 training wrappers
- Detection JSON export with environmental impact info

## Folder Structure
```
GenAI_Jovac/
  data/
    taco/
    coco/
    recyclenet/
  models/
  notebooks/
  main_utils.py
  requirements.txt
```

## Quick Start
Install dependencies (Python 3.9+ recommended). Run the setup script which creates a virtual environment and installs packages:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(Optionally set up Kaggle credentials for real dataset downloads.)

## Running the Workflow
Open notebooks in order inside `notebooks/`:
1. Prepare TACO
2. Prepare COCO subset
3. Prepare RecycleNet
4. Merge datasets & fine-tune
5. Run camera / image detection

Each notebook is designed to run even if real images are missing (uses stubs). For real training, replace placeholder annotations with actual datasets and ensure `images/` directories contain matching image files.

## Kaggle / Dataset Notes
Provide the actual dataset zip files or implement the download logic in `main_utils.py` where placeholder functions exist (`download_taco`, `download_coco_subset`, `download_recyclenet`). Ensure annotation JSON follows COCO-like structure.

## Training Parameters
Adjust epochs, batch size, and model name inside notebooks. Defaults are small for rapid iteration.

## Detection JSON Output
Example:
```json
{
  "metal": {
    "decompose_time": "50 years",
    "harm": "Energy-intensive production pollutes air and water.",
    "recycle_tip": "Recycle into cans, car parts, and tools.
",
    "timestamp": "2025-01-01T12:00:00"
  }
}
```

## Extending
- Replace placeholder downloads with actual HTTP or Kaggle API logic.
- Add validation dataset splits.
- Introduce advanced augmentation and hyperparameter tuning.
- Integrate a lightweight web UI (Streamlit / Gradio) for interactive detection.

## Environment Variables
Set these for production usage:
- `KAGGLE_USERNAME` / `KAGGLE_KEY` for Kaggle API

## Disclaimer
Current repo contains placeholder dataset logic for demonstration. Real performance requires full datasets with images.

## License
MIT (adjust as needed).
