"""Main utility helpers for GenAI_Jovac project.

This module centralizes dataset preparation, annotation conversion,
YOLO training, model inference and detection JSON export.

Design goals:
 - Fail gracefully with clear messages
 - Be runnable on CPU if GPU not available
 - Keep heavy imports (torch/ultralytics) inside functions to reduce import overhead
 - Allow notebooks to orchestrate high-level workflows
"""
from __future__ import annotations

import os
import json
import shutil
import time
import random
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("GenAI_Jovac")


# --------------------------------------------------------------------------------------
# General filesystem helpers
# --------------------------------------------------------------------------------------
def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_write_text(path: Path | str, text: str):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def has_gpu() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Dataset download placeholders
# --------------------------------------------------------------------------------------
def download_taco(dest: Path) -> None:
    """Download the TACO dataset if not already present.

    Actual TACO dataset (Trash Annotations in Context) is hosted on GitHub.
    To keep this script self-contained offline, we only scaffold expected structure
    if dataset not found. Users can manually place dataset or set TACO_ZIP env path.
    """
    ensure_dir(dest)
    ann_file = dest / "annotations.json"
    if ann_file.exists():
        logger.info("TACO dataset already present: %s", dest)
        return
    logger.warning("TACO dataset not found. Creating placeholder structure at %s", dest)
    # Create placeholder minimal annotation file
    placeholder = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "plastic"},
            {"id": 2, "name": "metal"},
            {"id": 3, "name": "paper"},
        ],
    }
    safe_write_text(ann_file, json.dumps(placeholder, indent=2))


def download_coco_subset(dest: Path, year: str = "2017", max_images: int = 200) -> None:
    """Download a small COCO subset for experimentation.

    Full COCO is large; we sample a subset of annotations (if pycocotools available).
    If download not feasible, create a placeholder similar to TACO stub.
    """
    ensure_dir(dest)
    ann_file = dest / f"instances_train{year}.json"
    if ann_file.exists():
        logger.info("COCO subset already present: %s", ann_file)
        return
    logger.warning("COCO subset not found. Creating placeholder subset at %s", ann_file)
    placeholder = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "bottle"},
            {"id": 2, "name": "cup"},
            {"id": 3, "name": "fork"},
            {"id": 4, "name": "knife"},
            {"id": 5, "name": "spoon"},
        ],
    }
    safe_write_text(ann_file, json.dumps(placeholder, indent=2))


def download_recyclenet(dest: Path) -> None:
    """Placeholder for RecycleNet dataset acquisition.
    Provide stub annotation categories.
    """
    ensure_dir(dest)
    ann_file = dest / "annotations.json"
    if ann_file.exists():
        logger.info("RecycleNet dataset already present: %s", dest)
        return
    logger.warning("RecycleNet dataset not found. Creating placeholder structure at %s", dest)
    placeholder = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "glass"},
            {"id": 2, "name": "cardboard"},
            {"id": 3, "name": "plastic"},
        ],
    }
    safe_write_text(ann_file, json.dumps(placeholder, indent=2))


# --------------------------------------------------------------------------------------
# Annotation conversion (COCO-like -> YOLO)
# --------------------------------------------------------------------------------------
def coco_to_yolo_bbox(width: float, height: float, bbox: List[float]) -> Tuple[float, float, float, float]:
    """Convert COCO bbox [x,y,w,h] to YOLO (cx, cy, w, h) normalized."""
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return cx / width, cy / height, w / width, h / height


def convert_coco_like_to_yolo(images: List[dict], annotations: List[dict], categories: List[dict],
                              images_dir: Path, yolo_out_dir: Path) -> Dict[int, str]:
    """Convert a COCO-like set of dictionaries to YOLO format labels.

    Returns mapping of category id to name.
    """
    ensure_dir(yolo_out_dir / 'images' / 'train')
    ensure_dir(yolo_out_dir / 'labels' / 'train')

    cat_id_to_name = {c['id']: c['name'] for c in categories}
    img_id_to_info = {img['id']: img for img in images}
    img_to_anns: Dict[int, List[dict]] = {}
    for ann in annotations:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)

    converted = 0
    for img_id, anns in img_to_anns.items():
        img_info = img_id_to_info.get(img_id)
        if not img_info:
            continue
        file_name = img_info.get('file_name') or f"{img_id}.jpg"
        width = img_info.get('width') or 1
        height = img_info.get('height') or 1
        label_lines = []
        for ann in anns:
            bbox = ann.get('bbox')
            cat_id = ann.get('category_id')
            if not bbox or cat_id not in cat_id_to_name:
                continue
            cx, cy, w, h = coco_to_yolo_bbox(width, height, bbox)
            class_index = list(cat_id_to_name.keys()).index(cat_id)
            label_lines.append(f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not label_lines:
            continue
        # Only write labels; copying images is optional (placeholder dataset)
        label_path = yolo_out_dir / 'labels' / 'train' / (Path(file_name).stem + '.txt')
        safe_write_text(label_path, "\n".join(label_lines))
        converted += 1
    logger.info("Converted %d images to YOLO labels (placeholder images may be missing)", converted)
    return cat_id_to_name


def build_dataset_yaml(name: str, yolo_dir: Path, classes: List[str]) -> Path:
    """Create a minimal dataset YAML for YOLO training."""
    yaml_content = (
        f"path: {yolo_dir}\n"  # root folder
        f"train: images/train\n"  # relative to path
        f"val: images/train\n"  # reuse train as val in placeholder scenario
        f"names: {classes}\n"
    )
    yaml_path = yolo_dir / f"{name}.yaml"
    safe_write_text(yaml_path, yaml_content)
    return yaml_path


# --------------------------------------------------------------------------------------
# YOLO training wrappers
# --------------------------------------------------------------------------------------
def train_yolo(dataset_yaml: Path, project_dir: Path, model_name: str = "yolov8n.pt", epochs: int = 1,
               imgsz: int = 640, batch: int = 4) -> Optional[Path]:
    """Train a YOLOv8 model with ultralytics. Returns path to best model or None on failure.

    Epochs default to 1 for quick placeholder training if dataset tiny.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        logger.error("Ultralytics not installed: %s", e)
        return None
    try:
        model = YOLO(model_name)
        results = model.train(data=str(dataset_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                              project=str(project_dir), name="train_" + dataset_yaml.stem,
                              pretrained=True, verbose=True, exist_ok=True)
        # Attempt to locate best model
        run_dir = Path(results.save_dir)
        best = run_dir / 'weights' / 'best.pt'
        if best.exists():
            logger.info("Training complete. Best model at %s", best)
            return best
        else:
            logger.warning("Best weights not found; returning last weights")
            last = run_dir / 'weights' / 'last.pt'
            return last if last.exists() else None
    except Exception as e:
        logger.exception("Training failed: %s", e)
        return None


# --------------------------------------------------------------------------------------
# Detection helpers
# --------------------------------------------------------------------------------------
ENV_INFO = {
    "plastic": {
        "decompose_time": "450 years",
        "harm": "Breaks into microplastics harming marine life and entering food chains.",
        "recycle_tip": "Rinse and place in appropriate plastic recycling stream.",
    },
    "metal": {
        "decompose_time": "50 years",
        "harm": "Energy-intensive production pollutes air and water.",
        "recycle_tip": "Recycle into cans, car parts, and tools.",
    },
    "paper": {
        "decompose_time": "2-6 weeks",
        "harm": "If not recycled, increases landfill methane emissions.",
        "recycle_tip": "Keep dry and clean for effective recycling.",
    },
    "glass": {
        "decompose_time": "Undetermined (can take thousands of years)",
        "harm": "Sharp fragments can injure wildlife; persists in environment.",
        "recycle_tip": "Recycle endlessly without quality loss.",
    },
    "cardboard": {
        "decompose_time": "2 months",
        "harm": "Takes space in landfill; can be recycled easily.",
        "recycle_tip": "Flatten boxes to save space and keep dry.",
    },
    "bottle": {
        "decompose_time": "450 years (plastic)",
        "harm": "Can leach chemicals; choking hazard for animals.",
        "recycle_tip": "Separate caps and crush to save space.",
    },
    "cup": {
        "decompose_time": "20-30 years (plastic coated)",
        "harm": "Composite materials reduce recyclability.",
        "recycle_tip": "Use reusable cups when possible.",
    },
    "fork": {
        "decompose_time": "100 years (plastic)",
        "harm": "Adds to plastic waste stream.",
        "recycle_tip": "Use metal reusable cutlery.",
    },
    "knife": {
        "decompose_time": "100 years (plastic)",
        "harm": "Adds to plastic waste stream.",
        "recycle_tip": "Use metal reusable cutlery.",
    },
    "spoon": {
        "decompose_time": "100 years (plastic)",
        "harm": "Adds to plastic waste stream.",
        "recycle_tip": "Use metal reusable cutlery.",
    },
}


def enrich_detection(label: str) -> Dict[str, str]:
    base = ENV_INFO.get(label.lower(), {
        "decompose_time": "Unknown",
        "harm": "No data available.",
        "recycle_tip": "Research local recycling guidelines.",
    })
    enriched = dict(base)
    enriched["timestamp"] = datetime.utcnow().isoformat(timespec='seconds')
    return enriched


def run_inference(model_path: Path, source: str | int = 0, conf: float = 0.25,
                  save: bool = False, save_dir: Optional[Path] = None):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        logger.error("Ultralytics not installed: %s", e)
        return []
    try:
        model = YOLO(str(model_path))
        results = model.predict(source=source, conf=conf, save=save, project=str(save_dir) if save_dir else None)
        return results
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        return []


def detections_to_json(results, skip_classes: Iterable[str] = ("person",), out_path: Path | None = None) -> Dict[str, Dict[str, str]]:
    """Convert YOLO results (ultralytics) to enriched JSON mapping, skipping specified classes."""
    output: Dict[str, Dict[str, str]] = {}
    try:
        for r in results:
            names = r.names
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for cls_id in boxes.cls.tolist():
                label = names.get(int(cls_id), str(cls_id))
                if label.lower() in (c.lower() for c in skip_classes):
                    continue
                if label not in output:  # one entry per class for summary
                    output[label] = enrich_detection(label)
    except Exception as e:
        logger.exception("Failed to parse detection results: %s", e)
    if out_path:
        safe_write_text(out_path, json.dumps(output, indent=2))
        logger.info("Saved detections JSON to %s", out_path)
    return output


# --------------------------------------------------------------------------------------
# Dataset merging
# --------------------------------------------------------------------------------------
def merge_label_dirs(yolo_dirs: List[Path], merged_dir: Path) -> None:
    """Merge YOLO directories (assumes same class ordering)."""
    for sub in ["images/train", "labels/train"]:
        ensure_dir(merged_dir / sub)
    for ydir in yolo_dirs:
        for sub in ["labels/train"]:  # skipping images if placeholders
            src = ydir / sub
            if not src.exists():
                continue
            for file in src.glob("*.txt"):
                dest = merged_dir / sub / file.name
                if dest.exists():
                    # Append lines to combine
                    dest.write_text(dest.read_text() + "\n" + file.read_text())
                else:
                    shutil.copy(file, dest)
    logger.info("Merged labels from %d datasets into %s", len(yolo_dirs), merged_dir)


__all__ = [
    'ensure_dir', 'download_taco', 'download_coco_subset', 'download_recyclenet',
    'convert_coco_like_to_yolo', 'build_dataset_yaml', 'train_yolo', 'run_inference',
    'detections_to_json', 'merge_label_dirs', 'has_gpu'
]
