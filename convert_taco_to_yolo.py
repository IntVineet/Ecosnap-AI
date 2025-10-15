#!/usr/bin/env python3
"""
Convert TACO dataset from COCO format to YOLO format
Maps TACO categories to 3 main classes: plastic, metal, paper
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
ANNOTATIONS_PATH = DATA_DIR / 'annotations.json'
YOLO_DIR = DATA_DIR / 'taco' / 'yolo'
IMAGES_DIR = YOLO_DIR / 'images' / 'train'
LABELS_DIR = YOLO_DIR / 'labels' / 'train'

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

print("🔄 Converting TACO dataset to YOLO format...")
print(f"📂 Reading annotations from: {ANNOTATIONS_PATH}")

# Load TACO annotations
with open(ANNOTATIONS_PATH, 'r') as f:
    taco_data = json.load(f)

# TACO category mapping to our 3 main classes
# Based on TACO category names, map to: 0=plastic, 1=metal, 2=paper
CATEGORY_MAPPING = {
    # Plastic items (class 0)
    'Plastic bag & wrapper': 0,
    'Bottle': 0,
    'Bottle cap': 0,
    'Cup': 0,
    'Lid': 0,
    'Straw': 0,
    'Food wrapper': 0,
    'Container': 0,
    'Plastic film': 0,
    'Styrofoam piece': 0,
    'Plastic': 0,
    'Plastic container': 0,
    'Plastic bottle': 0,
    'Plastic cup': 0,
    'Disposable plastic cup': 0,
    'Other plastic': 0,
    'Crisp packet': 0,
    'Plastic utensils': 0,
    'Plastic glooves': 0,
    'Six pack rings': 0,
    'Garbage bag': 0,
    
    # Metal items (class 1)
    'Can': 1,
    'Metal': 1,
    'Aluminium foil': 1,
    'Drink can': 1,
    'Aerosol': 1,
    'Pop tab': 1,
    'Scrap metal': 1,
    'Metal bottle cap': 1,
    'Metal lid': 1,
    'Aluminium blister pack': 1,
    
    # Paper items (class 2)
    'Paper': 2,
    'Carton': 2,
    'Magazine paper': 2,
    'Tissues': 2,
    'Wrapping paper': 2,
    'Paper bag': 2,
    'Toilet tube': 2,
    'Other plastic wrapper': 2,  # Some overlap, but mostly paper-based
    'Meal carton': 2,
    'Pizza box': 2,
    'Paper cup': 2,
    'Egg carton': 2,
    'Newspaper': 2,
    'Cardboard': 2,
    'Corrugated carton': 2,
}

# Class names for YOLO
CLASS_NAMES = ['plastic', 'metal', 'paper']

print(f"\n📊 Category Mapping:")
print(f"   • Class 0 (Plastic): {sum(1 for v in CATEGORY_MAPPING.values() if v == 0)} TACO categories")
print(f"   • Class 1 (Metal): {sum(1 for v in CATEGORY_MAPPING.values() if v == 1)} TACO categories")
print(f"   • Class 2 (Paper): {sum(1 for v in CATEGORY_MAPPING.values() if v == 2)} TACO categories")

# Create category ID to name mapping
cat_id_to_name = {}
for cat in taco_data['categories']:
    cat_id_to_name[cat['id']] = cat['name']

# Create image ID to file mapping
img_id_to_info = {}
for img in taco_data['images']:
    img_id_to_info[img['id']] = {
        'file_name': img['file_name'],
        'width': img['width'],
        'height': img['height']
    }

# Group annotations by image
annotations_by_image = defaultdict(list)
for ann in taco_data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

print(f"\n🖼️  Processing {len(img_id_to_info)} images...")

# Convert annotations to YOLO format
converted_count = 0
skipped_count = 0
class_counts = {0: 0, 1: 0, 2: 0}

for img_id, img_info in img_id_to_info.items():
    if img_id not in annotations_by_image:
        continue
    
    img_width = img_info['width']
    img_height = img_info['height']
    img_filename = img_info['file_name']
    
    # Check if image file exists (TACO images are in batch_* subdirectories)
    source_img_path = DATA_DIR / img_filename
    if not source_img_path.exists():
        # Try alternate path without nested directory structure
        source_img_path = DATA_DIR / Path(img_filename).name
        if not source_img_path.exists():
            skipped_count += 1
            continue
    
    # Prepare YOLO labels
    yolo_labels = []
    
    for ann in annotations_by_image[img_id]:
        cat_id = ann['category_id']
        cat_name = cat_id_to_name.get(cat_id, 'Unknown')
        
        # Map to our 3 classes
        if cat_name not in CATEGORY_MAPPING:
            continue  # Skip unmapped categories
        
        class_id = CATEGORY_MAPPING[cat_name]
        class_counts[class_id] += 1
        
        # Get bounding box in COCO format [x, y, width, height]
        bbox = ann['bbox']
        x, y, w, h = bbox
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        cx = (x + w / 2) / img_width
        cy = (y + h / 2) / img_height
        nw = w / img_width
        nh = h / img_height
        
        # Ensure values are within [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))
        
        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    
    if not yolo_labels:
        continue  # Skip images with no mapped annotations
    
    # Copy image to YOLO directory (flatten directory structure)
    target_img_filename = Path(img_filename).name
    target_img_path = IMAGES_DIR / target_img_filename
    if not target_img_path.exists():
        shutil.copy2(source_img_path, target_img_path)
    
    # Write YOLO label file
    label_filename = target_img_path.stem + '.txt'
    label_path = LABELS_DIR / label_filename
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_labels))
    
    converted_count += 1
    if converted_count % 100 == 0:
        print(f"   Converted {converted_count} images...")

print(f"\n✅ Conversion Complete!")
print(f"   • Images converted: {converted_count}")
print(f"   • Images skipped (not downloaded): {skipped_count}")
print(f"\n📊 Annotation Distribution:")
print(f"   • Plastic: {class_counts[0]} annotations")
print(f"   • Metal: {class_counts[1]} annotations")
print(f"   • Paper: {class_counts[2]} annotations")

# Create dataset YAML file
yaml_path = YOLO_DIR / 'taco.yaml'
yaml_content = f"""path: {YOLO_DIR}
train: images/train
val: images/train
names:
  0: plastic
  1: metal
  2: paper
"""

with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n📄 Created YOLO dataset config: {yaml_path}")
print(f"\n🎉 TACO dataset is ready for YOLO training!")
print(f"\n📂 Dataset structure:")
print(f"   • Images: {IMAGES_DIR}")
print(f"   • Labels: {LABELS_DIR}")
print(f"   • Config: {yaml_path}")
print(f"\n🚀 Next step: Run training with:")
print(f"   python3 train_waste_model.py")
