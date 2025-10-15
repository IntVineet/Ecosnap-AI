#!/usr/bin/env python3
"""
Train Waste Detection Model with Real Data
This script trains a YOLOv8 model specifically for waste detection
"""

import sys
from pathlib import Path
import shutil
from ultralytics import YOLO
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'taco'
YOLO_DATA = DATA_ROOT / 'yolo'
MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'train_taco'

# Waste categories
WASTE_CLASSES = ['plastic', 'metal', 'paper']

def create_realistic_training_data(num_images=50):
    """Create realistic synthetic training data with better variety"""
    print("🎨 Creating realistic training images...")
    
    images_dir = YOLO_DATA / 'images' / 'train'
    labels_dir = YOLO_DATA / 'labels' / 'train'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files
    for f in images_dir.glob('*.jpg'):
        f.unlink()
    for f in labels_dir.glob('*.txt'):
        f.unlink()
    
    created_count = 0
    
    for i in range(num_images):
        # Create more realistic image with varied backgrounds
        img = np.ones((640, 640, 3), dtype=np.uint8)
        
        # Varied backgrounds
        if i % 4 == 0:
            # White background
            img[:] = (240, 240, 240)
        elif i % 4 == 1:
            # Gray background
            img[:] = (180, 180, 180)
        elif i % 4 == 2:
            # Light blue background (simulating outdoor)
            img[:] = (220, 200, 180)
        else:
            # Gradient background
            for y in range(640):
                color_val = int(200 + (y / 640) * 50)
                img[y, :] = (color_val, color_val - 20, color_val - 10)
        
        # Add noise for realism
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create label file content
        labels = []
        
        # Add 2-5 objects per image
        num_objects = np.random.randint(2, 6)
        
        for _ in range(num_objects):
            class_id = np.random.randint(0, 3)  # 0=plastic, 1=metal, 2=paper
            
            # Random size and position
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.15, 0.4)
            cx = np.random.uniform(width/2, 1 - width/2)
            cy = np.random.uniform(height/2, 1 - height/2)
            
            # Draw realistic-looking object
            x1 = int((cx - width/2) * 640)
            y1 = int((cy - height/2) * 640)
            x2 = int((cx + width/2) * 640)
            y2 = int((cy + height/2) * 640)
            
            # Different shapes and colors based on class
            if class_id == 0:  # Plastic (bottles, containers)
                # Draw bottle-like shape
                color = (np.random.randint(50, 150), np.random.randint(150, 255), np.random.randint(150, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                # Add cap
                cap_height = int((y2 - y1) * 0.2)
                cv2.rectangle(img, (x1 + 10, y1 - cap_height), (x2 - 10, y1), 
                            (color[0] - 30, color[1] - 30, color[2] - 30), -1)
                # Add label area
                label_y = y1 + int((y2 - y1) * 0.4)
                cv2.rectangle(img, (x1 + 5, label_y), (x2 - 5, label_y + 20), 
                            (255, 255, 255), -1)
                
            elif class_id == 1:  # Metal (cans)
                # Draw can-like shape
                color = (np.random.randint(150, 200), np.random.randint(150, 200), np.random.randint(150, 200))
                cv2.ellipse(img, ((x1 + x2)//2, (y1 + y2)//2), 
                          ((x2 - x1)//2, (y2 - y1)//2), 0, 0, 360, color, -1)
                # Add shine effect
                shine_x = x1 + int((x2 - x1) * 0.3)
                cv2.line(img, (shine_x, y1 + 10), (shine_x, y2 - 10), (255, 255, 255), 3)
                # Add top rim
                cv2.ellipse(img, ((x1 + x2)//2, y1 + 10), 
                          ((x2 - x1)//2, 5), 0, 0, 360, (200, 200, 200), -1)
                
            else:  # Paper (sheets, documents)
                # Draw paper-like shape
                color = (np.random.randint(200, 255), np.random.randint(200, 255), np.random.randint(200, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                # Add text lines
                for line_y in range(y1 + 20, y2 - 10, 15):
                    cv2.line(img, (x1 + 10, line_y), (x2 - 10, line_y), 
                           (100, 100, 100), 1)
                # Add corner fold
                fold_size = min(20, (x2 - x1) // 4)
                pts = np.array([[x2, y1], [x2 - fold_size, y1], [x2, y1 + fold_size]], np.int32)
                cv2.fillPoly(img, [pts], (150, 150, 150))
            
            # Add slight shadow
            shadow_offset = 5
            cv2.rectangle(img, (x1 + shadow_offset, y1 + shadow_offset), 
                        (x2 + shadow_offset, y2 + shadow_offset), (0, 0, 0), 2)
            
            # Save label in YOLO format
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
        
        # Save image
        img_path = images_dir / f'realistic_{i:04d}.jpg'
        cv2.imwrite(str(img_path), img)
        
        # Save labels
        label_path = labels_dir / f'realistic_{i:04d}.txt'
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        created_count += 1
        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{num_images} images...")
    
    print(f"✅ Created {created_count} realistic training images with labels")
    return created_count

def create_dataset_yaml():
    """Create dataset YAML file"""
    yaml_content = f"""path: {YOLO_DATA}
train: images/train
val: images/train
names:
  0: plastic
  1: metal
  2: paper
"""
    
    yaml_path = YOLO_DATA / 'taco.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created dataset YAML: {yaml_path}")
    return yaml_path

def train_model(yaml_path, epochs=10, imgsz=640, batch=16):
    """Train YOLO model"""
    print("\n" + "="*60)
    print("🚀 TRAINING WASTE DETECTION MODEL (FAST MODE)")
    print("="*60)
    
    # Initialize model
    print("📦 Loading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')
    
    print(f"\n📊 Training Configuration:")
    print(f"  • Epochs: {epochs}")
    print(f"  • Batch Size: {batch}")
    print(f"  • Image Size: {imgsz}")
    print(f"  • Dataset: {yaml_path}")
    print(f"  • Classes: {', '.join(WASTE_CLASSES)}")
    
    # Train model
    print(f"\n🏋️ Starting training...\n")
    
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='train_taco',
            project=str(PROJECT_ROOT / 'models'),
            patience=10,
            save=True,
            plots=True,
            verbose=True,
            device='cpu'  # Use CPU for compatibility
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Get model path
        model_path = PROJECT_ROOT / 'models' / 'train_taco' / 'weights' / 'best.pt'
        
        if model_path.exists():
            print(f"\n📦 Trained model saved to:")
            print(f"   {model_path}")
            print(f"\n📊 Model size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
            return model_path
        else:
            print(f"\n⚠️  Model file not found at expected location")
            return None
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model(model_path):
    """Test the trained model"""
    if not model_path or not model_path.exists():
        print("⚠️  No model to test")
        return
    
    print("\n" + "="*60)
    print("🧪 TESTING TRAINED MODEL")
    print("="*60)
    
    # Load model
    model = YOLO(str(model_path))
    
    # Test on a few training images
    test_images = list((YOLO_DATA / 'images' / 'train').glob('*.jpg'))[:3]
    
    print(f"\n📸 Running inference on {len(test_images)} test images...\n")
    
    for img_path in test_images:
        results = model(str(img_path), verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  • {img_path.name}: {detections} objects detected")
    
    print("\n✅ Model testing completed!")

def main():
    """Main training workflow"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🗑️  WASTE DETECTION MODEL TRAINING                       ║
    ║      Train YOLOv8 for Plastic, Metal, Paper Detection    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("📋 Training Workflow:")
    print("  1. Using REAL TACO dataset images")
    print("  2. Prepare dataset configuration")
    print("  3. Train YOLOv8 model")
    print("  4. Test trained model")
    print("")
    
    # Check if TACO dataset exists
    yaml_path = YOLO_DATA / 'taco.yaml'
    if not yaml_path.exists():
        print("❌ TACO dataset not found. Please run:")
        print("   python3 convert_taco_to_yolo.py")
        return
    
    # Count existing images
    images_dir = YOLO_DATA / 'images' / 'train'
    labels_dir = YOLO_DATA / 'labels' / 'train'
    num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.JPG')))
    num_labels = len(list(labels_dir.glob('*.txt')))
    
    print(f"✅ TACO Dataset Ready:")
    print(f"   • Images: {num_images}")
    print(f"   • Labels: {num_labels}")
    print("")
    
    # Train model (reduced epochs for faster training)
    model_path = train_model(yaml_path, epochs=10, batch=16)
    
    # Test model
    if model_path:
        test_model(model_path)
        
        print("\n" + "="*60)
        print("🎉 TRAINING WORKFLOW COMPLETE!")
        print("="*60)
        print(f"\n✅ Your custom waste detection model is ready!")
        print(f"📦 Model location: {model_path}")
        print(f"\n🚀 Next step: Run camera detection")
        print(f"   python3 quick_start.py")
        print("\n💡 The camera will now use your trained model!")
    else:
        print("\n❌ Training workflow failed. Please check errors above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
