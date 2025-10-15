#!/usr/bin/env python3
"""
Test the trained waste detection model on sample images
"""

from ultralytics import YOLO
from pathlib import Path
import cv2

def test_model():
    """Test the trained model"""
    print("=" * 60)
    print("🧪 TESTING WASTE DETECTION MODEL")
    print("=" * 60)
    
    # Load model
    model_path = Path('models/train_taco4/weights/best.pt')
    
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        return
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"✅ Model loaded successfully!")
    print(f"\n📊 Model Information:")
    print(f"   Classes: {model.names}")
    print(f"   Number of classes: {len(model.names)}")
    print(f"   Task: {model.task}")
    
    # Test on training images
    print(f"\n🖼️  Testing on sample training images...")
    image_dir = Path('data/taco/yolo/images/train')
    
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    # Get first 5 images
    images = list(image_dir.glob('*.jpg'))[:5]
    
    if not images:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"   Found {len(images)} test images\n")
    
    for i, img_path in enumerate(images, 1):
        print(f"   [{i}] Testing: {img_path.name}")
        
        # Run detection
        results = model(str(img_path), conf=0.25, verbose=False)[0]
        
        # Count detections by class
        if results.boxes is not None and len(results.boxes) > 0:
            detections = {}
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                if class_name not in detections:
                    detections[class_name] = []
                detections[class_name].append(conf)
            
            # Print results
            for class_name, confs in detections.items():
                avg_conf = sum(confs) / len(confs)
                print(f"       • {class_name.capitalize()}: {len(confs)} detections (avg conf: {avg_conf:.2f})")
        else:
            print(f"       • No detections")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TEST COMPLETE")
    print("=" * 60)
    print("\n💡 Tips:")
    print("   • If detections look good, run: python3 quick_start.py")
    print("   • Point camera at plastic bottles, metal cans, or paper")
    print("   • Model expects waste objects similar to training data")
    print("   • Ensure good lighting for best results")

if __name__ == "__main__":
    test_model()
