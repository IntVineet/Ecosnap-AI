#!/usr/bin/env python3
"""
Comprehensive diagnostic check for waste detection system
"""

from pathlib import Path
from ultralytics import YOLO
import sys

def check_system():
    """Run all diagnostic checks"""
    
    print("\n" + "=" * 70)
    print("🔍 WASTE DETECTION SYSTEM DIAGNOSTIC CHECK")
    print("=" * 70)
    
    all_passed = True
    
    # Check 1: Model file exists
    print("\n[1/5] Checking model file...")
    model_path = Path('models/train_taco4/weights/best.pt')
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Model file exists: {model_path}")
        print(f"   ✅ File size: {size_mb:.2f} MB")
    else:
        print(f"   ❌ Model file NOT found: {model_path}")
        all_passed = False
    
    # Check 2: Load model
    print("\n[2/5] Loading model...")
    try:
        model = YOLO(str(model_path))
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Model classes: {model.names}")
        
        # Verify classes
        expected_classes = {0: 'plastic', 1: 'metal', 2: 'paper'}
        if model.names == expected_classes:
            print(f"   ✅ Classes match expected: {expected_classes}")
        else:
            print(f"   ⚠️  Classes mismatch!")
            print(f"      Expected: {expected_classes}")
            print(f"      Got: {model.names}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        all_passed = False
        return all_passed
    
    # Check 3: Test images exist
    print("\n[3/5] Checking test images...")
    image_dir = Path('data/taco/yolo/images/train')
    if image_dir.exists():
        images = list(image_dir.glob('*.jpg'))
        print(f"   ✅ Image directory exists: {image_dir}")
        print(f"   ✅ Found {len(images)} training images")
    else:
        print(f"   ❌ Image directory NOT found: {image_dir}")
        all_passed = False
    
    # Check 4: Test detection on image
    print("\n[4/5] Testing detection on sample image...")
    if image_dir.exists() and images:
        test_img = images[0]
        print(f"   Testing on: {test_img.name}")
        
        try:
            results = model(str(test_img), conf=0.25, verbose=False)[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                print(f"   ✅ Detection working! Found {len(results.boxes)} objects")
                
                # Show what was detected
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    print(f"      • {class_name.capitalize()}: {conf:.2f} confidence")
            else:
                print(f"   ⚠️  No detections on this image (may be normal)")
        except Exception as e:
            print(f"   ❌ Detection failed: {e}")
            all_passed = False
    
    # Check 5: Camera detection script
    print("\n[5/5] Checking camera detection script...")
    camera_script = Path('camera_detection.py')
    if camera_script.exists():
        print(f"   ✅ Camera detection script exists")
        
        # Check if it imports correctly
        try:
            from camera_detection import WasteDetector
            print(f"   ✅ Script imports successfully")
        except Exception as e:
            print(f"   ❌ Import failed: {e}")
            all_passed = False
    else:
        print(f"   ❌ Camera detection script NOT found")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\n🎯 Your system is ready for camera detection!")
        print("\n📋 Next steps:")
        print("   1. Get actual waste objects (plastic bottle, metal can, cardboard)")
        print("   2. Run: python3 quick_start.py")
        print("   3. Hold objects 30-50cm from camera")
        print("   4. Ensure good lighting")
        print("\n💡 Remember:")
        print("   • Model detects WASTE, not general objects")
        print("   • Yellow box = Plastic")
        print("   • Blue box = Metal")
        print("   • Green box = Paper")
        print("   • Check confidence scores (>0.5 is reliable)")
        
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        print("\n⚠️  Please fix the issues above before using camera detection")
        print("\nIf you need to retrain the model:")
        print("   python3 train_waste_model.py")
    
    print("\n" + "=" * 70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = check_system()
    sys.exit(0 if success else 1)
