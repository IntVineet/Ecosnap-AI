#!/usr/bin/env python3
"""
Enhanced Waste Detection with Live Stats
Shows what's being detected in real-time
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import time

# Setup
PROJECT_ROOT = Path(__file__).parent
MODEL_PATHS = [
    PROJECT_ROOT / 'models' / f'train_taco{i}' / 'weights' / 'best.pt'
    for i in ['5', '4', '3', '2', '']
]
MODEL_PATH = None
for path in MODEL_PATHS:
    if path.exists():
        MODEL_PATH = path
        break

# Colors (BGR)
COLORS = {
    'plastic': (0, 255, 255),    # Yellow
    'metal': (255, 0, 0),         # Blue
    'paper': (0, 255, 0)          # Green
}

def main():
    """Run detection with live stats"""
    
    print("\n" + "=" * 70)
    print("🗑️  WASTE DETECTION - LIVE VIEW")
    print("=" * 70)
    
    # Load model
    if MODEL_PATH and MODEL_PATH.exists():
        print(f"\n✅ Loading trained model: {MODEL_PATH.name}")
        model = YOLO(str(MODEL_PATH))
        print(f"✅ Model classes: {model.names}")
    else:
        print("\n❌ No trained model found!")
        print("   Run: python3 train_waste_model.py")
        return
    
    # Initialize camera
    print("\n📹 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Camera ready!")
    
    # Stats
    detection_counts = {'plastic': 0, 'metal': 0, 'paper': 0}
    frame_count = 0
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("🎥 DETECTION ACTIVE!")
    print("=" * 70)
    print("\n📋 What to do:")
    print("   1. Hold waste objects in front of camera (30-50cm away)")
    print("   2. Try: plastic bottle, metal can, or cardboard")
    print("   3. Watch for colored boxes:")
    print("      🟡 Yellow = Plastic")
    print("      🔵 Blue = Metal")
    print("      🟢 Green = Paper")
    print("\n⌨️  Controls:")
    print("   • Press 'Q' in camera window to quit")
    print("   • Press 'S' to save current frame")
    print("   • Press 'C' to clear detection counts")
    print("\n" + "=" * 70 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=0.25, verbose=False)[0]
            
            # Draw detections
            current_detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = results.names[cls]
                    display_name = class_name.capitalize()
                    color = COLORS.get(class_name, (255, 255, 255))
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label
                    label = f"{display_name}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Track detections
                    detection_counts[class_name] += 1
                    current_detections.append(f"{display_name} ({conf:.2f})")
            
            # Add info overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Info panel
            info_y = 30
            cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)
            cv2.putText(frame, "WASTE DETECTION ACTIVE", (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            info_y += 30
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_y += 25
            cv2.putText(frame, "Detections:", (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            info_y += 20
            cv2.putText(frame, f"  Plastic: {detection_counts['plastic']}", (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            info_y += 20
            cv2.putText(frame, f"  Metal: {detection_counts['metal']}", (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            info_y += 20
            cv2.putText(frame, f"  Paper: {detection_counts['paper']}", (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Show current detections
            if current_detections:
                print(f"\r🔍 Detecting: {', '.join(current_detections)}", end='', flush=True)
            else:
                print(f"\r⏳ Waiting for objects... (frames: {frame_count}, fps: {fps:.1f})", end='', flush=True)
            
            # Show frame
            cv2.imshow('Waste Detection - Press Q to quit', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n\n👋 Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n📸 Saved: {filename}")
            elif key == ord('c') or key == ord('C'):
                detection_counts = {'plastic': 0, 'metal': 0, 'paper': 0}
                print("\n🔄 Counts cleared!")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n\n" + "=" * 70)
        print("📊 DETECTION SUMMARY")
        print("=" * 70)
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {fps:.2f}")
        print(f"\n🗑️  Total Detections:")
        print(f"   🟡 Plastic: {detection_counts['plastic']}")
        print(f"   🔵 Metal: {detection_counts['metal']}")
        print(f"   🟢 Paper: {detection_counts['paper']}")
        print("=" * 70)
        print("\n✅ Detection ended successfully!\n")

if __name__ == "__main__":
    main()
