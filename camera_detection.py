#!/usr/bin/env python3
"""
Real-time Waste Detection using Camera
Detects waste objects (plastic, metal, paper) in real-time from webcam feed
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

# Configuration
PROJECT_ROOT = Path(__file__).parent

# Try multiple model paths (train_taco, train_taco2, train_taco3, etc.)
# Check in reverse order to get the latest model
MODEL_PATHS = [
    PROJECT_ROOT / 'models' / f'train_taco{i}' / 'weights' / 'best.pt'
    for i in ['5', '4', '3', '2', '']
]
MODEL_PATH = None
for path in MODEL_PATHS:
    if path.exists():
        MODEL_PATH = path
        break

FALLBACK_MODEL = 'yolov8n.pt'  # Pre-trained model as fallback

# Colors for bounding boxes (BGR format for OpenCV)
# Works with both lowercase and uppercase class names
COLORS = {
    'plastic': (0, 255, 255),    # Yellow
    'Plastic': (0, 255, 255),    # Yellow
    'metal': (255, 0, 0),         # Blue
    'Metal': (255, 0, 0),         # Blue
    'paper': (0, 255, 0),          # Green
    'Paper': (0, 255, 0)          # Green
}

class WasteDetector:
    def __init__(self):
        """Initialize the waste detector with YOLO model"""
        print("🚀 Initializing Waste Detection System...")
        
        # Load model
        if MODEL_PATH and MODEL_PATH.exists():
            print(f"✅ Loading trained model: {MODEL_PATH}")
            self.model = YOLO(str(MODEL_PATH))
            self.model_name = "Custom TACO Model"
        else:
            print(f"⚠️  Trained model not found. Using pre-trained YOLOv8n model")
            self.model = YOLO(FALLBACK_MODEL)
            self.model_name = "YOLOv8n (Pre-trained)"
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open camera. Please check your webcam connection.")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ Waste Detection System Ready!")
        print(f"📹 Camera initialized")
        print(f"🤖 Model: {self.model_name}")
        print("\n📋 Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 's' to save current frame")
        print("   • Press 'c' to clear detection count")
        print("\n🗑️  Waste Categories: Plastic (Yellow), Metal (Blue), Paper (Green)")
        
    def detect_objects(self, frame):
        """Run detection on a frame"""
        results = self.model(frame, conf=0.25, verbose=False)
        return results[0]
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Get class name directly from model
                class_name = results.names[cls] if cls < len(results.names) else f"Class {cls}"
                
                # Capitalize first letter for display
                display_name = class_name.capitalize()
                
                # Get color (works with both lowercase and uppercase)
                color = COLORS.get(class_name, COLORS.get(display_name, (255, 255, 255)))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                label = f"{display_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Store detection info
                detections.append({
                    'class': display_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return frame, detections
    
    def draw_info_panel(self, frame, detections):
        """Draw information panel on frame"""
        height, width = frame.shape[:2]
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        # Draw semi-transparent panel
        panel_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text information
        y_offset = 35
        cv2.putText(frame, f"Model: {self.model_name}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Detections: {len(detections)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw detection summary
        if detections:
            y_offset += 30
            detection_summary = {}
            for det in detections:
                class_name = det['class']
                detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
            
            summary_text = ", ".join([f"{k}: {v}" for k, v in detection_summary.items()])
            cv2.putText(frame, summary_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        saved_count = 0
        total_detections = {'Plastic': 0, 'Metal': 0, 'Paper': 0, 'Other': 0}
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Run detection
            results = self.detect_objects(frame)
            
            # Draw detections
            frame, detections = self.draw_detections(frame, results)
            
            # Update total detections
            for det in detections:
                class_name = det['class']
                if class_name in total_detections:
                    total_detections[class_name] += 1
                else:
                    total_detections['Other'] += 1
            
            # Draw info panel
            frame = self.draw_info_panel(frame, detections)
            
            # Display frame
            cv2.imshow('🗑️ Waste Detection - Press Q to Quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Stopping detection...")
                break
            elif key == ord('s'):
                # Save current frame
                output_dir = PROJECT_ROOT / 'detections'
                output_dir.mkdir(exist_ok=True)
                filename = output_dir / f'detection_{saved_count:04d}.jpg'
                cv2.imwrite(str(filename), frame)
                saved_count += 1
                print(f"📸 Saved frame to: {filename}")
            elif key == ord('c'):
                # Clear detection count
                total_detections = {'Plastic': 0, 'Metal': 0, 'Paper': 0, 'Other': 0}
                print("🔄 Detection count cleared")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*50)
        print("📊 DETECTION SUMMARY")
        print("="*50)
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Average FPS: {self.fps:.2f}")
        print(f"Frames Saved: {saved_count}")
        print("\n🗑️  Total Detections by Category:")
        for category, count in total_detections.items():
            if count > 0:
                print(f"   • {category}: {count}")
        print("="*50)
        print("✅ Detection session ended successfully!")

def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║     🗑️  WASTE DETECTION SYSTEM - CAMERA MODE     ║
    ║           Real-time Object Detection              ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    try:
        detector = WasteDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
