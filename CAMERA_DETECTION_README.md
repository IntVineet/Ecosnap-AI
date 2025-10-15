# 🗑️ Waste Detection System - Camera Detection Guide

## Quick Start Guide

### Option 1: Direct Camera Detection (Fastest)
Run real-time waste detection immediately:

```bash
python quick_start.py
```

### Option 2: Camera Detection Only
Run the full-featured camera detection:

```bash
python camera_detection.py
```

### Option 3: Complete Workflow
Run notebook preparation + camera detection:

```bash
python run_complete_workflow.py
```

## Features

### Real-time Detection
- ✅ Live camera feed with object detection
- ✅ Bounding boxes around detected waste objects
- ✅ Color-coded by category:
  - **Yellow** - Plastic
  - **Blue** - Metal  
  - **Green** - Paper
- ✅ Confidence scores displayed
- ✅ FPS counter and performance metrics

### Controls
- **Q** - Quit detection
- **S** - Save current frame with detections
- **C** - Clear detection count

### Output Information
- Real-time FPS display
- Detection count per category
- Confidence scores for each object
- Total detections summary on exit

## System Requirements

### Required Packages
```bash
pip install ultralytics opencv-python numpy
```

### Camera Access
- Webcam must be available (default camera 0)
- macOS users: Grant camera permissions when prompted

## Model Information

The system uses:
1. **Trained Model** (if available): `models/train_taco/weights/best.pt`
   - Custom trained on TACO waste dataset
   - Categories: Plastic, Metal, Paper

2. **Fallback Model**: YOLOv8n pre-trained
   - Used if custom model not found
   - Can detect 80+ COCO object classes

## Detection Output

### On-Screen Display
- Live video feed with bounding boxes
- Object labels and confidence scores
- Performance metrics (FPS, detection count)
- Detection summary panel

### Saved Frames
- Saved to: `detections/` folder
- Format: `detection_0001.jpg`, `detection_0002.jpg`, etc.
- Contains all drawn bounding boxes and labels

### Console Output
```
📊 DETECTION SUMMARY
==================================================
Total Frames Processed: 1245
Average FPS: 28.5
Frames Saved: 5

🗑️  Total Detections by Category:
   • Plastic: 45
   • Metal: 23
   • Paper: 12
==================================================
```

## Troubleshooting

### Camera Not Opening
- Check if camera is connected
- Try changing camera index in code (0, 1, 2, etc.)
- Grant camera permissions in System Preferences (macOS)

### Low FPS
- Use smaller image size
- Reduce confidence threshold
- Use lighter YOLO model (yolov8n)

### No Detections
- Ensure objects are clearly visible
- Adjust lighting conditions
- Lower confidence threshold (default: 0.25)

### Model Not Found
- System automatically uses pre-trained YOLOv8n
- Train custom model first by running notebook

## File Structure

```
GenAI_Jovac/
├── camera_detection.py       # Main camera detection script
├── quick_start.py            # Quick start launcher
├── run_complete_workflow.py  # Full workflow runner
├── models/
│   └── train_taco/
│       └── weights/
│           └── best.pt       # Trained model (if available)
├── detections/               # Saved detection frames
└── notebooks/
    └── 1_prepare_taco.ipynb # Dataset preparation notebook
```

## Advanced Configuration

### Adjust Detection Parameters
Edit `camera_detection.py`:

```python
# Line ~76: Adjust confidence threshold
results = self.model(frame, conf=0.25, verbose=False)  # 0.25 = 25% confidence

# Line ~98: Change camera resolution
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Line ~95: Change camera index
self.cap = cv2.VideoCapture(0)  # Try 1, 2, etc. for different cameras
```

### Customize Colors
Edit the `COLORS` dictionary (line ~23):

```python
COLORS = {
    'Plastic': (0, 255, 255),    # BGR: Yellow
    'Metal': (255, 0, 0),         # BGR: Blue
    'Paper': (0, 255, 0)          # BGR: Green
}
```

## Usage Examples

### Example 1: Quick Detection Session
```bash
# Start detection
python quick_start.py

# Point camera at waste objects
# Press 'S' to save interesting frames
# Press 'Q' to quit
```

### Example 2: Save All Detections
```bash
# Start detection
python camera_detection.py

# Continuously save frames with detections
# Press 'S' for each frame you want to save
# Check detections/ folder for saved images
```

### Example 3: Full Training + Detection
```bash
# Run complete workflow
python run_complete_workflow.py

# Choose option 1 for full workflow
# Notebook runs first (prepares data)
# Then camera detection launches automatically
```

## Next Steps

1. ✅ **Test Detection**: Run `python quick_start.py`
2. 📊 **Review Results**: Check `detections/` folder
3. 🎯 **Improve Model**: Train with more real waste images
4. 🔧 **Customize**: Adjust colors, confidence, resolution

## Tips for Best Results

- **Good Lighting**: Ensure well-lit environment
- **Clear Objects**: Hold objects clearly visible to camera
- **Distance**: Keep objects 30cm - 2m from camera
- **Background**: Use plain background for better detection
- **Movement**: Keep objects relatively still

---

**Happy Detecting! 🎉**

For issues or questions, check the troubleshooting section above.
