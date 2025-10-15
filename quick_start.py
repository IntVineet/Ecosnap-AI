#!/usr/bin/env python3
"""
Quick Start: Real-time Waste Detection
Directly run camera detection with YOLO model
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# Import and run camera detection
from camera_detection import WasteDetector

def main():
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║     🗑️  WASTE DETECTION - QUICK START             ║
    ║          Real-time Camera Detection               ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    print("📋 Starting real-time waste detection...")
    print("   • Categories: Plastic, Metal, Paper")
    print("   • Press 'Q' to quit")
    print("   • Press 'S' to save current frame")
    print("")
    
    try:
        detector = WasteDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n👋 Detection stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
