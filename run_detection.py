#!/usr/bin/env python3
"""
Continuous Waste Detection - Runs until manually stopped
This version keeps running and shows real-time detection stats
"""

from camera_detection import WasteDetector
import time

def main():
    """Run continuous waste detection"""
    
    print("\n" + "=" * 60)
    print("🗑️  CONTINUOUS WASTE DETECTION")
    print("=" * 60)
    print("\n📋 Instructions:")
    print("   1. Place waste objects in front of camera:")
    print("      • 🟡 Plastic bottle/container")
    print("      • 🔵 Metal can (aluminum/tin)")
    print("      • 🟢 Cardboard/paper")
    print("\n   2. Hold objects 30-50cm from camera")
    print("   3. Ensure good lighting")
    print("\n   4. Controls:")
    print("      • Press 'Q' to quit")
    print("      • Press 'S' to save frame")
    print("      • Press 'C' to clear counts")
    print("\n" + "=" * 60)
    
    input("\n⏸️  Press ENTER to start detection... ")
    
    try:
        # Initialize detector
        detector = WasteDetector()
        
        print("\n🎥 Camera is ON - Detection running!")
        print("   Point waste objects at camera now...")
        print("   (Press 'Q' in the camera window to stop)\n")
        
        # Run detection
        detector.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Detection interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   • Check camera is connected")
        print("   • Close other apps using camera")
        print("   • Ensure model file exists")
    finally:
        print("\n✅ Detection ended")

if __name__ == "__main__":
    main()
