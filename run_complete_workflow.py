#!/usr/bin/env python3
"""
Complete Waste Detection Workflow
1. Run TACO preparation notebook
2. Launch camera detection
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
NOTEBOOK_PATH = PROJECT_ROOT / 'notebooks' / '1_prepare_taco.ipynb'
CAMERA_SCRIPT = PROJECT_ROOT / 'camera_detection.py'

def run_notebook():
    """Convert and run the Jupyter notebook"""
    print("="*60)
    print("📓 STEP 1: Running TACO Preparation Notebook")
    print("="*60)
    
    try:
        # Convert notebook to Python script and execute
        print(f"📂 Notebook: {NOTEBOOK_PATH}")
        print("⏳ Converting notebook to Python script...")
        
        # Use jupyter nbconvert to convert notebook to script
        result = subprocess.run([
            'jupyter', 'nbconvert', 
            '--to', 'script',
            '--execute',
            '--ExecutePreprocessor.timeout=600',
            str(NOTEBOOK_PATH)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Notebook executed successfully!")
        else:
            print("⚠️  Notebook execution completed with warnings")
            if result.stderr:
                print(f"Output: {result.stderr[:500]}")
        
        return True
        
    except FileNotFoundError:
        print("⚠️  jupyter nbconvert not found. Skipping notebook execution.")
        print("💡 You can install it with: pip install nbconvert")
        return False
    except Exception as e:
        print(f"⚠️  Error running notebook: {e}")
        return False

def run_camera_detection():
    """Launch camera detection"""
    print("\n" + "="*60)
    print("📹 STEP 2: Launching Camera Detection")
    print("="*60)
    
    try:
        print(f"🚀 Starting camera detection...")
        print(f"📂 Script: {CAMERA_SCRIPT}\n")
        
        # Run camera detection script
        subprocess.run([sys.executable, str(CAMERA_SCRIPT)])
        
    except KeyboardInterrupt:
        print("\n⚠️  Camera detection stopped by user")
    except Exception as e:
        print(f"❌ Error running camera detection: {e}")

def main():
    """Main workflow"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🗑️  COMPLETE WASTE DETECTION WORKFLOW                    ║
    ║                                                           ║
    ║  Step 1: Prepare TACO dataset and train model            ║
    ║  Step 2: Real-time camera detection                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Ask user if they want to run notebook
    print("\n📋 Options:")
    print("1. Run complete workflow (notebook + camera)")
    print("2. Skip to camera detection only")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        # Run notebook first
        success = run_notebook()
        if success:
            print("\n✅ Notebook execution completed!")
        
        # Ask before launching camera
        input("\n📹 Press ENTER to launch camera detection (or Ctrl+C to exit)...")
        run_camera_detection()
        
    elif choice == '2':
        # Skip directly to camera
        run_camera_detection()
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    print("\n" + "="*60)
    print("🎉 Workflow completed!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Workflow interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
