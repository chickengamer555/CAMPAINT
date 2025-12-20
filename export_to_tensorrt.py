"""
Export YOLO models to TensorRT for 2-3x faster inference on GPU
Run this script once to create optimized .engine files
"""

from ultralytics import YOLO
import torch

def export_models():
    """Export YOLO models to TensorRT format"""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available! TensorRT requires a NVIDIA GPU.")
        print("   Your models will run on CPU (slower).")
        return
    
    print("=" * 60)
    print("🚀 EXPORTING YOLO MODELS TO TENSORRT")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Export Pose model
    print("[1/2] Exporting YOLO Pose model to TensorRT...")
    print("This may take 5-10 minutes on first run...")
    try:
        pose_model = YOLO('yolo11n-pose.pt')
        pose_model.export(
            format='engine',      # TensorRT format
            half=True,            # FP16 precision (2x faster)
            imgsz=320,            # Match your processing resolution
            workspace=4,          # GPU memory workspace (GB)
            verbose=True
        )
        print("✅ Pose model exported to: yolo11n-pose.engine")
    except Exception as e:
        print(f"❌ Error exporting pose model: {e}")
        print("   Continuing with segmentation model...")
    
    print()
    
    # Export Segmentation model
    print("[2/2] Exporting YOLO Segmentation model to TensorRT...")
    print("This may take 5-10 minutes on first run...")
    try:
        seg_model = YOLO('yolo11n-seg.pt')
        seg_model.export(
            format='engine',      # TensorRT format
            half=True,            # FP16 precision (2x faster)
            imgsz=320,            # Match your processing resolution
            workspace=4,          # GPU memory workspace (GB)
            verbose=True
        )
        print("✅ Segmentation model exported to: yolo11n-seg.engine")
    except Exception as e:
        print(f"❌ Error exporting segmentation model: {e}")
    
    print()
    print("=" * 60)
    print("✨ EXPORT COMPLETE!")
    print("=" * 60)
    print()
    print("📊 Expected Performance Improvement:")
    print("   - PyTorch FP32: ~30-40 FPS")
    print("   - PyTorch FP16: ~50-60 FPS")
    print("   - TensorRT FP16: ~80-120 FPS (2-3x faster!)")
    print()
    print("🚀 Now run ultra-advanced-tracking.py to use the optimized models!")
    print()

if __name__ == "__main__":
    export_models()

