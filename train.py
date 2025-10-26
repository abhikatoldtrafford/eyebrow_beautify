"""
YOLO Segmentation Training for Eyebrow and Eye Detection
Optimized augmentations for facial feature segmentation
"""

from ultralytics import YOLO
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset configuration
DATA_YAML = "annotated/data.yaml"  # Path to your dataset YAML file

# Model selection
MODEL_SIZE = "yolo11s-seg.pt"  # Options: yolo11n-seg.pt (fastest), yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt

# Training parameters
EPOCHS = 1000
BATCH_SIZE = 24  # Adjust based on GPU memory
IMAGE_SIZE = 800
PATIENCE = 100  # Early stopping patience

# Device
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ============================================================================
# AUGMENTATION PROFILES
# ============================================================================

# Profile 1: RECOMMENDED - Balanced augmentation for face/eyebrow segmentation
RECOMMENDED_AUGMENTATIONS = {
    # Color space augmentations (moderate - faces have varying lighting)
    "hsv_h": 0.015,      # Hue shift (default, subtle color variation)
    "hsv_s": 0.5,        # Saturation (reduced from default 0.7, faces don't vary as much)
    "hsv_v": 0.3,        # Brightness (reduced from default 0.4, important but controlled)
    
    # Geometric transformations (minimal - faces are usually upright)
    "degrees": 5.0,      # Small rotation (faces can tilt slightly)
    "translate": 0.1,    # Small translation (default, allows partial faces)
    "scale": 0.3,        # Moderate scaling (faces at different distances)
    "shear": 0.0,        # No shear (faces don't skew naturally)
    "perspective": 0.0,  # No perspective (faces are relatively flat)
    
    # Flipping
    "fliplr": 0.5,       # Horizontal flip (50% chance, faces can be mirrored)
    "flipud": 0.0,       # No vertical flip (faces are always upright)
    
    # Advanced augmentations
    "mosaic": 0.0,       # CRITICAL: No mosaic (eyebrows must stay with their face!)
    "mixup": 0.0,        # No mixup (would blend multiple faces)
    "copy_paste": 0.3,   # Moderate copy-paste (can help with sparse eyebrows)
    "copy_paste_mode": "flip",  # Copy from same image
    
    # Auto augmentation (disabled for segmentation)
    "auto_augment": None,
    
    # Erasing (disabled - would remove eyebrows)
    "erasing": 0.0,
}

# Profile 2: CONSERVATIVE - Minimal augmentation (use if overfitting isn't an issue)
CONSERVATIVE_AUGMENTATIONS = {
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.2,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.2,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "auto_augment": None,
    "erasing": 0.0,
}

# Profile 3: AGGRESSIVE - Strong augmentation (use if you have limited data)
AGGRESSIVE_AUGMENTATIONS = {
    "hsv_h": 0.03,       # More color variation
    "hsv_s": 0.7,        # Strong saturation changes
    "hsv_v": 0.5,        # Strong brightness changes
    "degrees": 30.0,     # More rotation
    "translate": 0.15,   # More translation
    "scale": 0.5,        # More scale variation
    "shear": 2.0,        # Slight shear
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.4,
    "mosaic": 0.5,       # Still no mosaic!
    "mixup": 0.2,
    "copy_paste": 0.25,   # More aggressive copy-paste
    "copy_paste_mode": "flip",
    "auto_augment": None,
    "erasing": 0.0,
}

# Profile 4: NO AUGMENTATION - Baseline (for comparison)
NO_AUGMENTATIONS = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "auto_augment": None,
    "erasing": 0.0,
}

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_eyebrow_model(
    data_yaml=DATA_YAML,
    model_size=MODEL_SIZE,
    augmentation_profile="recommended",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    device=DEVICE,
    project_name="eyebrow_training",
    experiment_name="run1"
):
    """
    Train YOLO segmentation model for eyebrow and eye detection
    
    Args:
        data_yaml: Path to dataset YAML file
        model_size: YOLO model size (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
        augmentation_profile: "recommended", "conservative", "aggressive", or "none"
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        device: GPU device ID or "cpu"
        project_name: Project name for saving results
        experiment_name: Experiment name
    """
    
    # Select augmentation profile
    profiles = {
        "recommended": RECOMMENDED_AUGMENTATIONS,
        "conservative": CONSERVATIVE_AUGMENTATIONS,
        "aggressive": AGGRESSIVE_AUGMENTATIONS,
        "none": NO_AUGMENTATIONS,
    }
    
    if augmentation_profile not in profiles:
        raise ValueError(f"Invalid profile. Choose from: {list(profiles.keys())}")
    
    augmentations = profiles[augmentation_profile]
    
    print(f"🚀 Starting training with {augmentation_profile.upper()} augmentation profile")
    print(f"📊 Model: {model_size}")
    print(f"📁 Dataset: {data_yaml}")
    print(f"⚙️  Device: {device}")
    print(f"🔄 Epochs: {epochs}, Batch: {batch_size}, Image size: {image_size}")
    print("\n" + "="*60)
    print("Augmentation settings:")
    for key, value in augmentations.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Load pretrained model
    model = YOLO(model_size)
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        patience=PATIENCE,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        project=project_name,
        name=experiment_name,
        
        # Augmentation parameters
        **augmentations,
        
        # Additional training parameters
        optimizer="auto",  # Auto-select optimizer
        verbose=True,
        seed=42,  # For reproducibility
        deterministic=False,  # Slight speed boost
        single_cls=False,  # Multi-class (eyebrows + eyes)
        rect=False,  # Rectangular training (can speed up)
        close_mosaic=10,  # Disable mosaic in last N epochs (we already disabled it)
        resume=False,  # Resume from last checkpoint if training interrupted
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,  # Use 100% of dataset
        profile=False,  # Don't profile ONNX/TensorRT
        freeze=None,  # Don't freeze layers
        lr0 = .0001,
        lrf = .1,
        # Validation
        val=True,
        plots=True,  # Generate training plots
        
        # Multi-GPU (if available)
        # workers=8,  # Number of dataloader workers
    )
    
    print("\n✅ Training complete!")
    print(f"📁 Results saved to: {project_name}/{experiment_name}")
    print(f"🏆 Best model: {project_name}/{experiment_name}/weights/best.pt")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Option 1: Train with RECOMMENDED profile (best for most cases)
    train_eyebrow_model(
        augmentation_profile="aggressive",
        experiment_name="eyebrow_recommended"
    )
    