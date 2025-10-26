"""
YOLO Detection Wrapper for Eyebrow Beautification.

Provides clean interface to YOLO11 segmentation model for detecting:
- eyebrows (class 2)
- eye (class 0)
- eye_box (class 1)
- hair (class 3)

Returns structured detection dictionaries organized by class.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Optional
import utils


def load_yolo_model(model_path: str = 'eyebrow_training/eyebrow_recommended/weights/best.pt'):
    """
    Load the YOLO11 segmentation model.

    Args:
        model_path: Path to YOLO model weights

    Returns:
        YOLO model object

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    return model


def detect_yolo(model, image_path: str, conf_threshold: float = 0.25) -> Dict:
    """
    Run YOLO detection and return structured detections organized by class.

    Args:
        model: YOLO model object
        image_path: Path to input image
        conf_threshold: Confidence threshold for predictions (default: 0.25)

    Returns:
        Dictionary with structure:
        {
            'eyebrows': [
                {
                    'class_id': 2,
                    'class_name': 'eyebrows',
                    'confidence': float (0-1),
                    'box': [x1, y1, x2, y2],  # absolute pixels, xyxy format
                    'box_width': float,
                    'box_height': float,
                    'center': (cx, cy),  # box center
                    'mask': np.ndarray,  # (H, W) binary uint8, 0 or 1
                    'mask_area': int,  # pixel count
                    'mask_centroid': (cx, cy)  # mask center of mass
                },
                ...
            ],
            'eye': [...],  # same structure
            'eye_box': [...],
            'hair': [...]
        }

        Classes:
        - 0 = eye
        - 1 = eye_box
        - 2 = eyebrows
        - 3 = hair

        Notes:
        - Masks are binary (0 or 1), same size as input image
        - Boxes are in xyxy format with absolute pixel coordinates
        - Empty classes return empty lists
    """
    # Class ID to name mapping
    CLASS_NAMES = {
        0: 'eye',
        1: 'eye_box',
        2: 'eyebrows',
        3: 'hair'
    }

    # Initialize output dictionary
    detections = {
        'eyebrows': [],
        'eye': [],
        'eye_box': [],
        'hair': []
    }

    # Run YOLO inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=0.7,
        imgsz=800,
        verbose=False
    )

    # Get first result (single image)
    result = results[0]

    # Read original image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_h, img_w = img.shape[:2]

    # Check if there are any detections
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections found.")
        return detections

    # Extract detection data
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (xyxy)
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Check if masks are available
    if result.masks is None:
        print("Warning: No segmentation masks found. Check if model is trained for segmentation.")
        return detections

    masks = result.masks.data.cpu().numpy()  # Segmentation masks

    # Process each detection
    for i, (box, conf, class_id, mask) in enumerate(zip(boxes, confidences, class_ids, masks)):
        # Get class name
        class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')

        # Skip unknown classes
        if class_name not in detections:
            continue

        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (img_w, img_h))

        # Binarize mask (threshold at 0.5)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Calculate mask properties
        mask_area = int(mask_binary.sum())
        mask_centroid = utils.calculate_centroid(mask_binary)

        # Extract box coordinates
        x1, y1, x2, y2 = box
        box_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Create detection dict
        detection = {
            'class_id': int(class_id),
            'class_name': class_name,
            'confidence': float(conf),
            'box': [float(x1), float(y1), float(x2), float(y2)],
            'box_width': float(x2 - x1),
            'box_height': float(y2 - y1),
            'center': (int(box_center[0]), int(box_center[1])),
            'mask': mask_binary,
            'mask_area': mask_area,
            'mask_centroid': mask_centroid
        }

        # Add to appropriate class list
        detections[class_name].append(detection)

    # Print summary
    total_detections = sum(len(dets) for dets in detections.values())
    print(f"YOLO detected {total_detections} objects:")
    for class_name, dets in detections.items():
        if len(dets) > 0:
            print(f"  - {class_name}: {len(dets)} detection(s)")

    return detections


def get_detection_summary(detections: Dict) -> Dict:
    """
    Get summary statistics for YOLO detections.

    Args:
        detections: Detection dict from detect_yolo()

    Returns:
        Dictionary with counts and average confidences per class
    """
    summary = {}

    for class_name, det_list in detections.items():
        if len(det_list) > 0:
            avg_conf = np.mean([d['confidence'] for d in det_list])
            summary[class_name] = {
                'count': len(det_list),
                'avg_confidence': float(avg_conf),
                'total_area': sum(d['mask_area'] for d in det_list)
            }
        else:
            summary[class_name] = {
                'count': 0,
                'avg_confidence': 0.0,
                'total_area': 0
            }

    return summary


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of YOLO detection wrapper."""

    # Load model
    model = load_yolo_model()

    # Run detection on test image
    test_image = "annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    detections = detect_yolo(model, test_image, conf_threshold=0.25)

    # Print summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)

    summary = get_detection_summary(detections)

    for class_name, stats in summary.items():
        print(f"\n{class_name.upper()}:")
        print(f"  Count: {stats['count']}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"  Total Area: {stats['total_area']} pixels")

    # Show details for eyebrows
    if len(detections['eyebrows']) > 0:
        print("\n" + "="*60)
        print("EYEBROW DETAILS")
        print("="*60)

        for i, eyebrow in enumerate(detections['eyebrows']):
            print(f"\nEyebrow {i+1}:")
            print(f"  Confidence: {eyebrow['confidence']:.3f}")
            print(f"  Box: {eyebrow['box']}")
            print(f"  Box dimensions: {eyebrow['box_width']:.1f} x {eyebrow['box_height']:.1f}")
            print(f"  Mask area: {eyebrow['mask_area']} pixels")
            print(f"  Mask centroid: {eyebrow['mask_centroid']}")
