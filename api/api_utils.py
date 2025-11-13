"""
Utility functions for API operations.
Handles file I/O, encoding/decoding, and data conversions.
"""

import base64
import io
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image


# =============================================================================
# FILE HANDLING
# =============================================================================

def save_uploaded_file(file_content: bytes, filename: str, temp_dir: str = "temp") -> str:
    """
    Save uploaded file to temp directory.

    Args:
        file_content: File content as bytes
        filename: Original filename
        temp_dir: Temporary directory path

    Returns:
        Path to saved file
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)

    # Generate unique filename
    import uuid
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = temp_path / unique_filename

    # Save file
    with open(file_path, 'wb') as f:
        f.write(file_content)

    return str(file_path)


def cleanup_temp_file(file_path: str):
    """Delete temporary file."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: Failed to delete temp file {file_path}: {e}")


# =============================================================================
# BASE64 ENCODING/DECODING
# =============================================================================

def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image (BGR).

    Args:
        base64_string: Base64 encoded image string

    Returns:
        OpenCV image (numpy array)

    Raises:
        ValueError: If decoding fails
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64
        img_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Convert to numpy array
        img_array = np.array(pil_img)

        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        return img_bgr

    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")


def image_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
    """
    Convert OpenCV image to base64 string.

    Args:
        image: OpenCV image (numpy array)
        format: Image format ('PNG' or 'JPEG')

    Returns:
        Base64 encoded string
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Convert to PIL Image
    pil_img = Image.fromarray(image_rgb.astype(np.uint8))

    # Encode to bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    img_bytes = buffer.getvalue()

    # Encode to base64
    base64_string = base64.b64encode(img_bytes).decode('utf-8')

    return base64_string


def mask_to_base64(mask: np.ndarray) -> str:
    """
    Convert binary mask to base64 PNG string.

    Args:
        mask: Binary mask (numpy array with 0 and 1)

    Returns:
        Base64 encoded PNG string
    """
    # Convert to 0-255 range
    mask_255 = (mask * 255).astype(np.uint8)

    return image_to_base64(mask_255, format='PNG')


def base64_to_mask(base64_string: str) -> np.ndarray:
    """
    Convert base64 PNG string to binary mask.

    Args:
        base64_string: Base64 encoded PNG string

    Returns:
        Binary mask (numpy array with 0 and 1)

    Raises:
        ValueError: If decoding fails
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64
        img_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Convert to numpy array
        img_array = np.array(pil_img)

        # Handle different image formats
        if len(img_array.shape) == 3:
            # RGB or RGBA image, take first channel
            mask = img_array[:, :, 0]
        else:
            # Grayscale
            mask = img_array

        # Convert to binary (0 or 1)
        binary_mask = (mask > 127).astype(np.uint8)

        return binary_mask

    except Exception as e:
        raise ValueError(f"Failed to decode base64 mask: {e}")


# Aliases for clarity in adjustment endpoints
def decode_mask_from_base64(base64_string: str) -> np.ndarray:
    """Alias for base64_to_mask."""
    return base64_to_mask(base64_string)


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Alias for mask_to_base64."""
    return mask_to_base64(mask)


# =============================================================================
# DATA CONVERSION
# =============================================================================

def convert_bbox_to_dict(bbox: list) -> dict:
    """Convert bbox list [x1, y1, x2, y2] to dict."""
    return {
        'x1': float(bbox[0]),
        'y1': float(bbox[1]),
        'x2': float(bbox[2]),
        'y2': float(bbox[3])
    }


def convert_yolo_detection(detection: dict, include_mask: bool = True) -> dict:
    """
    Convert YOLO detection dict to API-friendly format.

    Args:
        detection: Detection dict from yolo_pred.detect_yolo()
        include_mask: Whether to include base64 encoded mask

    Returns:
        Dict compatible with YOLODetection model
    """
    result = {
        'class_id': detection['class_id'],
        'class_name': detection['class_name'],
        'confidence': detection['confidence'],
        'box': convert_bbox_to_dict(detection['box']),
        'box_width': detection['box_width'],
        'box_height': detection['box_height'],
        'center': detection['center'],
        'mask_area': detection['mask_area'],
        'mask_centroid': detection['mask_centroid'],
    }

    if include_mask and 'mask' in detection:
        result['mask_base64'] = mask_to_base64(detection['mask'])

    return result


def convert_landmark_group(landmark_group: dict) -> dict:
    """
    Convert MediaPipe landmark group to API-friendly format.

    Args:
        landmark_group: Landmark dict from mediapipe_pred.detect_mediapipe()

    Returns:
        Dict compatible with LandmarkGroup model
    """
    bbox = landmark_group['bbox']

    return {
        'points': landmark_group['points'],
        'indices': landmark_group['indices'],
        'center': landmark_group['center'],
        'bbox': convert_bbox_to_dict(bbox)
    }


def convert_beautify_result(result: dict, include_masks: bool = True) -> dict:
    """
    Convert beautify result to API-friendly format.

    Args:
        result: Result dict from beautify.beautify_eyebrows()
        include_masks: Whether to include base64 encoded masks

    Returns:
        Dict compatible with EyebrowResult model
    """
    api_result = {
        'side': result['side'],
        'validation': result['validation'],
        'metadata': result['metadata']
    }

    if include_masks:
        api_result['original_mask_base64'] = mask_to_base64(result['masks']['original_yolo'])
        api_result['final_mask_base64'] = mask_to_base64(result['masks']['final_beautified'])

    return api_result


# =============================================================================
# VALIDATION
# =============================================================================

def validate_image_format(image: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate image format and dimensions.

    Args:
        image: OpenCV image

    Returns:
        (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"

    if not isinstance(image, np.ndarray):
        return False, "Image is not a numpy array"

    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image dimensions: {image.shape}"

    h, w = image.shape[:2]

    if h < 200 or w < 200:
        return False, f"Image too small: {w}x{h}. Minimum size is 200x200"

    if h > 8000 or w > 8000:
        return False, f"Image too large: {w}x{h}. Maximum size is 8000x8000"

    return True, None


# =============================================================================
# CONFIGURATION
# =============================================================================

def config_to_dict(config_obj) -> dict:
    """
    Convert BeautifyConfig Pydantic model to dict for beautify.py.

    Args:
        config_obj: BeautifyConfig instance

    Returns:
        Dict compatible with beautify.DEFAULT_CONFIG
    """
    return {
        'yolo_conf_threshold': config_obj.yolo_conf_threshold,
        'mediapipe_conf_threshold': config_obj.mediapipe_conf_threshold,
        'straightening_threshold': config_obj.straightening_threshold,
        'min_mp_coverage': config_obj.min_mp_coverage,
        'eye_dist_range': list(config_obj.eye_dist_range),
        'aspect_ratio_range': list(config_obj.aspect_ratio_range),
        'expansion_range': list(config_obj.expansion_range),
        'min_arch_thickness_pct': config_obj.min_arch_thickness_pct,
        'connection_thickness_pct': config_obj.connection_thickness_pct,
        'eye_buffer_kernel': config_obj.eye_buffer_kernel,
        'eye_buffer_iterations': config_obj.eye_buffer_iterations,
        'hair_overlap_threshold': config_obj.hair_overlap_threshold,
        'hair_distance_threshold': config_obj.hair_distance_threshold,
        'close_kernel': config_obj.close_kernel,
        'open_kernel': config_obj.open_kernel,
        'gaussian_kernel': config_obj.gaussian_kernel,
        'gaussian_sigma': config_obj.gaussian_sigma,
    }


# =============================================================================
# STENCIL EXTRACTION UTILITIES (v6.0)
# =============================================================================

def extract_stencils_from_image(image_path: str, yolo_model, config: dict) -> List[Dict]:
    """
    Extract stencil polygons from image using YOLO + MediaPipe.

    Args:
        image_path: Path to image file
        yolo_model: Loaded YOLO model
        config: Configuration dictionary

    Returns:
        List of stencil extraction results
    """
    import cv2
    import yolo_pred
    import mediapipe_pred
    import stencil_extract

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_shape = image.shape[:2]

    # Run YOLO detection
    yolo_result = yolo_pred.detect_yolo(yolo_model, image_path)

    # Run MediaPipe detection
    mp_result = mediapipe_pred.detect_mediapipe(image)

    # Extract stencil polygons
    stencils = stencil_extract.extract_stencils_from_detections(
        yolo_result,
        mp_result,
        image_shape,
        config
    )

    return stencils


def convert_stencil_result(stencil: Dict, image_shape: Tuple[int, int]):
    """
    Convert stencil extraction result to API response model.

    Args:
        stencil: Stencil result dict from stencil_extract
        image_shape: Image shape (height, width)

    Returns:
        StencilPolygon API model
    """
    from . import api_models
    from datetime import datetime

    return api_models.StencilPolygon(
        stencil_id=None,  # Not saved yet
        side=stencil['side'],
        polygon=stencil['polygon'],
        num_points=stencil['num_points'],
        source=stencil['source'],
        bbox=stencil['bbox'],
        alignment=api_models.PolygonAlignment(**stencil['alignment']),
        validation=api_models.PolygonValidation(**stencil['validation']),
        metadata=api_models.StencilMetadata(**stencil['metadata']),
        created_at=datetime.utcnow().isoformat() + 'Z',
        image_shape=image_shape
    )
