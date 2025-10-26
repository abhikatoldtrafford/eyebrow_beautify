"""
Streamlit App Utility Functions
Image encoding/decoding, mask overlay, display helpers
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List
import streamlit as st
from streamlit_config import COLORS, MP_POINT_RADIUS


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(b64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert binary mask to base64 PNG."""
    # Ensure binary (0 or 255)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    _, buffer = cv2.imencode('.png', mask)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_mask(b64_string: str) -> np.ndarray:
    """Convert base64 PNG to binary mask (0 or 1)."""
    img = base64_to_image(b64_string)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay a colored mask on an image.

    Args:
        image: Original image (BGR)
        mask: Binary mask (0 or 1, or 0-255)
        color: RGB color tuple
        alpha: Opacity (0-1)

    Returns:
        Image with overlay (BGR)
    """
    # Ensure mask is binary
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    # Create colored overlay
    overlay = image.copy()
    color_bgr = (color[2], color[1], color[0])  # RGB to BGR

    # Apply color where mask is non-zero
    overlay[mask > 0] = color_bgr

    # Blend with original
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return result


def draw_mediapipe_points(
    image: np.ndarray,
    landmarks: Dict,
    color: Tuple[int, int, int] = COLORS['mediapipe_points'],
    radius: int = MP_POINT_RADIUS
) -> np.ndarray:
    """
    Draw MediaPipe landmark points on image.

    Args:
        image: Original image (BGR)
        landmarks: Dict with 'left_eyebrow', 'right_eyebrow' containing 'points'
        color: RGB color
        radius: Point radius

    Returns:
        Image with points drawn
    """
    result = image.copy()
    color_bgr = (color[2], color[1], color[0])  # RGB to BGR

    for feature in ['left_eyebrow', 'right_eyebrow']:
        if feature in landmarks and landmarks[feature]:
            points = landmarks[feature].get('points', [])
            for point in points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(result, (x, y), radius, color_bgr, -1)
                # Draw outline for visibility
                cv2.circle(result, (x, y), radius + 1, (0, 0, 0), 1)

    return result


def create_comparison_view(
    original_image: np.ndarray,
    yolo_masks: Dict[str, np.ndarray],
    beautified_masks: Dict[str, np.ndarray],
    mediapipe_landmarks: Optional[Dict] = None,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create side-by-side comparison: Actual (YOLO+MP) vs Beautified.

    Args:
        original_image: Original image (BGR)
        yolo_masks: Dict with 'left' and 'right' YOLO masks
        beautified_masks: Dict with 'left' and 'right' beautified masks
        mediapipe_landmarks: MediaPipe landmarks dict
        alpha: Overlay opacity

    Returns:
        (actual_view, beautified_view) - both BGR images
    """
    # Create actual view (YOLO + MediaPipe)
    actual = original_image.copy()

    if yolo_masks.get('left') is not None:
        actual = overlay_mask_on_image(actual, yolo_masks['left'], COLORS['left_eyebrow'], alpha)

    if yolo_masks.get('right') is not None:
        actual = overlay_mask_on_image(actual, yolo_masks['right'], COLORS['right_eyebrow'], alpha)

    if mediapipe_landmarks:
        actual = draw_mediapipe_points(actual, mediapipe_landmarks)

    # Create beautified view
    beautified = original_image.copy()

    if beautified_masks.get('left') is not None:
        beautified = overlay_mask_on_image(beautified, beautified_masks['left'], COLORS['left_eyebrow'], alpha)

    if beautified_masks.get('right') is not None:
        beautified = overlay_mask_on_image(beautified, beautified_masks['right'], COLORS['right_eyebrow'], alpha)

    return actual, beautified


def display_validation_metrics(validation: Dict, side: str):
    """
    Display validation metrics with checkmarks/crosses.

    Args:
        validation: Validation dict from API response
        side: 'left' or 'right'
    """
    st.write(f"### {side.title()} Eyebrow Validation")

    # MediaPipe coverage (with availability indicator)
    mp_available = validation.get('mp_available', False)
    mp_coverage = validation.get('mp_coverage', 0)
    mp_pass = validation.get('mp_coverage_pass', False)

    if mp_available:
        icon = "✓" if mp_pass else "✗"
        color = "green" if mp_pass else "red"
        st.markdown(f":{color}[{icon}] **MediaPipe Coverage**: {mp_coverage:.1f}%")
    else:
        st.markdown(f":gray[○] **MediaPipe Coverage**: N/A (face not detected, using YOLO only)")

    # Eye distance (with availability indicator)
    eye_available = validation.get('eye_available', False)
    eye_dist = validation.get('eye_distance_pct', 0)
    eye_dist_pass = validation.get('eye_distance_pass', False)

    if eye_available:
        icon = "✓" if eye_dist_pass else "✗"
        color = "green" if eye_dist_pass else "red"
        st.markdown(f":{color}[{icon}] **Eye Distance**: {eye_dist:.1f}%")
    else:
        st.markdown(f":gray[○] **Eye Distance**: N/A (eye not detected by YOLO)")

    # Other checks (always applicable)
    checks = [
        ('aspect_ratio', 'Aspect Ratio', f"{validation.get('aspect_ratio', 0):.2f}"),
        ('eye_overlap', 'Eye Overlap', f"{validation.get('eye_overlap', 0)} px"),
        ('expansion_ratio', 'Expansion Ratio', f"{validation.get('expansion_ratio', 0):.2f}x"),
        ('thickness_ratio', 'Thickness Ratio', f"{validation.get('thickness_ratio', 0):.2f}"),
    ]

    for key, label, value in checks:
        pass_key = f"{key}_pass"
        passed = validation.get(pass_key, False)
        icon = "✓" if passed else "✗"
        color = "green" if passed else "red"

        st.markdown(f":{color}[{icon}] **{label}**: {value}")

    overall = validation.get('overall_pass', False)
    if overall:
        st.success("All validation checks passed!")
    else:
        st.warning("Some validation checks failed")


def display_statistics(metadata: Dict):
    """
    Display eyebrow statistics.

    Args:
        metadata: Metadata dict from API response
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("YOLO Confidence", f"{metadata.get('yolo_confidence', 0):.2f}")

    with col2:
        st.metric("YOLO Area", f"{metadata.get('yolo_area', 0):,} px")

    with col3:
        st.metric("Final Area", f"{metadata.get('final_area', 0):,} px")

    col4, col5, col6 = st.columns(3)

    with col4:
        has_eye = "✓" if metadata.get('has_eye', False) else "✗"
        st.write(f"**Eye Detected:** {has_eye}")

    with col5:
        has_box = "✓" if metadata.get('has_eye_box', False) else "✗"
        st.write(f"**Eye Box:** {has_box}")

    with col6:
        has_mp = "✓" if metadata.get('has_mediapipe', False) else "✗"
        st.write(f"**MediaPipe:** {has_mp}")


def resize_image_if_needed(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image if it exceeds max dimensions while preserving aspect ratio.

    Args:
        image: OpenCV image
        max_size: (max_width, max_height)

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size

    if w <= max_w and h <= max_h:
        return image

    # Calculate scaling factor
    scale = min(max_w / w, max_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_rotation_to_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate mask by given angle (degrees).

    Args:
        mask: Binary mask
        angle: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated mask
    """
    h, w = mask.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    return rotated


def apply_scale_to_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale mask by given factor.

    Args:
        mask: Binary mask
        scale: Scale factor (1.0 = no change)

    Returns:
        Scaled mask
    """
    h, w = mask.shape[:2]
    center = (w // 2, h // 2)

    # Get scaling matrix
    M = cv2.getRotationMatrix2D(center, 0, scale)

    # Apply scaling
    scaled = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    return scaled


def apply_translation_to_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Translate (move) mask by dx, dy pixels.

    Args:
        mask: Binary mask
        dx: Horizontal shift (pixels)
        dy: Vertical shift (pixels)

    Returns:
        Translated mask
    """
    h, w = mask.shape[:2]

    # Translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply translation
    translated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    return translated


def create_download_data(image: np.ndarray, filename: str = "result.png") -> bytes:
    """
    Prepare image for download.

    Args:
        image: OpenCV image (BGR)
        filename: Target filename

    Returns:
        Bytes for download
    """
    # Convert to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    # Save to bytes
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)

    return buf.getvalue()


def show_error(message: str):
    """Display error message."""
    st.error(f"❌ {message}")


def show_success(message: str):
    """Display success message."""
    st.success(f"✓ {message}")


def show_info(message: str):
    """Display info message."""
    st.info(f"ℹ️ {message}")


def show_warning(message: str):
    """Display warning message."""
    st.warning(f"⚠️ {message}")
