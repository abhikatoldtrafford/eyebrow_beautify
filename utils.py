"""
Utility functions for eyebrow beautification algorithm.
Includes geometry, transforms, face alignment, and mask operations.

This module provides all helper functions needed across the 7-phase pipeline.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional


# =============================================================================
# GEOMETRY & MEASUREMENT FUNCTIONS
# =============================================================================

def calculate_centroid(mask):
    """Calculate the centroid (center of mass) of a binary mask."""
    moments = cv2.moments(mask.astype(np.uint8))

    if moments['m00'] == 0:
        # Fallback to bounding box center
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return (0, 0)
        return (int(cols.mean()), int(rows.mean()))

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    return (cx, cy)


def get_bounding_box_from_mask(mask):
    """Get bounding box [x1, y1, x2, y2] from binary mask."""
    rows, cols = np.where(mask > 0)

    if len(rows) == 0:
        return [0, 0, 0, 0]

    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()

    return [x1, y1, x2, y2]


def get_bbox_from_points(points):
    """Calculate bounding box from list of (x, y) points."""
    if not points or len(points) == 0:
        return [0, 0, 0, 0]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return [min(xs), min(ys), max(xs), max(ys)]


def get_center(points_or_mask):
    """Get center from either list of points or binary mask."""
    if isinstance(points_or_mask, np.ndarray):
        # It's a mask
        return calculate_centroid(points_or_mask)
    else:
        # It's a list of points
        if not points_or_mask or len(points_or_mask) == 0:
            return (0, 0)
        xs = [p[0] for p in points_or_mask]
        ys = [p[1] for p in points_or_mask]
        return (int(np.mean(xs)), int(np.mean(ys)))


def calculate_box_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def estimate_mask_thickness(mask):
    """Estimate the thickness/height of a mask."""
    if mask.sum() == 0:
        return 0

    # Get bounding box
    bbox = get_bounding_box_from_mask(mask)
    height = bbox[3] - bbox[1]

    # Thickness is roughly 80% of the bounding box height
    return max(1, int(height * 0.8))


def smooth_mask_contours(mask, kernel_size=5, iterations=2):
    """
    Smooth mask boundaries perpendicular to contour (along normal vectors) while
    maintaining natural curvature.

    KEY INSIGHT: Use morphological operations with SMALL elliptical kernels.
    - Dilation/Erosion naturally shift boundaries along normal vectors (perpendicular)
    - Small kernels = gentle smoothing without destroying curvature
    - This removes zigzags/noise perpendicular to boundary
    - Preserves the natural arch shape of the eyebrow

    This is the correct approach for contour smoothing:
    1. Morphological operations work perpendicular to boundaries automatically
    2. Small kernels preserve curvature
    3. Multiple light passes are better than one heavy pass

    Parameters:
        mask: Binary mask (H, W), dtype=uint8
        kernel_size: Size of smoothing kernel (3, 5, or 7) - MUST be small
        iterations: Number of smoothing passes (1-3)

    Returns:
        Smoothed mask with natural curvature preserved and boundaries cleaned
    """
    if mask is None or np.sum(mask) == 0:
        return mask

    # Convert to uint8 if needed
    mask = mask.astype(np.uint8)

    # Use ELLIPTICAL kernel - matches eyebrow shape better than rectangle
    # Small kernel size is critical - too large will destroy curvature
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    smooth_mask = mask.copy()

    for _ in range(iterations):
        # Closing: dilation then erosion
        # Fills small gaps, smooths protrusions, works perpendicular to boundary
        smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel)

        # Opening: erosion then dilation
        # Removes small protrusions, smooths indentations, works perpendicular to boundary
        smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel)

    # Optional: Very light Gaussian blur on the boundary for sub-pixel smoothing
    # This further smooths pixelation without changing overall shape
    if kernel_size >= 5:
        smooth_float = smooth_mask.astype(np.float32)
        smooth_float = cv2.GaussianBlur(smooth_float, (3, 3), 0.8)
        smooth_mask = (smooth_float > 0.5).astype(np.uint8)

    return smooth_mask


def calculate_mask_thickness(mask):
    """
    Calculate average thickness (width) of a mask.

    Thickness = area / horizontal_span
    This gives average perpendicular width of the eyebrow.

    Parameters:
        mask: Binary mask (H, W)

    Returns:
        Average thickness in pixels
    """
    if mask is None or np.sum(mask) == 0:
        return 0.0

    # Calculate area
    area = np.sum(mask)

    # Calculate horizontal span
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return 0.0

    horizontal_span = np.max(x_coords) - np.min(x_coords) + 1

    if horizontal_span == 0:
        return 0.0

    # Thickness = area / length
    thickness = area / horizontal_span

    return thickness


def constrain_to_horizontal_bounds(mask, x_min, x_max):
    """
    Constrain mask to horizontal boundaries (x coordinates).

    Clips the mask to only include pixels within [x_min, x_max] range.
    This prevents eyebrow from extending beyond eye_box horizontally.

    Parameters:
        mask: Binary mask (H, W)
        x_min: Minimum x coordinate (inclusive)
        x_max: Maximum x coordinate (inclusive)

    Returns:
        Constrained mask
    """
    if mask is None:
        return mask

    h, w = mask.shape
    constrained = mask.copy()

    # Zero out everything outside horizontal bounds
    if x_min > 0:
        constrained[:, :x_min] = 0
    if x_max < w - 1:
        constrained[:, x_max+1:] = 0

    return constrained


# =============================================================================
# FACE ALIGNMENT & ROTATION FUNCTIONS
# =============================================================================

def detect_face_rotation(detections: Dict, mediapipe_landmarks: Dict) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """
    Detect face rotation angle using eye positions.

    Tries MediaPipe eye landmarks first (more accurate), falls back to YOLO eye detections.

    Args:
        detections: YOLO detection dict with 'eye' key
        mediapipe_landmarks: MediaPipe landmark dict with 'left_eye' and 'right_eye' keys

    Returns:
        (angle, left_eye_center, right_eye_center)
        - angle: rotation angle in degrees (positive = counterclockwise)
        - left_eye_center: (x, y) position of left eye
        - right_eye_center: (x, y) position of right eye
    """
    left_eye = None
    right_eye = None

    # Try MediaPipe first (preferred)
    if mediapipe_landmarks and 'left_eye' in mediapipe_landmarks and 'right_eye' in mediapipe_landmarks:
        if mediapipe_landmarks['left_eye'] and mediapipe_landmarks['right_eye']:
            left_eye = mediapipe_landmarks['left_eye']['center']
            right_eye = mediapipe_landmarks['right_eye']['center']

    # Fallback to YOLO eye detections
    if left_eye is None or right_eye is None:
        if detections and 'eye' in detections and len(detections['eye']) >= 2:
            # Sort eyes left to right (left eye has smaller x)
            eyes = sorted(detections['eye'], key=lambda e: e['mask_centroid'][0])
            left_eye = eyes[0]['mask_centroid']
            right_eye = eyes[1]['mask_centroid']

    # If still no eyes found, return 0 rotation
    if left_eye is None or right_eye is None:
        return 0.0, (0, 0), (0, 0)

    # Calculate rotation angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, left_eye, right_eye


def should_straighten_face(angle: float, threshold: float = 5.0) -> bool:
    """
    Determine if face should be straightened based on rotation angle.

    Args:
        angle: rotation angle in degrees
        threshold: threshold in degrees (default: 5.0)

    Returns:
        True if abs(angle) > threshold
    """
    return abs(angle) > threshold


def straighten_face(img: np.ndarray, angle: float, left_eye: Tuple[int, int],
                   right_eye: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Straighten face by rotating image to make eyes horizontal.

    Args:
        img: input image
        angle: rotation angle in degrees
        left_eye: (x, y) position of left eye
        right_eye: (x, y) position of right eye

    Returns:
        (straightened_image, transformation_matrix)
        - straightened_image: rotated image
        - transformation_matrix: 2x3 affine transformation matrix
    """
    h, w = img.shape[:2]

    # Calculate rotation center (midpoint between eyes)
    center_x = (left_eye[0] + right_eye[0]) // 2
    center_y = (left_eye[1] + right_eye[1]) // 2
    center = (center_x, center_y)

    # Get rotation matrix (rotate by -angle to straighten)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Apply rotation
    straightened = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return straightened, M


def transform_detections(detections: Dict, M: np.ndarray, img_shape: Tuple[int, int]) -> Dict:
    """
    Transform YOLO detections using affine transformation matrix.

    Applies transformation to bounding boxes, masks, centroids, and centers.

    Args:
        detections: YOLO detection dict
        M: 2x3 affine transformation matrix
        img_shape: (height, width) of image

    Returns:
        Transformed detection dict with same structure
    """
    transformed = {}

    for class_name, det_list in detections.items():
        transformed[class_name] = []

        for det in det_list:
            new_det = det.copy()

            # Transform bounding box
            if 'box' in det:
                new_det['box'] = rotate_bbox(det['box'], M, img_shape)
                x1, y1, x2, y2 = new_det['box']
                new_det['box_width'] = x2 - x1
                new_det['box_height'] = y2 - y1
                new_det['center'] = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Transform mask
            if 'mask' in det:
                new_det['mask'] = rotate_mask(det['mask'], M, img_shape)
                new_det['mask_area'] = int(new_det['mask'].sum())
                new_det['mask_centroid'] = calculate_centroid(new_det['mask'])

            transformed[class_name].append(new_det)

    return transformed


def transform_mediapipe(mediapipe_landmarks: Dict, M: np.ndarray) -> Dict:
    """
    Transform MediaPipe landmarks using affine transformation matrix.

    Applies transformation to all landmark points, centers, and bounding boxes.

    Args:
        mediapipe_landmarks: MediaPipe landmark dict
        M: 2x3 affine transformation matrix

    Returns:
        Transformed landmark dict with same structure
    """
    transformed = {}

    for key, landmark_group in mediapipe_landmarks.items():
        if key == 'all_landmarks':
            # Skip raw landmarks object (can't transform easily)
            transformed[key] = landmark_group
            continue

        if not landmark_group:
            transformed[key] = None
            continue

        new_group = {}

        # Transform points
        if 'points' in landmark_group:
            points = landmark_group['points']
            transformed_points = []

            for x, y in points:
                # Apply affine transformation
                new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                transformed_points.append((int(new_x), int(new_y)))

            new_group['points'] = transformed_points

            # Recalculate center and bbox from transformed points
            new_group['center'] = get_center(transformed_points)
            new_group['bbox'] = get_bbox_from_points(transformed_points)

        # Copy indices (unchanged)
        if 'indices' in landmark_group:
            new_group['indices'] = landmark_group['indices']

        transformed[key] = new_group

    return transformed


# =============================================================================
# MASK TRANSFORMATION FUNCTIONS
# =============================================================================

def rotate_bbox(box, M, img_shape):
    """Rotate bounding box using affine transformation matrix."""
    h, w = img_shape[:2]

    # Get box corners
    x1, y1, x2, y2 = box
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])

    # Add homogeneous coordinate
    ones = np.ones(shape=(len(corners), 1))
    corners_homogeneous = np.hstack([corners, ones])

    # Transform corners
    transformed_corners = M.dot(corners_homogeneous.T).T

    # Get new bounding box
    x_coords = transformed_corners[:, 0]
    y_coords = transformed_corners[:, 1]

    new_box = [
        max(0, float(x_coords.min())),
        max(0, float(y_coords.min())),
        min(w, float(x_coords.max())),
        min(h, float(y_coords.max()))
    ]

    return new_box


def rotate_mask(mask, M, img_shape):
    """Rotate binary mask using affine transformation matrix."""
    h, w = img_shape[:2]

    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_mask


# =============================================================================
# MASK PROCESSING FUNCTIONS
# =============================================================================

def remove_small_components(mask, min_size=50):
    """Remove small disconnected components from binary mask."""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    # Create output mask
    cleaned_mask = np.zeros_like(mask)

    # Keep components larger than min_size (skip label 0 = background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned_mask[labels == i] = 1

    return cleaned_mask


def fill_holes(mask):
    """Fill holes inside binary mask."""
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    filled = mask.copy()

    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # Has parent = is a hole
                cv2.drawContours(filled, contours, i, 1, -1)

    return filled


# =============================================================================
# SPLINE & ARCH CREATION FUNCTIONS
# =============================================================================

def create_arch_from_landmarks_parametric(mp_points, h, w, thickness):
    """
    Create arch-shaped mask using PARAMETRIC spline interpolation.
    This handles non-monotonic x-coordinates (curved eyebrows).

    Args:
        mp_points: List of (x, y) MediaPipe landmark points
        h: image height
        w: image width
        thickness: arch thickness in pixels

    Returns:
        Binary mask with drawn arch
    """
    from scipy.interpolate import splprep, splev

    # Sort points left to right
    mp_sorted = sorted(mp_points, key=lambda p: p[0])
    xs = np.array([p[0] for p in mp_sorted])
    ys = np.array([p[1] for p in mp_sorted])

    try:
        # Parametric spline (handles non-monotonic x)
        tck, u = splprep([xs, ys], s=len(xs)*3, k=min(3, len(xs)-1))

        # Sample dense points along curve
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)

    except Exception as e:
        print(f"Spline fitting failed: {e}, using linear interpolation")
        # Fallback to linear interpolation
        from scipy.interpolate import interp1d
        t = np.linspace(0, 1, len(xs))
        fx = interp1d(t, xs, kind='linear')
        fy = interp1d(t, ys, kind='linear')

        t_new = np.linspace(0, 1, 100)
        x_new = fx(t_new)
        y_new = fy(t_new)

    # Draw arch
    arch_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in zip(x_new, y_new):
        if 0 <= int(x) < w and 0 <= int(y) < h:
            cv2.circle(arch_mask, (int(x), int(y)), thickness // 2, 1, -1)

    return arch_mask


# =============================================================================
# DISTANCE TRANSFORM FUNCTIONS
# =============================================================================

def create_distance_map_from_mask(mask):
    """Create distance transform from a binary mask."""
    # Distance transform gives distance to nearest zero pixel
    dist = cv2.distanceTransform(
        (mask == 0).astype(np.uint8),
        cv2.DIST_L2, 5
    )

    # Normalize to [0, 1]
    if dist.max() > 0:
        dist = dist / dist.max()

    # Invert so that mask pixels have high values
    dist = 1 - dist

    return dist


# =============================================================================
# EYEBROW ADJUSTMENT FUNCTIONS (for API endpoints)
# =============================================================================

def adjust_eyebrow_thickness(mask, factor):
    """
    Adjust eyebrow thickness by factor while maintaining natural curvature.

    Uses morphological dilation/erosion which naturally works perpendicular to
    contours (along normal vectors). This creates parallel offsets that maintain
    the eyebrow arch shape.

    KEY INSIGHT: Morphological operations shift boundaries along normal vectors!
    - Dilation = expand perpendicular outward
    - Erosion = contract perpendicular inward

    Parameters:
        mask: Binary eyebrow mask (H, W), dtype=uint8
        factor: Thickness adjustment factor
                - factor > 1.0 = thicker (e.g., 1.05 = +5%)
                - factor < 1.0 = thinner (e.g., 0.95 = -5%)
                - factor = 1.0 = no change

    Returns:
        Adjusted mask with same shape, maintaining natural curvature

    Example:
        # Increase thickness by 5%
        thicker = adjust_eyebrow_thickness(mask, 1.05)

        # Decrease thickness by 5%
        thinner = adjust_eyebrow_thickness(mask, 0.95)
    """
    if mask is None or np.sum(mask) == 0:
        return mask

    mask = mask.astype(np.uint8)

    # Calculate current thickness
    current_thickness = calculate_mask_thickness(mask)

    if current_thickness == 0:
        return mask

    # Calculate target thickness change
    target_thickness = current_thickness * factor
    thickness_delta = target_thickness - current_thickness

    # Convert thickness change to kernel size for morphological operation
    # Each iteration of dilation/erosion shifts boundary by ~1 pixel
    kernel_size = max(3, int(abs(thickness_delta) / 2) + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Must be odd

    # Use elliptical kernel to match eyebrow shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if factor > 1.0:
        # INCREASE thickness - dilate (expand perpendicular outward)
        adjusted = cv2.dilate(mask, kernel, iterations=1)
    elif factor < 1.0:
        # DECREASE thickness - erode (contract perpendicular inward)
        adjusted = cv2.erode(mask, kernel, iterations=1)
    else:
        # No change
        adjusted = mask.copy()

    # Light smoothing to maintain clean edges
    adjusted = smooth_mask_contours(adjusted, kernel_size=3, iterations=1)

    return adjusted


def adjust_eyebrow_span_morphological(mask, factor, side='unknown'):
    """
    Adjust eyebrow span using simple morphological operations on the TAIL region only.

    KEY: Eyebrow curves/bows, so tail isn't leftmost/rightmost point!
    Solution: Take last 1/3 of eyebrow (the actual tail), apply erosion/dilation there.

    This is REVERSIBLE: increase then decrease returns to original.

    Parameters:
        mask: Binary eyebrow mask (H, W)
        factor: Span adjustment factor (>1.0 = increase, <1.0 = decrease)
        side: 'left' or 'right'

    Returns:
        Adjusted mask with tail extended/contracted
    """
    if mask is None or np.sum(mask) == 0:
        return mask

    if factor == 1.0:
        return mask

    mask = mask.astype(np.uint8)
    h, w = mask.shape

    # Get bounding box
    bbox = get_bounding_box_from_mask(mask)
    if bbox is None:
        return mask

    x_min, y_min, x_max, y_max = bbox
    span = x_max - x_min

    # Calculate kernel size based on factor (amplify 1.5x for visible but reasonable change)
    kernel_width = max(5, int(abs(span * (factor - 1.0)) * 1.5))
    if kernel_width % 2 == 0:
        kernel_width += 1  # Must be odd

    # Horizontal kernel (1 pixel tall for thin extension)
    kernel = np.ones((1, kernel_width), dtype=np.uint8)

    # Define tail region (last 1/3 of eyebrow)
    tail_fraction = 1.0 / 3.0
    tail_width = int(span * tail_fraction)

    # Create protection mask (protects center 2/3, modifies tail 1/3)
    protection_mask = np.zeros((h, w), dtype=np.uint8)

    if side == 'left':
        # Left eyebrow: tail is on the LEFT (small x values)
        # Protect RIGHT 2/3, modify LEFT 1/3
        tail_end = x_min + tail_width
        protection_mask[:, tail_end:] = 1
    else:  # side == 'right'
        # Right eyebrow: tail is on the RIGHT (large x values)
        # Protect LEFT 2/3, modify RIGHT 1/3
        tail_start = x_max - tail_width
        protection_mask[:, :tail_start] = 1

    # Apply morphological operation
    if factor > 1.0:
        # INCREASE span - dilate tail only
        full_dilate = cv2.dilate(mask, kernel, iterations=1)
        result = np.where(protection_mask > 0, mask, full_dilate).astype(np.uint8)
    else:
        # DECREASE span - erode tail only
        full_erode = cv2.erode(mask, kernel, iterations=1)
        result = np.where(protection_mask > 0, mask, full_erode).astype(np.uint8)

    # Light smoothing for natural appearance
    result = smooth_mask_contours(result, kernel_size=3, iterations=1)

    return result


def adjust_eyebrow_span(mask, factor, side='unknown', directional=True):
    """
    Adjust eyebrow horizontal span (width/length) using morphological operations on TAIL region.

    KEY INSIGHT: Eyebrow curves/bows, so tail isn't just leftmost/rightmost!
    - Takes last 1/3 of eyebrow (actual tail region based on bounding box)
    - Applies erosion/dilation ONLY to that tail region
    - Protects center 2/3 from modification
    - REVERSIBLE: increase then decrease returns to original!

    Parameters:
        mask: Binary eyebrow mask (H, W), dtype=uint8
        factor: Span adjustment factor
                - factor > 1.0 = longer (e.g., 1.05 = +5%)
                - factor < 1.0 = shorter (e.g., 0.95 = -5%)
                - factor = 1.0 = no change
        side: Eyebrow side ('left', 'right', or 'unknown')
              - 'left': tail is LEFT 1/3, center is RIGHT 2/3
              - 'right': tail is RIGHT 1/3, center is LEFT 2/3
        directional: If True, adjust only tail (default: True)

    Returns:
        Adjusted mask, reversible to original

    Example:
        # Increase span by 5%
        longer = adjust_eyebrow_span(mask, 1.05, side='left')
        # Decrease back
        original = adjust_eyebrow_span(longer, 0.95, side='left')  # Returns to original!
    """
    if mask is None or np.sum(mask) == 0:
        return mask

    # Use simple morphological approach (reversible!)
    if directional and side in ['left', 'right']:
        return adjust_eyebrow_span_morphological(mask, factor, side)

    # SYMMETRIC EXPANSION (old behavior, when side unknown)
    mask = mask.astype(np.uint8)
    bbox = get_bounding_box_from_mask(mask)
    if bbox is None:
        return mask

    span = bbox[2] - bbox[0]
    kernel_width = max(5, int(abs(span * (factor - 1.0))))
    if kernel_width % 2 == 0:
        kernel_width += 1

    kernel = np.ones((1, kernel_width), dtype=np.uint8)

    if factor > 1.0:
        adjusted = cv2.dilate(mask, kernel, iterations=1)
    elif factor < 1.0:
        adjusted = cv2.erode(mask, kernel, iterations=1)
    else:
        adjusted = mask.copy()

    adjusted = smooth_mask_contours(adjusted, kernel_size=3, iterations=1)
    return adjusted


def apply_eyebrow_adjustment(mask, adjustment_type, direction, increment=0.05, side='unknown'):
    """
    Generic eyebrow adjustment function - single entry point for all adjustments.

    This is the UNIFIED LOGIC that works for all adjustment operations.

    Parameters:
        mask: Binary eyebrow mask (H, W), dtype=uint8
        adjustment_type: Type of adjustment ('thickness' or 'span')
        direction: Direction of adjustment ('increase' or 'decrease')
        increment: Percentage increment per adjustment (default: 0.05 = 5%)
        side: Eyebrow side ('left', 'right', or 'unknown')
              Only used for 'span' adjustment to enable directional expansion

    Returns:
        Adjusted mask with natural curvature preserved

    Example:
        # Increase thickness by 5% (uniform)
        result = apply_eyebrow_adjustment(mask, 'thickness', 'increase')

        # Decrease span by 5% (directional - tail only)
        result = apply_eyebrow_adjustment(mask, 'span', 'decrease', side='left')

        # Custom increment (10%)
        result = apply_eyebrow_adjustment(mask, 'thickness', 'increase', increment=0.10)
    """
    if mask is None or np.sum(mask) == 0:
        return mask

    # Calculate adjustment factor
    if direction == 'increase':
        factor = 1.0 + increment  # e.g., 1.05 for +5%
    elif direction == 'decrease':
        factor = 1.0 - increment  # e.g., 0.95 for -5%
    else:
        raise ValueError(f"Invalid direction: {direction}. Use 'increase' or 'decrease'.")

    # Apply appropriate adjustment
    if adjustment_type == 'thickness':
        return adjust_eyebrow_thickness(mask, factor)
    elif adjustment_type == 'span':
        return adjust_eyebrow_span(mask, factor, side=side, directional=True)
    else:
        raise ValueError(f"Invalid adjustment_type: {adjustment_type}. Use 'thickness' or 'span'.")


def adjust_eyebrow_multiple_times(mask, adjustment_type, direction, num_clicks, increment=0.05, side='unknown'):
    """
    Apply eyebrow adjustment multiple times (for multiple button clicks).

    This simulates the user clicking the +/- button multiple times.
    Each click applies the increment independently.

    Parameters:
        mask: Binary eyebrow mask (H, W), dtype=uint8
        adjustment_type: 'thickness' or 'span'
        direction: 'increase' or 'decrease'
        num_clicks: Number of times to apply adjustment
        increment: Percentage increment per click (default: 0.05 = 5%)
        side: Eyebrow side ('left', 'right', or 'unknown')

    Returns:
        Adjusted mask after num_clicks applications

    Example:
        # User clicks "increase thickness" 3 times
        result = adjust_eyebrow_multiple_times(mask, 'thickness', 'increase', 3)
        # Result is 3 × 5% = 15% thicker

        # User clicks "increase span" 2 times (directional tail expansion)
        result = adjust_eyebrow_multiple_times(mask, 'span', 'increase', 2, side='left')
    """
    adjusted = mask.copy()

    for _ in range(num_clicks):
        adjusted = apply_eyebrow_adjustment(adjusted, adjustment_type, direction, increment, side=side)

    return adjusted
