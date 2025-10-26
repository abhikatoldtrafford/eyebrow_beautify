"""
Face Preprocessing Module

Comprehensive face preprocessing with:
- Multi-source angle calculation (MediaPipe, YOLO eyes, eye_box)
- Face sanity checks (eyes, eyebrows, quality)
- Asymmetry detection and correction
- Robust outlier removal using statistics
- Face alignment and normalization

This module runs BEFORE the main beautification pipeline to ensure
high-quality input and reject defective faces.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import mediapipe_pred
import yolo_pred


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PREPROCESS_CONFIG = {
    # Angle calculation
    'max_rotation_angle': 30.0,         # Reject if face rotation > this (degrees)
    'min_rotation_threshold': 2.5,      # Only correct rotation if > this (degrees) - increased from 1.0
    'angle_outlier_threshold': 2.0,     # IQR multiplier for outlier removal
    'min_angle_sources': 2,             # Minimum angle sources required
    'max_source_disagreement': 2.0,     # Max angle difference between sources (degrees)

    # Eye validation
    'require_both_eyes': True,          # Reject if both eyes not detected
    'min_eye_distance_pct': 0.15,       # Min eye distance (% of image width)
    'max_eye_distance_pct': 0.50,       # Max eye distance (% of image width)
    'max_eye_vertical_diff_pct': 0.05,  # Max vertical difference between eyes

    # Eyebrow validation
    'require_both_eyebrows': True,      # Reject if both eyebrows not detected
    'min_eyebrow_eye_overlap': 0.0,     # Min IoU between eyebrow and eye (should be 0)
    'max_eyebrow_eye_overlap': 0.05,    # Max IoU (slight overlap acceptable)
    'min_yolo_mp_overlap': 0.30,        # Min IoU between YOLO eyebrow and MP landmarks
    'eyebrow_above_eye_margin': 0.02,   # Eyebrow must be above eye by this % of height

    # Asymmetry detection
    'max_eyebrow_angle_diff': 10.0,     # Max angle difference between left/right (degrees)
    'max_eyebrow_position_diff_pct': 0.10,  # Max vertical position difference
    'correct_asymmetry': True,          # Auto-correct asymmetries if possible

    # Face quality
    'min_face_size': 200,               # Minimum face dimension (pixels)
    'mediapipe_min_confidence': 0.5,    # MediaPipe detection confidence
    'yolo_min_confidence': 0.25,        # YOLO detection confidence
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_angle_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate angle in degrees between two points.

    Args:
        p1: (x, y) first point
        p2: (x, y) second point

    Returns:
        Angle in degrees (0 = horizontal, positive = counterclockwise)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def remove_outliers_iqr(values: List[float], threshold: float = 2.0) -> List[float]:
    """
    Remove outliers using IQR method.

    Args:
        values: List of values
        threshold: IQR multiplier (default: 2.0)

    Returns:
        Filtered values without outliers
    """
    if len(values) < 3:
        return values

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    filtered = [v for v in values if lower_bound <= v <= upper_bound]

    return filtered if len(filtered) > 0 else values


# =============================================================================
# MULTI-SOURCE ANGLE CALCULATION
# =============================================================================

def calculate_eye_angle_mediapipe(mp_detections: Dict) -> Optional[float]:
    """
    Calculate eye angle using MediaPipe landmarks.

    Args:
        mp_detections: MediaPipe detection results

    Returns:
        Angle in degrees, or None if not available
    """
    if not mp_detections:
        return None

    left_eye = mp_detections.get('left_eye')
    right_eye = mp_detections.get('right_eye')

    if not left_eye or not right_eye:
        return None

    left_center = left_eye['center']
    right_center = right_eye['center']

    angle = calculate_angle_between_points(left_center, right_center)
    return angle


def calculate_eye_angle_yolo(yolo_detections: Dict) -> Optional[float]:
    """
    Calculate eye angle using YOLO eye detections.

    Args:
        yolo_detections: YOLO detection results

    Returns:
        Angle in degrees, or None if not available
    """
    eyes = yolo_detections.get('eye', [])

    if len(eyes) < 2:
        return None

    # Sort by x-coordinate (left to right)
    eyes_sorted = sorted(eyes, key=lambda e: e['mask_centroid'][0])
    left_eye = eyes_sorted[0]
    right_eye = eyes_sorted[-1]

    angle = calculate_angle_between_points(left_eye['mask_centroid'], right_eye['mask_centroid'])
    return angle


def calculate_eye_angle_eyebox(yolo_detections: Dict) -> Optional[float]:
    """
    Calculate eye angle using YOLO eye_box detections.

    Args:
        yolo_detections: YOLO detection results

    Returns:
        Angle in degrees, or None if not available
    """
    eye_boxes = yolo_detections.get('eye_box', [])

    if len(eye_boxes) < 2:
        return None

    # Sort by x-coordinate (left to right)
    boxes_sorted = sorted(eye_boxes, key=lambda e: e['mask_centroid'][0])
    left_box = boxes_sorted[0]
    right_box = boxes_sorted[-1]

    angle = calculate_angle_between_points(left_box['mask_centroid'], right_box['mask_centroid'])
    return angle


def calculate_eyebrow_angle_yolo(yolo_detections: Dict) -> Optional[float]:
    """
    Calculate rotation angle using YOLO eyebrow detections.
    Eyebrows are often more reliable than eyes for rotation detection
    since they span a wider horizontal distance.

    Args:
        yolo_detections: YOLO detection results

    Returns:
        Angle in degrees, or None if not available
    """
    eyebrows = yolo_detections.get('eyebrows', [])

    if len(eyebrows) < 2:
        return None

    # Sort by x-coordinate (left to right)
    eyebrows_sorted = sorted(eyebrows, key=lambda e: e['mask_centroid'][0])
    left_eyebrow = eyebrows_sorted[0]
    right_eyebrow = eyebrows_sorted[-1]

    angle = calculate_angle_between_points(left_eyebrow['mask_centroid'], right_eyebrow['mask_centroid'])
    return angle


def robust_angle_estimation(
    mp_detections: Dict,
    yolo_detections: Dict,
    config: Dict
) -> Tuple[Optional[float], Dict]:
    """
    Calculate robust angle estimate from multiple sources with outlier removal.

    Strategy:
    1. Collect angles from all available sources
    2. Remove outliers using IQR method
    3. Use median of remaining values
    4. Return angle + metadata

    Args:
        mp_detections: MediaPipe detection results
        yolo_detections: YOLO detection results
        config: Preprocessing configuration

    Returns:
        (final_angle, metadata_dict)
    """
    angles = []
    sources = []

    # MediaPipe eye angle (highest priority)
    mp_angle = calculate_eye_angle_mediapipe(mp_detections)
    if mp_angle is not None:
        angles.append(mp_angle)
        sources.append('mediapipe_eyes')

    # YOLO eye angle
    yolo_eye_angle = calculate_eye_angle_yolo(yolo_detections)
    if yolo_eye_angle is not None:
        angles.append(yolo_eye_angle)
        sources.append('yolo_eyes')

    # YOLO eye_box angle
    eyebox_angle = calculate_eye_angle_eyebox(yolo_detections)
    if eyebox_angle is not None:
        angles.append(eyebox_angle)
        sources.append('yolo_eyebox')

    # YOLO eyebrow angle (NEW - often most reliable)
    eyebrow_angle = calculate_eyebrow_angle_yolo(yolo_detections)
    if eyebrow_angle is not None:
        angles.append(eyebrow_angle)
        sources.append('yolo_eyebrows')

    metadata = {
        'raw_angles': angles.copy(),
        'sources': sources.copy(),
        'num_sources': len(angles)
    }

    # Need minimum number of sources
    if len(angles) < config['min_angle_sources']:
        metadata['status'] = 'insufficient_sources'
        metadata['final_angle'] = None
        return None, metadata

    # Check source agreement (NEW - reject if sources disagree too much)
    if len(angles) >= 2:
        max_diff = max(angles) - min(angles)
        metadata['source_agreement'] = max_diff

        if max_diff > config['max_source_disagreement']:
            metadata['status'] = 'source_disagreement'
            metadata['final_angle'] = None
            metadata['warning'] = f'Sources disagree by {max_diff:.2f}° (>{config["max_source_disagreement"]}°)'
            return None, metadata

    # Remove outliers using IQR
    filtered_angles = remove_outliers_iqr(angles, config['angle_outlier_threshold'])
    metadata['filtered_angles'] = filtered_angles
    metadata['outliers_removed'] = len(angles) - len(filtered_angles)

    if len(filtered_angles) == 0:
        metadata['status'] = 'all_outliers'
        metadata['final_angle'] = None
        return None, metadata

    # Use median of filtered values
    final_angle = np.median(filtered_angles)
    metadata['final_angle'] = final_angle
    metadata['angle_std'] = np.std(filtered_angles) if len(filtered_angles) > 1 else 0.0
    metadata['status'] = 'success'

    return final_angle, metadata


# =============================================================================
# FACE SANITY CHECKS
# =============================================================================

def validate_eyes(
    mp_detections: Dict,
    yolo_detections: Dict,
    img_shape: Tuple[int, int],
    config: Dict
) -> Tuple[bool, Dict]:
    """
    Validate that both eyes are detected and positioned correctly.

    Checks:
    1. Both eyes detected (MediaPipe or YOLO)
    2. Eye distance is reasonable
    3. Eyes are approximately horizontal

    Args:
        mp_detections: MediaPipe detection results
        yolo_detections: YOLO detection results
        img_shape: (height, width)
        config: Preprocessing configuration

    Returns:
        (is_valid, validation_details)
    """
    h, w = img_shape
    details = {}

    # Check MediaPipe eyes
    mp_has_eyes = bool(
        mp_detections and
        'left_eye' in mp_detections and
        'right_eye' in mp_detections and
        mp_detections['left_eye'] is not None and
        mp_detections['right_eye'] is not None
    )

    # Check YOLO eyes
    yolo_eyes = yolo_detections.get('eye', [])
    yolo_has_eyes = len(yolo_eyes) >= 2

    details['mediapipe_has_eyes'] = mp_has_eyes
    details['yolo_has_eyes'] = yolo_has_eyes
    details['yolo_eye_count'] = len(yolo_eyes)

    if config['require_both_eyes'] and not (mp_has_eyes or yolo_has_eyes):
        details['status'] = 'missing_eyes'
        details['is_valid'] = False
        return False, details

    # Use MediaPipe if available, otherwise YOLO
    if mp_has_eyes:
        left_center = mp_detections['left_eye']['center']
        right_center = mp_detections['right_eye']['center']
        details['eye_source'] = 'mediapipe'
    elif yolo_has_eyes:
        eyes_sorted = sorted(yolo_eyes, key=lambda e: e['mask_centroid'][0])
        left_center = eyes_sorted[0]['mask_centroid']
        right_center = eyes_sorted[-1]['mask_centroid']
        details['eye_source'] = 'yolo'
    else:
        details['status'] = 'no_eye_source'
        details['is_valid'] = False
        return False, details

    # Calculate eye distance
    eye_distance = np.sqrt((right_center[0] - left_center[0])**2 +
                           (right_center[1] - left_center[1])**2)
    eye_distance_pct = eye_distance / w

    details['eye_distance_px'] = eye_distance
    details['eye_distance_pct'] = eye_distance_pct

    # Check eye distance range
    if not (config['min_eye_distance_pct'] <= eye_distance_pct <= config['max_eye_distance_pct']):
        details['status'] = 'eye_distance_invalid'
        details['is_valid'] = False
        return False, details

    # Check vertical alignment
    vertical_diff = abs(right_center[1] - left_center[1])
    vertical_diff_pct = vertical_diff / h

    details['vertical_diff_px'] = vertical_diff
    details['vertical_diff_pct'] = vertical_diff_pct

    if vertical_diff_pct > config['max_eye_vertical_diff_pct']:
        details['status'] = 'eyes_not_horizontal'
        details['is_valid'] = False
        return False, details

    details['status'] = 'valid'
    details['is_valid'] = True
    return True, details


def validate_eyebrows(
    mp_detections: Dict,
    yolo_detections: Dict,
    img_shape: Tuple[int, int],
    config: Dict
) -> Tuple[bool, Dict]:
    """
    Validate that eyebrows are detected and positioned correctly.

    Checks:
    1. Both eyebrows detected
    2. Eyebrows are above eyes (not overlapping)
    3. YOLO eyebrow and MediaPipe landmarks overlap

    Args:
        mp_detections: MediaPipe detection results
        yolo_detections: YOLO detection results
        img_shape: (height, width)
        config: Preprocessing configuration

    Returns:
        (is_valid, validation_details)
    """
    h, w = img_shape
    details = {}

    # Check YOLO eyebrows
    yolo_eyebrows = yolo_detections.get('eyebrows', [])
    details['yolo_eyebrow_count'] = len(yolo_eyebrows)

    if config['require_both_eyebrows'] and len(yolo_eyebrows) < 2:
        details['status'] = 'missing_eyebrows'
        details['is_valid'] = False
        return False, details

    # Check MediaPipe eyebrows
    mp_has_eyebrows = bool(
        mp_detections and
        'left_eyebrow' in mp_detections and
        'right_eyebrow' in mp_detections and
        mp_detections['left_eyebrow'] is not None and
        mp_detections['right_eyebrow'] is not None
    )

    details['mediapipe_has_eyebrows'] = mp_has_eyebrows

    # Check YOLO eyebrows don't overlap with eyes
    yolo_eyes = yolo_detections.get('eye', [])

    if len(yolo_eyebrows) > 0 and len(yolo_eyes) > 0:
        # Sort eyebrows and eyes by x-coordinate
        eyebrows_sorted = sorted(yolo_eyebrows, key=lambda e: e['mask_centroid'][0])
        eyes_sorted = sorted(yolo_eyes, key=lambda e: e['mask_centroid'][0])

        overlaps = []
        above_eyes = []

        for i, eyebrow in enumerate(eyebrows_sorted[:2]):  # Check up to 2 eyebrows
            if i < len(eyes_sorted):
                eye = eyes_sorted[i]

                # Calculate IoU
                eyebrow_box = eyebrow['box']
                eye_box = eye['box']
                iou = calculate_iou(eyebrow_box, eye_box)
                overlaps.append(iou)

                # Check if eyebrow is above eye
                eyebrow_y = eyebrow['mask_centroid'][1]
                eye_y = eye['mask_centroid'][1]
                margin_px = config['eyebrow_above_eye_margin'] * h
                is_above = eyebrow_y < (eye_y - margin_px)
                above_eyes.append(is_above)

        details['eyebrow_eye_overlaps'] = overlaps
        details['eyebrows_above_eyes'] = above_eyes

        # Check overlap constraint
        if any(iou > config['max_eyebrow_eye_overlap'] for iou in overlaps):
            details['status'] = 'eyebrow_overlaps_eye'
            details['is_valid'] = False
            return False, details

        # Check position constraint
        if not all(above_eyes):
            details['status'] = 'eyebrow_not_above_eye'
            details['is_valid'] = False
            return False, details

    # Check YOLO-MediaPipe overlap (if both available)
    if mp_has_eyebrows and len(yolo_eyebrows) >= 2:
        mp_overlaps = []

        for side in ['left', 'right']:
            mp_eyebrow = mp_detections.get(f'{side}_eyebrow')
            if mp_eyebrow:
                mp_bbox = mp_eyebrow['bbox']
                # bbox is a list: [x1, y1, x2, y2]
                mp_box = mp_bbox if isinstance(mp_bbox, list) else [mp_bbox['x1'], mp_bbox['y1'], mp_bbox['x2'], mp_bbox['y2']]

                # Find corresponding YOLO eyebrow (by position)
                yolo_eyebrow = eyebrows_sorted[0] if side == 'left' else eyebrows_sorted[-1]
                yolo_box = yolo_eyebrow['box']

                iou = calculate_iou(mp_box, yolo_box)
                mp_overlaps.append(iou)

        details['yolo_mediapipe_overlaps'] = mp_overlaps

        if any(iou < config['min_yolo_mp_overlap'] for iou in mp_overlaps):
            details['status'] = 'yolo_mediapipe_mismatch'
            details['is_valid'] = False
            return False, details

    details['status'] = 'valid'
    details['is_valid'] = True
    return True, details


def validate_face_quality(
    mp_detections: Dict,
    yolo_detections: Dict,
    img_shape: Tuple[int, int],
    config: Dict
) -> Tuple[bool, Dict]:
    """
    Validate overall face quality.

    Checks:
    1. Face size is adequate
    2. Detection confidence is adequate
    3. No extreme rotations

    Args:
        mp_detections: MediaPipe detection results
        yolo_detections: YOLO detection results
        img_shape: (height, width)
        config: Preprocessing configuration

    Returns:
        (is_valid, validation_details)
    """
    h, w = img_shape
    details = {}

    # Check minimum face size
    min_dim = min(h, w)
    details['image_size'] = (h, w)
    details['min_dimension'] = min_dim

    if min_dim < config['min_face_size']:
        details['status'] = 'face_too_small'
        details['is_valid'] = False
        return False, details

    # Check MediaPipe confidence
    mp_confidence = 1.0  # MediaPipe doesn't provide overall confidence, assume 1.0 if detected
    if mp_detections:
        details['mediapipe_detected'] = True
        details['mediapipe_confidence'] = mp_confidence
    else:
        details['mediapipe_detected'] = False
        details['mediapipe_confidence'] = 0.0

    # Check YOLO confidence
    yolo_eyebrows = yolo_detections.get('eyebrows', [])
    if yolo_eyebrows:
        avg_confidence = np.mean([eb['confidence'] for eb in yolo_eyebrows])
        details['yolo_avg_confidence'] = avg_confidence

        if avg_confidence < config['yolo_min_confidence']:
            details['status'] = 'low_yolo_confidence'
            details['is_valid'] = False
            return False, details
    else:
        details['yolo_avg_confidence'] = 0.0

    details['status'] = 'valid'
    details['is_valid'] = True
    return True, details


# =============================================================================
# ASYMMETRY DETECTION
# =============================================================================

def detect_eyebrow_asymmetry(
    mp_detections: Dict,
    yolo_detections: Dict,
    img_shape: Tuple[int, int],
    config: Dict
) -> Dict:
    """
    Detect asymmetries between left and right eyebrows.

    Detects:
    1. Angle asymmetry (different slopes)
    2. Vertical position asymmetry
    3. Horizontal span asymmetry

    Args:
        mp_detections: MediaPipe detection results
        yolo_detections: YOLO detection results
        img_shape: (height, width)
        config: Preprocessing configuration

    Returns:
        Asymmetry detection results
    """
    h, w = img_shape
    results = {
        'has_asymmetry': False,
        'angle_asymmetry': False,
        'position_asymmetry': False,
        'span_asymmetry': False
    }

    # Get eyebrows
    yolo_eyebrows = yolo_detections.get('eyebrows', [])
    if len(yolo_eyebrows) < 2:
        results['status'] = 'insufficient_eyebrows'
        return results

    # Sort by x-coordinate
    eyebrows_sorted = sorted(yolo_eyebrows, key=lambda e: e['mask_centroid'][0])
    left_eyebrow = eyebrows_sorted[0]
    right_eyebrow = eyebrows_sorted[-1]

    # 1. Angle asymmetry (using MediaPipe if available)
    if mp_detections:
        left_mp = mp_detections.get('left_eyebrow')
        right_mp = mp_detections.get('right_eyebrow')

        if left_mp and right_mp and len(left_mp['points']) >= 2 and len(right_mp['points']) >= 2:
            # Calculate angle using first and last points of each eyebrow
            left_angle = calculate_angle_between_points(
                left_mp['points'][0], left_mp['points'][-1]
            )
            right_angle = calculate_angle_between_points(
                right_mp['points'][0], right_mp['points'][-1]
            )

            angle_diff = abs(left_angle - right_angle)
            results['left_angle'] = left_angle
            results['right_angle'] = right_angle
            results['angle_difference'] = angle_diff

            if angle_diff > config['max_eyebrow_angle_diff']:
                results['angle_asymmetry'] = True
                results['has_asymmetry'] = True

    # 2. Vertical position asymmetry
    left_y = left_eyebrow['mask_centroid'][1]
    right_y = right_eyebrow['mask_centroid'][1]
    position_diff = abs(left_y - right_y)
    position_diff_pct = position_diff / h

    results['left_y_position'] = left_y
    results['right_y_position'] = right_y
    results['position_difference_pct'] = position_diff_pct

    if position_diff_pct > config['max_eyebrow_position_diff_pct']:
        results['position_asymmetry'] = True
        results['has_asymmetry'] = True

    # 3. Span asymmetry
    left_box = left_eyebrow['box']
    right_box = right_eyebrow['box']
    left_span = left_box[2] - left_box[0]  # x2 - x1
    right_span = right_box[2] - right_box[0]

    span_diff = abs(left_span - right_span)
    span_diff_pct = span_diff / w

    results['left_span'] = left_span
    results['right_span'] = right_span
    results['span_difference_pct'] = span_diff_pct

    # Note: Span asymmetry is common and natural, so we just report it
    # We don't flag it as problematic unless it's extreme (>20%)
    if span_diff_pct > 0.20:
        results['span_asymmetry'] = True
        results['has_asymmetry'] = True

    results['status'] = 'success'
    return results


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_face(
    image_path: str,
    model: Any,
    config: Optional[Dict] = None
) -> Dict:
    """
    Comprehensive face preprocessing pipeline.

    Pipeline:
    1. Detect face features (YOLO + MediaPipe)
    2. Validate eyes
    3. Validate eyebrows
    4. Validate face quality
    5. Calculate robust rotation angle
    6. Detect asymmetries
    7. Return preprocessed data + validation results

    Args:
        image_path: Path to image file
        model: YOLO model
        config: Preprocessing configuration (uses defaults if None)

    Returns:
        {
            'valid': bool,
            'image': np.ndarray,
            'image_shape': (h, w),
            'yolo_detections': dict,
            'mediapipe_detections': dict,
            'rotation_angle': float or None,
            'angle_metadata': dict,
            'eye_validation': dict,
            'eyebrow_validation': dict,
            'quality_validation': dict,
            'asymmetry_detection': dict,
            'rejection_reason': str or None,
            'warnings': list
        }
    """
    if config is None:
        config = DEFAULT_PREPROCESS_CONFIG.copy()

    result = {
        'valid': False,
        'rejection_reason': None,
        'warnings': []
    }

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        result['rejection_reason'] = 'failed_to_load_image'
        return result

    h, w = image.shape[:2]
    result['image'] = image
    result['image_shape'] = (h, w)

    # 1. Detect face features
    try:
        yolo_detections = yolo_pred.detect_yolo(
            model, image_path, conf_threshold=config['yolo_min_confidence']
        )
        result['yolo_detections'] = yolo_detections
    except Exception as e:
        result['rejection_reason'] = f'yolo_detection_failed: {str(e)}'
        return result

    try:
        mediapipe_detections = mediapipe_pred.detect_mediapipe(
            image, conf_threshold=config['mediapipe_min_confidence']
        )
        result['mediapipe_detections'] = mediapipe_detections
    except Exception as e:
        result['mediapipe_detections'] = None
        result['warnings'].append(f'mediapipe_detection_failed: {str(e)}')

    # 2. Validate eyes
    eyes_valid, eye_validation = validate_eyes(
        result['mediapipe_detections'],
        yolo_detections,
        (h, w),
        config
    )
    result['eye_validation'] = eye_validation

    if not eyes_valid:
        result['rejection_reason'] = eye_validation['status']
        return result

    # 3. Validate eyebrows
    eyebrows_valid, eyebrow_validation = validate_eyebrows(
        result['mediapipe_detections'],
        yolo_detections,
        (h, w),
        config
    )
    result['eyebrow_validation'] = eyebrow_validation

    if not eyebrows_valid:
        result['rejection_reason'] = eyebrow_validation['status']
        return result

    # 4. Validate face quality
    quality_valid, quality_validation = validate_face_quality(
        result['mediapipe_detections'],
        yolo_detections,
        (h, w),
        config
    )
    result['quality_validation'] = quality_validation

    if not quality_valid:
        result['rejection_reason'] = quality_validation['status']
        return result

    # 5. Calculate robust rotation angle
    rotation_angle, angle_metadata = robust_angle_estimation(
        result['mediapipe_detections'],
        yolo_detections,
        config
    )
    result['rotation_angle'] = rotation_angle
    result['angle_metadata'] = angle_metadata

    if rotation_angle is None:
        result['warnings'].append(f'rotation_angle_unavailable: {angle_metadata["status"]}')
    elif abs(rotation_angle) > config['max_rotation_angle']:
        result['rejection_reason'] = f'excessive_rotation: {rotation_angle:.1f} degrees'
        return result

    # 6. Detect asymmetries
    asymmetry_detection = detect_eyebrow_asymmetry(
        result['mediapipe_detections'],
        yolo_detections,
        (h, w),
        config
    )
    result['asymmetry_detection'] = asymmetry_detection

    if asymmetry_detection['has_asymmetry']:
        result['warnings'].append('eyebrow_asymmetry_detected')

    # All validations passed
    result['valid'] = True
    return result


# =============================================================================
# CORRECTION FUNCTIONS
# =============================================================================

def correct_face_rotation(
    preprocess_result: Dict,
    apply_correction: bool = True
) -> Dict:
    """
    Correct face rotation based on preprocessing results.

    Args:
        preprocess_result: Output from preprocess_face()
        apply_correction: Whether to apply correction (or just return angle)

    Returns:
        {
            'corrected_image': np.ndarray or None,
            'rotation_applied': float,
            'status': str
        }
    """
    result = {
        'corrected_image': None,
        'rotation_applied': 0.0,
        'status': 'not_applied'
    }

    if not preprocess_result['valid']:
        result['status'] = 'preprocessing_failed'
        return result

    rotation_angle = preprocess_result['rotation_angle']

    if rotation_angle is None:
        result['status'] = 'no_rotation_angle'
        return result

    if not apply_correction:
        result['rotation_applied'] = rotation_angle
        result['status'] = 'correction_disabled'
        return result

    # Apply rotation
    image = preprocess_result['image']
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate to make horizontal (negate the angle)
    rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
    corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

    result['corrected_image'] = corrected_image
    result['rotation_applied'] = -rotation_angle
    result['status'] = 'success'

    return result


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_preprocessing_report(preprocess_result: Dict) -> str:
    """
    Generate human-readable preprocessing report.

    Args:
        preprocess_result: Output from preprocess_face()

    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("FACE PREPROCESSING REPORT")
    report_lines.append("=" * 60)

    # Overall status
    if preprocess_result['valid']:
        report_lines.append("✓ FACE VALID - Ready for processing")
    else:
        report_lines.append(f"✗ FACE REJECTED - Reason: {preprocess_result['rejection_reason']}")

    report_lines.append("")

    # Image info
    if 'image_shape' in preprocess_result:
        h, w = preprocess_result['image_shape']
        report_lines.append(f"Image size: {w}x{h}")

    # Eye validation
    if 'eye_validation' in preprocess_result:
        ev = preprocess_result['eye_validation']
        report_lines.append(f"\nEye Validation: {ev.get('status', 'N/A')}")
        if 'eye_distance_pct' in ev:
            report_lines.append(f"  Eye distance: {ev['eye_distance_pct']*100:.1f}% of width")
        if 'vertical_diff_pct' in ev:
            report_lines.append(f"  Vertical difference: {ev['vertical_diff_pct']*100:.1f}% of height")

    # Eyebrow validation
    if 'eyebrow_validation' in preprocess_result:
        ebv = preprocess_result['eyebrow_validation']
        report_lines.append(f"\nEyebrow Validation: {ebv.get('status', 'N/A')}")
        if 'yolo_eyebrow_count' in ebv:
            report_lines.append(f"  YOLO eyebrows detected: {ebv['yolo_eyebrow_count']}")
        if 'eyebrow_eye_overlaps' in ebv:
            report_lines.append(f"  Eyebrow-eye overlaps: {[f'{o:.3f}' for o in ebv['eyebrow_eye_overlaps']]}")

    # Rotation angle
    if 'angle_metadata' in preprocess_result:
        am = preprocess_result['angle_metadata']
        report_lines.append(f"\nRotation Angle: {am.get('status', 'N/A')}")
        if 'final_angle' in am and am['final_angle'] is not None:
            report_lines.append(f"  Final angle: {am['final_angle']:.2f}°")
            report_lines.append(f"  Sources: {am.get('num_sources', 0)} ({', '.join(am.get('sources', []))})")
            if 'outliers_removed' in am:
                report_lines.append(f"  Outliers removed: {am['outliers_removed']}")

    # Asymmetry
    if 'asymmetry_detection' in preprocess_result:
        ad = preprocess_result['asymmetry_detection']
        report_lines.append(f"\nAsymmetry Detection:")
        if ad.get('has_asymmetry'):
            report_lines.append("  ⚠ Asymmetries detected:")
            if ad.get('angle_asymmetry'):
                report_lines.append(f"    - Angle difference: {ad.get('angle_difference', 0):.1f}°")
            if ad.get('position_asymmetry'):
                report_lines.append(f"    - Position difference: {ad.get('position_difference_pct', 0)*100:.1f}%")
            if ad.get('span_asymmetry'):
                report_lines.append(f"    - Span difference: {ad.get('span_difference_pct', 0)*100:.1f}%")
        else:
            report_lines.append("  ✓ No significant asymmetries")

    # Warnings
    if preprocess_result.get('warnings'):
        report_lines.append(f"\nWarnings:")
        for warning in preprocess_result['warnings']:
            report_lines.append(f"  ⚠ {warning}")

    report_lines.append("=" * 60)

    return "\n".join(report_lines)
