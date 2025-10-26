"""
Eyebrow Beautification Algorithm - Main Pipeline

7-Phase Multi-Source Fusion Algorithm:
1. Pre-processing
2. Source Collection (YOLO + MediaPipe)
3. Face Alignment & Normalization
4. Eyebrow Pairing & Association
5. Multi-Source Fusion (CORE)
6. Validation & Quality Control
7. Output Generation

Intelligently combines YOLO segmentation with MediaPipe landmarks
to create complete, natural eyebrow masks.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import utils
import yolo_pred
import mediapipe_pred
import preprocess


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Preprocessing (NEW: comprehensive face validation and correction)
    'enable_preprocessing': True,  # Enable comprehensive preprocessing
    'reject_invalid_faces': True,  # Reject faces that fail preprocessing validation
    'auto_correct_rotation': True,  # Auto-correct face rotation
    'min_rotation_threshold': 2.5,  # Only correct rotation if > this (degrees) - increased from 1.0

    # Detection thresholds
    'yolo_conf_threshold': 0.25,
    'mediapipe_conf_threshold': 0.3,  # Lowered from 0.5 for better detection

    # Face alignment (legacy - now handled by preprocessing)
    'straightening_threshold': 5.0,  # degrees

    # Validation thresholds (from empirical data)
    'min_mp_coverage': 80.0,  # percentage (only checked if MediaPipe available)
    'eye_dist_range': [4.0, 8.0],  # percentage of height
    'aspect_ratio_range': [3.0, 10.0],  # relaxed from 4.0
    'expansion_range': [0.9, 2.0],  # expansion ratio
    'thickness_range': [0.7, 1.3],  # NEW: thickness ratio range (70-130%, max 30% change)

    # Extension parameters
    'min_arch_thickness_pct': 0.015,  # 1.5% of image height
    'connection_thickness_pct': 0.01,  # 1% of image height

    # Eye exclusion
    'eye_buffer_kernel': (15, 15),
    'eye_buffer_iterations': 2,

    # Hair filtering
    'hair_overlap_threshold': 0.15,  # 15%
    'hair_distance_threshold': 0.3,  # normalized distance

    # Beautification
    'close_kernel': (7, 7),
    'open_kernel': (5, 5),
    'gaussian_kernel': (9, 9),
    'gaussian_sigma': 2.0,

    # Smooth contour parameters (NEW: for natural curved boundaries)
    'smooth_kernel_size': 5,  # Smoothing kernel size (3, 5, or 7)
    'smooth_iterations': 2,  # Morphological smoothing passes (1-3)
}


# =============================================================================
# PHASE 1: PRE-PROCESSING
# =============================================================================

def load_and_validate_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Load and validate input image.

    Args:
        image_path: Path to input image

    Returns:
        (image, img_shape) where img_shape is (height, width, channels)

    Raises:
        ValueError: If image is invalid or too small
    """
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = img.shape[:2]

    # Minimum size check
    if h < 200 or w < 200:
        raise ValueError(f"Image too small: {w}x{h}. Minimum size is 200x200.")

    # Warning for very large images
    if h > 4000 or w > 4000:
        print(f"Warning: Large image ({w}x{h}). Processing may be slow.")

    img_shape = img.shape

    return img, img_shape


# =============================================================================
# PHASE 4: EYEBROW PAIRING & ASSOCIATION
# =============================================================================

def determine_side(eyebrow_center: Tuple[int, int], img_width: int) -> str:
    """Determine if eyebrow is on left or right side of face."""
    if eyebrow_center[0] < img_width / 2:
        return 'left'
    else:
        return 'right'


def find_closest_detection(source_center: Tuple[int, int],
                          candidates: List[Dict],
                          side_constraint: Optional[str] = None,
                          img_width: Optional[int] = None) -> Optional[Dict]:
    """
    Find closest detection to source, optionally constraining to same side.

    Args:
        source_center: (x, y) center of source detection
        candidates: List of detection dicts (with 'mask_centroid')
        side_constraint: 'left' or 'right' to constrain search
        img_width: Image width (required if side_constraint is set)

    Returns:
        Closest detection dict, or None if no candidates
    """
    if not candidates:
        return None

    # Filter by side if constraint provided
    if side_constraint and img_width:
        filtered = []
        for det in candidates:
            det_side = determine_side(det['mask_centroid'], img_width)
            if det_side == side_constraint:
                filtered.append(det)
        candidates = filtered if filtered else candidates  # Fall back to all if none match

    if not candidates:
        return None

    # Find closest by Euclidean distance
    min_dist = float('inf')
    closest = None

    sx, sy = source_center

    for det in candidates:
        cx, cy = det['mask_centroid']
        dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)

        if dist < min_dist:
            min_dist = dist
            closest = det

    return closest


def find_containing_eye_box(eyebrow: Dict, eye_boxes: List[Dict]) -> Optional[Dict]:
    """Find eye_box that contains the eyebrow (highest IoU)."""
    if not eye_boxes:
        return None

    eyebrow_box = utils.get_bounding_box_from_mask(eyebrow['mask'])

    max_iou = 0
    best_eye_box = None

    for eye_box in eye_boxes:
        iou = utils.calculate_box_iou(eyebrow_box, eye_box['box'])
        if iou > max_iou:
            max_iou = iou
            best_eye_box = eye_box

    return best_eye_box


def find_overlapping_hair(eyebrow: Dict, hair_dets: List[Dict]) -> List[Dict]:
    """Find all hair regions that overlap with eyebrow."""
    if not hair_dets:
        return []

    overlapping = []
    eyebrow_mask = eyebrow['mask']

    for hair in hair_dets:
        # Check overlap
        overlap = np.logical_and(eyebrow_mask, hair['mask'])
        overlap_area = overlap.sum()

        if overlap_area > 0:
            overlapping.append(hair)

    return overlapping


def pair_eyebrows_with_context(detections: Dict, mediapipe_landmarks: Optional[Dict],
                               img_shape: Tuple[int, int, int]) -> List[Dict]:
    """
    Pair each YOLO eyebrow with all associated detections and landmarks.

    For each eyebrow, finds:
    - Closest eye (same side)
    - Containing eye_box
    - Overlapping hair regions
    - Corresponding MediaPipe eyebrow landmarks
    - Corresponding MediaPipe eye landmarks

    Args:
        detections: YOLO detection dict
        mediapipe_landmarks: MediaPipe landmark dict (optional)
        img_shape: Image shape (height, width, channels)

    Returns:
        List of eyebrow pair dicts, each containing:
        {
            'side': 'left' or 'right',
            'eyebrow': YOLO eyebrow detection,
            'eye': YOLO eye detection (or None),
            'eye_box': YOLO eye_box detection (or None),
            'hair': List of overlapping hair detections,
            'mp_eyebrow': MediaPipe eyebrow landmarks (or None),
            'mp_eye': MediaPipe eye landmarks (or None),
            'mp_coverage': Coverage statistics dict,
            'img_shape': Image shape
        }
    """
    h, w = img_shape[:2]
    pairs = []

    # Get eyebrows
    eyebrows = detections.get('eyebrows', [])

    if not eyebrows:
        print("No eyebrows detected!")
        return pairs

    # Process each eyebrow
    for eyebrow in eyebrows:
        # Determine side
        side = determine_side(eyebrow['mask_centroid'], w)

        # Find associated detections
        eye = find_closest_detection(
            eyebrow['mask_centroid'],
            detections.get('eye', []),
            side_constraint=side,
            img_width=w
        )

        eye_box = find_containing_eye_box(eyebrow, detections.get('eye_box', []))

        hair_list = find_overlapping_hair(eyebrow, detections.get('hair', []))

        # Find MediaPipe landmarks
        mp_eyebrow = None
        mp_eye = None
        mp_coverage = None

        if mediapipe_landmarks:
            # Get corresponding MediaPipe eyebrow
            mp_key = f'{side}_eyebrow'
            if mp_key in mediapipe_landmarks and mediapipe_landmarks[mp_key]:
                mp_eyebrow = mediapipe_landmarks[mp_key]

                # Calculate coverage
                mp_coverage = mediapipe_pred.calculate_mediapipe_coverage(
                    mp_eyebrow['points'],
                    eyebrow['mask']
                )

            # Get corresponding MediaPipe eye
            mp_eye_key = f'{side}_eye'
            if mp_eye_key in mediapipe_landmarks and mediapipe_landmarks[mp_eye_key]:
                mp_eye = mediapipe_landmarks[mp_eye_key]

        # Create pair dict
        pair = {
            'side': side,
            'eyebrow': eyebrow,
            'eye': eye,
            'eye_box': eye_box,
            'hair': hair_list,
            'mp_eyebrow': mp_eyebrow,
            'mp_eye': mp_eye,
            'mp_coverage': mp_coverage,
            'img_shape': img_shape
        }

        pairs.append(pair)

    print(f"\nPaired {len(pairs)} eyebrows with context:")
    for i, pair in enumerate(pairs):
        print(f"  Eyebrow {i+1} ({pair['side']}):")
        print(f"    - Eye: {'✓' if pair['eye'] else '✗'}")
        print(f"    - Eye_box: {'✓' if pair['eye_box'] else '✗'}")
        print(f"    - Hair overlap: {len(pair['hair'])} region(s)")
        print(f"    - MediaPipe eyebrow: {'✓' if pair['mp_eyebrow'] else '✗'}")
        if pair['mp_coverage']:
            print(f"    - MP coverage: {pair['mp_coverage']['coverage_percent']:.1f}%")

    return pairs


# =============================================================================
# PHASE 5: MULTI-SOURCE FUSION (CORE ALGORITHM)
# =============================================================================

def create_foundation_mask(pair: Dict, config: Dict) -> np.ndarray:
    """
    Phase 5.1: Create foundation mask from YOLO detection.

    Starts with YOLO mask, cleans small components, applies light smoothing.

    Args:
        pair: Eyebrow pair dict
        config: Configuration dict

    Returns:
        Cleaned foundation mask
    """
    # Start with YOLO mask
    foundation = pair['eyebrow']['mask'].copy()

    # Remove small disconnected components
    foundation = utils.remove_small_components(foundation, min_size=50)

    # Light smoothing to reduce pixelation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foundation = cv2.morphologyEx(foundation.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return foundation


def create_mediapipe_extension(pair: Dict, config: Dict) -> np.ndarray:
    """
    Phase 5.2: Create extension mask guided by MediaPipe landmarks.

    If MediaPipe points exist outside YOLO mask:
    - Method A: Fit parametric spline through all 10 MP points → draw arch
    - Method B: Create connection paths from YOLO edge to missing MP points
    - Combine both methods

    Args:
        pair: Eyebrow pair dict
        config: Configuration dict

    Returns:
        Extension mask (binary)
    """
    h, w = pair['img_shape'][:2]
    extension = np.zeros((h, w), dtype=np.uint8)

    # Check if we have MediaPipe eyebrow landmarks
    if not pair['mp_eyebrow']:
        return extension

    mp_points = pair['mp_eyebrow']['points']

    if not mp_points:
        return extension

    # Check if there are points outside YOLO mask
    if pair['mp_coverage'] and pair['mp_coverage']['points_outside'] > 0:
        outside_points = pair['mp_coverage']['outside_points']

        # Method A: Parametric spline through all MP points
        arch_thickness = max(3, int(h * config['min_arch_thickness_pct']))
        arch_mask = utils.create_arch_from_landmarks_parametric(
            mp_points, h, w, arch_thickness
        )

        # Method B: Connection paths from YOLO to missing points
        connection_thickness = max(2, int(h * config['connection_thickness_pct']))
        connection_mask = np.zeros((h, w), dtype=np.uint8)

        yolo_mask = pair['eyebrow']['mask']

        for outside_pt in outside_points:
            # Find nearest point in YOLO mask
            if yolo_mask.sum() == 0:
                continue

            # Get YOLO mask points
            yolo_ys, yolo_xs = np.where(yolo_mask > 0)

            if len(yolo_xs) == 0:
                continue

            # Find closest YOLO point
            ox, oy = outside_pt
            distances = np.sqrt((yolo_xs - ox)**2 + (yolo_ys - oy)**2)
            min_idx = np.argmin(distances)
            nearest_x = yolo_xs[min_idx]
            nearest_y = yolo_ys[min_idx]

            # Draw connection line
            cv2.line(connection_mask, (nearest_x, nearest_y), (ox, oy),
                    1, thickness=connection_thickness)

        # Combine methods (logical OR)
        extension = np.logical_or(arch_mask, connection_mask).astype(np.uint8)

    return extension


def create_candidate_region(foundation: np.ndarray, extension: np.ndarray,
                           pair: Dict, config: Dict) -> np.ndarray:
    """
    Phase 5.3: Create candidate region by combining foundation and extension.

    - Union foundation + extension
    - Apply eye_box constraint (upper 35% ± 5% margins)
    - Force include all MediaPipe points

    Args:
        foundation: Foundation mask from phase 5.1
        extension: Extension mask from phase 5.2
        pair: Eyebrow pair dict
        config: Configuration dict

    Returns:
        Candidate mask with constraints applied
    """
    h, w = pair['img_shape'][:2]

    # Union of foundation and extension
    candidate = np.logical_or(foundation, extension).astype(np.uint8)

    # Apply eye_box constraint if available
    if pair['eye_box']:
        # Create allowed zone: upper 35% of eye_box ± 5% margins
        eye_box = pair['eye_box']['box']
        x1, y1, x2, y2 = [int(c) for c in eye_box]
        box_height = y2 - y1

        # Upper 35% with margins
        allowed_y_start = max(0, y1 - int(box_height * 0.05))
        allowed_y_end = min(h, int(y1 + box_height * 0.35 + box_height * 0.05))

        allowed_x_start = max(0, x1 - int((x2 - x1) * 0.05))
        allowed_x_end = min(w, x2 + int((x2 - x1) * 0.05))

        # Create allowed zone mask
        allowed_zone = np.zeros((h, w), dtype=np.uint8)
        allowed_zone[allowed_y_start:allowed_y_end, allowed_x_start:allowed_x_end] = 1

        # Constrain candidate to allowed zone
        candidate = np.logical_and(candidate, allowed_zone).astype(np.uint8)

    # Strict horizontal span constraint using eye_box (NEW)
    # Ensure eyebrow doesn't extend beyond eye_box horizontally
    if pair['eye_box']:
        eye_box = pair['eye_box']['box']
        x1_box, _, x2_box, _ = [int(c) for c in eye_box]

        # Constrain to eye_box horizontal bounds
        candidate = utils.constrain_to_horizontal_bounds(candidate, x1_box, x2_box)

    # Force include all MediaPipe points (even if filtered by constraint)
    if pair['mp_eyebrow']:
        for x, y in pair['mp_eyebrow']['points']:
            if 0 <= x < w and 0 <= y < h:
                # Draw small circle around each MP point
                cv2.circle(candidate, (x, y), 3, 1, -1)

    return candidate


def apply_exclusions(candidate: np.ndarray, pair: Dict, config: Dict) -> np.ndarray:
    """
    Phase 5.4: Apply exclusion rules.

    - Eye exclusion: Remove eye mask with buffer
    - Hair filtering: Distance-based filtering

    Args:
        candidate: Candidate mask from phase 5.3
        pair: Eyebrow pair dict
        config: Configuration dict

    Returns:
        Refined mask after exclusions
    """
    h, w = pair['img_shape'][:2]
    refined = candidate.copy()

    # Eye exclusion
    if pair['eye']:
        eye_mask = pair['eye']['mask']

        # Dilate eye mask to create buffer zone
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['eye_buffer_kernel'])
        eye_buffered = cv2.dilate(eye_mask.astype(np.uint8), kernel,
                                  iterations=config['eye_buffer_iterations'])

        # Subtract from candidate
        refined = np.logical_and(refined, np.logical_not(eye_buffered)).astype(np.uint8)

        # Check if mask became empty after eye exclusion
        if refined.sum() == 0:
            print(f"⚠️  Warning: {pair['side']} eyebrow mask became empty after eye exclusion!")
            print(f"   Falling back to foundation mask (YOLO only, no eye exclusion)")
            # Fallback to foundation mask without eye exclusion
            foundation = pair['eyebrow']['mask']
            refined = foundation.copy()

    # Hair filtering (distance-based)
    if len(pair['hair']) > 0:
        # Calculate hair overlap ratio
        total_hair_mask = np.zeros((h, w), dtype=np.uint8)
        for hair in pair['hair']:
            total_hair_mask = np.logical_or(total_hair_mask, hair['mask']).astype(np.uint8)

        overlap = np.logical_and(refined, total_hair_mask)
        overlap_ratio = overlap.sum() / refined.sum() if refined.sum() > 0 else 0

        # Apply filtering only if significant hair overlap
        if overlap_ratio > config['hair_overlap_threshold']:
            # Get foundation mask for distance reference
            foundation = pair['eyebrow']['mask']

            # Calculate distance transform from foundation
            dist_transform = cv2.distanceTransform(
                (foundation == 0).astype(np.uint8),
                cv2.DIST_L2, 5
            )

            # Normalize distance
            if dist_transform.max() > 0:
                dist_transform = dist_transform / dist_transform.max()

            # Keep only regions close to foundation (within threshold)
            proximity_mask = (dist_transform <= config['hair_distance_threshold']).astype(np.uint8)

            # Apply proximity constraint to hair regions only
            hair_regions = np.logical_and(refined, total_hair_mask)
            filtered_hair = np.logical_and(hair_regions, proximity_mask)

            # Combine: keep non-hair parts + filtered hair parts
            non_hair = np.logical_and(refined, np.logical_not(total_hair_mask))
            refined = np.logical_or(non_hair, filtered_hair).astype(np.uint8)

    # Validation: ensure MediaPipe points not wrongly excluded
    if pair['mp_eyebrow']:
        for x, y in pair['mp_eyebrow']['points']:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(refined, (x, y), 2, 1, -1)

    return refined


def beautify_shape(mask: np.ndarray, pair: Dict, config: Dict) -> np.ndarray:
    """
    Phase 5.5: Shape beautification using morphological operations + smart hull.

    1. Close small gaps
    2. Remove small protrusions
    3. Fill interior holes
    4. Smooth boundaries (Gaussian blur)
    5. Apply smart hull for beautiful curved lines (NEW)

    Args:
        mask: Refined mask from phase 5.4
        pair: Eyebrow pair dict
        config: Configuration dict

    Returns:
        Beautified final mask with smooth, curved boundaries
    """
    beautified = mask.copy().astype(np.uint8)

    # 1. Close small gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['close_kernel'])
    beautified = cv2.morphologyEx(beautified, cv2.MORPH_CLOSE, close_kernel)

    # 2. Remove small protrusions
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['open_kernel'])
    beautified = cv2.morphologyEx(beautified, cv2.MORPH_OPEN, open_kernel)

    # 3. Fill interior holes
    beautified = utils.fill_holes(beautified)

    # 4. Smooth boundaries (Gaussian blur)
    beautified_float = beautified.astype(np.float32)
    beautified_float = cv2.GaussianBlur(
        beautified_float,
        config['gaussian_kernel'],
        config['gaussian_sigma']
    )

    # Threshold back to binary (>0.5)
    beautified = (beautified_float > 0.5).astype(np.uint8)

    # 5. Smooth boundaries uniformly while maintaining natural curvature (NEW)
    # This removes zigzags/noise perpendicular to boundaries without straightening curves
    kernel_size = config.get('smooth_kernel_size', 5)
    iterations = config.get('smooth_iterations', 2)

    beautified = utils.smooth_mask_contours(
        beautified,
        kernel_size=kernel_size,
        iterations=iterations
    )

    return beautified


# =============================================================================
# PHASE 6: VALIDATION & QUALITY CONTROL
# =============================================================================

def validate_eyebrow_mask(final_mask: np.ndarray, pair: Dict, config: Dict) -> Dict:
    """
    Phase 6: Comprehensive quality control with 6 metrics.

    1. MediaPipe coverage (target: 80-100%)
    2. Eye distance (target: 4-8% of image height)
    3. Aspect ratio (target: 4-10)
    4. Eye overlap (target: 0 pixels)
    5. Expansion ratio (target: 0.9-2.0x)
    6. Overall pass (all checks)

    Args:
        final_mask: Beautified mask from phase 5.5
        pair: Eyebrow pair dict
        config: Configuration dict

    Returns:
        Validation results dict with all metrics + pass/fail flags
    """
    h, w = pair['img_shape'][:2]

    results = {}

    # 1. MediaPipe coverage (optional - only checked if MediaPipe detected face)
    if pair['mp_eyebrow']:
        mp_coverage = mediapipe_pred.calculate_mediapipe_coverage(
            pair['mp_eyebrow']['points'],
            final_mask
        )
        results['mp_coverage'] = mp_coverage['coverage_percent']
        results['mp_coverage_pass'] = results['mp_coverage'] >= config['min_mp_coverage']
        results['mp_available'] = True
    else:
        results['mp_coverage'] = 0.0
        results['mp_coverage_pass'] = True  # Pass if MediaPipe not available (graceful degradation)
        results['mp_available'] = False

    # 2. Eye distance (only checked if eye detected)
    if pair['eye']:
        final_centroid = utils.calculate_centroid(final_mask)
        eye_centroid = pair['eye']['mask_centroid']

        vertical_dist = abs(final_centroid[1] - eye_centroid[1])
        eye_dist_pct = (vertical_dist / h) * 100

        results['eye_distance_pct'] = eye_dist_pct
        results['eye_distance_pass'] = (config['eye_dist_range'][0] <= eye_dist_pct <= config['eye_dist_range'][1])
        results['eye_available'] = True
    else:
        results['eye_distance_pct'] = 0.0
        results['eye_distance_pass'] = True  # Pass if no eye detected (can't validate without reference)
        results['eye_available'] = False

    # 3. Aspect ratio
    final_bbox = utils.get_bounding_box_from_mask(final_mask)
    final_width = final_bbox[2] - final_bbox[0]
    final_height = final_bbox[3] - final_bbox[1]

    aspect_ratio = final_width / final_height if final_height > 0 else 0

    results['aspect_ratio'] = aspect_ratio
    results['aspect_ratio_pass'] = (config['aspect_ratio_range'][0] <= aspect_ratio <= config['aspect_ratio_range'][1])

    # 4. Eye overlap
    if pair['eye']:
        overlap = np.logical_and(final_mask, pair['eye']['mask'])
        eye_overlap = int(overlap.sum())
        results['eye_overlap'] = eye_overlap
        results['eye_overlap_pass'] = (eye_overlap == 0)
    else:
        results['eye_overlap'] = 0
        results['eye_overlap_pass'] = True

    # 5. Expansion ratio
    original_area = pair['eyebrow']['mask_area']
    final_area = int(final_mask.sum())

    expansion_ratio = final_area / original_area if original_area > 0 else 0

    results['expansion_ratio'] = expansion_ratio
    results['expansion_ratio_pass'] = (config['expansion_range'][0] <= expansion_ratio <= config['expansion_range'][1])

    # 6. Thickness ratio (new constraint: thickness shouldn't change more than 20-30%)
    original_thickness = utils.calculate_mask_thickness(pair['eyebrow']['mask'])
    final_thickness = utils.calculate_mask_thickness(final_mask)

    thickness_ratio = final_thickness / original_thickness if original_thickness > 0 else 1.0
    thickness_range = config.get('thickness_range', (0.7, 1.3))  # Default: 70-130% (30% change max)

    results['thickness_ratio'] = thickness_ratio
    results['thickness_ratio_pass'] = (thickness_range[0] <= thickness_ratio <= thickness_range[1])

    # 7. Overall pass
    results['overall_pass'] = all([
        results['mp_coverage_pass'],
        results['eye_distance_pass'],
        results['aspect_ratio_pass'],
        results['eye_overlap_pass'],
        results['expansion_ratio_pass'],
        results['thickness_ratio_pass']
    ])

    # Print warnings for failed checks
    if not results['overall_pass']:
        print(f"\n⚠ Validation warnings for {pair['side']} eyebrow:")
        if not results['mp_coverage_pass'] and results['mp_available']:
            print(f"  - MediaPipe coverage: {results['mp_coverage']:.1f}% (target: ≥{config['min_mp_coverage']}%)")
        if not results['eye_distance_pass'] and results['eye_available']:
            print(f"  - Eye distance: {results['eye_distance_pct']:.1f}% (target: {config['eye_dist_range']})")
        if not results['aspect_ratio_pass']:
            print(f"  - Aspect ratio: {results['aspect_ratio']:.2f} (target: {config['aspect_ratio_range']})")
        if not results['eye_overlap_pass']:
            print(f"  - Eye overlap: {results['eye_overlap']} pixels (target: 0)")
        if not results['expansion_ratio_pass']:
            print(f"  - Expansion ratio: {results['expansion_ratio']:.2f}x (target: {config['expansion_range']})")
        if not results['thickness_ratio_pass']:
            print(f"  - Thickness ratio: {results['thickness_ratio']:.2f}x (target: {thickness_range})")

    return results


# =============================================================================
# PHASE 7: OUTPUT GENERATION
# =============================================================================

def generate_output(pair: Dict, final_mask: np.ndarray, validation_results: Dict) -> Dict:
    """
    Phase 7: Package results with all metadata.

    Args:
        pair: Eyebrow pair dict
        final_mask: Beautified mask from phase 5.5
        validation_results: Validation results from phase 6

    Returns:
        Output dict with structure:
        {
            'side': 'left' or 'right',
            'masks': {
                'original_yolo': np.ndarray,
                'final_beautified': np.ndarray
            },
            'validation': {...},
            'metadata': {...}
        }
    """
    output = {
        'side': pair['side'],
        'masks': {
            'original_yolo': pair['eyebrow']['mask'],
            'final_beautified': final_mask
        },
        'validation': validation_results,
        'metadata': {
            'yolo_confidence': pair['eyebrow']['confidence'],
            'yolo_area': pair['eyebrow']['mask_area'],
            'final_area': int(final_mask.sum()),
            'has_eye': pair['eye'] is not None,
            'has_eye_box': pair['eye_box'] is not None,
            'hair_regions': len(pair['hair']),
            'has_mediapipe': pair['mp_eyebrow'] is not None,
        }
    }

    return output


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def beautify_eyebrows(image_path: str, model, config: Optional[Dict] = None) -> List[Dict]:
    """
    Complete eyebrow beautification pipeline (7 phases).

    Args:
        image_path: Path to input image
        model: YOLO model object (from yolo_pred.load_yolo_model())
        config: Configuration dict (optional, uses DEFAULT_CONFIG if None)

    Returns:
        List of result dicts (one per eyebrow detected), each containing:
        - 'side': 'left' or 'right'
        - 'masks': {'original_yolo': mask, 'final_beautified': mask}
        - 'validation': validation metrics dict
        - 'metadata': additional information

    Example:
        model = yolo_pred.load_yolo_model()
        results = beautify_eyebrows('image.jpg', model)

        for result in results:
            print(f"Side: {result['side']}")
            print(f"Validation passed: {result['validation']['overall_pass']}")
            final_mask = result['masks']['final_beautified']
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()

    print("="*60)
    print("EYEBROW BEAUTIFICATION PIPELINE")
    print("="*60)

    # Phase 0: Comprehensive Preprocessing (NEW)
    preprocessing_result = None
    rotation_applied = False

    if config.get('enable_preprocessing', True):
        print("\n[Phase 0] Comprehensive Face Preprocessing...")
        print("-" * 60)

        # Create preprocessing config from beautify config
        preprocess_config = preprocess.DEFAULT_PREPROCESS_CONFIG.copy()
        preprocess_config['yolo_min_confidence'] = config['yolo_conf_threshold']
        preprocess_config['mediapipe_min_confidence'] = config['mediapipe_conf_threshold']

        # Run preprocessing
        preprocessing_result = preprocess.preprocess_face(image_path, model, preprocess_config)

        # Check if face is valid
        if not preprocessing_result['valid']:
            if config.get('reject_invalid_faces', True):
                print(f"\n✗ FACE REJECTED: {preprocessing_result['rejection_reason']}")
                print("\nPreprocessing Report:")
                print(preprocess.generate_preprocessing_report(preprocessing_result))
                return []  # Return empty list for invalid faces
            else:
                print(f"\n⚠ WARNING: Face validation failed ({preprocessing_result['rejection_reason']})")
                print("  Continuing anyway (reject_invalid_faces=False)")
        else:
            print("✓ Face validation passed")

            # Print preprocessing summary
            if preprocessing_result['rotation_angle'] is not None:
                print(f"  Rotation angle: {preprocessing_result['rotation_angle']:.2f}°")
            if preprocessing_result['asymmetry_detection'].get('has_asymmetry'):
                print(f"  ⚠ Asymmetries detected")

            # Apply rotation correction if enabled
            if config.get('auto_correct_rotation', True) and preprocessing_result['rotation_angle'] is not None:
                min_threshold = config.get('min_rotation_threshold', 1.0)
                if abs(preprocessing_result['rotation_angle']) > min_threshold:
                    print(f"\n  Applying rotation correction: {-preprocessing_result['rotation_angle']:.2f}°...")
                    correction_result = preprocess.correct_face_rotation(preprocessing_result, apply_correction=True)

                    if correction_result['status'] == 'success' and correction_result['corrected_image'] is not None:
                        # Save corrected image temporarily
                        import tempfile
                        import os
                        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
                        os.close(temp_fd)
                        cv2.imwrite(temp_path, correction_result['corrected_image'])

                        # Update image_path to use corrected version
                        image_path = temp_path
                        rotation_applied = True
                        print(f"  ✓ Rotation corrected")
                else:
                    print(f"  ℹ Rotation angle {preprocessing_result['rotation_angle']:.2f}° below threshold ({min_threshold}°), skipping correction")

    # Phase 1: Load and validate image
    print("\n[Phase 1] Loading image...")
    img, img_shape = load_and_validate_image(image_path)
    print(f"✓ Image loaded: {img_shape[1]}x{img_shape[0]}")

    # Phase 2: Source collection
    print("\n[Phase 2] Running detections...")

    # OPTIMIZATION: Reuse detections from preprocessing if available and image unchanged
    if preprocessing_result is not None and not rotation_applied:
        # Reuse detections from preprocessing (image unchanged, no need to re-detect)
        print("  ℹ Reusing detections from preprocessing (no image rotation applied)")
        yolo_detections = preprocessing_result.get('yolo_detections', {})
        mediapipe_landmarks = preprocessing_result.get('mediapipe_detections')
    else:
        # Run fresh detections (preprocessing disabled OR image was rotated)
        if rotation_applied:
            print("  ℹ Re-running detections on rotated image")
        yolo_detections = yolo_pred.detect_yolo(model, image_path, config['yolo_conf_threshold'])
        mediapipe_landmarks = mediapipe_pred.detect_mediapipe(img, config['mediapipe_conf_threshold'])

    if mediapipe_landmarks:
        print("✓ MediaPipe landmarks detected")
    else:
        print("⚠ No MediaPipe landmarks (will use YOLO only)")

    # Phase 3: Face alignment (LEGACY - now handled by preprocessing)
    if rotation_applied:
        print("\n[Phase 3] Face alignment (skipped - already corrected in preprocessing)")
    else:
        print("\n[Phase 3] Checking face alignment...")
        angle, left_eye, right_eye = utils.detect_face_rotation(yolo_detections, mediapipe_landmarks)
        print(f"Face rotation: {angle:.2f}°")

        if utils.should_straighten_face(angle, config['straightening_threshold']):
            print(f"Straightening face (threshold: {config['straightening_threshold']}°)...")
            img, M = utils.straighten_face(img, angle, left_eye, right_eye)
            yolo_detections = utils.transform_detections(yolo_detections, M, img_shape)
            if mediapipe_landmarks:
                mediapipe_landmarks = utils.transform_mediapipe(mediapipe_landmarks, M)
            print("✓ Face straightened")
        else:
            print("✓ Face alignment acceptable")

    # Phase 4: Pairing
    print("\n[Phase 4] Pairing eyebrows with context...")
    eyebrow_pairs = pair_eyebrows_with_context(yolo_detections, mediapipe_landmarks, img_shape)

    if not eyebrow_pairs:
        print("No eyebrows to process!")
        return []

    # Phase 5-7: Process each eyebrow
    results = []
    errors = []

    for i, pair in enumerate(eyebrow_pairs):
        print(f"\n{'='*60}")
        print(f"Processing {pair['side']} eyebrow ({i+1}/{len(eyebrow_pairs)})")
        print(f"{'='*60}")

        try:
            # Phase 5: Multi-source fusion
            print("\n[Phase 5] Fusing YOLO + MediaPipe...")

            print("  5.1: Creating foundation mask...")
            foundation = create_foundation_mask(pair, config)

            print("  5.2: Creating MediaPipe extension...")
            extension = create_mediapipe_extension(pair, config)

            print("  5.3: Creating candidate region...")
            candidate = create_candidate_region(foundation, extension, pair, config)

            print("  5.4: Applying exclusions...")
            refined = apply_exclusions(candidate, pair, config)

            print("  5.5: Beautifying shape...")
            beautified = beautify_shape(refined, pair, config)

            print("✓ Fusion complete")

            # Phase 6: Validation
            print("\n[Phase 6] Validating result...")
            validation = validate_eyebrow_mask(beautified, pair, config)

            if validation['overall_pass']:
                print("✓ All validation checks passed")
            else:
                print("⚠ Some validation checks failed (see warnings above)")

            # Phase 7: Output
            print("\n[Phase 7] Generating output...")
            output = generate_output(pair, beautified, validation)

            # Add preprocessing results if available
            if preprocessing_result is not None:
                output['preprocessing'] = {
                    'valid': preprocessing_result['valid'],
                    'rotation_angle': preprocessing_result.get('rotation_angle'),
                    'rotation_corrected': rotation_applied,
                    'eye_validation': preprocessing_result.get('eye_validation', {}),
                    'eyebrow_validation': preprocessing_result.get('eyebrow_validation', {}),
                    'asymmetry_detection': preprocessing_result.get('asymmetry_detection', {}),
                    'warnings': preprocessing_result.get('warnings', [])
                }

            results.append(output)

            print("✓ Output generated")

        except Exception as e:
            # Capture error for this eyebrow
            error_info = {
                'side': pair['side'],
                'error': str(e),
                'type': type(e).__name__
            }
            errors.append(error_info)
            print(f"✗ Failed to process {pair['side']} eyebrow: {e}")
            import traceback
            traceback.print_exc()

    # Handle results based on success/failure
    print(f"\n{'='*60}")

    if errors:
        if len(errors) == len(eyebrow_pairs):
            # All eyebrows failed - this is a complete failure
            print(f"PIPELINE FAILED: All {len(eyebrow_pairs)} eyebrow(s) failed processing")
            print(f"{'='*60}\n")
            error_details = "\n".join([f"  - {err['side']}: {err['error']}" for err in errors])
            raise ValueError(f"All eyebrows failed processing:\n{error_details}")
        else:
            # Partial success - some eyebrows succeeded, some failed
            print(f"PIPELINE PARTIAL SUCCESS: {len(results)}/{len(eyebrow_pairs)} eyebrow(s) processed")
            print(f"⚠️  {len(errors)} eyebrow(s) failed:")
            for err in errors:
                print(f"     - {err['side']}: {err['error']}")
            print(f"{'='*60}\n")
            print(f"⚠️  WARNING: Returning partial results only. Check errors above.")
    else:
        # All succeeded
        print(f"PIPELINE COMPLETE: Processed {len(results)} eyebrow(s) successfully")
        print(f"{'='*60}\n")

    return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of beautification pipeline."""

    # Load YOLO model
    print("Loading YOLO model...")
    model = yolo_pred.load_yolo_model()

    # Test image
    test_image = "annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Run beautification
    results = beautify_eyebrows(test_image, model)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for i, result in enumerate(results):
        print(f"\nEyebrow {i+1} ({result['side']}):")
        print(f"  Validation passed: {result['validation']['overall_pass']}")
        print(f"  MediaPipe coverage: {result['validation']['mp_coverage']:.1f}%")
        print(f"  Expansion ratio: {result['validation']['expansion_ratio']:.2f}x")
        print(f"  Final area: {result['metadata']['final_area']} pixels")
