"""
Stencil Extraction Module (v6.0)

This module implements the 4-phase polygon extraction algorithm for converting
YOLO masks and MediaPipe landmarks into precise brow stencil polygons.

Core Concept: "Grounding"
--------------------------
Combining YOLO's dense detection with MediaPipe's precise landmarks to create
a single, accurate polygon boundary for eyebrow stencils.

Algorithm Phases:
    Phase 1: Extract YOLO Polygon - Convert mask to contour
    Phase 2: Check Alignment - Validate YOLO vs MediaPipe agreement
    Phase 3: Merge or Fallback - Combine or use MP-only
    Phase 4: Validate & Return - Quality checks and packaging

Author: Brow Stencil System
Date: 2025-01-13
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import utils


# Default configuration
DEFAULT_CONFIG = {
    # YOLO polygon extraction
    'yolo_simplify_epsilon': 0.005,  # Douglas-Peucker factor (0.5% of perimeter)

    # Alignment thresholds
    'alignment_iou_threshold': 0.3,  # Minimum IoU for "aligned"
    'alignment_distance_threshold': 20.0,  # Maximum avg distance (pixels)

    # Polygon validation
    'min_polygon_points': 5,
    'max_polygon_points': 50,

    # Grounding strategy
    'grounding_method': 'insert_mp',  # Insert MediaPipe into YOLO
    'fallback_to_mp': True,  # Use MP-only if misaligned
}


def extract_stencil_polygon(
    yolo_mask: np.ndarray,
    mp_landmarks: List[List[int]],
    image_shape: Tuple[int, int],
    config: Optional[Dict] = None
) -> Dict:
    """
    Extract polygon stencil by grounding YOLO against MediaPipe.

    This is the main entry point for the stencil extraction system.
    It combines YOLO's dense segmentation with MediaPipe's precise landmarks
    to create an accurate boundary polygon for eyebrow shaping.

    Parameters:
        yolo_mask: Binary mask from YOLO detection (H, W), dtype=uint8, values 0 or 1
        mp_landmarks: List of 10 MediaPipe points [[x, y], ...] per eyebrow
        image_shape: (height, width) of original image
        config: Optional configuration dict (uses defaults if None)

    Returns:
        {
            'polygon': [[x, y], ...],           # Final polygon coordinates
            'source': str,                      # 'merged' or 'mediapipe_only'
            'num_points': int,                  # Number of vertices
            'alignment': {...},                 # Alignment metrics
            'validation': {...},                # Validation results
            'bbox': [x1, y1, x2, y2],          # Bounding box
            'metadata': {...}                   # Additional info
        }

    Example:
        >>> yolo_mask = yolo_detections['eyebrows'][0]['mask']
        >>> mp_landmarks = mp_result['left_eyebrow']['points']
        >>> image_shape = (600, 800)
        >>>
        >>> result = extract_stencil_polygon(yolo_mask, mp_landmarks, image_shape)
        >>> print(f"Polygon: {len(result['polygon'])} points")
        >>> print(f"Source: {result['source']}")
        >>> print(f"Aligned: {result['alignment']['aligned']}")

    Algorithm Phases:
        1. Extract YOLO Polygon: cv2.findContours + Douglas-Peucker simplification
        2. Check Alignment: Calculate IoU and distance metrics
        3. Merge or Fallback:
           - If aligned: Insert MP landmarks into YOLO polygon
           - If misaligned: Use MP landmarks only
        4. Validate: Check polygon properties (point count, closed, etc.)
    """
    # Use default config if not provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults (provided config overrides defaults)
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config

    # =========================================================================
    # PHASE 1: EXTRACT YOLO POLYGON
    # =========================================================================
    # Convert binary mask to simplified polygon contour
    yolo_polygon = utils.extract_yolo_contour(
        yolo_mask,
        epsilon_factor=config['yolo_simplify_epsilon']
    )

    # Handle case where YOLO extraction fails
    if not yolo_polygon or len(yolo_polygon) < 3:
        # No valid YOLO contour - must use MediaPipe only
        return _create_fallback_result(mp_landmarks, image_shape, config, reason='yolo_extraction_failed')

    # =========================================================================
    # PHASE 2: CHECK ALIGNMENT (GROUNDING VALIDATION)
    # =========================================================================
    # Determine if YOLO and MediaPipe agree
    alignment = utils.calculate_alignment_score(
        yolo_polygon,
        mp_landmarks,
        image_shape
    )

    # =========================================================================
    # PHASE 3: MERGE (if aligned) OR FALLBACK (if mismatch)
    # =========================================================================
    # NEW LOGIC: If all MP points are inside YOLO (with 10% buffer), keep YOLO as-is
    if alignment.get('all_mp_inside_with_buffer', False):
        # CASE A: All MediaPipe landmarks inside YOLO polygon (within 10% buffer)
        # No need to complicate - YOLO already covers everything
        final_polygon = yolo_polygon
        source = 'yolo_only'
        merge_info = {
            'yolo_vertices': len(yolo_polygon),
            'mp_landmarks': len(mp_landmarks),
            'final_vertices': len(final_polygon),
            'source': source,
            'merged': False,
            'mp_inside_with_buffer_count': alignment['mp_inside_with_buffer_count'],
            'mp_inside_with_buffer_ratio': alignment['mp_inside_with_buffer_ratio'],
            'buffer_distance': alignment['buffer_distance'],
            'reason': 'all_mp_inside_yolo_with_buffer'
        }
    elif alignment['aligned'] and config['grounding_method'] == 'insert_mp':
        # CASE B: Aligned but some MP points outside - Merge YOLO + MediaPipe
        final_polygon = utils.insert_mp_into_polygon(yolo_polygon, mp_landmarks)
        source = 'merged'
        merge_info = {
            'yolo_vertices': len(yolo_polygon),
            'mp_landmarks': len(mp_landmarks),
            'final_vertices': len(final_polygon),
            'source': source,
            'merged': True,
            'mp_inside_count': alignment['mp_inside_count'],
            'mp_inside_ratio': alignment['mp_inside_ratio']
        }
    else:
        # CASE C: Misaligned or config disabled - Use MediaPipe only
        if config['fallback_to_mp']:
            final_polygon = mp_landmarks.copy()
            source = 'mediapipe_only'
            merge_info = {
                'yolo_vertices': len(yolo_polygon),
                'mp_landmarks': len(mp_landmarks),
                'final_vertices': len(final_polygon),
                'source': source,
                'merged': False,
                'mp_inside_count': alignment.get('mp_inside_count', 0),
                'mp_inside_ratio': alignment.get('mp_inside_ratio', 0.0),
                'fallback_reason': 'alignment_failed' if not alignment['aligned'] else 'config_disabled'
            }
        else:
            # Fallback disabled - use YOLO only (rare case)
            final_polygon = yolo_polygon
            source = 'yolo_only'
            merge_info = {
                'yolo_vertices': len(yolo_polygon),
                'mp_landmarks': len(mp_landmarks),
                'final_vertices': len(final_polygon),
                'source': source,
                'merged': False,
                'mp_inside_count': alignment.get('mp_inside_count', 0),
                'mp_inside_ratio': alignment.get('mp_inside_ratio', 0.0),
                'fallback_reason': 'fallback_disabled'
            }

    # =========================================================================
    # PHASE 4: VALIDATE & RETURN
    # =========================================================================
    # Validate polygon properties
    validation = utils.validate_polygon(final_polygon, config)

    # Calculate bounding box
    bbox = utils.calculate_bbox(final_polygon)

    # Package complete result
    result = {
        'polygon': final_polygon,
        'source': source,
        'num_points': len(final_polygon),
        'alignment': alignment,
        'validation': validation,
        'bbox': bbox,
        'metadata': {
            **merge_info,
            'aligned': alignment['aligned'],
            'iou': alignment['iou'],
            'avg_distance': alignment['avg_distance'],
            'image_shape': image_shape,
            'config_used': {
                'epsilon': config['yolo_simplify_epsilon'],
                'iou_threshold': config['alignment_iou_threshold'],
                'distance_threshold': config['alignment_distance_threshold']
            }
        }
    }

    return result


def _create_fallback_result(
    mp_landmarks: List[List[int]],
    image_shape: Tuple[int, int],
    config: Dict,
    reason: str = 'unknown'
) -> Dict:
    """
    Create result using MediaPipe landmarks only (fallback).

    Used when YOLO extraction fails or alignment check cannot be performed.

    Parameters:
        mp_landmarks: List of 10 MediaPipe points
        image_shape: (height, width)
        config: Configuration dict
        reason: Why fallback was triggered

    Returns:
        Result dict in same format as extract_stencil_polygon()
    """
    final_polygon = mp_landmarks.copy()

    # Validate polygon
    validation = utils.validate_polygon(final_polygon, config)

    # Calculate bounding box
    bbox = utils.calculate_bbox(final_polygon)

    result = {
        'polygon': final_polygon,
        'source': 'mediapipe_only',
        'num_points': len(final_polygon),
        'alignment': {
            'aligned': False,
            'iou': 0.0,
            'avg_distance': float('inf')
        },
        'validation': validation,
        'bbox': bbox,
        'metadata': {
            'yolo_vertices': 0,
            'mp_landmarks': len(mp_landmarks),
            'final_vertices': len(final_polygon),
            'merged': False,
            'fallback_reason': reason,
            'image_shape': image_shape
        }
    }

    return result


def extract_stencils_from_detections(
    yolo_detections: Dict,
    mp_detections: Dict,
    image_shape: Tuple[int, int],
    config: Optional[Dict] = None
) -> List[Dict]:
    """
    Extract stencil polygons from complete detection results.

    This is a convenience function that processes both left and right eyebrows
    from YOLO and MediaPipe detection dictionaries.

    Parameters:
        yolo_detections: Result from yolo_pred.detect_yolo()
                        Must contain 'eyebrows' key with list of detections
        mp_detections: Result from mediapipe_pred.detect_mediapipe()
                      Must contain 'left_eyebrow' and 'right_eyebrow' keys
        image_shape: (height, width) of original image
        config: Optional configuration dict

    Returns:
        List of result dicts (one per eyebrow), each containing:
            {
                'side': 'left' or 'right',
                'polygon': {...},
                'alignment': {...},
                'validation': {...},
                'metadata': {...}
            }

    Example:
        >>> yolo_result = detect_yolo(model, image_path)
        >>> mp_result = detect_mediapipe(image)
        >>> stencils = extract_stencils_from_detections(
        ...     yolo_result,
        ...     mp_result,
        ...     image.shape[:2]
        ... )
        >>> for stencil in stencils:
        ...     print(f"{stencil['side']}: {len(stencil['polygon'])} points")
    """
    results = []

    # Process eyebrows (match YOLO with MediaPipe)
    yolo_eyebrows = yolo_detections.get('eyebrows', [])

    if not yolo_eyebrows:
        # No YOLO detections - try MediaPipe only
        if 'left_eyebrow' in mp_detections:
            left_result = _create_fallback_result(
                mp_detections['left_eyebrow']['points'],
                image_shape,
                config or DEFAULT_CONFIG,
                reason='no_yolo_detections'
            )
            left_result['side'] = 'left'
            results.append(left_result)

        if 'right_eyebrow' in mp_detections:
            right_result = _create_fallback_result(
                mp_detections['right_eyebrow']['points'],
                image_shape,
                config or DEFAULT_CONFIG,
                reason='no_yolo_detections'
            )
            right_result['side'] = 'right'
            results.append(right_result)

        return results

    # Match YOLO detections with MediaPipe landmarks by position
    image_width = image_shape[1]
    midpoint = image_width / 2

    for yolo_eyebrow in yolo_eyebrows:
        # Determine side based on centroid x-coordinate
        centroid_x = yolo_eyebrow.get('mask_centroid', [0, 0])[0]
        side = 'left' if centroid_x < midpoint else 'right'

        # Get corresponding MediaPipe landmarks
        mp_key = f'{side}_eyebrow'
        if mp_key in mp_detections and mp_detections[mp_key]:
            mp_landmarks = mp_detections[mp_key]['points']

            # Extract polygon
            result = extract_stencil_polygon(
                yolo_eyebrow['mask'],
                mp_landmarks,
                image_shape,
                config
            )

            # Add side information
            result['side'] = side

            # Add YOLO confidence
            result['metadata']['yolo_confidence'] = yolo_eyebrow.get('confidence', 0.0)

            results.append(result)
        else:
            # No MediaPipe for this side - use YOLO only
            yolo_polygon = utils.extract_yolo_contour(
                yolo_eyebrow['mask'],
                epsilon_factor=(config or DEFAULT_CONFIG)['yolo_simplify_epsilon']
            )

            result = {
                'side': side,
                'polygon': yolo_polygon,
                'source': 'yolo_only',
                'num_points': len(yolo_polygon),
                'alignment': {'aligned': False, 'iou': 0.0, 'avg_distance': float('inf')},
                'validation': utils.validate_polygon(yolo_polygon, config or DEFAULT_CONFIG),
                'bbox': utils.calculate_bbox(yolo_polygon),
                'metadata': {
                    'yolo_vertices': len(yolo_polygon),
                    'mp_landmarks': 0,
                    'final_vertices': len(yolo_polygon),
                    'merged': False,
                    'fallback_reason': 'no_mediapipe_for_side',
                    'yolo_confidence': yolo_eyebrow.get('confidence', 0.0)
                }
            }

            results.append(result)

    return results


def visualize_stencil_extraction(
    image: np.ndarray,
    yolo_mask: np.ndarray,
    mp_landmarks: List[List[int]],
    result: Dict,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization of stencil extraction process.

    Draws:
        - Original image (background)
        - YOLO mask (red overlay, semi-transparent)
        - MediaPipe landmarks (blue circles)
        - Final polygon (green line)
        - Alignment info (text overlay)

    Parameters:
        image: Original image (H, W, 3)
        yolo_mask: YOLO binary mask (H, W)
        mp_landmarks: MediaPipe points [[x, y], ...]
        result: Result dict from extract_stencil_polygon()
        output_path: Optional path to save image

    Returns:
        Visualization image (H, W, 3)
    """
    vis = image.copy()

    # Draw YOLO mask (red overlay)
    yolo_overlay = np.zeros_like(vis)
    yolo_overlay[:, :, 2] = (yolo_mask * 255).astype(np.uint8)  # Red channel
    vis = cv2.addWeighted(vis, 0.7, yolo_overlay, 0.3, 0)

    # Draw MediaPipe landmarks (blue circles)
    for mp_point in mp_landmarks:
        cv2.circle(vis, tuple(mp_point), 5, (255, 0, 0), -1)  # Blue

    # Draw final polygon (green line)
    polygon_points = np.array(result['polygon'], dtype=np.int32)
    cv2.polylines(vis, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Add text info
    info_lines = [
        f"Source: {result['source']}",
        f"Points: {result['num_points']}",
        f"IoU: {result['alignment']['iou']:.2f}",
        f"Distance: {result['alignment']['avg_distance']:.1f}px",
        f"Aligned: {result['alignment']['aligned']}"
    ]

    y_offset = 30
    for line in info_lines:
        cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += 25

    # Save if path provided
    if output_path:
        cv2.imwrite(output_path, vis)

    return vis


# =============================================================================
# MAIN ENTRY POINT (for testing/CLI)
# =============================================================================

if __name__ == '__main__':
    """
    Test the stencil extraction on a sample image.

    Usage:
        python stencil_extract.py
    """
    import sys
    from yolo_pred import load_yolo_model, detect_yolo
    from mediapipe_pred import detect_mediapipe

    print("=" * 70)
    print("Stencil Extraction Test")
    print("=" * 70)

    # Load test image
    test_image_path = 'annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg'
    print(f"\nLoading test image: {test_image_path}")

    image = cv2.imread(test_image_path)
    if image is None:
        print(f"ERROR: Could not load image from {test_image_path}")
        sys.exit(1)

    image_shape = image.shape[:2]
    print(f"Image shape: {image_shape}")

    # Run YOLO detection
    print("\nRunning YOLO detection...")
    model = load_yolo_model()
    yolo_result = detect_yolo(model, test_image_path)
    print(f"YOLO eyebrows detected: {len(yolo_result['eyebrows'])}")

    # Run MediaPipe detection
    print("\nRunning MediaPipe detection...")
    mp_result = detect_mediapipe(image)
    print(f"MediaPipe left eyebrow: {len(mp_result.get('left_eyebrow', {}).get('points', []))} points")
    print(f"MediaPipe right eyebrow: {len(mp_result.get('right_eyebrow', {}).get('points', []))} points")

    # Extract stencils
    print("\nExtracting stencil polygons...")
    stencils = extract_stencils_from_detections(yolo_result, mp_result, image_shape)

    print(f"\nExtracted {len(stencils)} stencils:")
    for stencil in stencils:
        print(f"\n  {stencil['side'].upper()} Eyebrow:")
        print(f"    Polygon: {stencil['num_points']} points")
        print(f"    Source: {stencil['source']}")
        print(f"    Aligned: {stencil['alignment']['aligned']}")
        print(f"    IoU: {stencil['alignment']['iou']:.3f}")
        print(f"    Avg Distance: {stencil['alignment']['avg_distance']:.1f}px")
        print(f"    Valid: {stencil['validation']['valid']}")

        # Create visualization
        if yolo_result['eyebrows']:
            idx = 0 if stencil['side'] == 'left' else min(1, len(yolo_result['eyebrows']) - 1)
            if idx < len(yolo_result['eyebrows']):
                yolo_mask = yolo_result['eyebrows'][idx]['mask']
                mp_key = f"{stencil['side']}_eyebrow"
                if mp_key in mp_result:
                    mp_landmarks = mp_result[mp_key]['points']

                    vis_path = f"test_span_output/stencil_extraction_{stencil['side']}.jpg"
                    visualize_stencil_extraction(image, yolo_mask, mp_landmarks, stencil, vis_path)
                    print(f"    Visualization saved: {vis_path}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
