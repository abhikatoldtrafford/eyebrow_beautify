"""
MediaPipe Face Mesh Detection Wrapper for Eyebrow Beautification.

Provides clean interface to MediaPipe Face Mesh (468 landmarks) for detecting:
- Left eyebrow (10 landmarks)
- Right eyebrow (10 landmarks)
- Left eye (8 landmarks)
- Right eye (8 landmarks)
- Face oval (36 landmarks)

Returns structured landmark dictionaries organized by facial feature.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, RuntimeError):
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")

import utils


# MediaPipe landmark indices for facial features
LANDMARK_INDICES = {
    'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    'left_eye': [33, 133, 160, 159, 158, 157, 173, 246],
    'right_eye': [362, 263, 387, 386, 385, 384, 398, 466],
    'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
}


def detect_mediapipe(image: np.ndarray, conf_threshold: float = 0.5) -> Optional[Dict]:
    """
    Run MediaPipe Face Mesh detection and return structured landmarks.

    Args:
        image: Input image (BGR format, from cv2.imread)
        conf_threshold: Minimum detection confidence (default: 0.5)

    Returns:
        Dictionary with structure:
        {
            'left_eyebrow': {
                'points': [(x1, y1), ..., (x10, y10)],  # 10 landmarks
                'indices': [70, 63, 105, ...],
                'center': (cx, cy),  # mean of points
                'bbox': [x1, y1, x2, y2]  # bounding box
            },
            'right_eyebrow': {...},  # same structure
            'left_eye': {...},  # 8 landmarks
            'right_eye': {...},  # 8 landmarks
            'face_oval': {...},  # 36 landmarks
            'all_landmarks': <MediaPipe landmarks object>  # raw 468 points
        }

        Returns None if:
        - MediaPipe not available
        - No face detected
        - Detection error

        Notes:
        - Coordinates are in absolute pixels (x, y)
        - Points are converted from MediaPipe's normalized 0-1 format
        - Left/right are from the person's perspective (left = their left)
    """
    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe not available")
        return None

    try:
        mp_face_mesh = mp.solutions.face_mesh

        # Run face mesh detection
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=conf_threshold
        ) as face_mesh:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                results = face_mesh.process(image_rgb)
            except AttributeError as e:
                # Known protobuf compatibility issue - this is a SYSTEM ERROR, not image issue
                print(f"❌ SYSTEM ERROR: MediaPipe library issue detected")
                print(f"   Error: {e}")
                print(f"   This is a protobuf compatibility problem, NOT an issue with your image")
                print(f"   Tip: Try: pip install --upgrade protobuf mediapipe")
                return None
            except Exception as e:
                # Unknown processing error - could be system or image related
                print(f"⚠️  MediaPipe processing failed: {e}")
                print(f"   This may be a system issue or image format problem")
                return None

            # Check if face detected
            if not results.multi_face_landmarks:
                print("ℹ️  No face detected by MediaPipe (image is valid, no landmarks found)")
                return None

            # Get first face (we only process one face)
            landmarks = results.multi_face_landmarks[0]

            # Extract organized landmark groups
            landmark_dict = extract_landmark_groups(landmarks, image.shape)

            # Add raw landmarks for advanced use
            landmark_dict['all_landmarks'] = landmarks

            return landmark_dict

    except Exception as e:
        # Outer exception handler for initialization errors
        print(f"❌ SYSTEM ERROR: Failed to initialize MediaPipe: {e}")
        return None


def extract_landmark_groups(landmarks, img_shape: Tuple[int, int, int]) -> Dict:
    """
    Extract and organize MediaPipe landmarks into facial feature groups.

    Args:
        landmarks: MediaPipe landmarks object (468 points)
        img_shape: Image shape (height, width, channels)

    Returns:
        Dictionary with organized landmark groups (eyebrows, eyes, face oval)
    """
    h, w = img_shape[:2]

    landmark_dict = {}

    # Process each facial feature group
    for feature_name, indices in LANDMARK_INDICES.items():
        # Extract points for this feature
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]

            # Convert normalized coordinates to absolute pixels
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

        # Calculate center (mean of points)
        center = utils.get_center(points)

        # Calculate bounding box
        bbox = utils.get_bbox_from_points(points)

        # Store organized data
        landmark_dict[feature_name] = {
            'points': points,
            'indices': indices,
            'center': center,
            'bbox': bbox
        }

    return landmark_dict


def calculate_mediapipe_coverage(mediapipe_points: List[Tuple[int, int]],
                                 yolo_mask: np.ndarray) -> Dict:
    """
    Calculate how many MediaPipe points are inside YOLO mask.

    Args:
        mediapipe_points: List of (x, y) landmark points
        yolo_mask: Binary mask from YOLO (H, W), values 0 or 1

    Returns:
        {
            'total_points': int,
            'points_inside': int,
            'points_outside': int,
            'coverage_percent': float,
            'outside_points': [(x, y), ...]  # points not in mask
        }
    """
    total_points = len(mediapipe_points)
    points_inside = 0
    outside_points = []

    h, w = yolo_mask.shape

    for x, y in mediapipe_points:
        # Check bounds
        if 0 <= x < w and 0 <= y < h:
            if yolo_mask[y, x] > 0:
                points_inside += 1
            else:
                outside_points.append((x, y))
        else:
            # Out of image bounds
            outside_points.append((x, y))

    points_outside = total_points - points_inside
    coverage_percent = (points_inside / total_points * 100) if total_points > 0 else 0.0

    return {
        'total_points': total_points,
        'points_inside': points_inside,
        'points_outside': points_outside,
        'coverage_percent': coverage_percent,
        'outside_points': outside_points
    }


def draw_landmarks_on_image(image: np.ndarray, landmark_dict: Dict,
                           draw_eyebrows: bool = True,
                           draw_eyes: bool = True,
                           draw_all: bool = False) -> np.ndarray:
    """
    Draw MediaPipe landmarks on image.

    Args:
        image: Input image (BGR)
        landmark_dict: Landmark dict from detect_mediapipe()
        draw_eyebrows: Draw eyebrow landmarks (magenta, size 4)
        draw_eyes: Draw eye landmarks (yellow, size 3)
        draw_all: Draw all 468 face landmarks (gray, size 1)

    Returns:
        Image with landmarks drawn
    """
    img_with_landmarks = image.copy()
    h, w = image.shape[:2]

    # Draw all 468 landmarks as small dots
    if draw_all and 'all_landmarks' in landmark_dict:
        landmarks = landmark_dict['all_landmarks']
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img_with_landmarks, (x, y), 1, (150, 150, 150), -1)  # Gray

    # Draw eyebrow landmarks (emphasized)
    if draw_eyebrows:
        if 'left_eyebrow' in landmark_dict and landmark_dict['left_eyebrow']:
            for x, y in landmark_dict['left_eyebrow']['points']:
                cv2.circle(img_with_landmarks, (x, y), 4, (255, 0, 255), -1)  # Magenta

        if 'right_eyebrow' in landmark_dict and landmark_dict['right_eyebrow']:
            for x, y in landmark_dict['right_eyebrow']['points']:
                cv2.circle(img_with_landmarks, (x, y), 4, (255, 0, 255), -1)  # Magenta

    # Draw eye landmarks
    if draw_eyes:
        if 'left_eye' in landmark_dict and landmark_dict['left_eye']:
            for x, y in landmark_dict['left_eye']['points']:
                cv2.circle(img_with_landmarks, (x, y), 3, (0, 255, 255), -1)  # Yellow

        if 'right_eye' in landmark_dict and landmark_dict['right_eye']:
            for x, y in landmark_dict['right_eye']['points']:
                cv2.circle(img_with_landmarks, (x, y), 3, (0, 255, 255), -1)  # Yellow

    return img_with_landmarks


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of MediaPipe detection wrapper."""

    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe not installed. Install with: pip install mediapipe")
        exit(1)

    # Load test image
    test_image = "annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"
    img = cv2.imread(test_image)

    if img is None:
        print(f"Error: Could not load image {test_image}")
        exit(1)

    print(f"Loaded image: {test_image}")
    print(f"Image shape: {img.shape}")

    # Run MediaPipe detection
    print("\nRunning MediaPipe Face Mesh detection...")
    landmarks = detect_mediapipe(img, conf_threshold=0.5)

    if landmarks is None:
        print("No landmarks detected!")
        exit(1)

    # Print summary
    print("\n" + "="*60)
    print("MEDIAPIPE LANDMARK SUMMARY")
    print("="*60)

    for feature_name in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye']:
        if feature_name in landmarks and landmarks[feature_name]:
            feature = landmarks[feature_name]
            print(f"\n{feature_name.upper()}:")
            print(f"  Points: {len(feature['points'])}")
            print(f"  Center: {feature['center']}")
            print(f"  Bbox: {feature['bbox']}")
            print(f"  Indices: {feature['indices'][:3]}... (showing first 3)")

    # Test coverage calculation with a dummy mask
    print("\n" + "="*60)
    print("TESTING COVERAGE CALCULATION")
    print("="*60)

    # Create dummy mask (center region)
    h, w = img.shape[:2]
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    dummy_mask[h//4:3*h//4, w//4:3*w//4] = 1

    if 'left_eyebrow' in landmarks and landmarks['left_eyebrow']:
        coverage = calculate_mediapipe_coverage(
            landmarks['left_eyebrow']['points'],
            dummy_mask
        )
        print(f"\nLeft eyebrow coverage (with dummy mask):")
        print(f"  Total points: {coverage['total_points']}")
        print(f"  Points inside: {coverage['points_inside']}")
        print(f"  Points outside: {coverage['points_outside']}")
        print(f"  Coverage: {coverage['coverage_percent']:.1f}%")

    # Draw landmarks
    img_with_landmarks = draw_landmarks_on_image(
        img, landmarks,
        draw_eyebrows=True,
        draw_eyes=True,
        draw_all=True
    )

    # Save result
    output_path = "mediapipe_test_output.jpg"
    cv2.imwrite(output_path, img_with_landmarks)
    print(f"\nSaved visualization to: {output_path}")
