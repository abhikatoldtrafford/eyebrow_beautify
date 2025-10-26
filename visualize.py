"""
Visualization Functions for Eyebrow Beautification

Provides visualization tools including:
- 6-panel comparative visualization
- Detection overlay drawing
- Mask comparisons and difference maps
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

import utils


# Color palette (BGR format)
COLORS = {
    'eyebrows': (0, 0, 255),      # Red
    'eye': (255, 0, 0),           # Blue
    'eye_box': (0, 255, 0),       # Green
    'hair': (255, 255, 0),        # Cyan
    'mediapipe': (255, 0, 255),   # Magenta
    'expansion': (255, 100, 0),   # Blue (for difference map)
    'reduction': (0, 165, 255),   # Orange (for difference map)
}


def draw_detections(img: np.ndarray, detections: Dict,
                   colors: Optional[Dict] = None,
                   draw_masks: bool = True,
                   draw_boxes: bool = True,
                   draw_labels: bool = True,
                   mask_alpha: float = 0.4) -> np.ndarray:
    """
    Draw YOLO detections on image with colored overlays.

    Args:
        img: Input image (BGR)
        detections: YOLO detection dict (from yolo_pred.detect_yolo)
        colors: Color dict for each class (BGR format)
        draw_masks: Whether to draw segmentation masks
        draw_boxes: Whether to draw bounding boxes
        draw_labels: Whether to draw class labels
        mask_alpha: Transparency for mask overlay (0-1)

    Returns:
        Image with detections drawn
    """
    if colors is None:
        colors = COLORS

    img_with_detections = img.copy()
    overlay = img.copy()

    # Draw masks first (as overlay)
    if draw_masks:
        for class_name, det_list in detections.items():
            color = colors.get(class_name, (255, 255, 255))

            for det in det_list:
                if 'mask' not in det:
                    continue

                mask = det['mask']

                # Create colored mask
                colored_mask = np.zeros_like(img)
                colored_mask[mask == 1] = color

                # Blend with overlay
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, mask_alpha, 0)

        # Blend overlay with original
        img_with_detections = cv2.addWeighted(img_with_detections, 1.0 - mask_alpha,
                                             overlay, mask_alpha, 0)

    # Draw boxes and labels
    if draw_boxes or draw_labels:
        for class_name, det_list in detections.items():
            color = colors.get(class_name, (255, 255, 255))

            for det in det_list:
                if 'box' not in det:
                    continue

                x1, y1, x2, y2 = [int(c) for c in det['box']]

                # Draw bounding box
                if draw_boxes:
                    cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)

                # Draw label
                if draw_labels and 'class_name' in det and 'confidence' in det:
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y1_label = max(y1, label_size[1] + 10)

                    # Draw label background
                    cv2.rectangle(
                        img_with_detections,
                        (x1, y1_label - label_size[1] - 10),
                        (x1 + label_size[0], y1_label),
                        color,
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        img_with_detections,
                        label,
                        (x1, y1_label - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

    return img_with_detections


def create_mask_overlay(img: np.ndarray, mask: np.ndarray,
                       color: Tuple[int, int, int],
                       alpha: float = 0.5) -> np.ndarray:
    """
    Create image with colored mask overlay.

    Args:
        img: Input image (BGR)
        mask: Binary mask
        color: Overlay color (BGR)
        alpha: Transparency (0-1)

    Returns:
        Image with mask overlay
    """
    overlay = img.copy()

    # Create colored mask
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 1] = color

    # Blend
    result = cv2.addWeighted(img, 1.0 - alpha, colored_mask, alpha, 0)

    return result


def create_difference_map(original_mask: np.ndarray, final_mask: np.ndarray,
                         background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Create difference map showing expansion and reduction.

    Args:
        original_mask: Original YOLO mask
        final_mask: Final beautified mask
        background_color: Background color (BGR)

    Returns:
        RGB image with:
        - Blue: expansion (final - original)
        - Orange: reduction (original - final)
        - Gray: unchanged (intersection)
    """
    h, w = original_mask.shape
    diff_map = np.full((h, w, 3), background_color, dtype=np.uint8)

    # Expansion: in final but not in original (blue)
    expansion = np.logical_and(final_mask == 1, original_mask == 0)
    diff_map[expansion] = COLORS['expansion']

    # Reduction: in original but not in final (orange)
    reduction = np.logical_and(original_mask == 1, final_mask == 0)
    diff_map[reduction] = COLORS['reduction']

    # Unchanged: in both (gray)
    intersection = np.logical_and(original_mask == 1, final_mask == 1)
    diff_map[intersection] = (128, 128, 128)

    return diff_map


def draw_mediapipe_landmarks(img: np.ndarray, mp_eyebrow: Optional[Dict],
                            draw_points: bool = True,
                            draw_arch: bool = True) -> np.ndarray:
    """
    Draw MediaPipe eyebrow landmarks on image.

    Args:
        img: Input image (BGR)
        mp_eyebrow: MediaPipe eyebrow dict (with 'points')
        draw_points: Whether to draw landmark points
        draw_arch: Whether to draw fitted arch line

    Returns:
        Image with MediaPipe landmarks drawn
    """
    if not mp_eyebrow:
        return img

    result = img.copy()
    h, w = img.shape[:2]

    points = mp_eyebrow.get('points', [])

    if not points:
        return result

    # Draw fitted arch (connecting line through points)
    if draw_arch:
        # Sort points left to right
        sorted_points = sorted(points, key=lambda p: p[0])

        # Draw connecting lines
        for i in range(len(sorted_points) - 1):
            pt1 = sorted_points[i]
            pt2 = sorted_points[i + 1]
            cv2.line(result, pt1, pt2, COLORS['mediapipe'], 2)

    # Draw landmark points
    if draw_points:
        for x, y in points:
            cv2.circle(result, (x, y), 4, COLORS['mediapipe'], -1)

    return result


def create_clean_result(mask: np.ndarray,
                       background_color: Tuple[int, int, int] = (255, 255, 255),
                       mask_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Create clean visualization of mask on solid background.

    Args:
        mask: Binary mask
        background_color: Background color (BGR)
        mask_color: Mask color (BGR)

    Returns:
        RGB image with mask on solid background
    """
    h, w = mask.shape
    clean = np.full((h, w, 3), background_color, dtype=np.uint8)
    clean[mask == 1] = mask_color

    return clean


def create_6panel_visualization(pair: Dict, final_mask: np.ndarray,
                                original_image: np.ndarray) -> np.ndarray:
    """
    Create 6-panel comparative visualization.

    Layout (2 rows x 3 columns):
    1. Original Image (eyebrow region highlighted)
    2. YOLO Mask (red overlay)
    3. Final Mask (green overlay)
    4. Difference Map (blue=expansion, orange=reduction)
    5. MediaPipe Overlay (points + fitted arch)
    6. Clean Result (final mask on white background)

    Args:
        pair: Eyebrow pair dict from beautify.py
        final_mask: Final beautified mask
        original_image: Original input image

    Returns:
        6-panel visualization image (H x W x 3)
    """
    h, w = original_image.shape[:2]

    # Extract masks and data
    yolo_mask = pair['eyebrow']['mask']
    mp_eyebrow = pair['mp_eyebrow']

    # Get eyebrow bounding box for cropping
    final_bbox = utils.get_bounding_box_from_mask(final_mask)
    x1, y1, x2, y2 = final_bbox

    # Add padding (20%)
    padding_x = int((x2 - x1) * 0.2)
    padding_y = int((y2 - y1) * 0.2)

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w, x2 + padding_x)
    y2 = min(h, y2 + padding_y)

    # Helper function to crop region
    def crop_region(img):
        if len(img.shape) == 2:
            return img[y1:y2, x1:x2]
        else:
            return img[y1:y2, x1:x2, :]

    # Panel 1: Original Image (eyebrow region highlighted)
    panel1 = original_image.copy()
    cv2.rectangle(panel1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    panel1 = crop_region(panel1)

    # Panel 2: YOLO Mask (red overlay)
    panel2 = create_mask_overlay(original_image, yolo_mask, COLORS['eyebrows'], alpha=0.5)
    panel2 = crop_region(panel2)

    # Panel 3: Final Mask (green overlay)
    panel3 = create_mask_overlay(original_image, final_mask, (0, 255, 0), alpha=0.5)
    panel3 = crop_region(panel3)

    # Panel 4: Difference Map
    panel4 = create_difference_map(yolo_mask, final_mask)
    panel4 = crop_region(panel4)

    # Panel 5: MediaPipe Overlay
    panel5 = original_image.copy()
    panel5 = draw_mediapipe_landmarks(panel5, mp_eyebrow, draw_points=True, draw_arch=True)
    panel5 = crop_region(panel5)

    # Panel 6: Clean Result (black mask on white background)
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    clean_mask[y1:y2, x1:x2] = final_mask[y1:y2, x1:x2]
    panel6 = create_clean_result(clean_mask, background_color=(255, 255, 255),
                                 mask_color=(0, 0, 0))
    panel6 = crop_region(panel6)

    # Ensure all panels have the same size (resize to match panel1)
    target_h, target_w = panel1.shape[:2]

    panels = [panel1, panel2, panel3, panel4, panel5, panel6]
    resized_panels = []

    for panel in panels:
        if panel.shape[:2] != (target_h, target_w):
            panel = cv2.resize(panel, (target_w, target_h))
        resized_panels.append(panel)

    # Add labels to each panel
    labels = [
        "1. Original Image",
        "2. YOLO Mask (Red)",
        "3. Final Mask (Green)",
        "4. Difference Map",
        "5. MediaPipe Overlay",
        "6. Clean Result"
    ]

    labeled_panels = []
    for panel, label in zip(resized_panels, labels):
        # Add white bar at top for label
        label_h = 30
        labeled = np.ones((panel.shape[0] + label_h, panel.shape[1], 3), dtype=np.uint8) * 255

        # Add label text
        cv2.putText(
            labeled, label,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        # Add panel below label
        labeled[label_h:, :, :] = panel

        labeled_panels.append(labeled)

    # Arrange in 2x3 grid
    row1 = np.hstack(labeled_panels[0:3])
    row2 = np.hstack(labeled_panels[3:6])
    grid = np.vstack([row1, row2])

    # Add title bar at top
    title_h = 50
    title_bar = np.ones((title_h, grid.shape[1], 3), dtype=np.uint8) * 240

    title = f"Eyebrow Beautification - {pair['side'].upper()} Side"
    cv2.putText(
        title_bar, title,
        (grid.shape[1] // 2 - 200, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Final visualization
    visualization = np.vstack([title_bar, grid])

    return visualization


def save_visualization(visualization: np.ndarray, output_path: str):
    """
    Save visualization to file.

    Args:
        visualization: Visualization image
        output_path: Output file path
    """
    cv2.imwrite(output_path, visualization)
    print(f"Saved visualization to: {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of visualization functions."""

    import yolo_pred
    import mediapipe_pred
    import beautify

    # Load model and run beautification
    print("Loading YOLO model...")
    model = yolo_pred.load_yolo_model()

    test_image = "annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    print(f"Running beautification on: {test_image}")
    results = beautify.beautify_eyebrows(test_image, model)

    # Load original image
    original_img = cv2.imread(test_image)

    # Create visualizations for each result
    for i, result in enumerate(results):
        print(f"\nCreating visualization for {result['side']} eyebrow...")

        # We need the pair dict to create 6-panel visualization
        # For now, just demonstrate mask overlay

        final_mask = result['masks']['final_beautified']
        yolo_mask = result['masks']['original_yolo']

        # Create comparison visualization
        vis_final = create_mask_overlay(original_img, final_mask, (0, 255, 0), alpha=0.5)
        vis_yolo = create_mask_overlay(original_img, yolo_mask, (0, 0, 255), alpha=0.5)

        # Create difference map
        diff_map = create_difference_map(yolo_mask, final_mask)

        # Stack horizontally
        comparison = np.hstack([vis_yolo, vis_final, diff_map])

        # Save
        output_path = f"visualization_{result['side']}_eyebrow.jpg"
        cv2.imwrite(output_path, comparison)
        print(f"Saved comparison to: {output_path}")

    print("\nDone!")
