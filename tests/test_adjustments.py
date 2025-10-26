"""
Test script for eyebrow adjustment functions.

Tests all 4 adjustment operations:
1. Increase thickness
2. Decrease thickness
3. Increase span (directional - tail only)
4. Decrease span (directional - tail only)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt

import yolo_pred
import utils


def visualize_adjustment(original_mask, adjusted_mask, title="Adjustment"):
    """Visualize before/after adjustment with overlay."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Adjusted
    axes[1].imshow(adjusted_mask, cmap='gray')
    axes[1].set_title("Adjusted")
    axes[1].axis('off')

    # Overlay comparison
    overlay = np.zeros((*original_mask.shape, 3), dtype=np.uint8)
    overlay[original_mask > 0] = [255, 0, 0]  # Red: original
    overlay[adjusted_mask > 0] = [0, 255, 0]  # Green: adjusted
    overlap = np.logical_and(original_mask, adjusted_mask)
    overlay[overlap] = [255, 255, 0]  # Yellow: overlap

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red=Original, Green=Adjusted, Yellow=Both)")
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def test_all_adjustments(model, test_image_path, output_dir="tests/output/adjustments"):
    """Test all 4 adjustment operations on eyebrows."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EYEBROW ADJUSTMENT TESTS")
    print("="*60)

    # Load image and detect
    print(f"\n1. Loading image: {test_image_path}")
    img = cv2.imread(test_image_path)
    h, w = img.shape[:2]

    print("2. Running YOLO detection...")
    detections = yolo_pred.detect_yolo(model, test_image_path, conf_threshold=0.25)

    if 'eyebrows' not in detections or len(detections['eyebrows']) == 0:
        print("ERROR: No eyebrows detected!")
        return

    print(f"   Found {len(detections['eyebrows'])} eyebrow(s)")

    # Test each eyebrow
    for i, eyebrow in enumerate(detections['eyebrows']):
        # Determine side (simple heuristic)
        centroid = eyebrow['mask_centroid']
        side = 'left' if centroid[0] < w / 2 else 'right'

        print(f"\n{'='*60}")
        print(f"Testing Eyebrow {i+1} ({side} side)")
        print(f"{'='*60}")

        original_mask = eyebrow['mask']
        original_area = original_mask.sum()
        original_thickness = utils.calculate_mask_thickness(original_mask)
        bbox = utils.get_bounding_box_from_mask(original_mask)
        original_span = bbox[2] - bbox[0]

        print(f"\n3. Original metrics:")
        print(f"   Area: {original_area} pixels")
        print(f"   Thickness: {original_thickness:.2f} pixels")
        print(f"   Span: {original_span} pixels")

        # Test configurations
        tests = [
            {
                "name": "Increase Thickness (+5%)",
                "adjustment_type": "thickness",
                "direction": "increase",
                "increment": 0.05
            },
            {
                "name": "Decrease Thickness (-5%)",
                "adjustment_type": "thickness",
                "direction": "decrease",
                "increment": 0.05
            },
            {
                "name": "Increase Span (+5%, Tail Only)",
                "adjustment_type": "span",
                "direction": "increase",
                "increment": 0.05
            },
            {
                "name": "Decrease Span (-5%, Tail Only)",
                "adjustment_type": "span",
                "direction": "decrease",
                "increment": 0.05
            },
        ]

        for test in tests:
            print(f"\n4. Testing: {test['name']}")

            # Apply adjustment
            adjusted_mask = utils.apply_eyebrow_adjustment(
                mask=original_mask,
                adjustment_type=test['adjustment_type'],
                direction=test['direction'],
                increment=test['increment'],
                side=side
            )

            # Calculate metrics (convert to float to avoid overflow)
            adjusted_area = float(adjusted_mask.sum())
            adjusted_thickness = utils.calculate_mask_thickness(adjusted_mask)
            bbox_adj = utils.get_bounding_box_from_mask(adjusted_mask)
            adjusted_span = bbox_adj[2] - bbox_adj[0]

            area_change = ((adjusted_area - original_area) / original_area * 100) if original_area > 0 else 0
            thickness_change = ((adjusted_thickness - original_thickness) / original_thickness * 100) if original_thickness > 0 else 0
            span_change = ((adjusted_span - original_span) / original_span * 100) if original_span > 0 else 0

            print(f"   Area: {adjusted_area} pixels ({area_change:+.1f}%)")
            print(f"   Thickness: {adjusted_thickness:.2f} pixels ({thickness_change:+.1f}%)")
            print(f"   Span: {adjusted_span} pixels ({span_change:+.1f}%)")

            # Validate results
            if test['adjustment_type'] == 'thickness':
                if test['direction'] == 'increase' and area_change > 0:
                    print(f"   ✓ Thickness increased as expected")
                elif test['direction'] == 'decrease' and area_change < 0:
                    print(f"   ✓ Thickness decreased as expected")
                else:
                    print(f"   ⚠ Unexpected result!")

            elif test['adjustment_type'] == 'span':
                if test['direction'] == 'increase' and span_change > 0:
                    print(f"   ✓ Span increased as expected (tail side)")
                elif test['direction'] == 'decrease' and span_change < 0:
                    print(f"   ✓ Span decreased as expected (tail side)")
                else:
                    print(f"   ⚠ Unexpected result!")

            # Visualize
            fig = visualize_adjustment(
                original_mask,
                adjusted_mask,
                title=f"Eyebrow {i+1} ({side}) - {test['name']}"
            )

            # Save
            filename = f"eyebrow_{i+1}_{side}_{test['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
            output_file = output_path / filename
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"   Saved: {output_file}")

    print(f"\n{'='*60}")
    print(f"TESTS COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Load model
    print("Loading YOLO model...")
    model = yolo_pred.load_yolo_model()

    # Test on sample image
    test_image = "annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    if not Path(test_image).exists():
        print(f"ERROR: Test image not found: {test_image}")
    else:
        test_all_adjustments(model, test_image)
