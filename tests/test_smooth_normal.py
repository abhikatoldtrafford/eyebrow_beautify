"""
Test script to verify that the new smoothing function preserves natural eyebrow curvature.

This tests the normal-vector-based smoothing approach.
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


def visualize_smoothing_comparison(original_mask, smoothed_mask, title="Smoothing Comparison"):
    """Visualize before/after smoothing with contour overlay."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title("Original YOLO Mask")
    axes[0].axis('off')

    # Smoothed
    axes[1].imshow(smoothed_mask, cmap='gray')
    axes[1].set_title("Smoothed (Normal-Vector Preserving)")
    axes[1].axis('off')

    # Overlay comparison
    overlay = np.zeros((*original_mask.shape, 3), dtype=np.uint8)
    overlay[original_mask > 0] = [255, 0, 0]  # Red: original
    overlay[smoothed_mask > 0] = [0, 255, 0]  # Green: smoothed
    overlap = np.logical_and(original_mask, smoothed_mask)
    overlay[overlap] = [255, 255, 0]  # Yellow: overlap

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red=Original, Green=Smoothed, Yellow=Both)")
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def extract_contour_curvature(mask):
    """
    Extract contour and calculate local curvature at each point.

    Returns:
        contour: (N, 2) array of contour points
        curvatures: (N,) array of curvature values
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None, None

    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour.squeeze()

    if len(contour_points.shape) != 2:
        return None, None

    # Calculate curvature at each point
    n = len(contour_points)
    curvatures = []

    for i in range(n):
        # Get three consecutive points
        p0 = contour_points[(i - 1) % n]
        p1 = contour_points[i]
        p2 = contour_points[(i + 1) % n]

        # Calculate curvature using the Menger curvature formula
        # κ = 4 * Area(triangle) / (|p0-p1| * |p1-p2| * |p2-p0|)

        # Side lengths
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p0 - p2)

        # Triangle area using cross product
        area = 0.5 * abs(np.cross(p1 - p0, p2 - p0))

        # Curvature
        if a * b * c > 0:
            k = 4 * area / (a * b * c)
        else:
            k = 0

        curvatures.append(k)

    return contour_points, np.array(curvatures)


def test_curvature_preservation(model, test_image_path, output_dir="tests/output/smoothing"):
    """
    Test that smoothing preserves natural curvature.

    Compares curvature distribution before/after smoothing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("NORMAL-VECTOR SMOOTHING TEST")
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
        print(f"\n{'='*60}")
        print(f"Testing Eyebrow {i+1}")
        print(f"{'='*60}")

        original_mask = eyebrow['mask']

        # Calculate original metrics
        original_area = original_mask.sum()
        original_thickness = utils.calculate_mask_thickness(original_mask)
        original_contour, original_curvatures = extract_contour_curvature(original_mask)

        print(f"\n3. Original mask metrics:")
        print(f"   Area: {original_area} pixels")
        print(f"   Thickness: {original_thickness:.2f} pixels")
        if original_curvatures is not None:
            print(f"   Mean curvature: {original_curvatures.mean():.6f}")
            print(f"   Max curvature: {original_curvatures.max():.6f}")
            print(f"   Curvature std: {original_curvatures.std():.6f}")

        # Test different smoothing levels
        test_configs = [
            {"kernel_size": 3, "iterations": 1, "name": "Light (3x1)"},
            {"kernel_size": 5, "iterations": 2, "name": "Medium (5x2)"},
            {"kernel_size": 7, "iterations": 2, "name": "Heavy (7x2)"},
        ]

        for config in test_configs:
            print(f"\n4. Testing smoothing: {config['name']}")

            # Apply smoothing
            smoothed_mask = utils.smooth_mask_contours(
                original_mask,
                kernel_size=config['kernel_size'],
                iterations=config['iterations']
            )

            # Calculate smoothed metrics
            smoothed_area = smoothed_mask.sum()
            smoothed_thickness = utils.calculate_mask_thickness(smoothed_mask)
            smoothed_contour, smoothed_curvatures = extract_contour_curvature(smoothed_mask)

            # Calculate changes
            area_change = (smoothed_area - original_area) / original_area * 100
            thickness_change = (smoothed_thickness - original_thickness) / original_thickness * 100

            print(f"   Area: {smoothed_area} pixels ({area_change:+.1f}%)")
            print(f"   Thickness: {smoothed_thickness:.2f} pixels ({thickness_change:+.1f}%)")

            if smoothed_curvatures is not None and original_curvatures is not None:
                curvature_change = (smoothed_curvatures.mean() - original_curvatures.mean()) / original_curvatures.mean() * 100
                print(f"   Mean curvature: {smoothed_curvatures.mean():.6f} ({curvature_change:+.1f}%)")
                print(f"   Max curvature: {smoothed_curvatures.max():.6f}")
                print(f"   Curvature std: {smoothed_curvatures.std():.6f}")

                # Check if curvature is preserved (within 20% is good)
                if abs(curvature_change) < 20:
                    print(f"   ✓ Curvature preserved (change < 20%)")
                else:
                    print(f"   ⚠ Curvature changed significantly (change > 20%)")

            # Visualize
            fig = visualize_smoothing_comparison(
                original_mask,
                smoothed_mask,
                title=f"Eyebrow {i+1} - {config['name']}"
            )

            # Save
            output_file = output_path / f"eyebrow_{i+1}_{config['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"   Saved: {output_file}")

    print(f"\n{'='*60}")
    print(f"TEST COMPLETE")
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
        test_curvature_preservation(model, test_image)
