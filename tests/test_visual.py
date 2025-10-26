"""
Visual validation tests

Generates and saves mask visualizations:
- Original YOLO mask vs final beautified mask
- Side-by-side comparisons
- Difference maps (expansion/reduction)
- MediaPipe overlay
"""

import requests
import base64
import cv2
import numpy as np
from pathlib import Path
from test_config import API_URL, TEST_IMAGES, VIZ_DIR


def base64_to_mask(b64_string):
    """Decode base64 string to binary mask."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Convert to binary
    _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return mask


def create_comparison_visualization(original_img, original_mask, final_mask, eyebrow_data, output_path):
    """
    Create 4-panel comparison visualization.

    Panels:
    1. Original image with original YOLO mask overlay (red)
    2. Original image with final beautified mask overlay (green)
    3. Difference map (blue=expansion, orange=reduction)
    4. Statistics text
    """
    h, w = original_img.shape[:2]

    # Create 2x2 grid
    panel_h, panel_w = h, w
    canvas = np.ones((panel_h * 2, panel_w * 2, 3), dtype=np.uint8) * 255

    # Panel 1: Original mask overlay
    panel1 = original_img.copy()
    overlay = panel1.copy()
    overlay[original_mask > 0] = [0, 0, 255]  # Red
    panel1 = cv2.addWeighted(panel1, 0.7, overlay, 0.3, 0)
    cv2.putText(panel1, "Original YOLO Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    canvas[0:panel_h, 0:panel_w] = panel1

    # Panel 2: Final mask overlay
    panel2 = original_img.copy()
    overlay = panel2.copy()
    overlay[final_mask > 0] = [0, 255, 0]  # Green
    panel2 = cv2.addWeighted(panel2, 0.7, overlay, 0.3, 0)
    cv2.putText(panel2, "Beautified Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    canvas[0:panel_h, panel_w:panel_w*2] = panel2

    # Panel 3: Difference map
    panel3 = np.ones((h, w, 3), dtype=np.uint8) * 255
    expansion = (final_mask > 0) & (original_mask == 0)
    reduction = (original_mask > 0) & (final_mask == 0)
    unchanged = (final_mask > 0) & (original_mask > 0)

    panel3[expansion] = [255, 165, 0]  # Orange for expansion
    panel3[reduction] = [255, 0, 0]    # Blue for reduction
    panel3[unchanged] = [200, 200, 200]  # Gray for unchanged

    cv2.putText(panel3, "Difference Map", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(panel3, "Orange=Expansion", (10, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    cv2.putText(panel3, "Blue=Reduction", (10, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    canvas[panel_h:panel_h*2, 0:panel_w] = panel3

    # Panel 4: Statistics
    panel4 = np.ones((h, w, 3), dtype=np.uint8) * 255

    val = eyebrow_data['validation']
    meta = eyebrow_data['metadata']
    side = eyebrow_data['side']

    y_pos = 50
    line_height = 40

    texts = [
        f"{side.upper()} EYEBROW STATISTICS",
        "",
        f"Original Area: {meta['yolo_area']} px",
        f"Final Area: {meta['final_area']} px",
        f"Expansion: {val['expansion_ratio']:.2f}x",
        "",
        f"MP Coverage: {val['mp_coverage']:.1f}%",
        f"Pass: {val['mp_coverage_pass']}",
        "",
        f"Eye Distance: {val['eye_distance_pct']:.2f}%",
        f"Pass: {val['eye_distance_pass']}",
        "",
        f"Aspect Ratio: {val['aspect_ratio']:.2f}",
        f"Pass: {val['aspect_ratio_pass']}",
        "",
        f"Eye Overlap: {val['eye_overlap']} px",
        f"Pass: {val['eye_overlap_pass']}",
        "",
        f"OVERALL: {'PASS' if val['overall_pass'] else 'FAIL'}"
    ]

    for i, text in enumerate(texts):
        color = (0, 150, 0) if "PASS" in text and "FAIL" not in text else (0, 0, 0)
        if "FAIL" in text:
            color = (0, 0, 255)
        if i == 0 or i == len(texts) - 1:
            font_scale = 0.8
            thickness = 2
        else:
            font_scale = 0.6
            thickness = 1

        cv2.putText(panel4, text, (20, y_pos + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    canvas[panel_h:panel_h*2, panel_w:panel_w*2] = panel4

    # Save
    cv2.imwrite(output_path, canvas)
    print(f"Saved visualization: {output_path}")


def test_visual_validation(test_image):
    """Generate visual comparison for a test image."""
    print("\n" + "="*70)
    print(f"VISUAL VALIDATION: {Path(test_image).name}")
    print("="*70)

    # Load original image
    original_img = cv2.imread(test_image)

    # Call beautify endpoint
    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/beautify", files=files)

    if response.status_code != 200:
        print(f"FAIL: Beautify request failed: {response.text}")
        return False

    data = response.json()

    if not data['success']:
        print(f"FAIL: Beautification failed")
        return False

    print(f"Processing Time: {data['processing_time_ms']:.1f}ms")
    print(f"Eyebrows Found: {len(data['eyebrows'])}")

    # Create visualizations for each eyebrow
    for i, eyebrow in enumerate(data['eyebrows']):
        # Decode masks
        original_mask = base64_to_mask(eyebrow['original_mask_base64'])
        final_mask = base64_to_mask(eyebrow['final_mask_base64'])

        # Generate visualization
        img_name = Path(test_image).stem
        output_name = f"{img_name}_{eyebrow['side']}_comparison.jpg"
        output_path = str(Path(VIZ_DIR) / output_name)

        create_comparison_visualization(
            original_img,
            original_mask,
            final_mask,
            eyebrow,
            output_path
        )

        print(f"  {eyebrow['side']} eyebrow: {output_name}")

    print("PASS - Visual validation complete")
    return True


def run_all_tests():
    """Run visual tests on all test images."""
    print("\n" + "="*70)
    print("VISUAL VALIDATION TESTS")
    print("="*70)

    # Create output directory
    Path(VIZ_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {VIZ_DIR}\n")

    results = []

    for test_image in TEST_IMAGES:
        if not Path(test_image).exists():
            print(f"SKIP: {test_image} not found")
            continue

        success = test_visual_validation(test_image)
        results.append((Path(test_image).name, success))

    # Summary
    print("\n" + "="*70)
    print("VISUAL TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Visualizations saved to: {VIZ_DIR}/")

    if passed == total:
        print("\nALL VISUAL TESTS PASSED")
        return True
    else:
        print(f"\n{total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys

    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API server")
        print(f"Make sure the server is running at {API_URL}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
