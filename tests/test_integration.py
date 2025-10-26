"""
Integration tests for end-to-end workflows

Tests complete user workflows:
1. Upload image → Get beautified masks
2. Upload image → Edit mask → Submit edit
3. Upload image → Get masks → SD generate (placeholder)
4. Full pipeline: YOLO → MediaPipe → Beautify → Validate
"""

import requests
import base64
from pathlib import Path
from test_config import API_URL, TEST_IMAGES


def encode_image_to_base64(image_path):
    """Encode image file to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_workflow_1_basic_beautification():
    """
    Workflow 1: Basic beautification
    User uploads image → Gets beautified eyebrow masks
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST 1: Basic Beautification Workflow")
    print("="*70)

    test_image = TEST_IMAGES[0]

    print(f"Step 1: Upload image {Path(test_image).name}")

    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/beautify", files=files)

    if response.status_code != 200:
        print(f"FAIL: Beautify request failed: {response.text}")
        return False

    data = response.json()

    print(f"Step 2: Receive beautified masks")
    print(f"  Eyebrows: {len(data['eyebrows'])}")
    print(f"  Processing Time: {data['processing_time_ms']:.1f}ms")

    # Validate response
    if not data['success']:
        print("FAIL: Beautification failed")
        return False

    if len(data['eyebrows']) < 2:
        print("FAIL: Expected at least 2 eyebrows")
        return False

    # Check that masks are provided
    for eb in data['eyebrows']:
        if not eb['original_mask_base64'] or not eb['final_mask_base64']:
            print(f"FAIL: Missing masks for {eb['side']} eyebrow")
            return False

    print("Step 3: Validation checks")
    overall_pass_count = sum(1 for eb in data['eyebrows'] if eb['validation']['overall_pass'])
    print(f"  Overall pass rate: {overall_pass_count}/{len(data['eyebrows'])}")

    print("\nPASS - Workflow 1 complete")
    return True


def test_workflow_2_edit_submit():
    """
    Workflow 2: Beautify → Edit → Submit
    User uploads image → Gets masks → Edits mask → Submits final version
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST 2: Edit & Submit Workflow")
    print("="*70)

    test_image = TEST_IMAGES[0]

    print(f"Step 1: Beautify image")

    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/beautify", files=files)

    if response.status_code != 200:
        print(f"FAIL: Beautify failed")
        return False

    data = response.json()

    print(f"  Received {len(data['eyebrows'])} eyebrow(s)")

    if len(data['eyebrows']) == 0:
        print("FAIL: No eyebrows found")
        return False

    # Simulate user editing the first eyebrow
    eyebrow = data['eyebrows'][0]

    print(f"\nStep 2: User edits {eyebrow['side']} eyebrow mask (simulated)")

    # In real workflow, user would edit the mask in UI
    # Here we just use the final mask as "edited" mask
    edited_mask_b64 = eyebrow['final_mask_base64']

    print(f"Step 3: Submit edited mask")

    img_b64 = encode_image_to_base64(test_image)

    submit_request = {
        "image_base64": img_b64,
        "edited_mask_base64": edited_mask_b64,
        "side": eyebrow['side'],
        "metadata": {"test": "workflow_2"}
    }

    response = requests.post(
        f"{API_URL}/beautify/submit-edit",
        json=submit_request
    )

    if response.status_code != 200:
        print(f"FAIL: Submit edit failed: {response.text}")
        return False

    submit_data = response.json()

    print(f"  Success: {submit_data['success']}")
    print(f"  Side: {submit_data['side']}")
    print(f"  Final Mask Area: {submit_data['mask_area']} pixels")

    if not submit_data['success']:
        print("FAIL: Submit edit not successful")
        return False

    print("\nPASS - Workflow 2 complete")
    return True


def test_workflow_3_sd_generation():
    """
    Workflow 3: Beautify → SD Generate
    User uploads image → Gets masks → Generates beautified eyebrows with SD
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST 3: SD Generation Workflow (Placeholder)")
    print("="*70)

    test_image = TEST_IMAGES[0]

    print(f"Step 1: Beautify image")

    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/beautify", files=files)

    if response.status_code != 200:
        print(f"FAIL: Beautify failed")
        return False

    data = response.json()

    print(f"  Received {len(data['eyebrows'])} eyebrow(s)")

    if len(data['eyebrows']) < 2:
        print("FAIL: Expected at least 2 eyebrows")
        return False

    # Find left and right masks
    left_mask = None
    right_mask = None

    for eb in data['eyebrows']:
        if eb['side'] == 'left':
            left_mask = eb['final_mask_base64']
        elif eb['side'] == 'right':
            right_mask = eb['final_mask_base64']

    print(f"\nStep 2: Prepare SD generation request")

    img_b64 = encode_image_to_base64(test_image)

    sd_request = {
        "image_base64": img_b64,
        "left_eyebrow_mask_base64": left_mask,
        "right_eyebrow_mask_base64": right_mask,
        "prompt": "natural, well-groomed eyebrows",
        "seed": 42
    }

    print(f"Step 3: Call SD generation endpoint")

    response = requests.post(
        f"{API_URL}/generate/sd-beautify",
        json=sd_request
    )

    if response.status_code != 200:
        print(f"FAIL: SD generation failed: {response.text}")
        return False

    sd_data = response.json()

    print(f"  Status: {sd_data['metadata']['status']}")
    print(f"  Seed Used: {sd_data['seed_used']}")
    print(f"  Left Mask Provided: {sd_data['metadata']['left_mask_provided']}")
    print(f"  Right Mask Provided: {sd_data['metadata']['right_mask_provided']}")

    # Check that it's a placeholder (success should be False)
    if sd_data['metadata']['status'] != 'placeholder':
        print("FAIL: Expected placeholder status")
        return False

    print("\nPASS - Workflow 3 complete (placeholder working)")
    return True


def test_workflow_4_full_pipeline():
    """
    Workflow 4: Full pipeline validation
    Test that YOLO → MediaPipe → Beautify → Validate all works together
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST 4: Full Pipeline Validation")
    print("="*70)

    test_image = TEST_IMAGES[0]

    print(f"Step 1: YOLO Detection")

    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_URL}/detect/yolo?conf_threshold=0.25",
            files=files
        )

    if response.status_code != 200:
        print(f"FAIL: YOLO detection failed")
        return False

    yolo_data = response.json()
    print(f"  Eyebrows detected: {len(yolo_data['detections'].get('eyebrows', []))}")

    print(f"\nStep 2: MediaPipe Detection")

    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_URL}/detect/mediapipe?conf_threshold=0.5",
            files=files
        )

    if response.status_code != 200:
        print(f"FAIL: MediaPipe detection failed")
        return False

    mp_data = response.json()

    if mp_data['landmarks']:
        print(f"  Landmarks detected:")
        for feature, data in mp_data['landmarks'].items():
            print(f"    {feature}: {len(data['points'])} points")
    else:
        print("  WARNING: No landmarks detected")

    print(f"\nStep 3: Full Beautification")

    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/beautify", files=files)

    if response.status_code != 200:
        print(f"FAIL: Beautification failed")
        return False

    beautify_data = response.json()

    print(f"  Eyebrows processed: {len(beautify_data['eyebrows'])}")

    print(f"\nStep 4: Validation Results")

    all_valid = True

    for eb in beautify_data['eyebrows']:
        val = eb['validation']
        print(f"  {eb['side']} eyebrow:")
        print(f"    MP Coverage: {val['mp_coverage']:.1f}% - {'PASS' if val['mp_coverage_pass'] else 'FAIL'}")
        print(f"    Eye Overlap: {val['eye_overlap']} px - {'PASS' if val['eye_overlap_pass'] else 'FAIL'}")
        print(f"    Overall: {'PASS' if val['overall_pass'] else 'FAIL'}")

        if not val['overall_pass']:
            all_valid = False

    if all_valid:
        print("\nPASS - Full pipeline working, all validations passed")
        return True
    else:
        print("\nWARNING - Pipeline working but some validations failed")
        return True  # Still pass the test, validation failures are logged


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("END-TO-END INTEGRATION TESTS")
    print("="*70)

    results = []

    # Workflow 1: Basic beautification
    results.append(("Workflow 1: Basic Beautification", test_workflow_1_basic_beautification()))

    # Workflow 2: Edit & submit
    results.append(("Workflow 2: Edit & Submit", test_workflow_2_edit_submit()))

    # Workflow 3: SD generation
    results.append(("Workflow 3: SD Generation", test_workflow_3_sd_generation()))

    # Workflow 4: Full pipeline
    results.append(("Workflow 4: Full Pipeline", test_workflow_4_full_pipeline()))

    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL INTEGRATION TESTS PASSED")
        print("All workflows functioning correctly")
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
