"""
Test Preprocessing Optimization

Verifies:
1. Models called exactly once when no rotation applied
2. Models called twice when rotation applied (once before, once after)
3. Rotation threshold works correctly (<1° = passthrough, >1° = correct)
4. Rotation angle detection is accurate
"""

import cv2
import numpy as np
import yolo_pred
import beautify
import os
from unittest.mock import patch


def create_rotated_image(input_path, output_path, angle_degrees):
    """
    Create a rotated version of an image.

    Args:
        input_path: Path to original image
        output_path: Path to save rotated image
        angle_degrees: Rotation angle in degrees
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotate image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite(output_path, rotated)
    print(f"  ✓ Created rotated image: {output_path} (angle: {angle_degrees}°)")
    return output_path


def test_model_call_count():
    """
    Test that models are called the correct number of times.
    """
    print("\n" + "="*80)
    print("TEST 1: Model Call Count Optimization")
    print("="*80)

    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Load model
    print("\nLoading YOLO model...")
    model = yolo_pred.load_yolo_model()
    print("✓ Model loaded\n")

    # Test case 1: No rotation (models should be called once in preprocessing, reused in beautify)
    print("\n--- Case 1: No Rotation Applied (Passthrough) ---")
    print("Testing with original image (should have minimal rotation)\n")

    # Patch detect_yolo and detect_mediapipe to count calls
    yolo_call_count = {'count': 0}
    mp_call_count = {'count': 0}

    original_yolo = yolo_pred.detect_yolo
    original_mp = __import__('mediapipe_pred').detect_mediapipe

    def counted_yolo(*args, **kwargs):
        yolo_call_count['count'] += 1
        print(f"  [YOLO CALL #{yolo_call_count['count']}]")
        return original_yolo(*args, **kwargs)

    def counted_mp(*args, **kwargs):
        mp_call_count['count'] += 1
        print(f"  [MediaPipe CALL #{mp_call_count['count']}]")
        return original_mp(*args, **kwargs)

    # Monkey patch
    yolo_pred.detect_yolo = counted_yolo
    import mediapipe_pred
    mediapipe_pred.detect_mediapipe = counted_mp

    # Run beautification with HIGH threshold to avoid rotation
    config = beautify.DEFAULT_CONFIG.copy()
    config['min_rotation_threshold'] = 5.0  # High threshold to avoid rotation on test image

    results = beautify.beautify_eyebrows(test_image, model, config)

    # Restore original functions
    yolo_pred.detect_yolo = original_yolo
    mediapipe_pred.detect_mediapipe = original_mp

    print(f"\n✓ Test completed")
    print(f"  YOLO calls: {yolo_call_count['count']}")
    print(f"  MediaPipe calls: {mp_call_count['count']}")

    # Expected: 1 call each (preprocessing only, beautify reuses)
    if yolo_call_count['count'] == 1 and mp_call_count['count'] == 1:
        print("  ✓ PASS: Models called exactly once (optimal!)")
    else:
        print(f"  ✗ FAIL: Expected 1 call each, got YOLO={yolo_call_count['count']}, MP={mp_call_count['count']}")

    return yolo_call_count['count'] == 1 and mp_call_count['count'] == 1


def test_rotation_threshold():
    """
    Test rotation threshold (passthrough vs correct) using threshold adjustment.
    """
    print("\n" + "="*80)
    print("TEST 2: Rotation Threshold (Passthrough vs Correction)")
    print("="*80)

    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Load model
    print("\nLoading YOLO model...")
    model = yolo_pred.load_yolo_model()
    print("✓ Model loaded\n")

    # Test cases: different threshold values to trigger passthrough vs correction
    test_cases = [
        {'threshold': 10.0, 'should_correct': False, 'name': 'High threshold (passthrough)'},
        {'threshold': 0.5, 'should_correct': True, 'name': 'Low threshold (correction)'},
    ]

    results = []

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} (threshold={test_case['threshold']}°) ---")

        config = beautify.DEFAULT_CONFIG.copy()
        config['min_rotation_threshold'] = test_case['threshold']

        # Count model calls
        yolo_calls = {'count': 0}
        mp_calls = {'count': 0}

        original_yolo = yolo_pred.detect_yolo
        original_mp = __import__('mediapipe_pred').detect_mediapipe

        def counted_yolo(*args, **kwargs):
            yolo_calls['count'] += 1
            return original_yolo(*args, **kwargs)

        def counted_mp(*args, **kwargs):
            mp_calls['count'] += 1
            return original_mp(*args, **kwargs)

        yolo_pred.detect_yolo = counted_yolo
        import mediapipe_pred as mp_pred
        mp_pred.detect_mediapipe = counted_mp

        # Run beautification
        beautify_results = beautify.beautify_eyebrows(test_image, model, config)

        # Restore
        yolo_pred.detect_yolo = original_yolo
        mp_pred.detect_mediapipe = original_mp

        # Check if rotation was applied based on model call count
        rotation_applied = (yolo_calls['count'] == 2)  # Called twice = rotation applied

        print(f"  Model calls: YOLO={yolo_calls['count']}, MP={mp_calls['count']}")
        print(f"  Rotation applied: {rotation_applied}")

        # Validate logic
        passed = (rotation_applied == test_case['should_correct'])

        if passed:
            if test_case['should_correct']:
                print(f"  ✓ PASS: Rotation corrected as expected")
            else:
                print(f"  ✓ PASS: Passthrough as expected")
        else:
            print(f"  ✗ FAIL: Expected {'correction' if test_case['should_correct'] else 'passthrough'}")

        results.append({
            'name': test_case['name'],
            'threshold': test_case['threshold'],
            'rotation_applied': rotation_applied,
            'should_correct': test_case['should_correct'],
            'passed': passed
        })

    # Summary
    print("\n" + "="*80)
    print("ROTATION THRESHOLD TEST SUMMARY")
    print("="*80)

    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        action = "corrected" if r['rotation_applied'] else "passthrough"
        print(f"{status}: {r['name']:30s} → {action:>11s}")

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    print(f"\nResult: {passed_count}/{total_count} tests passed")

    return passed_count == total_count


def test_angle_detection_accuracy():
    """
    Test that rotation angle detection is accurate.
    """
    print("\n" + "="*80)
    print("TEST 3: Rotation Angle Detection Accuracy")
    print("="*80)

    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Create test images with known rotation angles
    os.makedirs("./temp_rotated", exist_ok=True)

    test_angles = [2.0, 3.0, 5.0, 10.0, 15.0]

    print("\nCreating rotated test images...")
    rotated_images = []
    for angle in test_angles:
        output_path = f"./temp_rotated/rotated_{angle:.1f}deg.jpg"
        create_rotated_image(test_image, output_path, angle)
        rotated_images.append((angle, output_path))

    # Load model
    print("\nLoading YOLO model...")
    model = yolo_pred.load_yolo_model()
    print("✓ Model loaded\n")

    # Test angle detection
    import preprocess

    results = []
    for expected_angle, img_path in rotated_images:
        print(f"\n--- Testing {expected_angle}° rotation ---")

        # Run preprocessing only
        preprocess_config = preprocess.DEFAULT_PREPROCESS_CONFIG.copy()
        result = preprocess.preprocess_face(img_path, model, preprocess_config)

        detected_angle = result.get('rotation_angle')

        if detected_angle is not None:
            # Account for direction (negative angle means face rotated clockwise)
            error = abs(abs(detected_angle) - expected_angle)
            tolerance = 2.0  # ±2° tolerance

            passed = error <= tolerance

            print(f"  Expected: {expected_angle:.2f}°")
            print(f"  Detected: {detected_angle:.2f}°")
            print(f"  Error: {error:.2f}°")

            if passed:
                print(f"  ✓ PASS: Within {tolerance}° tolerance")
            else:
                print(f"  ✗ FAIL: Error exceeds {tolerance}° tolerance")

            results.append({
                'expected': expected_angle,
                'detected': detected_angle,
                'error': error,
                'passed': passed
            })
        else:
            print(f"  ✗ FAIL: No rotation angle detected")
            results.append({
                'expected': expected_angle,
                'detected': None,
                'error': None,
                'passed': False
            })

    # Clean up
    print("\nCleaning up temporary files...")
    for _, img_path in rotated_images:
        os.remove(img_path)
    os.rmdir("./temp_rotated")

    # Summary
    print("\n" + "="*80)
    print("ANGLE DETECTION ACCURACY SUMMARY")
    print("="*80)

    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        if r['detected'] is not None:
            print(f"{status}: Expected {r['expected']:>5.1f}°, Detected {r['detected']:>6.2f}°, Error {r['error']:>4.2f}°")
        else:
            print(f"{status}: Expected {r['expected']:>5.1f}°, No detection")

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    print(f"\nResult: {passed_count}/{total_count} tests passed")

    return passed_count == total_count


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PREPROCESSING OPTIMIZATION TEST SUITE")
    print("="*80)

    # Run all tests
    test1_passed = test_model_call_count()
    test2_passed = test_rotation_threshold()
    test3_passed = test_angle_detection_accuracy()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    tests = [
        ("Model Call Count Optimization", test1_passed),
        ("Rotation Threshold Logic", test2_passed),
        ("Angle Detection Accuracy", test3_passed)
    ]

    for name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, passed in tests if passed)
    total_count = len(tests)

    print(f"\nOverall: {passed_count}/{total_count} test suites passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠ SOME TESTS FAILED")
