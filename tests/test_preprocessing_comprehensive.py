"""
Comprehensive Preprocessing Integration Test Suite

Tests all preprocessing components:
1. Core preprocessing (face validation, rotation detection, asymmetry)
2. Beautify integration (detection reuse, rotation correction)
3. API endpoint (POST /preprocess)
4. End-to-end pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yolo_pred
import beautify
import preprocess
import requests
import base64
import time


def image_to_base64(image_path):
    """Convert image to Base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


class TestResults:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()

    def add(self, name, passed, details=""):
        self.tests.append({'name': name, 'passed': passed, 'details': details})
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        elapsed = time.time() - self.start_time
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        for test in self.tests:
            status = "✓ PASS" if test['passed'] else "✗ FAIL"
            print(f"{status}: {test['name']}")
            if test['details']:
                print(f"       {test['details']}")

        print("\n" + "="*80)
        print(f"Results: {self.passed}/{len(self.tests)} passed, {self.failed} failed")
        print(f"Time: {elapsed:.2f}s")
        print("="*80)

        return self.failed == 0


def test_1_core_preprocessing(results, model):
    """Test 1: Core preprocessing functionality."""
    print("\n" + "="*80)
    print("TEST 1: Core Preprocessing (Face Validation, Rotation, Asymmetry)")
    print("="*80)

    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Test preprocessing
    result = preprocess.preprocess_face(test_image, model)

    # Check face validity
    if result['valid']:
        results.add("1.1 Face validation", True, f"Face valid: {result['valid']}")
    else:
        results.add("1.1 Face validation", False, f"Face rejected: {result['rejection_reason']}")
        return

    # Check rotation angle detection
    if result['rotation_angle'] is not None:
        results.add("1.2 Rotation angle detection", True, f"Angle: {result['rotation_angle']:.2f}°")
    else:
        results.add("1.2 Rotation angle detection", False, "No angle detected")

    # Check eye validation
    eye_val = result['eye_validation']
    if eye_val['is_valid']:
        results.add("1.3 Eye validation", True, f"Status: {eye_val['status']}")
    else:
        results.add("1.3 Eye validation", False, f"Status: {eye_val['status']}")

    # Check eyebrow validation
    eb_val = result['eyebrow_validation']
    if eb_val['is_valid']:
        results.add("1.4 Eyebrow validation", True, f"Status: {eb_val['status']}")
    else:
        results.add("1.4 Eyebrow validation", False, f"Status: {eb_val['status']}")

    # Check asymmetry detection runs (doesn't need to find asymmetry)
    asymm = result['asymmetry_detection']
    results.add("1.5 Asymmetry detection", True, f"Has asymmetry: {asymm['has_asymmetry']}")

    # Check boolean types (critical for API)
    mp_has_eyes = eye_val.get('mediapipe_has_eyes')
    yolo_has_eyes = eye_val.get('yolo_has_eyes')

    if isinstance(mp_has_eyes, bool) and isinstance(yolo_has_eyes, bool):
        results.add("1.6 Boolean type validation", True, "All booleans are proper type")
    else:
        results.add("1.6 Boolean type validation", False,
                   f"mp_has_eyes: {type(mp_has_eyes)}, yolo_has_eyes: {type(yolo_has_eyes)}")


def test_2_detection_reuse(results, model):
    """Test 2: Detection reuse optimization."""
    print("\n" + "="*80)
    print("TEST 2: Detection Reuse Optimization")
    print("="*80)

    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

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

    # Monkey patch
    yolo_pred.detect_yolo = counted_yolo
    import mediapipe_pred
    mediapipe_pred.detect_mediapipe = counted_mp

    # Run beautify with high threshold (no rotation)
    config = beautify.DEFAULT_CONFIG.copy()
    config['min_rotation_threshold'] = 10.0  # High threshold = no rotation

    beautify_results = beautify.beautify_eyebrows(test_image, model, config)

    # Restore
    yolo_pred.detect_yolo = original_yolo
    mediapipe_pred.detect_mediapipe = original_mp

    # Verify single call
    if yolo_calls['count'] == 1 and mp_calls['count'] == 1:
        results.add("2.1 Detection reuse (no rotation)", True,
                   f"YOLO: 1 call, MP: 1 call (optimal)")
    else:
        results.add("2.1 Detection reuse (no rotation)", False,
                   f"YOLO: {yolo_calls['count']} calls, MP: {mp_calls['count']} calls")

    # Test with rotation
    yolo_calls['count'] = 0
    mp_calls['count'] = 0

    yolo_pred.detect_yolo = counted_yolo
    mediapipe_pred.detect_mediapipe = counted_mp

    config['min_rotation_threshold'] = 0.5  # Low threshold = rotation applied

    beautify_results = beautify.beautify_eyebrows(test_image, model, config)

    yolo_pred.detect_yolo = original_yolo
    mediapipe_pred.detect_mediapipe = original_mp

    # Verify double call (before and after rotation)
    if yolo_calls['count'] == 2 and mp_calls['count'] == 2:
        results.add("2.2 Re-detection after rotation", True,
                   f"YOLO: 2 calls, MP: 2 calls (correct)")
    else:
        results.add("2.2 Re-detection after rotation", False,
                   f"YOLO: {yolo_calls['count']} calls, MP: {mp_calls['count']} calls")


def test_3_rotation_threshold(results, model):
    """Test 3: Rotation threshold logic."""
    print("\n" + "="*80)
    print("TEST 3: Rotation Threshold Logic")
    print("="*80)

    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Test passthrough (high threshold)
    config = beautify.DEFAULT_CONFIG.copy()
    config['min_rotation_threshold'] = 10.0

    beautify_results = beautify.beautify_eyebrows(test_image, model, config)

    if len(beautify_results) > 0:
        preprocessing = beautify_results[0].get('preprocessing', {})
        rotation_corrected = preprocessing.get('rotation_corrected', False)

        if not rotation_corrected:
            results.add("3.1 High threshold (passthrough)", True,
                       "Rotation not applied (correct)")
        else:
            results.add("3.1 High threshold (passthrough)", False,
                       "Rotation applied (should be passthrough)")
    else:
        results.add("3.1 High threshold (passthrough)", False, "No results returned")

    # Test correction (low threshold)
    config['min_rotation_threshold'] = 0.5

    beautify_results = beautify.beautify_eyebrows(test_image, model, config)

    if len(beautify_results) > 0:
        preprocessing = beautify_results[0].get('preprocessing', {})
        rotation_corrected = preprocessing.get('rotation_corrected', False)

        if rotation_corrected:
            results.add("3.2 Low threshold (correction)", True,
                       "Rotation applied (correct)")
        else:
            results.add("3.2 Low threshold (correction)", False,
                       "Rotation not applied (should be corrected)")
    else:
        results.add("3.2 Low threshold (correction)", False, "No results returned")


def test_4_api_endpoint(results):
    """Test 4: API endpoint."""
    print("\n" + "="*80)
    print("TEST 4: POST /preprocess API Endpoint")
    print("="*80)

    api_url = "http://localhost:8000"
    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Check API health
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code == 200:
            results.add("4.1 API health check", True, "API is healthy")
        else:
            results.add("4.1 API health check", False, f"Status: {health_response.status_code}")
            return
    except Exception as e:
        results.add("4.1 API health check", False, f"Error: {str(e)}")
        return

    # Test preprocessing endpoint
    try:
        image_b64 = image_to_base64(test_image)
        request_data = {"image_base64": image_b64, "config": None}

        response = requests.post(f"{api_url}/preprocess", json=request_data, timeout=30)

        if response.status_code == 200:
            results.add("4.2 POST /preprocess endpoint", True, "Request successful")

            result = response.json()

            # Validate response structure
            required_fields = ['success', 'valid', 'rotation_angle', 'eye_validation',
                             'eyebrow_validation', 'processing_time_ms', 'report']

            missing = [f for f in required_fields if f not in result]
            if not missing:
                results.add("4.3 Response structure", True, "All required fields present")
            else:
                results.add("4.3 Response structure", False, f"Missing: {', '.join(missing)}")

            # Check success
            if result.get('success'):
                results.add("4.4 Preprocessing success", True, f"Valid: {result.get('valid')}")
            else:
                results.add("4.4 Preprocessing success", False, "Preprocessing failed")

        else:
            results.add("4.2 POST /preprocess endpoint", False,
                       f"Status: {response.status_code}")
    except Exception as e:
        results.add("4.2 POST /preprocess endpoint", False, f"Error: {str(e)}")


def test_5_end_to_end(results, model):
    """Test 5: End-to-end pipeline."""
    print("\n" + "="*80)
    print("TEST 5: End-to-End Pipeline")
    print("="*80)

    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    # Run complete beautification pipeline
    config = beautify.DEFAULT_CONFIG.copy()
    config['enable_preprocessing'] = True
    config['auto_correct_rotation'] = True
    config['min_rotation_threshold'] = 1.0

    beautify_results = beautify.beautify_eyebrows(test_image, model, config)

    if len(beautify_results) > 0:
        results.add("5.1 Beautification with preprocessing", True,
                   f"Processed {len(beautify_results)} eyebrow(s)")

        # Check preprocessing data included
        first_result = beautify_results[0]
        if 'preprocessing' in first_result:
            results.add("5.2 Preprocessing data included", True,
                       "Preprocessing results attached")

            preprocessing = first_result['preprocessing']
            if 'rotation_angle' in preprocessing:
                results.add("5.3 Rotation data present", True,
                           f"Angle: {preprocessing['rotation_angle']:.2f}°")
            else:
                results.add("5.3 Rotation data present", False,
                           "No rotation data")
        else:
            results.add("5.2 Preprocessing data included", False,
                       "No preprocessing data")

        # Check validation passed
        validation = first_result.get('validation', {})
        if validation.get('overall_pass'):
            results.add("5.4 Validation passed", True, "All checks passed")
        else:
            results.add("5.4 Validation passed", False, "Validation failed")
    else:
        results.add("5.1 Beautification with preprocessing", False,
                   "No results returned")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE PREPROCESSING TEST SUITE")
    print("="*80)
    print("\nTesting all preprocessing components:")
    print("  1. Core preprocessing (validation, rotation, asymmetry)")
    print("  2. Detection reuse optimization")
    print("  3. Rotation threshold logic")
    print("  4. API endpoint")
    print("  5. End-to-end pipeline")
    print("\n" + "="*80)

    results = TestResults()

    # Load model once
    print("\nLoading YOLO model...")
    model = yolo_pred.load_yolo_model()
    print("✓ Model loaded\n")

    # Run all tests
    test_1_core_preprocessing(results, model)
    test_2_detection_reuse(results, model)
    test_3_rotation_threshold(results, model)
    test_4_api_endpoint(results)
    test_5_end_to_end(results, model)

    # Print summary
    success = results.print_summary()

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
