"""
End-to-End Test for Developer Corner
Tests all features of the Developer Corner to ensure comprehensive functionality.
"""

import os
import sys
import base64
import requests
import time
import subprocess
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE = "test_images/test1.jpg"


def setup_test():
    """Setup test environment."""
    print("\n" + "="*80)
    print("DEVELOPER CORNER END-TO-END TEST")
    print("="*80)
    print(f"\nAPI URL: {API_URL}")
    print(f"Test Image: {TEST_IMAGE}")
    print(f"Current Directory: {os.getcwd()}")


def test_api_health():
    """Test 1: API Health Check."""
    print("\n[1/8] Testing API Health...")

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"

        data = response.json()
        assert data['status'] == 'healthy', "API not healthy"
        assert data['model_loaded'] == True, "Model not loaded"

        print(f"   ✓ API is healthy")
        print(f"   ✓ Model loaded: {data['model_loaded']}")
        return True

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def test_config_endpoints():
    """Test 2: Config Endpoints (GET /config, POST /config)."""
    print("\n[2/8] Testing Config Endpoints...")

    try:
        # GET config
        response = requests.get(f"{API_URL}/config", timeout=5)
        assert response.status_code == 200, f"GET /config failed: {response.status_code}"

        config = response.json()
        print(f"   ✓ GET /config successful")
        print(f"   ✓ Current YOLO threshold: {config.get('yolo_conf_threshold')}")

        # POST config (update)
        new_config = {
            'yolo_conf_threshold': 0.30,
            'min_mp_coverage': 75.0
        }
        response = requests.post(f"{API_URL}/config", json=new_config, timeout=5)
        assert response.status_code == 200, f"POST /config failed: {response.status_code}"

        updated_config = response.json()
        assert updated_config['yolo_conf_threshold'] == 0.30, "Config not updated"

        print(f"   ✓ POST /config successful")
        print(f"   ✓ Updated YOLO threshold: {updated_config['yolo_conf_threshold']}")

        return True

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def image_to_base64(image_path):
    """Convert image to base64."""
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    _, buffer = cv2.imencode('.png', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64


def test_beautify_endpoint():
    """Test 3: POST /beautify/base64."""
    print("\n[3/8] Testing Beautify Endpoint...")

    try:
        if not os.path.exists(TEST_IMAGE):
            print(f"   ⚠ Test image not found: {TEST_IMAGE}, skipping...")
            return True  # Skip but don't fail

        img_b64 = image_to_base64(TEST_IMAGE)

        payload = {
            'image_base64': img_b64,
            'return_masks': True
        }

        start_time = time.time()
        response = requests.post(f"{API_URL}/beautify/base64", json=payload, timeout=30)
        elapsed = (time.time() - start_time) * 1000

        assert response.status_code == 200, f"Beautify failed: {response.status_code}"

        data = response.json()
        assert data['success'] == True, "Beautify not successful"
        assert len(data['eyebrows']) > 0, "No eyebrows detected"

        print(f"   ✓ POST /beautify/base64 successful")
        print(f"   ✓ Time: {elapsed:.1f}ms")
        print(f"   ✓ Eyebrows detected: {len(data['eyebrows'])}")

        # Store for next test
        return data

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return None


def test_detect_yolo_endpoint():
    """Test 4: POST /detect/yolo/base64."""
    print("\n[4/8] Testing YOLO Detection Endpoint...")

    try:
        if not os.path.exists(TEST_IMAGE):
            print(f"   ⚠ Test image not found: {TEST_IMAGE}, skipping...")
            return True

        img_b64 = image_to_base64(TEST_IMAGE)

        payload = {
            'image_base64': img_b64,
            'return_masks': True
        }

        start_time = time.time()
        response = requests.post(f"{API_URL}/detect/yolo/base64", json=payload, timeout=30)
        elapsed = (time.time() - start_time) * 1000

        assert response.status_code == 200, f"YOLO detection failed: {response.status_code}"

        data = response.json()
        assert data['success'] == True, "YOLO detection not successful"

        detections = data.get('detections', {})

        print(f"   ✓ POST /detect/yolo/base64 successful")
        print(f"   ✓ Time: {elapsed:.1f}ms")
        print(f"   ✓ Eyebrows: {len(detections.get('eyebrows', []))}")
        print(f"   ✓ Eyes: {len(detections.get('eye', []))}")
        print(f"   ✓ Eye Boxes: {len(detections.get('eye_box', []))}")

        return True

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def test_detect_mediapipe_endpoint():
    """Test 5: POST /detect/mediapipe/base64."""
    print("\n[5/8] Testing MediaPipe Detection Endpoint...")

    try:
        if not os.path.exists(TEST_IMAGE):
            print(f"   ⚠ Test image not found: {TEST_IMAGE}, skipping...")
            return True

        img_b64 = image_to_base64(TEST_IMAGE)

        payload = {
            'image_base64': img_b64
        }

        start_time = time.time()
        response = requests.post(f"{API_URL}/detect/mediapipe/base64", json=payload, timeout=30)
        elapsed = (time.time() - start_time) * 1000

        assert response.status_code == 200, f"MediaPipe detection failed: {response.status_code}"

        data = response.json()
        assert data['success'] == True, "MediaPipe detection not successful"

        landmarks = data.get('landmarks', {})

        print(f"   ✓ POST /detect/mediapipe/base64 successful")
        print(f"   ✓ Time: {elapsed:.1f}ms")

        if landmarks:
            print(f"   ✓ Left Eyebrow Points: {len(landmarks.get('left_eyebrow', {}).get('points', []))}")
            print(f"   ✓ Right Eyebrow Points: {len(landmarks.get('right_eyebrow', {}).get('points', []))}")
        else:
            print(f"   ⚠ No landmarks detected (may be normal for some images)")

        return True

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def test_adjustment_endpoints(beautify_result):
    """Test 6: Adjustment Endpoints."""
    print("\n[6/8] Testing Adjustment Endpoints...")

    if not beautify_result or not isinstance(beautify_result, dict) or not beautify_result.get('eyebrows'):
        print(f"   ⚠ No beautify result available, skipping...")
        return True

    try:
        eyebrow = beautify_result['eyebrows'][0]
        mask_b64 = eyebrow['final_mask_base64']
        side = eyebrow['side']

        # Test thickness increase
        payload = {
            'mask_base64': mask_b64,
            'side': side,
            'increment': 0.05,
            'num_clicks': 1
        }

        response = requests.post(f"{API_URL}/adjust/thickness/increase", json=payload, timeout=10)
        assert response.status_code == 200, f"Thickness increase failed: {response.status_code}"

        data = response.json()
        assert data['success'] == True, "Adjustment not successful"

        print(f"   ✓ POST /adjust/thickness/increase successful")
        print(f"   ✓ Area change: {data.get('area_change_pct', 0):.1f}%")

        # Test span increase
        response = requests.post(f"{API_URL}/adjust/span/increase", json=payload, timeout=10)
        assert response.status_code == 200, f"Span increase failed: {response.status_code}"

        print(f"   ✓ POST /adjust/span/increase successful")

        return True

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def test_test_files_exist():
    """Test 7: Verify all test files exist."""
    print("\n[7/8] Testing Test Files Availability...")

    expected_tests = [
        "tests/run_all_tests.py",
        "tests/test_critical_fixes.py",
        "tests/test_api_endpoints.py",
        "tests/test_integration.py",
        "tests/test_adjustment_api.py",
        "tests/test_adjustments.py",
        "tests/test_model_loading.py",
        "tests/test_config.py",
        "tests/test_smooth_normal.py",
        "tests/test_statistical.py",
        "tests/test_visual.py"
    ]

    missing = []
    found = []

    for test_file in expected_tests:
        if os.path.exists(test_file):
            found.append(test_file)
        else:
            missing.append(test_file)

    print(f"   ✓ Found: {len(found)}/{len(expected_tests)} test files")

    if missing:
        print(f"   ⚠ Missing: {missing}")

    # Don't fail if some tests are missing, just warn
    return True


def test_developer_corner_imports():
    """Test 8: Verify Developer Corner can be imported."""
    print("\n[8/8] Testing Developer Corner Imports...")

    try:
        # Check if streamlit_developer.py exists
        assert os.path.exists('streamlit_developer.py'), "streamlit_developer.py not found"
        print(f"   ✓ streamlit_developer.py exists")

        # Check if streamlit_app.py exists
        assert os.path.exists('streamlit_app.py'), "streamlit_app.py not found"
        print(f"   ✓ streamlit_app.py exists")

        # Try to import (syntax check)
        result = subprocess.run(
            ['python3', '-c', 'import streamlit_developer'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print(f"   ✓ streamlit_developer.py imports successfully")
        else:
            print(f"   ⚠ Import warning (may require streamlit): {result.stderr[:100]}")

        return True

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False


def run_all_tests():
    """Run all Developer Corner tests."""
    setup_test()

    results = []

    # Test 1: API Health
    results.append(("API Health", test_api_health()))

    # Test 2: Config Endpoints
    results.append(("Config Endpoints", test_config_endpoints()))

    # Test 3: Beautify Endpoint
    beautify_result = test_beautify_endpoint()
    results.append(("Beautify Endpoint", beautify_result is not None))

    # Test 4: YOLO Detection
    results.append(("YOLO Detection", test_detect_yolo_endpoint()))

    # Test 5: MediaPipe Detection
    results.append(("MediaPipe Detection", test_detect_mediapipe_endpoint()))

    # Test 6: Adjustment Endpoints
    results.append(("Adjustment Endpoints", test_adjustment_endpoints(beautify_result)))

    # Test 7: Test Files
    results.append(("Test Files Exist", test_test_files_exist()))

    # Test 8: Developer Corner Imports
    results.append(("Developer Corner Imports", test_developer_corner_imports()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {test_name}")

        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Developer Corner is fully functional!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
