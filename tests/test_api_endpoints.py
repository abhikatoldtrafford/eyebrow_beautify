"""
Comprehensive API endpoint testing with real HTTP calls

Tests all endpoints:
- GET /health
- GET /config
- POST /config
- POST /beautify
- POST /beautify/base64
- POST /detect/yolo
- POST /detect/mediapipe
- POST /beautify/submit-edit
- POST /generate/sd-beautify
"""

import requests
import json
import base64
from pathlib import Path
from test_config import API_URL, TEST_IMAGES


def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


class TestResults:
    """Track test results across all tests."""
    def __init__(self):
        self.results = []

    def add(self, name, passed):
        self.results.append((name, passed))

    def summary(self):
        passed = sum(1 for _, p in self.results if p)
        total = len(self.results)
        return passed, total


# Global test results
results = TestResults()


def test_root_endpoint():
    """Test GET / endpoint."""
    print("\n" + "="*70)
    print("TEST: GET / (Root/Info)")
    print("="*70)

    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Name: {data.get('name')}")
        print(f"Version: {data.get('version')}")
        print(f"Docs: {data.get('docs')}")

        passed = 'name' in data and 'version' in data
        results.add("GET /", passed)
        print("PASS" if passed else "FAIL")
        return passed
    else:
        results.add("GET /", False)
        print(f"FAIL: {response.text}")
        return False


def test_health_endpoint():
    """Test GET /health endpoint."""
    print("\n" + "="*70)
    print("TEST: GET /health")
    print("="*70)

    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Model Loaded: {data['model_loaded']}")
        print(f"MediaPipe: {data['mediapipe_available']}")

        passed = data['model_loaded'] == True and data['status'] == 'healthy'
        results.add("GET /health", passed)
        print("PASS" if passed else "FAIL")
        return passed
    else:
        results.add("GET /health", False)
        print(f"FAIL: {response.text}")
        return False


def test_get_config():
    """Test GET /config endpoint."""
    print("\n" + "="*70)
    print("TEST: GET /config")
    print("="*70)

    response = requests.get(f"{API_URL}/config")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"YOLO Confidence: {data.get('yolo_conf_threshold')}")
        print(f"MediaPipe Confidence: {data.get('mediapipe_conf_threshold')}")
        print(f"Min MP Coverage: {data.get('min_mp_coverage')}%")

        passed = 'yolo_conf_threshold' in data and 'min_mp_coverage' in data
        results.add("GET /config", passed)
        print("PASS" if passed else "FAIL")
        return passed
    else:
        results.add("GET /config", False)
        print(f"FAIL: {response.text}")
        return False


def test_post_config():
    """Test POST /config endpoint."""
    print("\n" + "="*70)
    print("TEST: POST /config")
    print("="*70)

    # Update config
    new_config = {
        "yolo_conf_threshold": 0.3,
        "mediapipe_conf_threshold": 0.6,
        "min_mp_coverage": 85.0
    }

    response = requests.post(
        f"{API_URL}/config",
        json=new_config
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Updated YOLO Conf: {data.get('yolo_conf_threshold')}")
        print(f"Updated MP Conf: {data.get('mediapipe_conf_threshold')}")
        print(f"Updated Min Coverage: {data.get('min_mp_coverage')}%")

        passed = data.get('yolo_conf_threshold') == 0.3
        results.add("POST /config", passed)
        print("PASS" if passed else "FAIL")

        # Reset to defaults
        requests.post(f"{API_URL}/config", json={
            "yolo_conf_threshold": 0.25,
            "mediapipe_conf_threshold": 0.5,
            "min_mp_coverage": 80.0
        })

        return passed
    else:
        results.add("POST /config", False)
        print(f"FAIL: {response.text}")
        return False


def test_yolo_detection(image_path):
    """Test POST /detect/yolo endpoint."""
    print("\n" + "="*70)
    print("TEST: POST /detect/yolo")
    print("="*70)

    if not Path(image_path).exists():
        print(f"FAIL: Test image not found: {image_path}")
        results.add("POST /detect/yolo", False)
        return False

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_URL}/detect/yolo?conf_threshold=0.25",
            files=files
        )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Processing Time: {data['processing_time_ms']:.1f}ms")

        detections = data['detections']
        for class_name, dets in detections.items():
            if len(dets) > 0:
                print(f"  {class_name}: {len(dets)} detection(s)")

        passed = data['success'] and len(detections.get('eyebrows', [])) >= 2
        results.add("POST /detect/yolo", passed)
        print("PASS - Found eyebrows" if passed else "FAIL - No eyebrows")
        return passed
    else:
        results.add("POST /detect/yolo", False)
        print(f"FAIL: {response.text}")
        return False


def test_mediapipe_detection(image_path):
    """Test POST /detect/mediapipe endpoint."""
    print("\n" + "="*70)
    print("TEST: POST /detect/mediapipe")
    print("="*70)

    if not Path(image_path).exists():
        print(f"FAIL: Test image not found: {image_path}")
        results.add("POST /detect/mediapipe", False)
        return False

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_URL}/detect/mediapipe?conf_threshold=0.5",
            files=files
        )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Message: {data['message']}")
        print(f"Processing Time: {data['processing_time_ms']:.1f}ms")

        if data['landmarks']:
            for feature, details in data['landmarks'].items():
                print(f"  {feature}: {len(details['points'])} points")

        passed = data['success'] and data['landmarks'] is not None
        results.add("POST /detect/mediapipe", passed)
        print("PASS - Found landmarks" if passed else "FAIL - No landmarks")
        return passed
    else:
        results.add("POST /detect/mediapipe", False)
        print(f"FAIL: {response.text}")
        return False


def test_beautify(image_path):
    """Test POST /beautify endpoint."""
    print("\n" + "="*70)
    print("TEST: POST /beautify")
    print("="*70)

    if not Path(image_path).exists():
        print(f"FAIL: Test image not found: {image_path}")
        results.add("POST /beautify", False)
        return False, None

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_URL}/beautify",
            files=files
        )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Message: {data['message']}")
        print(f"Eyebrows: {len(data['eyebrows'])}")
        print(f"Processing Time: {data['processing_time_ms']:.1f}ms")

        for eb in data['eyebrows']:
            print(f"\n  {eb['side']} eyebrow:")
            val = eb['validation']
            print(f"    MP Coverage: {val['mp_coverage']:.1f}% (pass: {val['mp_coverage_pass']})")
            print(f"    Eye Distance: {val['eye_distance_pct']:.2f}% (pass: {val['eye_distance_pass']})")
            print(f"    Aspect Ratio: {val['aspect_ratio']:.2f} (pass: {val['aspect_ratio_pass']})")
            print(f"    Eye Overlap: {val['eye_overlap']} px (pass: {val['eye_overlap_pass']})")
            print(f"    Expansion: {val['expansion_ratio']:.2f}x (pass: {val['expansion_ratio_pass']})")
            print(f"    Overall: {'PASS' if val['overall_pass'] else 'FAIL'}")

        passed = data['success'] and len(data['eyebrows']) >= 2
        results.add("POST /beautify", passed)
        print("\nPASS - Beautification successful" if passed else "FAIL")
        return passed, data
    else:
        results.add("POST /beautify", False)
        print(f"FAIL: {response.text}")
        return False, None


def test_beautify_base64(image_path):
    """Test POST /beautify/base64 endpoint."""
    print("\n" + "="*70)
    print("TEST: POST /beautify/base64")
    print("="*70)

    if not Path(image_path).exists():
        print(f"FAIL: Test image not found: {image_path}")
        results.add("POST /beautify/base64", False)
        return False

    # Encode image to base64
    img_b64 = encode_image_to_base64(image_path)

    request_data = {
        "image_base64": img_b64,
        "return_masks": True
    }

    response = requests.post(
        f"{API_URL}/beautify/base64",
        json=request_data
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Eyebrows: {len(data['eyebrows'])}")
        print(f"Processing Time: {data['processing_time_ms']:.1f}ms")

        passed = data['success'] and len(data['eyebrows']) >= 2
        results.add("POST /beautify/base64", passed)
        print("PASS" if passed else "FAIL")
        return passed
    else:
        results.add("POST /beautify/base64", False)
        print(f"FAIL: {response.text}")
        return False


def test_submit_edit(image_path, beautify_data):
    """Test POST /beautify/submit-edit endpoint."""
    print("\n" + "="*70)
    print("TEST: POST /beautify/submit-edit")
    print("="*70)

    if beautify_data is None or len(beautify_data['eyebrows']) == 0:
        print("SKIP: No beautify data available")
        results.add("POST /beautify/submit-edit", False)
        return False

    # Use the first eyebrow's mask as "edited" mask
    eyebrow = beautify_data['eyebrows'][0]
    img_b64 = encode_image_to_base64(image_path)

    request_data = {
        "image_base64": img_b64,
        "edited_mask_base64": eyebrow['final_mask_base64'],
        "side": eyebrow['side'],
        "metadata": {"test": "endpoint_test"}
    }

    response = requests.post(
        f"{API_URL}/beautify/submit-edit",
        json=request_data
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Message: {data['message']}")
        print(f"Side: {data['side']}")
        print(f"Mask Area: {data['mask_area']} pixels")

        passed = data['success']
        results.add("POST /beautify/submit-edit", passed)
        print("PASS" if passed else "FAIL")
        return passed
    else:
        results.add("POST /beautify/submit-edit", False)
        print(f"FAIL: {response.text}")
        return False


def test_sd_beautify(image_path, beautify_data):
    """Test POST /generate/sd-beautify endpoint."""
    print("\n" + "="*70)
    print("TEST: POST /generate/sd-beautify (Placeholder)")
    print("="*70)

    if beautify_data is None or len(beautify_data['eyebrows']) < 2:
        print("SKIP: No beautify data available")
        results.add("POST /generate/sd-beautify", False)
        return False

    img_b64 = encode_image_to_base64(image_path)

    # Find left and right eyebrows
    left_mask = None
    right_mask = None
    for eb in beautify_data['eyebrows']:
        if eb['side'] == 'left':
            left_mask = eb['final_mask_base64']
        elif eb['side'] == 'right':
            right_mask = eb['final_mask_base64']

    request_data = {
        "image_base64": img_b64,
        "left_eyebrow_mask_base64": left_mask,
        "right_eyebrow_mask_base64": right_mask,
        "prompt": "natural eyebrows",
        "seed": 42
    }

    response = requests.post(
        f"{API_URL}/generate/sd-beautify",
        json=request_data
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}  (Expected: False - placeholder)")
        print(f"Message: {data['message']}")
        print(f"Seed: {data['seed_used']}")
        print(f"Metadata: {data['metadata'].get('status')}")

        # This should return False (placeholder) but still work
        passed = response.status_code == 200 and data['metadata']['status'] == 'placeholder'
        results.add("POST /generate/sd-beautify", passed)
        print("PASS - Placeholder endpoint working" if passed else "FAIL")
        return passed
    else:
        results.add("POST /generate/sd-beautify", False)
        print(f"FAIL: {response.text}")
        return False


def run_all_tests():
    """Run all endpoint tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE API ENDPOINT TESTS")
    print("="*70)

    # Test basic endpoints
    test_root_endpoint()
    test_health_endpoint()
    test_get_config()
    test_post_config()

    # Use first test image
    test_image = TEST_IMAGES[0]

    # Test detection endpoints
    test_yolo_detection(test_image)
    test_mediapipe_detection(test_image)

    # Test beautification
    beautify_success, beautify_data = test_beautify(test_image)

    # Test base64 variant
    test_beautify_base64(test_image)

    # Test submit edit (requires beautify data)
    test_submit_edit(test_image, beautify_data)

    # Test SD endpoint (placeholder)
    test_sd_beautify(test_image, beautify_data)

    # Summary
    print("\n" + "="*70)
    print("ENDPOINT TEST SUMMARY")
    print("="*70)

    passed, total = results.summary()

    for test_name, result in results.results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL ENDPOINT TESTS PASSED")
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
        print("Start it with: ./start_api.sh")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
