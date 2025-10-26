"""
Test API Preprocessing Endpoint

Tests the POST /preprocess endpoint with Base64 encoded images.
"""

import requests
import base64
import cv2
import json


def image_to_base64(image_path):
    """Convert image file to Base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_preprocess_endpoint():
    """Test the POST /preprocess endpoint."""
    print("="*80)
    print("TESTING POST /preprocess ENDPOINT")
    print("="*80)

    # API URL
    api_url = "http://localhost:8000"
    endpoint = f"{api_url}/preprocess"

    # Test image
    test_image = "./annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    print(f"\n1. Loading test image: {test_image}")
    try:
        img = cv2.imread(test_image)
        if img is None:
            print("✗ Failed to load image")
            return False
        print(f"   ✓ Image loaded: {img.shape[1]}x{img.shape[0]}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return False

    print("\n2. Converting to Base64...")
    try:
        image_b64 = image_to_base64(test_image)
        print(f"   ✓ Base64 encoded ({len(image_b64)} characters)")
    except Exception as e:
        print(f"✗ Error encoding image: {e}")
        return False

    print("\n3. Sending POST request to /preprocess...")
    try:
        # Prepare request
        request_data = {
            "image_base64": image_b64,
            "config": None  # Use default config
        }

        # Send request
        response = requests.post(endpoint, json=request_data, timeout=30)

        print(f"   Status code: {response.status_code}")

        if response.status_code != 200:
            print(f"   ✗ Request failed: {response.text}")
            return False

        print("   ✓ Request successful")

    except Exception as e:
        print(f"✗ Error sending request: {e}")
        print("\n⚠ Make sure API is running:")
        print("  ./start_api.sh")
        print("  OR")
        print("  uvicorn api.api_main:app --reload --host 0.0.0.0 --port 8000")
        return False

    print("\n4. Parsing response...")
    try:
        result = response.json()
        print("   ✓ Response parsed")
    except Exception as e:
        print(f"✗ Error parsing response: {e}")
        return False

    print("\n5. Validating response structure...")
    required_fields = [
        'success', 'valid', 'image_shape', 'rotation_angle',
        'eye_validation', 'eyebrow_validation', 'quality_validation',
        'angle_metadata', 'asymmetry_detection', 'processing_time_ms', 'report'
    ]

    missing = []
    for field in required_fields:
        if field not in result:
            missing.append(field)

    if missing:
        print(f"   ✗ Missing fields: {', '.join(missing)}")
        return False

    print("   ✓ All required fields present")

    print("\n6. Preprocessing Results:")
    print("   " + "-"*76)
    print(f"   Success: {result['success']}")
    print(f"   Face Valid: {result['valid']}")
    print(f"   Image Shape: {result['image_shape']}")
    print(f"   Rotation Angle: {result['rotation_angle']}°" if result['rotation_angle'] else "   Rotation Angle: None")
    print(f"   Eye Validation: {result['eye_validation']['status']}")
    print(f"   Eyebrow Validation: {result['eyebrow_validation']['status']}")
    print(f"   Quality Validation: {result['quality_validation']['status']}")
    print(f"   Asymmetry Detected: {result['asymmetry_detection']['has_asymmetry']}")
    print(f"   Warnings: {len(result['warnings'])}")
    print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")

    print("\n7. Angle Metadata:")
    angle_meta = result['angle_metadata']
    print(f"   Final Angle: {angle_meta['final_angle']}°" if angle_meta['final_angle'] else "   Final Angle: None")
    print(f"   Status: {angle_meta['status']}")
    print(f"   Sources Used: {angle_meta['num_sources']}")
    if angle_meta['all_sources']:
        print(f"   All Sources: {', '.join(angle_meta['all_sources'])}")
    print(f"   Outliers Removed: {angle_meta['outliers_removed']}")

    print("\n8. Full Preprocessing Report:")
    print("   " + "-"*76)
    # Print report with indentation
    for line in result['report'].split('\n'):
        print(f"   {line}")

    print("\n" + "="*80)
    print("✓ TEST PASSED: Preprocessing endpoint working correctly!")
    print("="*80)

    return True


if __name__ == "__main__":
    success = test_preprocess_endpoint()

    if not success:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        exit(1)
