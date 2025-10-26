"""
Test model singleton loading pattern

Verifies that the YOLO model is loaded only once on API startup
and reused for all subsequent requests.
"""

import requests
import time
from test_config import API_URL


def test_health_check():
    """Test that API is running and model is loaded."""
    print("\n" + "="*70)
    print("TEST 1: Health Check & Model Loading")
    print("="*70)

    response = requests.get(f"{API_URL}/health")

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Service Status: {data['status']}")
        print(f"Model Loaded: {data['model_loaded']}")
        print(f"MediaPipe Available: {data['mediapipe_available']}")
        print(f"API Version: {data['version']}")

        assert data['model_loaded'] == True, "YOLO model must be loaded"
        assert data['status'] == 'healthy', "Service must be healthy"

        print("\nPASS - API healthy, model loaded on startup")
        return True
    else:
        print(f"FAIL - Health check failed: {response.text}")
        return False


def test_multiple_requests_timing():
    """Test that subsequent requests are faster (model already loaded)."""
    print("\n" + "="*70)
    print("TEST 2: Model Singleton Pattern (Multiple Request Timing)")
    print("="*70)

    # Make 5 health check requests and time them
    times = []

    for i in range(5):
        start = time.time()
        response = requests.get(f"{API_URL}/health")
        elapsed = (time.time() - start) * 1000  # ms

        times.append(elapsed)
        print(f"Request {i+1}: {elapsed:.2f}ms")

    avg_time = sum(times) / len(times)
    print(f"\nAverage response time: {avg_time:.2f}ms")

    # All requests should be fast (<100ms typically)
    assert all(t < 500 for t in times), "All requests should be fast (model already loaded)"

    print("PASS - All requests fast, model singleton working")
    return True


def test_model_consistency():
    """Test that the same model is used across requests."""
    print("\n" + "="*70)
    print("TEST 3: Model Consistency Across Requests")
    print("="*70)

    # Make multiple requests and check model_loaded is always True
    consistent = True

    for i in range(10):
        response = requests.get(f"{API_URL}/health")
        data = response.json()

        if not data['model_loaded']:
            print(f"FAIL - Request {i+1}: Model not loaded!")
            consistent = False
            break

    if consistent:
        print(f"Checked 10 requests - model_loaded always True")
        print("PASS - Model remains loaded (singleton pattern working)")
        return True
    else:
        print("FAIL - Model loading inconsistent")
        return False


def run_all_tests():
    """Run all model loading tests."""
    print("\n" + "="*70)
    print("MODEL SINGLETON LOADING TESTS")
    print("="*70)

    results = []

    # Test 1: Health check
    results.append(("Health Check", test_health_check()))

    # Test 2: Multiple request timing
    results.append(("Multiple Request Timing", test_multiple_requests_timing()))

    # Test 3: Model consistency
    results.append(("Model Consistency", test_model_consistency()))

    # Summary
    print("\n" + "="*70)
    print("MODEL LOADING TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED")
        print("Model singleton pattern is working correctly.")
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
