"""
Test adjustment API endpoints with REAL API calls.
"""

import requests
import json
import base64
from pathlib import Path

API_URL = "http://localhost:8000"

def test_adjustment_endpoints():
    """Test all 4 adjustment endpoints with real API calls."""

    print("="*70)
    print("TESTING ADJUSTMENT API ENDPOINTS - REAL API CALLS")
    print("="*70)

    # Step 1: Get eyebrow masks from beautify endpoint
    print("\n[Step 1] Getting eyebrow masks from beautify endpoint...")
    test_image = "annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg"

    with open(test_image, 'rb') as f:
        response = requests.post(
            f"{API_URL}/beautify",
            files={'file': f}
        )

    if response.status_code != 200:
        print(f"✗ Beautify failed: {response.text}")
        return False

    data = response.json()
    print(f"✓ Got {len(data['eyebrows'])} eyebrow masks")

    # Get left eyebrow mask
    left_eyebrow = [eb for eb in data['eyebrows'] if eb['side'] == 'left'][0]
    left_mask_base64 = left_eyebrow['final_mask_base64']
    left_original_area = left_eyebrow['metadata']['final_area']

    print(f"✓ Left eyebrow: {left_original_area} pixels")
    print(f"✓ Mask length: {len(left_mask_base64)} chars")

    # Test all 4 adjustment endpoints
    adjustments = [
        {
            'name': 'Increase Thickness',
            'endpoint': '/adjust/thickness/increase',
            'expected_change': 'positive'
        },
        {
            'name': 'Decrease Thickness',
            'endpoint': '/adjust/thickness/decrease',
            'expected_change': 'negative'
        },
        {
            'name': 'Increase Span (Tail Only)',
            'endpoint': '/adjust/span/increase',
            'expected_change': 'positive'
        },
        {
            'name': 'Decrease Span (Tail Only)',
            'endpoint': '/adjust/span/decrease',
            'expected_change': 'negative'
        }
    ]

    results = []

    for adj in adjustments:
        print(f"\n{'='*70}")
        print(f"[Test] {adj['name']}")
        print(f"{'='*70}")

        # Prepare request
        payload = {
            'mask_base64': left_mask_base64,
            'side': 'left',
            'increment': 0.05,  # 5%
            'num_clicks': 1
        }

        # Make API call
        print(f"POST {adj['endpoint']}")
        print(f"  - side: left")
        print(f"  - increment: 5%")
        print(f"  - num_clicks: 1")

        response = requests.post(
            f"{API_URL}{adj['endpoint']}",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        print(f"  - Status: {response.status_code}")

        if response.status_code != 200:
            print(f"✗ Request failed: {response.text[:200]}")
            results.append(False)
            continue

        result = response.json()

        # Verify response
        print(f"\nResponse:")
        print(f"  ✓ Success: {result['success']}")
        print(f"  ✓ Message: {result['message']}")
        print(f"  ✓ Adjustment Type: {result['adjustment_type']}")
        print(f"  ✓ Direction: {result['direction']}")
        print(f"  ✓ Increment Applied: {result['increment_applied']*100}%")
        print(f"  ✓ Original Area: {result['original_area']} pixels")
        print(f"  ✓ Adjusted Area: {result['adjusted_area']} pixels")
        print(f"  ✓ Area Change: {result['area_change_pct']:+.1f}%")
        print(f"  ✓ Processing Time: {result['processing_time_ms']:.1f}ms")
        print(f"  ✓ Adjusted Mask Length: {len(result['adjusted_mask_base64'])} chars")

        # Validate
        area_change = result['area_change_pct']
        if adj['expected_change'] == 'positive' and area_change > 0:
            print(f"\n  ✅ PASS - Area increased as expected ({area_change:+.1f}%)")
            results.append(True)
        elif adj['expected_change'] == 'negative' and area_change < 0:
            print(f"\n  ✅ PASS - Area decreased as expected ({area_change:+.1f}%)")
            results.append(True)
        else:
            print(f"\n  ⚠️ WARNING - Unexpected area change: {area_change:+.1f}%")
            results.append(False)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passed = sum(results)
    total = len(results)

    for adj, result in zip(adjustments, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{adj['name']}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL ADJUSTMENT ENDPOINT TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys

    print("Eyebrow Adjustment API Test")
    print(f"API URL: {API_URL}")
    print()

    try:
        success = test_adjustment_endpoints()
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API server")
        print("Make sure the server is running at", API_URL)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
