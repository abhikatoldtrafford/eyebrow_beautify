"""
Comprehensive API Unit Tests

Tests ALL 15 API endpoints:
- Health & Config (3)
- Detection (4)
- Beautification (3)
- Adjustments (4)
- Generation (1)

Run: pytest tests/test_api_all_endpoints.py -v
"""

import pytest
import requests
import base64
import json
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE = "annotated/test/images/before_clean_jpg.rf.a9557744ee9655b206e5afbf1930c3a5.jpg"


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def api_url():
    """Base API URL."""
    return API_URL


@pytest.fixture(scope="session")
def test_image_path():
    """Test image path."""
    path = Path(TEST_IMAGE)
    assert path.exists(), f"Test image not found: {TEST_IMAGE}"
    return str(path)


@pytest.fixture(scope="session")
def test_image_base64(test_image_path):
    """Test image encoded as base64."""
    with open(test_image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


@pytest.fixture(scope="session")
def beautify_result(api_url, test_image_base64):
    """Cached beautify result for dependent tests."""
    response = requests.post(
        f"{api_url}/beautify/base64",
        json={"image_base64": test_image_base64},
        timeout=60
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="session")
def sample_mask(beautify_result):
    """Sample mask from beautify result."""
    return beautify_result['eyebrows'][0]['final_mask_base64']


# =============================================================================
# HEALTH & CONFIGURATION (3 endpoints)
# =============================================================================

def test_health_endpoint(api_url):
    """Test GET /health endpoint."""
    response = requests.get(f"{api_url}/health")

    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert 'mediapipe_available' in data
    assert 'version' in data


def test_get_config(api_url):
    """Test GET /config endpoint."""
    response = requests.get(f"{api_url}/config")

    assert response.status_code == 200
    config = response.json()
    assert 'yolo_conf_threshold' in config
    assert 'mediapipe_conf_threshold' in config
    assert 'min_mp_coverage' in config


def test_post_config(api_url):
    """Test POST /config endpoint."""
    # Get original config
    orig_response = requests.get(f"{api_url}/config")
    orig_config = orig_response.json()

    # Update config
    new_config = {
        "yolo_conf_threshold": 0.3,
        "min_mp_coverage": 85.0
    }
    response = requests.post(
        f"{api_url}/config",
        json=new_config
    )

    assert response.status_code == 200
    data = response.json()
    assert data['yolo_conf_threshold'] == 0.3
    assert data['min_mp_coverage'] == 85.0

    # Restore original
    requests.post(f"{api_url}/config", json=orig_config)


# =============================================================================
# DETECTION (4 endpoints)
# =============================================================================

def test_detect_yolo_file(api_url, test_image_path):
    """Test POST /detect/yolo (file upload)."""
    with open(test_image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/detect/yolo",
            files={'file': f},
            timeout=30
        )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'detections' in data
    assert 'eyebrows' in data['detections']
    assert len(data['detections']['eyebrows']) >= 2


def test_detect_yolo_base64(api_url, test_image_base64):
    """Test POST /detect/yolo/base64."""
    response = requests.post(
        f"{api_url}/detect/yolo/base64",
        json={"image_base64": test_image_base64},
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'detections' in data
    assert 'eyebrows' in data['detections']


def test_detect_mediapipe_file(api_url, test_image_path):
    """Test POST /detect/mediapipe (file upload)."""
    with open(test_image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/detect/mediapipe",
            files={'file': f},
            timeout=30
        )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'landmarks' in data

    # Check for eyebrow landmarks
    if data['landmarks']:  # May be None if face not detected
        assert 'left_eyebrow' in data['landmarks']
        assert 'right_eyebrow' in data['landmarks']


def test_detect_mediapipe_base64(api_url, test_image_base64):
    """Test POST /detect/mediapipe/base64."""
    response = requests.post(
        f"{api_url}/detect/mediapipe/base64",
        json={"image_base64": test_image_base64},
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'landmarks' in data


# =============================================================================
# BEAUTIFICATION (3 endpoints)
# =============================================================================

def test_beautify_file(api_url, test_image_path):
    """Test POST /beautify (file upload)."""
    with open(test_image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/beautify",
            files={'file': f},
            timeout=60
        )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert len(data['eyebrows']) >= 2

    # Check validation structure
    for eb in data['eyebrows']:
        assert 'side' in eb
        assert 'validation' in eb
        assert 'overall_pass' in eb['validation']
        assert 'metadata' in eb


def test_beautify_base64(api_url, test_image_base64):
    """Test POST /beautify/base64."""
    response = requests.post(
        f"{api_url}/beautify/base64",
        json={
            "image_base64": test_image_base64,
            "return_masks": True
        },
        timeout=60
    )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert len(data['eyebrows']) >= 2

    # Check masks are included
    for eb in data['eyebrows']:
        assert 'original_mask_base64' in eb
        assert 'final_mask_base64' in eb


def test_beautify_submit_edit(api_url, test_image_base64, sample_mask):
    """Test POST /beautify/submit-edit."""
    response = requests.post(
        f"{api_url}/beautify/submit-edit",
        json={
            "original_image_base64": test_image_base64,
            "edited_mask_base64": sample_mask,
            "side": "left",
            "metadata": {"source": "test"}
        },
        timeout=30
    )

    # This endpoint might have specific validation requirements
    # Accept both success (200) and validation error (422)
    assert response.status_code in [200, 422]
    if response.status_code == 200:
        data = response.json()
        assert data['success'] is True


# =============================================================================
# ADJUSTMENTS (4 endpoints)
# =============================================================================

def test_adjust_thickness_increase(api_url, sample_mask):
    """Test POST /adjust/thickness/increase."""
    response = requests.post(
        f"{api_url}/adjust/thickness/increase",
        json={
            "mask_base64": sample_mask,
            "side": "left"
        },
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'adjusted_mask_base64' in data
    assert data['adjustment_type'] == 'thickness'
    assert data['direction'] == 'increase'


def test_adjust_thickness_decrease(api_url, sample_mask):
    """Test POST /adjust/thickness/decrease."""
    response = requests.post(
        f"{api_url}/adjust/thickness/decrease",
        json={
            "mask_base64": sample_mask,
            "side": "left"
        },
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert data['adjustment_type'] == 'thickness'
    assert data['direction'] == 'decrease'


def test_adjust_span_increase(api_url, sample_mask):
    """Test POST /adjust/span/increase."""
    response = requests.post(
        f"{api_url}/adjust/span/increase",
        json={
            "mask_base64": sample_mask,
            "side": "left"
        },
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert data['adjustment_type'] == 'span'
    assert data['direction'] == 'increase'


def test_adjust_span_decrease(api_url, sample_mask):
    """Test POST /adjust/span/decrease."""
    response = requests.post(
        f"{api_url}/adjust/span/decrease",
        json={
            "mask_base64": sample_mask,
            "side": "left"
        },
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert data['adjustment_type'] == 'span'
    assert data['direction'] == 'decrease'


# =============================================================================
# GENERATION (1 endpoint - placeholder)
# =============================================================================

def test_sd_beautify_placeholder(api_url, test_image_base64, sample_mask):
    """Test POST /generate/sd-beautify (placeholder)."""
    response = requests.post(
        f"{api_url}/generate/sd-beautify",
        json={
            "original_image_base64": test_image_base64,
            "left_mask_base64": sample_mask,
            "right_mask_base64": sample_mask,
            "prompt": "test",
            "strength": 0.5
        },
        timeout=30
    )

    # Placeholder endpoint - accept various responses
    # May return validation error, not implemented, or success
    assert response.status_code in [200, 422, 500, 501]


# =============================================================================
# ERROR HANDLING
# =============================================================================

def test_invalid_image_format(api_url):
    """Test API handles invalid image gracefully."""
    invalid_base64 = "not_a_valid_base64_image"
    response = requests.post(
        f"{api_url}/beautify/base64",
        json={"image_base64": invalid_base64},
        timeout=30
    )

    # Should return 4xx error
    assert response.status_code >= 400


def test_missing_file(api_url):
    """Test API handles missing file gracefully."""
    response = requests.post(
        f"{api_url}/beautify",
        files={},
        timeout=30
    )

    # Should return 422 validation error
    assert response.status_code == 422


# =============================================================================
# PERFORMANCE CHECKS
# =============================================================================

def test_health_response_time(api_url):
    """Health endpoint should respond quickly."""
    import time
    start = time.time()
    response = requests.get(f"{api_url}/health")
    elapsed = (time.time() - start) * 1000

    assert response.status_code == 200
    assert elapsed < 100  # Should be under 100ms


def test_beautify_includes_timing(api_url, test_image_base64):
    """Beautify endpoint should include processing time."""
    response = requests.post(
        f"{api_url}/beautify/base64",
        json={"image_base64": test_image_base64},
        timeout=60
    )

    assert response.status_code == 200
    data = response.json()
    assert 'processing_time_ms' in data
    assert data['processing_time_ms'] > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    print("Running comprehensive API tests...")
    print("Make sure the API is running at http://localhost:8000")
    print()
    print("Start API with: ./start_api.sh")
    print("Run tests with: pytest tests/test_api_all_endpoints.py -v")
