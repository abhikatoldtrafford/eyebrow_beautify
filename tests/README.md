# API Testing Strategy

## Quick Start

```bash
# 1. Start the API server
./start_api.sh

# 2. Run API tests
pytest tests/test_api_all_endpoints.py -v

# 3. View results
# ✅ All 19 endpoint tests should pass
```

---

## Test Files Overview

### ✅ **API Unit Tests** (Focus here!)

| File | Purpose | Status | Tests |
|------|---------|--------|-------|
| **`test_api_all_endpoints.py`** | **Comprehensive API testing** | ✅ **Primary** | **19 tests covering all 15 endpoints** |
| `test_api_endpoints.py` | Legacy API tests | ⚠️  Deprecated | 10 tests (subset of above) |
| `test_adjustment_api.py` | Adjustment endpoints only | ⚠️  Deprecated | Covered in comprehensive |
| `test_api_preprocessing.py` | Preprocessing endpoint | ⚠️  Deprecated | Covered in comprehensive |

### ⚠️ **Algorithm Tests** (Not API tests - for internal development)

| File | Purpose | Note |
|------|---------|------|
| `test_integration.py` | Tests beautify pipeline directly | Internal testing |
| `test_adjustments.py` | Tests utils.adjust_* functions | Internal testing |
| `test_smooth_normal.py` | Tests smoothing algorithms | Internal testing |
| `test_statistical.py` | Tests algorithm metrics | Internal testing |
| `test_visual.py` | Generates visual comparisons | Internal testing |
| `test_critical_fixes.py` | Regression tests for bugs | Internal testing |
| `test_preprocessing_*.py` | Tests preprocessing modules | Internal testing |
| `test_model_loading.py` | Tests YOLO model loading | Internal testing |
| `test_config.py` | Tests configuration | Internal testing |
| `test_developer_corner_e2e.py` | Streamlit E2E tests | UI testing |

---

## Recommended Testing Approach

### For API Development

**Use: `test_api_all_endpoints.py`**

This single file tests ALL 15 API endpoints:

```bash
pytest tests/test_api_all_endpoints.py -v
```

**Coverage:**
- ✅ Health & Config (3 endpoints)
- ✅ Detection (4 endpoints: YOLO/MediaPipe, file/base64)
- ✅ Beautification (3 endpoints: beautify, submit-edit)
- ✅ Adjustments (4 endpoints: thickness/span, increase/decrease)
- ✅ Generation (1 endpoint: SD placeholder)
- ✅ Error handling (invalid inputs, missing files)
- ✅ Performance checks (response times)

### For Algorithm Development

Use the internal test files when modifying core algorithms:

```bash
# Test specific algorithm changes
pytest tests/test_integration.py -v
pytest tests/test_adjustments.py -v
pytest tests/test_smooth_normal.py -v
```

---

## Test Organization

### API Tests (External Interface)

**Philosophy:** Test the API contract, not the implementation.

- Test endpoints return correct HTTP status codes
- Test response JSON structure matches API specification
- Test error handling (4xx/5xx responses)
- Test all endpoints are reachable
- **Do NOT** test internal algorithm correctness here

**Example:**
```python
def test_beautify_base64(api_url, test_image_base64):
    """Test POST /beautify/base64."""
    response = requests.post(
        f"{api_url}/beautify/base64",
        json={"image_base64": test_image_base64},
        timeout=60
    )

    assert response.status_code == 200  # API contract
    data = response.json()
    assert data['success'] is True  # API contract
    assert 'eyebrows' in data  # API contract
    # NOT testing: algorithm quality, mask accuracy, etc.
```

### Algorithm Tests (Internal Implementation)

**Philosophy:** Test correctness of algorithms and transformations.

- Test beautification pipeline produces valid masks
- Test adjustment functions modify masks correctly
- Test validation metrics are calculated accurately
- Test edge cases and boundary conditions
- **Do NOT** test via HTTP endpoints

**Example:**
```python
def test_thickness_adjustment():
    """Test thickness adjustment algorithm."""
    mask = create_test_mask()
    adjusted = adjust_eyebrow_thickness(mask, factor=1.05)

    assert adjusted.shape == mask.shape
    assert np.sum(adjusted) > np.sum(mask)  # Should increase
    assert check_mask_validity(adjusted)  # Should be valid
```

---

## Running Tests

### Quick Test (API Only)

```bash
pytest tests/test_api_all_endpoints.py -v
```

**Expected output:**
```
19 passed in 2.66s
```

### Full Test Suite (All Tests)

```bash
pytest tests/ -v --ignore=tests/test_developer_corner_e2e.py
```

### Specific Test

```bash
# Test single endpoint
pytest tests/test_api_all_endpoints.py::test_beautify_base64 -v

# Test category
pytest tests/test_api_all_endpoints.py -k "adjust" -v
```

### With Coverage

```bash
pytest tests/test_api_all_endpoints.py --cov=api --cov-report=html
```

---

## Continuous Integration

For CI/CD pipelines, use the comprehensive API test:

```yaml
# .github/workflows/test.yml
- name: Start API
  run: ./start_api.sh &

- name: Wait for API
  run: sleep 5

- name: Test API
  run: pytest tests/test_api_all_endpoints.py -v

- name: Stop API
  run: pkill -f uvicorn
```

---

## Test Fixtures

The comprehensive API test uses session-scoped fixtures for efficiency:

```python
@pytest.fixture(scope="session")
def beautify_result(api_url, test_image_base64):
    """Cached beautify result - only runs once."""
    response = requests.post(...)
    return response.json()
```

**Benefits:**
- Faster test execution (shared setup)
- Consistent test data across tests
- Reduced API load

---

## Deprecated Tests

These files can be removed or archived:

- ❌ `test_api_endpoints.py` → Use `test_api_all_endpoints.py`
- ❌ `test_adjustment_api.py` → Use `test_api_all_endpoints.py`
- ❌ `test_api_preprocessing.py` → Use `test_api_all_endpoints.py`

**Why deprecate?**
- Redundant coverage (subset of comprehensive test)
- Harder to maintain multiple test files
- Incomplete endpoint coverage

**Migration:**
```bash
# Old way (incomplete, 3 files):
pytest tests/test_api_endpoints.py tests/test_adjustment_api.py tests/test_api_preprocessing.py

# New way (complete, 1 file):
pytest tests/test_api_all_endpoints.py
```

---

## Best Practices

### ✅ DO

- Test all API endpoints in one comprehensive file
- Use session fixtures for shared setup
- Test HTTP contracts (status codes, JSON structure)
- Test error handling
- Keep API tests fast (< 5 seconds total)

### ❌ DON'T

- Mix API tests with algorithm tests
- Test internal implementation details via API
- Create multiple small test files for each endpoint
- Make API tests depend on algorithm correctness
- Use algorithm tests for API validation

---

## Summary

**For API Development:**
- Use: `test_api_all_endpoints.py` ✅
- Run: `pytest tests/test_api_all_endpoints.py -v`
- Time: ~3 seconds
- Coverage: All 15 endpoints

**For Algorithm Development:**
- Use: Individual algorithm test files
- Run: `pytest tests/test_integration.py -v` (etc.)
- Time: Varies
- Coverage: Algorithm correctness

**Clean separation = Faster development + Easier maintenance!** 🎯
