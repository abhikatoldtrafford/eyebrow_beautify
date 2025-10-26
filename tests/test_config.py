"""
Test configuration and shared utilities
"""

API_URL = "http://localhost:8000"

# Test images (from annotated/test/images/)
TEST_IMAGES = [
    "annotated/test/images/After_jpg.rf.46aeb3ac6f2ed5beb66e9a92cbe8ee73.jpg",
    "annotated/test/images/After_jpg.rf.cbd89b09083c32e288c10eafa5595963.jpg",
    "annotated/test/images/After_jpg.rf.f7e8c4e7f96af28dc72fb5e6dfb0af3d.jpg"
]

# Expected validation ranges (from empirical data)
EXPECTED_METRICS = {
    'mp_coverage': (80.0, 100.0),         # 80-100% MediaPipe coverage
    'eye_distance': (4.0, 8.0),           # 4-8% of image height
    'aspect_ratio': (3.0, 10.0),          # 3-10 (relaxed from 4-10)
    'eye_overlap': (0, 0),                # Must be 0
    'expansion_ratio': (0.9, 2.0),        # 0.9-2.0x expansion
}

# Output directories
OUTPUT_DIR = "tests/output"
VIZ_DIR = "tests/output/visualizations"
REPORT_DIR = "tests/output/reports"
