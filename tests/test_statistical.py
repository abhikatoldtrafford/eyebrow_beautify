"""
Statistical validation tests

Validates that metrics fall within expected empirical ranges:
- MediaPipe coverage: 80-100%
- Eye distance: 4-8% of image height
- Aspect ratio: 3-10
- Eye overlap: 0 pixels
- Expansion ratio: 0.9-2.0x

Collects statistics across multiple images and generates report.
"""

import requests
import numpy as np
from pathlib import Path
from test_config import API_URL, TEST_IMAGES, EXPECTED_METRICS


class MetricsCollector:
    """Collect and analyze metrics across multiple images."""

    def __init__(self):
        self.all_metrics = {
            'mp_coverage': [],
            'eye_distance_pct': [],
            'aspect_ratio': [],
            'eye_overlap': [],
            'expansion_ratio': []
        }
        self.pass_counts = {
            'mp_coverage_pass': 0,
            'eye_distance_pass': 0,
            'aspect_ratio_pass': 0,
            'eye_overlap_pass': 0,
            'expansion_ratio_pass': 0,
            'overall_pass': 0
        }
        self.total = 0

    def add(self, validation_data):
        """Add validation metrics from one eyebrow."""
        self.all_metrics['mp_coverage'].append(validation_data['mp_coverage'])
        self.all_metrics['eye_distance_pct'].append(validation_data['eye_distance_pct'])
        self.all_metrics['aspect_ratio'].append(validation_data['aspect_ratio'])
        self.all_metrics['eye_overlap'].append(validation_data['eye_overlap'])
        self.all_metrics['expansion_ratio'].append(validation_data['expansion_ratio'])

        # Count passes
        self.pass_counts['mp_coverage_pass'] += int(validation_data['mp_coverage_pass'])
        self.pass_counts['eye_distance_pass'] += int(validation_data['eye_distance_pass'])
        self.pass_counts['aspect_ratio_pass'] += int(validation_data['aspect_ratio_pass'])
        self.pass_counts['eye_overlap_pass'] += int(validation_data['eye_overlap_pass'])
        self.pass_counts['expansion_ratio_pass'] += int(validation_data['expansion_ratio_pass'])
        self.pass_counts['overall_pass'] += int(validation_data['overall_pass'])

        self.total += 1

    def get_statistics(self):
        """Calculate statistics for each metric."""
        stats = {}

        for metric_name, values in self.all_metrics.items():
            if len(values) == 0:
                continue

            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

        return stats

    def check_all_in_range(self):
        """Check if all metrics fall within expected ranges."""
        failures = []

        for metric_name, values in self.all_metrics.items():
            if metric_name not in EXPECTED_METRICS:
                continue

            expected_range = EXPECTED_METRICS[metric_name]

            for i, val in enumerate(values):
                if not (expected_range[0] <= val <= expected_range[1]):
                    failures.append({
                        'metric': metric_name,
                        'value': val,
                        'expected': expected_range,
                        'eyebrow_index': i
                    })

        return len(failures) == 0, failures

    def print_summary(self):
        """Print comprehensive statistics summary."""
        print("\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)

        stats = self.get_statistics()

        print(f"\nTotal Eyebrows Analyzed: {self.total}\n")

        # Per-metric statistics
        for metric_name in ['mp_coverage', 'eye_distance_pct', 'aspect_ratio', 'eye_overlap', 'expansion_ratio']:
            if metric_name not in stats:
                continue

            s = stats[metric_name]
            expected = EXPECTED_METRICS.get(metric_name, (None, None))

            print(f"{metric_name.upper()}:")
            print(f"  Mean: {s['mean']:.2f}")
            print(f"  Std:  {s['std']:.2f}")
            print(f"  Range: [{s['min']:.2f}, {s['max']:.2f}]")
            print(f"  Median: {s['median']:.2f}")

            if expected[0] is not None:
                print(f"  Expected Range: [{expected[0]}, {expected[1]}]")

                # Check if mean is in range
                if expected[0] <= s['mean'] <= expected[1]:
                    print(f"  Mean Status: IN RANGE")
                else:
                    print(f"  Mean Status: OUT OF RANGE")

            print()

        # Pass rates
        print("VALIDATION PASS RATES:")
        for check_name, count in self.pass_counts.items():
            rate = (count / self.total * 100) if self.total > 0 else 0
            print(f"  {check_name}: {count}/{self.total} ({rate:.1f}%)")

        print()

        # Overall assessment
        all_in_range, failures = self.check_all_in_range()

        if all_in_range:
            print("RANGE CHECK: ALL METRICS IN EXPECTED RANGES")
        else:
            print(f"RANGE CHECK: {len(failures)} VALUE(S) OUT OF RANGE")
            print("\nOut-of-range values:")
            for f in failures[:5]:  # Show first 5
                print(f"  {f['metric']}: {f['value']:.2f} (expected {f['expected']})")

        return all_in_range


# Global collector
collector = MetricsCollector()


def test_image_statistics(test_image):
    """Collect statistics from one test image."""
    print("\n" + "="*70)
    print(f"STATISTICAL TEST: {Path(test_image).name}")
    print("="*70)

    # Call beautify endpoint
    with open(test_image, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/beautify", files=files)

    if response.status_code != 200:
        print(f"FAIL: Request failed: {response.text}")
        return False

    data = response.json()

    if not data['success']:
        print(f"FAIL: Beautification failed")
        return False

    print(f"Eyebrows Found: {len(data['eyebrows'])}")

    # Collect metrics from each eyebrow
    for eyebrow in data['eyebrows']:
        validation = eyebrow['validation']
        collector.add(validation)

        print(f"\n{eyebrow['side']} eyebrow metrics:")
        print(f"  MP Coverage: {validation['mp_coverage']:.1f}%")
        print(f"  Eye Distance: {validation['eye_distance_pct']:.2f}%")
        print(f"  Aspect Ratio: {validation['aspect_ratio']:.2f}")
        print(f"  Eye Overlap: {validation['eye_overlap']} px")
        print(f"  Expansion: {validation['expansion_ratio']:.2f}x")
        print(f"  Overall Pass: {validation['overall_pass']}")

    print("PASS - Metrics collected")
    return True


def run_all_tests():
    """Run statistical tests on all test images."""
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION TESTS")
    print("="*70)

    results = []

    for test_image in TEST_IMAGES:
        if not Path(test_image).exists():
            print(f"SKIP: {test_image} not found")
            continue

        success = test_image_statistics(test_image)
        results.append((Path(test_image).name, success))

    # Print comprehensive summary
    all_in_range = collector.print_summary()

    # Test summary
    print("\n" + "="*70)
    print("STATISTICAL TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} images processed")
    print(f"Total Eyebrows: {collector.total}")

    if passed == total and all_in_range:
        print("\nALL STATISTICAL TESTS PASSED")
        print("All metrics within expected empirical ranges")
        return True
    else:
        if passed < total:
            print(f"\n{total - passed} image(s) failed")
        if not all_in_range:
            print("Some metrics are out of expected range (see summary above)")
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
