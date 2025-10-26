"""
Test Suite Orchestrator

Runs all tests in sequence and generates comprehensive report:
1. Model Loading Tests
2. Endpoint Tests
3. Visual Validation
4. Statistical Validation
5. Integration Tests
6. Adjustment Tests (Thickness & Span)
7. Smoothing Tests (Curvature Preservation)

Generates markdown report with results.

Note: test_api_client.py requires API server running - run separately.
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime


class TestOrchestrator:
    """Orchestrate all test suites and generate report."""

    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None

    def run_test_suite(self, name, script_path):
        """Run a test suite and capture results."""
        print("\n" + "="*70)
        print(f"RUNNING: {name}")
        print("="*70)

        start = time.time()

        try:
            # Run test script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            elapsed = time.time() - start

            passed = result.returncode == 0

            self.results.append({
                'name': name,
                'passed': passed,
                'duration': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr
            })

            if passed:
                print(f"\nPASS - {name} ({elapsed:.1f}s)")
            else:
                print(f"\nFAIL - {name} ({elapsed:.1f}s)")
                if result.stderr:
                    print(f"Error: {result.stderr[:500]}")

            return passed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"\nFAIL - {name} (TIMEOUT after {elapsed:.1f}s)")

            self.results.append({
                'name': name,
                'passed': False,
                'duration': elapsed,
                'stdout': '',
                'stderr': 'Test timeout'
            })

            return False

        except Exception as e:
            elapsed = time.time() - start
            print(f"\nFAIL - {name} (ERROR: {str(e)})")

            self.results.append({
                'name': name,
                'passed': False,
                'duration': elapsed,
                'stdout': '',
                'stderr': str(e)
            })

            return False

    def generate_report(self, output_path):
        """Generate comprehensive markdown report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0

        passed_count = sum(1 for r in self.results if r['passed'])
        total_count = len(self.results)

        report = []
        report.append("# Eyebrow Beautification API - Test Report")
        report.append("")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Duration**: {total_duration:.1f}s")
        report.append("")

        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total Test Suites**: {total_count}")
        report.append(f"- **Passed**: {passed_count}")
        report.append(f"- **Failed**: {total_count - passed_count}")
        report.append(f"- **Pass Rate**: {(passed_count/total_count*100):.1f}%")
        report.append("")

        if passed_count == total_count:
            report.append("**Status**: ALL TESTS PASSED")
        else:
            report.append(f"**Status**: {total_count - passed_count} TEST SUITE(S) FAILED")

        report.append("")

        # Results Table
        report.append("## Test Results")
        report.append("")
        report.append("| Test Suite | Status | Duration |")
        report.append("|------------|--------|----------|")

        for result in self.results:
            status = "PASS" if result['passed'] else "FAIL"
            duration = f"{result['duration']:.1f}s"
            report.append(f"| {result['name']} | {status} | {duration} |")

        report.append("")

        # Detailed Results
        report.append("## Detailed Results")
        report.append("")

        for result in self.results:
            report.append(f"### {result['name']}")
            report.append("")
            report.append(f"- **Status**: {'PASS' if result['passed'] else 'FAIL'}")
            report.append(f"- **Duration**: {result['duration']:.1f}s")

            if not result['passed'] and result['stderr']:
                report.append(f"- **Error**: {result['stderr'][:200]}")

            report.append("")

            # Include last few lines of stdout
            if result['stdout']:
                lines = result['stdout'].split('\n')
                last_lines = lines[-15:]  # Last 15 lines
                report.append("**Output** (last 15 lines):")
                report.append("```")
                report.append('\n'.join(last_lines))
                report.append("```")
                report.append("")

        # Write report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"\nReport saved to: {output_path}")

    def run_all(self):
        """Run all test suites."""
        self.start_time = time.time()

        print("="*70)
        print("EYEBROW BEAUTIFICATION API - COMPREHENSIVE TEST SUITE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")

        # Define test suites
        test_suites = [
            ("Model Loading Tests", "tests/test_model_loading.py"),
            ("Endpoint Tests", "tests/test_endpoints.py"),
            ("Visual Validation", "tests/test_visual.py"),
            ("Statistical Validation", "tests/test_statistical.py"),
            ("Integration Tests", "tests/test_integration.py"),
            ("Adjustment Tests", "tests/test_adjustments.py"),
            ("Smoothing Tests", "tests/test_smooth_normal.py")
        ]

        # Run each suite
        for name, script_path in test_suites:
            if not Path(script_path).exists():
                print(f"\nSKIP: {name} - Script not found: {script_path}")
                continue

            self.run_test_suite(name, script_path)

        self.end_time = time.time()

        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)

        passed_count = sum(1 for r in self.results if r['passed'])
        total_count = len(self.results)

        print(f"\nTest Suites: {passed_count}/{total_count} passed")
        print(f"Total Duration: {self.end_time - self.start_time:.1f}s")

        for result in self.results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {result['name']}: {status} ({result['duration']:.1f}s)")

        # Generate report
        report_path = "tests/output/reports/test_report.md"
        self.generate_report(report_path)

        print("\n" + "="*70)

        if passed_count == total_count:
            print("ALL TEST SUITES PASSED")
            print("="*70)
            return True
        else:
            print(f"{total_count - passed_count} TEST SUITE(S) FAILED")
            print("="*70)
            return False


def main():
    """Main entry point."""
    orchestrator = TestOrchestrator()

    try:
        success = orchestrator.run_all()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
