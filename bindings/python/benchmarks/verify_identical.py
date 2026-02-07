#!/usr/bin/env python3
"""
Correctness Verification Suite

Ensures that optimizations produce bit-exact identical results.
This is CRITICAL - any deviation means the optimization broke something.

Usage:
    python verify_identical.py save     # Save reference outputs
    python verify_identical.py verify   # Verify against reference
"""

import sys
import json
import numpy as np
from pathlib import Path

try:
    from tdigest_rs import TDigest
except ImportError:
    print("ERROR: Could not import tdigest_rs. Make sure Python bindings are built.")
    print("Run: cd bindings/python && maturin develop --release")
    sys.exit(1)


REFERENCE_FILE = Path(__file__).parent / "reference_outputs.json"


def serialize_array(arr):
    """Convert numpy array to list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def arrays_equal(arr1, arr2, tolerance=0.0):
    """Check if two arrays are exactly equal (or within tolerance)."""
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    if arr1.shape != arr2.shape:
        return False

    if tolerance > 0:
        return np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance)
    else:
        # Bit-exact comparison
        return np.array_equal(arr1, arr2)


def test_basic_creation():
    """Test basic digest creation."""
    np.random.seed(42)
    data = np.random.randn(1000).astype(np.float32) * 100.0

    digest = TDigest.from_array(data, delta=100.0)

    return {
        'test_name': 'basic_creation',
        'means': serialize_array(digest.means),
        'weights': serialize_array(digest.weights),
        'size': len(digest),
    }


def test_quantiles():
    """Test quantile computation."""
    np.random.seed(123)
    data = np.random.randn(5000).astype(np.float32) * 50.0

    digest = TDigest.from_array(data, delta=100.0)

    quantiles = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    results = [digest.quantile(q) for q in quantiles]

    return {
        'test_name': 'quantiles',
        'quantiles': quantiles,
        'results': serialize_array(results),
        'digest_size': len(digest),
    }


def test_merge():
    """Test digest merging."""
    np.random.seed(456)
    data1 = np.random.randn(2000).astype(np.float32) * 30.0
    data2 = np.random.randn(3000).astype(np.float32) * 40.0

    digest1 = TDigest.from_array(data1, delta=100.0)
    digest2 = TDigest.from_array(data2, delta=100.0)

    merged = digest1.merge(digest2, delta=100.0)

    return {
        'test_name': 'merge',
        'means': serialize_array(merged.means),
        'weights': serialize_array(merged.weights),
        'size': len(merged),
    }


def test_update():
    """Test digest update operation."""
    np.random.seed(789)
    initial_data = np.random.randn(1000).astype(np.float32) * 20.0
    update_data = np.random.randn(5000).astype(np.float32) * 60.0

    digest = TDigest.from_array(initial_data, delta=100.0)
    updated = digest.update(update_data, delta=100.0)

    return {
        'test_name': 'update',
        'means': serialize_array(updated.means),
        'weights': serialize_array(updated.weights),
        'size': len(updated),
    }


def test_large_batch():
    """Test large batch (production scenario)."""
    np.random.seed(999)
    data = np.random.randn(32_000).astype(np.float32) * 100.0

    digest = TDigest.from_array(data, delta=100.0)

    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    results = [digest.quantile(q) for q in quantiles]

    return {
        'test_name': 'large_batch',
        'means': serialize_array(digest.means),
        'weights': serialize_array(digest.weights),
        'size': len(digest),
        'quantiles': quantiles,
        'quantile_results': serialize_array(results),
    }


def test_edge_cases():
    """Test edge cases."""
    results = {}

    # Single value
    digest_single = TDigest.from_array(np.array([42.0], dtype=np.float32), delta=100.0)
    results['single_value'] = {
        'means': serialize_array(digest_single.means),
        'weights': serialize_array(digest_single.weights),
        'quantile_median': float(digest_single.quantile(0.5)),
    }

    # Two values
    digest_two = TDigest.from_array(np.array([10.0, 20.0], dtype=np.float32), delta=100.0)
    results['two_values'] = {
        'means': serialize_array(digest_two.means),
        'weights': serialize_array(digest_two.weights),
    }

    # Sorted data
    np.random.seed(111)
    sorted_data = np.sort(np.random.randn(1000).astype(np.float32) * 50.0)
    digest_sorted = TDigest.from_array(sorted_data, delta=100.0)
    results['sorted_data'] = {
        'means': serialize_array(digest_sorted.means),
        'weights': serialize_array(digest_sorted.weights),
    }

    # Reverse sorted data
    reverse_sorted_data = sorted_data[::-1].copy()
    digest_reverse = TDigest.from_array(reverse_sorted_data, delta=100.0)
    results['reverse_sorted'] = {
        'means': serialize_array(digest_reverse.means),
        'weights': serialize_array(digest_reverse.weights),
    }

    return {
        'test_name': 'edge_cases',
        'cases': results,
    }


def run_all_tests():
    """Run all test cases and return results."""
    print("Running test suite...")

    tests = [
        test_basic_creation,
        test_quantiles,
        test_merge,
        test_update,
        test_large_batch,
        test_edge_cases,
    ]

    results = []
    for test_func in tests:
        print(f"  Running {test_func.__name__}...")
        result = test_func()
        results.append(result)

    return results


def save_reference():
    """Save reference outputs to file."""
    print("\n" + "="*70)
    print("SAVING REFERENCE OUTPUTS")
    print("="*70 + "\n")

    results = run_all_tests()

    with open(REFERENCE_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Reference outputs saved to: {REFERENCE_FILE}")
    print("\nThese outputs represent the CORRECT behavior.")
    print("All optimizations must produce IDENTICAL results.\n")


def verify_against_reference():
    """Verify current implementation against reference."""
    print("\n" + "="*70)
    print("VERIFYING AGAINST REFERENCE")
    print("="*70 + "\n")

    if not REFERENCE_FILE.exists():
        print(f"ERROR: Reference file not found: {REFERENCE_FILE}")
        print("Run: python verify_identical.py save")
        return False

    # Load reference
    with open(REFERENCE_FILE, 'r') as f:
        reference = json.load(f)

    # Run current tests
    current = run_all_tests()

    # Compare
    all_passed = True
    for ref_result, cur_result in zip(reference, current):
        test_name = ref_result['test_name']
        print(f"Verifying {test_name}...", end=" ")

        passed = compare_results(ref_result, cur_result)

        if passed:
            print("✓ PASS")
        else:
            print("✗ FAIL")
            all_passed = False
            print(f"  ERROR: Results differ for {test_name}")
            show_differences(ref_result, cur_result)

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Results are IDENTICAL")
        print("="*70 + "\n")
        return True
    else:
        print("✗ VERIFICATION FAILED - Results differ!")
        print("="*70)
        print("\nWARNING: The optimization changed behavior!")
        print("This is NOT acceptable. Revert the changes.\n")
        return False


def compare_results(ref, cur, path=""):
    """Recursively compare two result dictionaries."""
    if isinstance(ref, dict):
        if not isinstance(cur, dict):
            return False
        for key in ref.keys():
            if key not in cur:
                return False
            if not compare_results(ref[key], cur[key], f"{path}.{key}"):
                return False
        return True
    elif isinstance(ref, list):
        if not isinstance(cur, list):
            return False
        # For arrays, check bit-exact equality
        return arrays_equal(ref, cur, tolerance=0.0)
    else:
        # For scalar values
        return ref == cur


def show_differences(ref, cur, path="", max_depth=3, current_depth=0):
    """Show differences between reference and current results."""
    if current_depth >= max_depth:
        return

    if isinstance(ref, dict):
        for key in ref.keys():
            if key in cur and not compare_results(ref[key], cur[key]):
                print(f"    Difference in {path}.{key}")
                if isinstance(ref[key], list):
                    ref_arr = np.array(ref[key])
                    cur_arr = np.array(cur[key])
                    if ref_arr.shape == cur_arr.shape:
                        max_diff = np.max(np.abs(ref_arr - cur_arr))
                        print(f"      Max absolute difference: {max_diff}")
                    else:
                        print(f"      Shape mismatch: {ref_arr.shape} vs {cur_arr.shape}")
                show_differences(ref[key], cur[key], f"{path}.{key}",
                               max_depth, current_depth + 1)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python verify_identical.py save     # Save reference outputs")
        print("  python verify_identical.py verify   # Verify against reference")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'save':
        save_reference()
    elif command == 'verify':
        success = verify_against_reference()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print("Use 'save' or 'verify'")
        sys.exit(1)


if __name__ == '__main__':
    main()
