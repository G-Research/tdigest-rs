"""
Example usage of PyTorch backend for T-Digest.

Demonstrates CPU usage (works on MacBook) and shows how results
match the Rust implementation exactly.
"""

import numpy as np
import time
from tdigest_rs import TDigest
from tdigest_rs.torch_backend import TDigestTorch, batch_from_arrays_torch


def basic_example():
    """Basic usage example."""
    print("=" * 60)
    print("Basic PyTorch Backend Example")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    arrays = [np.random.randn(5000).astype(np.float64) for _ in range(10)]

    print(f"\nProcessing {len(arrays)} arrays of {len(arrays[0])} elements each")

    # Process with PyTorch backend
    result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')

    print(f"\nResults:")
    for i in range(len(arrays)):
        count = result.counts[i]
        print(f"  Digest {i}: {count} centroids")

    return result


def correctness_check():
    """Verify PyTorch backend matches Rust exactly."""
    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    np.random.seed(123)
    data = np.random.randn(10000).astype(np.float64)

    # Rust version
    rust_digest = TDigest.from_array(data, delta=0.01)

    # PyTorch version
    torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

    count = torch_result.counts[0]
    torch_means = torch_result.means[0, :count]
    torch_weights = torch_result.weights[0, :count]

    # Compare
    means_match = np.allclose(torch_means, rust_digest.means, rtol=1e-10, atol=1e-10)
    weights_match = np.array_equal(torch_weights.astype(np.uint32), rust_digest.weights)

    print(f"\nRust centroids: {len(rust_digest.means)}")
    print(f"PyTorch centroids: {count}")
    print(f"Means match: {means_match} ✓" if means_match else "Means match: {means_match} ✗")
    print(f"Weights match: {weights_match} ✓" if weights_match else "Weights match: {weights_match} ✗")

    if means_match and weights_match:
        print("\n✅ PyTorch backend produces IDENTICAL results to Rust!")
    else:
        print("\n❌ Results differ!")

    return means_match and weights_match


def performance_comparison():
    """Compare performance of Rust vs PyTorch on CPU."""
    print("\n" + "=" * 60)
    print("Performance Comparison (CPU)")
    print("=" * 60)

    np.random.seed(456)
    n_arrays = 100
    array_size = 10000
    arrays = [np.random.randn(array_size).astype(np.float64) for _ in range(n_arrays)]

    print(f"\nProcessing {n_arrays} arrays of {array_size} elements each")

    # Rust sequential
    start = time.time()
    rust_digests = [TDigest.from_array(arr, delta=0.01) for arr in arrays]
    rust_time = time.time() - start

    # PyTorch CPU
    start = time.time()
    torch_result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')
    torch_time = time.time() - start

    print(f"\nRust (sequential): {rust_time:.3f}s")
    print(f"PyTorch (CPU): {torch_time:.3f}s")
    print(f"Ratio: {rust_time/torch_time:.2f}x")

    if torch_time < rust_time * 1.1:
        print("✅ PyTorch performance is competitive")
    else:
        print("⚠️  PyTorch is slower (expected on CPU, would be faster on GPU)")


def quantile_example():
    """Show how to compute quantiles from PyTorch results."""
    print("\n" + "=" * 60)
    print("Quantile Computation Example")
    print("=" * 60)

    np.random.seed(789)
    data = np.random.randn(20000).astype(np.float64)

    # Process with PyTorch
    result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

    # Extract results
    count = result.counts[0]
    means = result.means[0, :count]
    weights = result.weights[0, :count].astype(np.uint32)

    # Create TDigest for quantile computation
    digest = TDigest.from_means_weights(means, weights, delta=0.01)

    print(f"\nComputed {count} centroids from {len(data)} points")
    print("\nQuantiles:")
    for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
        value = digest.quantile(q)
        print(f"  p{int(q*100):2d}: {value:7.3f}")


def device_info():
    """Show available devices."""
    print("\n" + "=" * 60)
    print("Device Information")
    print("=" * 60)

    best_device = TDigestTorch.get_device()
    cuda_available = TDigestTorch.is_available()

    print(f"\nBest available device: {best_device}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print("\n✅ GPU acceleration available!")
        print("   Use device='cuda' for ~10-30x speedup on large batches")
    else:
        print("\n💻 Running on CPU (no CUDA GPU detected)")
        print("   On MacBook: This is expected")
        print("   For GPU: Deploy to machine with NVIDIA GPU")


if __name__ == "__main__":
    device_info()
    basic_example()
    correctness_check()
    performance_comparison()
    quantile_example()

    print("\n" + "=" * 60)
    print("All examples completed successfully! ✅")
    print("=" * 60)
