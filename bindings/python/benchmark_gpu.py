"""
Benchmark script for GPU performance testing.

Run this on a machine with CUDA GPU to test the torch.compile + vmap optimization.
"""

import torch
import time
import numpy as np
from tdigest_rs.torch_impl import TDigestTorch

def benchmark_gpu():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires a CUDA GPU.")
        print(f"Available device: {TDigestTorch.get_device()}")
        return

    print("=" * 70)
    print("GPU Performance Benchmark - torch.compile + vmap")
    print("=" * 70)

    # Configuration
    batch_size = 1000
    n_elements = 32000
    delta = 0.01

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Elements per digest: {n_elements}")
    print(f"  Delta: {delta}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Create test data on GPU
    print(f"\nGenerating test data...")
    torch.manual_seed(42)
    data = torch.randn(batch_size, n_elements, dtype=torch.float64, device='cuda')

    # Warmup (torch.compile needs to JIT compile on first run)
    print(f"\nWarming up (torch.compile JIT compilation)...")
    warmup_start = time.time()
    _ = TDigestTorch.batch_from_tensor(data[:10], delta=delta, use_compile=True)
    warmup_time = time.time() - warmup_start
    print(f"  Warmup time: {warmup_time:.2f}s (this is one-time JIT compilation cost)")

    # Benchmark: Sequential CPU-like path (use_compile=False)
    print(f"\n{'─' * 70}")
    print("Benchmark 1: Sequential path (use_compile=False)")
    print('─' * 70)
    torch.cuda.synchronize()
    start = time.time()
    result_seq = TDigestTorch.batch_from_tensor(data, delta=delta, use_compile=False)
    torch.cuda.synchronize()
    time_seq = time.time() - start

    print(f"Time: {time_seq:.3f}s")
    print(f"Throughput: {batch_size * n_elements / time_seq / 1e6:.2f} M elements/sec")

    # Benchmark: Compiled + vmap path (use_compile=True)
    print(f"\n{'─' * 70}")
    print("Benchmark 2: torch.compile + vmap (use_compile=True)")
    print('─' * 70)
    torch.cuda.synchronize()
    start = time.time()
    result_compiled = TDigestTorch.batch_from_tensor(data, delta=delta, use_compile=True)
    torch.cuda.synchronize()
    time_compiled = time.time() - start

    print(f"Time: {time_compiled:.3f}s")
    print(f"Throughput: {batch_size * n_elements / time_compiled / 1e6:.2f} M elements/sec")

    # Speedup
    speedup = time_seq / time_compiled
    print(f"\n{'=' * 70}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'=' * 70}")

    # Verify correctness
    print(f"\nVerifying correctness...")
    counts_match = np.array_equal(result_seq.counts, result_compiled.counts)
    means_match = np.allclose(result_seq.means, result_compiled.means, rtol=1e-10, atol=1e-10)
    weights_match = np.allclose(result_seq.weights, result_compiled.weights, rtol=1e-10, atol=1e-10)

    print(f"  Counts match: {counts_match} {'✓' if counts_match else '✗'}")
    print(f"  Means match: {means_match} {'✓' if means_match else '✗'}")
    print(f"  Weights match: {weights_match} {'✓' if weights_match else '✗'}")

    if counts_match and means_match and weights_match:
        print("\n✅ Results are identical!")
    else:
        print("\n❌ Results differ!")

    # Large batch test
    print(f"\n{'=' * 70}")
    print("Large Batch Test (20,000 digests)")
    print('=' * 70)

    batch_size_large = 20000
    print(f"\nGenerating large batch ({batch_size_large} × {n_elements})...")
    data_large = torch.randn(batch_size_large, n_elements, dtype=torch.float64, device='cuda')

    print(f"\nProcessing with torch.compile + vmap...")
    torch.cuda.synchronize()
    start = time.time()
    result_large = TDigestTorch.batch_from_tensor(data_large, delta=delta, use_compile=True)
    torch.cuda.synchronize()
    time_large = time.time() - start

    total_elements = batch_size_large * n_elements
    print(f"Time: {time_large:.3f}s")
    print(f"Throughput: {total_elements / time_large / 1e6:.2f} M elements/sec")
    print(f"Average time per digest: {time_large / batch_size_large * 1000:.3f}ms")

    print(f"\n✅ Benchmark complete!")


if __name__ == "__main__":
    benchmark_gpu()
