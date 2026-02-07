#!/usr/bin/env python3
"""
Production Scenario Benchmark

Tests the exact user workload:
- 20,000 individual t-digests
- Batch updates of ~32,000 elements each
- Coordinated through Python thread pools

Measures throughput (elements/second) and total processing time.
"""

import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tdigest_rs import TDigest
except ImportError:
    print("ERROR: Could not import tdigest_rs. Make sure Python bindings are built.")
    print("Run: cd bindings/python && maturin develop --release")
    sys.exit(1)


def create_batch_data(batch_size=32_000, seed=None):
    """Generate a batch of random data for testing."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(batch_size).astype(np.float32) * 100.0


def process_digest(digest_id, batch_data, delta=100.0):
    """Process a single digest with batch update."""
    # Create initial digest
    digest = TDigest.from_array(np.random.randn(100).astype(np.float32) * 50.0, delta=delta)

    # Update with batch
    digest = digest.update(batch_data, delta=delta)

    # Perform some queries to simulate real usage
    quantiles = [digest.quantile(q) for q in [0.01, 0.25, 0.5, 0.75, 0.99]]

    return digest_id, len(batch_data), quantiles


def run_sequential_benchmark(num_digests=1000, batch_size=32_000, delta=100.0):
    """Run benchmark sequentially (baseline for comparison)."""
    print(f"\n{'='*70}")
    print(f"Sequential Benchmark: {num_digests} digests × {batch_size:,} elements")
    print(f"{'='*70}")

    batch_data = create_batch_data(batch_size, seed=42)

    start_time = time.time()
    total_elements = 0

    for i in range(num_digests):
        _, elements, _ = process_digest(i, batch_data, delta)
        total_elements += elements

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            throughput = total_elements / elapsed
            print(f"  Progress: {i+1:,}/{num_digests:,} digests | "
                  f"Throughput: {throughput:,.0f} elements/sec")

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = total_elements / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Average throughput: {throughput:,.0f} elements/second")
    print(f"  Time per digest: {elapsed/num_digests*1000:.2f} ms")

    return {
        'elapsed': elapsed,
        'throughput': throughput,
        'total_elements': total_elements,
        'num_digests': num_digests
    }


def run_parallel_benchmark(num_digests=20_000, batch_size=32_000,
                          delta=100.0, max_workers=8):
    """Run benchmark with thread pool (production scenario)."""
    print(f"\n{'='*70}")
    print(f"Parallel Benchmark: {num_digests:,} digests × {batch_size:,} elements")
    print(f"Workers: {max_workers}")
    print(f"{'='*70}")

    batch_data = create_batch_data(batch_size, seed=42)

    start_time = time.time()
    total_elements = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_digest, i, batch_data, delta): i
            for i in range(num_digests)
        }

        # Process results as they complete
        for future in as_completed(futures):
            digest_id, elements, _ = future.result()
            total_elements += elements
            completed += 1

            if completed % 1000 == 0:
                elapsed = time.time() - start_time
                throughput = total_elements / elapsed
                print(f"  Progress: {completed:,}/{num_digests:,} digests | "
                      f"Throughput: {throughput:,.0f} elements/sec")

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = total_elements / elapsed

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Average throughput: {throughput:,.0f} elements/second")
    print(f"  Time per digest: {elapsed/num_digests*1000:.2f} ms")

    return {
        'elapsed': elapsed,
        'throughput': throughput,
        'total_elements': total_elements,
        'num_digests': num_digests,
        'max_workers': max_workers
    }


def main():
    print("\n" + "="*70)
    print("T-Digest Production Scenario Benchmark")
    print("="*70)

    # Quick sequential test first (smaller scale)
    print("\n[1/2] Running sequential benchmark (warmup)...")
    seq_results = run_sequential_benchmark(
        num_digests=100,
        batch_size=32_000,
        delta=100.0
    )

    # Full parallel production scenario
    print("\n[2/2] Running parallel production scenario...")
    par_results = run_parallel_benchmark(
        num_digests=20_000,
        batch_size=32_000,
        delta=100.0,
        max_workers=8
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nSequential (100 digests):")
    print(f"  Throughput: {seq_results['throughput']:,.0f} elements/sec")
    print(f"\nParallel (20,000 digests):")
    print(f"  Throughput: {par_results['throughput']:,.0f} elements/sec")
    print(f"  Total time: {par_results['elapsed']:.2f} seconds")
    print(f"  Speedup vs sequential: {seq_results['throughput']/par_results['throughput']:.2f}x")

    print(f"\n{'='*70}")
    print("Baseline established! Save these numbers for comparison.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
