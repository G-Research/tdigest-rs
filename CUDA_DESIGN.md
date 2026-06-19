# CUDA Backend Design for T-Digest

## Executive Summary

**Estimated Speedup: 20-100x for batch workloads (10,000+ digests)**

- Current CPU: ~55 M elements/sec per core
- GPU Target: 1-5 B elements/sec (datacenter GPU)
- Best for: Processing 1,000-100,000 digests simultaneously

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Python API Layer                                       │
│  - TDigestGPU class                                     │
│  - Automatic CPU/GPU dispatch                           │
└─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼────┐                    ┌─────▼─────┐
    │ CPU Path│                    │ GPU Path  │
    │ (Rust)  │                    │ (PyTorch/ │
    │         │                    │  Triton)  │
    └─────────┘                    └───────────┘
```

## Recommended Implementation: PyTorch + Triton

### Why PyTorch/Triton?

✅ **Can develop on MacBook without NVIDIA GPU**
   - Use CPU backend for development/testing
   - Use MPS (Metal) backend on Mac for basic GPU testing
   - Deploy to CUDA without code changes

✅ **CUDA version agnostic**
   - PyTorch handles CUDA 11.x, 12.x, 13.x automatically
   - Single binary works across all supported CUDA versions
   - No recompilation needed

✅ **Triton for custom kernels**
   - Python-based kernel language (easy to develop)
   - Auto-optimizes for different GPU architectures
   - Can simulate on CPU during development

✅ **Production ready**
   - Battle-tested at scale (Meta, OpenAI, etc.)
   - Excellent performance for custom algorithms
   - Good profiling/debugging tools

## Implementation Design

### 1. Data Layout

```python
# Batched input: process N digests simultaneously
batch_size = 10000
elements_per_digest = 32000

# Shape: [batch_size, elements_per_digest]
data = torch.tensor(..., device='cuda')
weights = torch.ones(batch_size, elements_per_digest, device='cuda')

# Output: variable-length centroids per digest
# Use padded representation
max_centroids = 500  # Typical compression
means = torch.zeros(batch_size, max_centroids, device='cuda')
centroid_weights = torch.zeros(batch_size, max_centroids, device='cuda')
centroid_counts = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
```

### 2. GPU Kernel Structure (Triton)

```python
# Pseudocode for Triton kernel
@triton.jit
def tdigest_clustering_kernel(
    data_ptr, weights_ptr,           # Input: [batch, elements]
    means_ptr, cent_weights_ptr,     # Output: [batch, max_centroids]
    counts_ptr,                      # Output: [batch]
    batch_size, n_elements, delta,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one digest
    batch_idx = tl.program_id(0)

    # Load data for this digest into shared memory
    data = tl.load(data_ptr + batch_idx * n_elements + offsets)
    weights = tl.load(weights_ptr + batch_idx * n_elements + offsets)

    # Parallel sort (bitonic sort or radix sort)
    sorted_data, sorted_weights = parallel_sort(data, weights)

    # Sequential clustering (inherently sequential)
    # But parallelized across batch dimension
    cumulative_weight = 0.0
    sigma_mean = sorted_data[0]
    sigma_weight = sorted_weights[0]
    centroid_count = 0

    for i in range(1, n_elements):
        q = (cumulative_weight + sigma_weight + sorted_weights[i]) / total_weight
        q_limit = compute_q_limit(q, delta, n_elements)

        if q <= q_limit:
            # Merge into current centroid
            sigma_mean = (sigma_mean * sigma_weight +
                         sorted_data[i] * sorted_weights[i]) / \
                         (sigma_weight + sorted_weights[i])
            sigma_weight += sorted_weights[i]
        else:
            # Save current centroid
            tl.store(means_ptr + batch_idx * max_centroids + centroid_count,
                    sigma_mean)
            tl.store(cent_weights_ptr + batch_idx * max_centroids + centroid_count,
                    sigma_weight)
            centroid_count += 1

            # Start new centroid
            cumulative_weight += sigma_weight
            sigma_mean = sorted_data[i]
            sigma_weight = sorted_weights[i]

    # Store final centroid
    tl.store(means_ptr + batch_idx * max_centroids + centroid_count, sigma_mean)
    tl.store(cent_weights_ptr + batch_idx * max_centroids + centroid_count,
            sigma_weight)
    tl.store(counts_ptr + batch_idx, centroid_count + 1)
```

### 3. PyTorch High-Level API

```python
# bindings/python/tdigest_rs/cuda_backend.py

import torch
import triton
import triton.language as tl

class TDigestGPU:
    """GPU-accelerated T-Digest for batch processing."""

    @staticmethod
    def batch_from_arrays(
        arrays: list[np.ndarray],
        delta: float = 0.01,
        device: str = 'cuda'
    ) -> list[TDigest]:
        """Create multiple T-Digests on GPU in parallel."""

        # Convert to tensor
        batch_size = len(arrays)
        max_len = max(len(arr) for arr in arrays)

        # Pad arrays to same length
        data = torch.zeros(batch_size, max_len, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)

        for i, arr in enumerate(arrays):
            data[i, :len(arr)] = torch.tensor(arr, device=device)
            lengths[i] = len(arr)

        # Launch kernel
        max_centroids = int(max_len / 100) + 100  # Conservative estimate
        means = torch.zeros(batch_size, max_centroids, device=device)
        weights = torch.zeros(batch_size, max_centroids, device=device)
        counts = torch.zeros(batch_size, dtype=torch.int32, device=device)

        grid = (batch_size,)
        tdigest_clustering_kernel[grid](
            data, lengths,
            means, weights, counts,
            batch_size, max_len, delta,
            BLOCK_SIZE=256
        )

        # Convert back to CPU TDigest objects
        results = []
        for i in range(batch_size):
            n = counts[i].item()
            digest = TDigest(
                means=means[i, :n].cpu().numpy(),
                weights=weights[i, :n].cpu().numpy()
            )
            results.append(digest)

        return results

    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
```

### 4. Automatic Dispatch

```python
# bindings/python/tdigest_rs/lib.py

class TDigest:
    @classmethod
    def batch_from_arrays(
        cls,
        arrays: list[np.ndarray],
        delta: float = 0.01,
        use_gpu: bool = 'auto'
    ) -> list["TDigest"]:
        """
        Create multiple digests in parallel.
        Automatically uses GPU if available and beneficial.
        """

        # Auto-detect best backend
        if use_gpu == 'auto':
            use_gpu = (
                TDigestGPU.is_available() and
                len(arrays) >= 100  # GPU overhead only worth it for large batches
            )

        if use_gpu:
            return TDigestGPU.batch_from_arrays(arrays, delta, device='cuda')
        else:
            # Fall back to Rust CPU implementation
            return cls._batch_from_arrays_cpu(arrays, delta)
```

## Development Workflow (MacBook)

### 1. Initial Development (CPU)
```python
# Develop and test on CPU first
device = 'cpu'  # Works on any machine
digests = TDigestGPU.batch_from_arrays(arrays, delta, device=device)
```

### 2. Local GPU Testing (Metal)
```python
# Test on MacBook GPU using Metal backend
if torch.backends.mps.is_available():
    device = 'mps'  # Apple Metal
    digests = TDigestGPU.batch_from_arrays(arrays, delta, device=device)
```

### 3. Production Deployment (CUDA)
```python
# Automatically uses CUDA in production
device = 'cuda'  # NVIDIA GPU
digests = TDigestGPU.batch_from_arrays(arrays, delta, device=device)
```

## Performance Analysis

### Theoretical Limits

**CPU (Current):**
- Memory bandwidth: ~22 GB/s (M1/M2 MacBook)
- Throughput: 55 M elements/sec
- Parallel efficiency: 47% (11 cores)

**GPU (H100):**
- Memory bandwidth: 3,000 GB/s (136x higher)
- Compute: 60 TFLOPS FP64 (4000x higher than CPU core)
- Parallel units: 16,896 CUDA cores

**GPU (A100):**
- Memory bandwidth: 2,000 GB/s (91x higher)
- Compute: 19.5 TFLOPS FP64
- Parallel units: 6,912 CUDA cores

### Expected Speedup

| Scenario | Batch Size | CPU Time | GPU Time (est) | Speedup |
|----------|------------|----------|----------------|---------|
| Small | 100 digests | 20ms | 5ms | 4x |
| Medium | 1,000 digests | 200ms | 10ms | 20x |
| Large | 10,000 digests | 2,000ms | 50ms | 40x |
| Huge | 100,000 digests | 20,000ms | 200ms | 100x |

**Key factors:**
1. **GPU overhead:** ~2-5ms for data transfer and kernel launch
2. **Sweet spot:** 1,000-100,000 digests per batch
3. **Memory bound:** Limited by bandwidth, not compute
4. **Scaling:** Linear with batch size (GPU has 10,000+ parallel units)

### Speedup Formula

```
Speedup = (CPU_time / GPU_time)
        ≈ (batch_size × element_size × time_per_element_cpu) /
          (GPU_overhead + (batch_size × element_size × time_per_element_gpu))

For batch_size = 10,000, element_size = 32,000:
  CPU: 10,000 × 32,000 / 55M = ~5.8 seconds
  GPU: 5ms + (10,000 × 32,000 / 2000M) = 5ms + 160ms = 165ms
  Speedup: 5,800ms / 165ms ≈ 35x
```

## Bottleneck Analysis

### What GPU Parallelizes Well:
✅ **Sorting:** Bitonic/radix sort parallelizes perfectly (10-100x)
✅ **Batch processing:** Process 10,000 digests simultaneously
✅ **Memory bandwidth:** 100x higher than CPU
✅ **Weight conversions:** Parallel across all elements

### What Doesn't Parallelize:
❌ **Sequential clustering loop:** Inherently sequential (each step depends on previous)
❌ **Small batches:** GPU overhead dominates for <100 digests

### Optimization Strategies:

1. **Maximize parallelism across batch dimension**
   - Process 10,000+ digests simultaneously
   - Each GPU block handles one digest

2. **Optimize sorting**
   - Use CUB library's fast radix sort
   - Or implement bitonic sort in Triton
   - Sorting can be 50-100x faster on GPU

3. **Minimize data transfer**
   - Keep data on GPU between operations
   - Only transfer final centroids back to CPU

4. **Pipeline operations**
   - Overlap compute and data transfer
   - Use CUDA streams for async operations

## Implementation Plan

### Phase 1: Prototype (2-3 days)
```python
# Simple PyTorch implementation (no custom kernels)
def tdigest_batch_gpu_simple(data, delta):
    # Sort on GPU (built-in)
    sorted_data, indices = torch.sort(data, dim=1)

    # Clustering in PyTorch (not optimal but works)
    # Use vectorized operations where possible
    ...
```

### Phase 2: Triton Kernels (1 week)
```python
# Custom Triton kernels for clustering
@triton.jit
def parallel_sort_kernel(...):
    # Optimized sorting

@triton.jit
def clustering_kernel(...):
    # Custom clustering logic
```

### Phase 3: Optimization (1 week)
- Profile with NSight Systems
- Optimize memory access patterns
- Tune block sizes and grid dimensions
- Implement CUDA streams

### Phase 4: Production (3-5 days)
- Error handling
- Comprehensive testing
- Documentation
- CI/CD with GPU runners

## File Structure

```
tdigest-rs/
├── bindings/python/
│   └── tdigest_rs/
│       ├── lib.py              # Main API with auto-dispatch
│       ├── cuda_backend.py     # GPU implementation
│       ├── kernels/
│       │   ├── sort.py         # Triton sorting kernels
│       │   ├── cluster.py      # Triton clustering kernels
│       │   └── utils.py        # Helper functions
│       └── tests/
│           ├── test_gpu.py     # GPU-specific tests
│           └── test_parity.py  # CPU/GPU result comparison
```

## Dependencies

```toml
# Cargo.toml - no changes needed for Rust core

# pyproject.toml or requirements.txt
torch >= 2.0.0          # PyTorch with CUDA support
triton >= 2.1.0         # For custom kernels
numpy >= 1.24.0
pytest >= 7.0.0
pytest-benchmark >= 4.0.0
```

## Development Requirements

### For MacBook Development:
```bash
# Install PyTorch with CPU/MPS support
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Triton
pip install triton

# Develop and test on CPU
pytest tests/test_gpu.py --device=cpu
```

### For CUDA Deployment:
```bash
# Install PyTorch with CUDA 12.x support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Triton automatically supports CUDA
# No additional configuration needed
```

## Testing Strategy

### 1. Correctness Tests
```python
def test_gpu_cpu_parity():
    """Ensure GPU and CPU produce identical results."""
    arrays = [np.random.randn(32000) for _ in range(100)]

    cpu_results = TDigest.batch_from_arrays(arrays, use_gpu=False)
    gpu_results = TDigest.batch_from_arrays(arrays, use_gpu=True)

    for cpu, gpu in zip(cpu_results, gpu_results):
        np.testing.assert_allclose(cpu.means, gpu.means, rtol=1e-6)
        np.testing.assert_equal(cpu.weights, gpu.weights)
```

### 2. Performance Tests
```python
@pytest.mark.benchmark
def test_gpu_speedup():
    """Measure GPU speedup vs CPU."""
    arrays = [np.random.randn(32000) for _ in range(10000)]

    # CPU baseline
    start = time.time()
    TDigest.batch_from_arrays(arrays, use_gpu=False)
    cpu_time = time.time() - start

    # GPU
    start = time.time()
    TDigest.batch_from_arrays(arrays, use_gpu=True)
    gpu_time = time.time() - start

    speedup = cpu_time / gpu_time
    print(f"GPU Speedup: {speedup:.1f}x")
    assert speedup > 10  # Expect at least 10x for large batches
```

### 3. Device Compatibility
```python
def test_all_devices():
    """Test on all available devices."""
    arrays = [np.random.randn(1000) for _ in range(10)]

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')

    for device in devices:
        results = TDigestGPU.batch_from_arrays(arrays, device=device)
        assert len(results) == 10
```

## Cost-Benefit Analysis

### Development Cost:
- **Phase 1 (Prototype):** 2-3 days
- **Phase 2 (Triton):** 1 week
- **Phase 3 (Optimize):** 1 week
- **Phase 4 (Production):** 3-5 days
- **Total:** ~3-4 weeks

### Benefits:
- **20-100x speedup** for large batches (10,000+ digests)
- Enables real-time processing of massive datasets
- GPU hardware already available in data centers
- Works on MacBook for development (CPU/Metal)

### When It's Worth It:
✅ Processing >1,000 digests per batch regularly
✅ Working with streaming data at scale
✅ Have access to datacenter GPUs (A100/H100)
✅ Latency-critical applications

❌ Small batches (<100 digests)
❌ No GPU access
❌ Current CPU performance is sufficient

## Conclusion

**Recommended Approach:**
1. Use PyTorch + Triton for implementation
2. Develop on MacBook using CPU backend
3. Test on any GPU (Metal/CUDA) before deploying
4. Deploy to CUDA 12/13 without code changes

**Expected Results:**
- 20-40x speedup for typical workloads (10,000 digests × 32k elements)
- Up to 100x for very large batches (100,000+ digests)
- Zero development overhead for CUDA version compatibility
- Works on MacBook for entire development cycle

**Next Steps if You Want to Proceed:**
1. Create prototype with simple PyTorch implementation
2. Validate correctness vs Rust implementation
3. Benchmark on CPU to establish baseline
4. Implement custom Triton kernels
5. Optimize and deploy to CUDA GPUs
