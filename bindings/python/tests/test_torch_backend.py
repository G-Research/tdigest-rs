"""
Tests for PyTorch-based T-Digest backend.

Verifies numerical equivalence with Rust implementation.
"""

import pytest
import numpy as np
from tdigest_rs import TDigest
from tdigest_rs.torch_backend import TDigestTorch, batch_from_arrays_torch


class TestTorchBackendCorrectness:
    """Test that PyTorch backend produces identical results to Rust."""

    def test_single_array_basic(self):
        """Test single array produces correct centroids."""
        data = np.random.randn(1000).astype(np.float64)

        # Rust version
        rust_digest = TDigest.from_array(data, delta=0.01)

        # PyTorch version
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        torch_means = torch_result.means[0, :count]
        torch_weights = torch_result.weights[0, :count]

        # Should have same number of centroids
        assert len(rust_digest.means) == count
        assert len(rust_digest.weights) == count

        # Means and weights should match exactly
        np.testing.assert_allclose(torch_means, rust_digest.means, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(torch_weights.astype(np.uint32), rust_digest.weights)

    def test_batch_equivalence(self):
        """Test batch processing matches individual processing."""
        np.random.seed(42)
        arrays = [np.random.randn(5000).astype(np.float64) for _ in range(10)]
        delta = 0.01

        # Rust individual
        rust_digests = [TDigest.from_array(arr, delta=delta) for arr in arrays]

        # PyTorch batch
        torch_result = TDigestTorch.batch_from_arrays(arrays, delta=delta, device='cpu')

        # Compare each digest
        for i, rust_digest in enumerate(rust_digests):
            count = torch_result.counts[i]
            torch_means = torch_result.means[i, :count]
            torch_weights = torch_result.weights[i, :count]

            assert len(rust_digest.means) == count, f"Digest {i}: centroid count mismatch"

            np.testing.assert_allclose(
                torch_means,
                rust_digest.means,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Digest {i}: means mismatch"
            )

            np.testing.assert_array_equal(
                torch_weights.astype(np.uint32),
                rust_digest.weights,
                err_msg=f"Digest {i}: weights mismatch"
            )

    def test_quantile_equivalence(self):
        """Test that quantiles computed from PyTorch digests match Rust."""
        np.random.seed(123)
        data = np.random.randn(10000).astype(np.float64)
        delta = 0.01

        # Create digests
        rust_digest = TDigest.from_array(data, delta=delta)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=delta, device='cpu')

        # Extract PyTorch digest
        count = torch_result.counts[0]
        torch_means = torch_result.means[0, :count]
        torch_weights = torch_result.weights[0, :count].astype(np.uint32)

        # Create a TDigest from PyTorch results for quantile computation
        torch_digest = TDigest.from_means_weights(torch_means, torch_weights, delta=delta)

        # Test various quantiles
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for q in quantiles:
            rust_q = rust_digest.quantile(q)
            torch_q = torch_digest.quantile(q)
            np.testing.assert_allclose(
                torch_q,
                rust_q,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Quantile {q} mismatch"
            )

    def test_sorted_data(self):
        """Test with already sorted data."""
        data = np.sort(np.random.randn(5000).astype(np.float64))

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        torch_means = torch_result.means[0, :count]
        torch_weights = torch_result.weights[0, :count]

        np.testing.assert_allclose(torch_means, rust_digest.means, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(torch_weights.astype(np.uint32), rust_digest.weights)

    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        data = np.random.uniform(0, 100, 10000).astype(np.float64)

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        assert len(rust_digest.means) == count
        np.testing.assert_allclose(
            torch_result.means[0, :count],
            rust_digest.means,
            rtol=1e-10,
            atol=1e-10
        )

    def test_exponential_distribution(self):
        """Test with exponential distribution."""
        data = np.random.exponential(2.0, 10000).astype(np.float64)

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        np.testing.assert_allclose(
            torch_result.means[0, :count],
            rust_digest.means,
            rtol=1e-10,
            atol=1e-10
        )


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_empty_array(self):
        """Test with empty array."""
        result = TDigestTorch.batch_from_arrays([], delta=0.01, device='cpu')
        assert len(result.means) == 0
        assert len(result.weights) == 0
        assert len(result.counts) == 0

    def test_single_element(self):
        """Test with single element."""
        data = np.array([42.0])

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        assert count == 1
        assert torch_result.means[0, 0] == 42.0
        assert torch_result.weights[0, 0] == 1.0

    def test_identical_values(self):
        """Test with all identical values."""
        data = np.full(1000, 42.0)

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        # Should compress to single centroid
        assert count == len(rust_digest.means)
        np.testing.assert_allclose(
            torch_result.means[0, :count],
            rust_digest.means,
            rtol=1e-10,
            atol=1e-10
        )

    def test_with_infinities(self):
        """Test handling of positive and negative infinities."""
        data = np.array([-np.inf, -100, 0, 100, np.inf, -np.inf, np.inf])

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        assert count == len(rust_digest.means)

        # Check that infinities are handled correctly
        torch_means = torch_result.means[0, :count]
        assert torch_means[0] == float('-inf')
        assert torch_means[-1] == float('inf')

        np.testing.assert_allclose(
            torch_means,
            rust_digest.means,
            rtol=1e-10,
            atol=1e-10
        )

    def test_with_nans(self):
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        # NaNs should be filtered out
        torch_means = torch_result.means[0, :count]
        assert not np.any(np.isnan(torch_means))


class TestDifferentSizes:
    """Test with various array sizes."""

    @pytest.mark.parametrize("size", [10, 100, 1000, 10000, 32000])
    def test_various_sizes(self, size):
        """Test with different array sizes."""
        np.random.seed(42)
        data = np.random.randn(size).astype(np.float64)

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        assert count == len(rust_digest.means)

        np.testing.assert_allclose(
            torch_result.means[0, :count],
            rust_digest.means,
            rtol=1e-10,
            atol=1e-10
        )

    @pytest.mark.parametrize("batch_size", [1, 10, 100])
    def test_various_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        np.random.seed(42)
        arrays = [np.random.randn(1000).astype(np.float64) for _ in range(batch_size)]

        rust_digests = [TDigest.from_array(arr, delta=0.01) for arr in arrays]
        torch_result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')

        assert len(torch_result.counts) == batch_size

        for i in range(batch_size):
            count = torch_result.counts[i]
            assert count == len(rust_digests[i].means)

            np.testing.assert_allclose(
                torch_result.means[i, :count],
                rust_digests[i].means,
                rtol=1e-10,
                atol=1e-10
            )


class TestDifferentDeltas:
    """Test with various delta values."""

    @pytest.mark.parametrize("delta", [0.001, 0.01, 0.05, 0.1, 0.5])
    def test_various_deltas(self, delta):
        """Test with different delta (compression) values."""
        np.random.seed(42)
        data = np.random.randn(10000).astype(np.float64)

        rust_digest = TDigest.from_array(data, delta=delta)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=delta, device='cpu')

        count = torch_result.counts[0]
        assert count == len(rust_digest.means)

        np.testing.assert_allclose(
            torch_result.means[0, :count],
            rust_digest.means,
            rtol=1e-10,
            atol=1e-10
        )


class TestConvenienceFunctions:
    """Test convenience functions and utilities."""

    def test_convenience_function(self):
        """Test batch_from_arrays_torch convenience function."""
        np.random.seed(42)
        arrays = [np.random.randn(1000).astype(np.float64) for _ in range(5)]

        result = batch_from_arrays_torch(arrays, delta=0.01, device='cpu')

        assert len(result.counts) == 5
        assert result.means.shape[0] == 5
        assert result.weights.shape[0] == 5

    def test_device_detection(self):
        """Test device detection functions."""
        device = TDigestTorch.get_device()
        assert device in ['cpu', 'cuda', 'mps']

        # is_available should not crash
        _ = TDigestTorch.is_available()


@pytest.mark.skipif(
    not TDigestTorch.is_available(),
    reason="CUDA not available"
)
class TestGPUExecution:
    """Tests that require GPU. Skipped if CUDA not available."""

    def test_gpu_equivalence(self):
        """Test that GPU execution matches CPU."""
        np.random.seed(42)
        arrays = [np.random.randn(5000).astype(np.float64) for _ in range(10)]

        cpu_result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')
        gpu_result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cuda')

        # Results should be identical
        np.testing.assert_array_equal(cpu_result.counts, gpu_result.counts)

        for i in range(len(arrays)):
            count = cpu_result.counts[i]
            np.testing.assert_allclose(
                cpu_result.means[i, :count],
                gpu_result.means[i, :count],
                rtol=1e-10,
                atol=1e-10
            )
            np.testing.assert_allclose(
                cpu_result.weights[i, :count],
                gpu_result.weights[i, :count],
                rtol=1e-10,
                atol=1e-10
            )

    def test_large_batch_gpu(self):
        """Test large batch on GPU."""
        np.random.seed(42)
        arrays = [np.random.randn(1000).astype(np.float64) for _ in range(1000)]

        result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cuda')

        assert len(result.counts) == 1000
        assert all(count > 0 for count in result.counts)
