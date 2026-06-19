"""Tests for PyTorch T-Digest implementation."""

import pytest
import numpy as np
import torch
from tdigest_rs import TDigest
from tdigest_rs.torch_impl import TDigestTorch, batch_from_arrays_torch, batch_from_tensor_torch


class TestTensorAPI:
    """Test the new tensor-based API."""

    def test_tensor_single(self):
        data = torch.randn(1, 1000, dtype=torch.float64)

        rust_digest = TDigest.from_array(data[0].numpy(), delta=0.01)
        torch_result = TDigestTorch.batch_from_tensor(data, delta=0.01)

        count = torch_result.counts[0]
        torch_means = torch_result.means[0, :count]
        torch_weights = torch_result.weights[0, :count]

        assert len(rust_digest.means) == count
        np.testing.assert_allclose(torch_means, rust_digest.means, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(torch_weights.astype(np.uint32), rust_digest.weights)

    def test_tensor_batch(self):
        torch.manual_seed(42)
        data = torch.randn(10, 5000, dtype=torch.float64)

        rust_digests = [TDigest.from_array(data[i].numpy(), delta=0.01) for i in range(10)]
        torch_result = TDigestTorch.batch_from_tensor(data, delta=0.01)

        for i, rust_digest in enumerate(rust_digests):
            count = torch_result.counts[i]
            assert len(rust_digest.means) == count
            np.testing.assert_allclose(
                torch_result.means[i, :count],
                rust_digest.means,
                rtol=1e-10,
                atol=1e-10
            )

    def test_tensor_device_inference(self):
        data_cpu = torch.randn(5, 1000, dtype=torch.float64)
        result_cpu = TDigestTorch.batch_from_tensor(data_cpu, delta=0.01)
        assert result_cpu.counts.shape == (5,)

        if torch.cuda.is_available():
            data_gpu = data_cpu.cuda()
            result_gpu = TDigestTorch.batch_from_tensor(data_gpu, delta=0.01)
            np.testing.assert_array_equal(result_cpu.counts, result_gpu.counts)

    def test_tensor_convenience_function(self):
        data = torch.randn(5, 1000, dtype=torch.float64)
        result = batch_from_tensor_torch(data, delta=0.01)
        assert len(result.counts) == 5

    def test_tensor_invalid_shape(self):
        data_1d = torch.randn(1000, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be 2D"):
            TDigestTorch.batch_from_tensor(data_1d, delta=0.01)

        data_3d = torch.randn(10, 100, 100, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be 2D"):
            TDigestTorch.batch_from_tensor(data_3d, delta=0.01)


class TestCorrectness:
    """Verify numerical equivalence with Rust implementation."""

    def test_single_array(self):
        data = np.random.randn(1000).astype(np.float64)

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        torch_means = torch_result.means[0, :count]
        torch_weights = torch_result.weights[0, :count]

        assert len(rust_digest.means) == count
        np.testing.assert_allclose(torch_means, rust_digest.means, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(torch_weights.astype(np.uint32), rust_digest.weights)

    def test_batch(self):
        np.random.seed(42)
        arrays = [np.random.randn(5000).astype(np.float64) for _ in range(10)]

        rust_digests = [TDigest.from_array(arr, delta=0.01) for arr in arrays]
        torch_result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')

        for i, rust_digest in enumerate(rust_digests):
            count = torch_result.counts[i]
            assert len(rust_digest.means) == count

            np.testing.assert_allclose(
                torch_result.means[i, :count],
                rust_digest.means,
                rtol=1e-10,
                atol=1e-10
            )
            np.testing.assert_array_equal(
                torch_result.weights[i, :count].astype(np.uint32),
                rust_digest.weights
            )

    def test_quantiles(self):
        np.random.seed(123)
        data = np.random.randn(10000).astype(np.float64)

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        torch_digest = TDigest.from_means_weights(
            torch_result.means[0, :count],
            torch_result.weights[0, :count].astype(np.uint32),
            delta=0.01
        )

        for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            np.testing.assert_allclose(
                torch_digest.quantile(q),
                rust_digest.quantile(q),
                rtol=1e-10,
                atol=1e-10
            )

    def test_sorted_data(self):
        data = np.sort(np.random.randn(5000).astype(np.float64))

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        np.testing.assert_allclose(
            torch_result.means[0, :count],
            rust_digest.means,
            rtol=1e-10,
            atol=1e-10
        )

    @pytest.mark.parametrize("distribution", ["uniform", "exponential"])
    def test_distributions(self, distribution):
        if distribution == "uniform":
            data = np.random.uniform(0, 100, 10000).astype(np.float64)
        else:
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

    def test_empty_batch(self):
        result = TDigestTorch.batch_from_arrays([], delta=0.01, device='cpu')
        assert len(result.means) == 0
        assert len(result.weights) == 0
        assert len(result.counts) == 0

    def test_single_element(self):
        data = np.array([42.0])

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        assert torch_result.counts[0] == 1
        assert torch_result.means[0, 0] == 42.0
        assert torch_result.weights[0, 0] == 1.0

    def test_identical_values(self):
        data = np.full(1000, 42.0)

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        assert count == len(rust_digest.means)

    def test_infinities(self):
        data = np.array([-np.inf, -100, 0, 100, np.inf, -np.inf, np.inf])

        rust_digest = TDigest.from_array(data, delta=0.01)
        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        torch_means = torch_result.means[0, :count]

        assert torch_means[0] == float('-inf')
        assert torch_means[-1] == float('inf')
        np.testing.assert_allclose(torch_means, rust_digest.means, rtol=1e-10, atol=1e-10)

    def test_nans(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])

        torch_result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu')

        count = torch_result.counts[0]
        torch_means = torch_result.means[0, :count]
        assert not np.any(np.isnan(torch_means))


class TestValidation:
    """Test input validation and error handling."""

    def test_invalid_delta_zero(self):
        arrays = [np.array([1.0, 2.0, 3.0])]
        with pytest.raises(ValueError, match="delta must be in"):
            TDigestTorch.batch_from_arrays(arrays, delta=0.0, device='cpu')

    def test_invalid_delta_negative(self):
        arrays = [np.array([1.0, 2.0, 3.0])]
        with pytest.raises(ValueError, match="delta must be in"):
            TDigestTorch.batch_from_arrays(arrays, delta=-0.1, device='cpu')

    def test_invalid_delta_too_large(self):
        arrays = [np.array([1.0, 2.0, 3.0])]
        with pytest.raises(ValueError, match="delta must be in"):
            TDigestTorch.batch_from_arrays(arrays, delta=1.1, device='cpu')

    def test_mixed_length_arrays(self):
        arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0, 4.0])
        ]
        with pytest.raises(ValueError, match="same length"):
            TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')

    def test_max_centroids_parameter(self):
        data = np.random.randn(1000).astype(np.float64)
        result = TDigestTorch.batch_from_arrays([data], delta=0.01, device='cpu', max_centroids=1000)
        assert result.counts[0] <= 1000


class TestSizes:
    """Test with various array and batch sizes."""

    @pytest.mark.parametrize("size", [10, 100, 1000, 10000, 32000])
    def test_array_sizes(self, size):
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
    def test_batch_sizes(self, batch_size):
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


class TestDeltas:
    """Test with various delta values."""

    @pytest.mark.parametrize("delta", [0.001, 0.01, 0.05, 0.1, 0.5])
    def test_delta_values(self, delta):
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


class TestUtilities:
    """Test convenience functions and utilities."""

    def test_convenience_function(self):
        np.random.seed(42)
        arrays = [np.random.randn(1000).astype(np.float64) for _ in range(5)]

        result = batch_from_arrays_torch(arrays, delta=0.01, device='cpu')

        assert len(result.counts) == 5
        assert result.means.shape[0] == 5
        assert result.weights.shape[0] == 5

    def test_device_detection(self):
        device = TDigestTorch.get_device()
        assert device in ['cpu', 'cuda', 'mps']

        _ = TDigestTorch.is_available()


@pytest.mark.skipif(
    not TDigestTorch.is_available(),
    reason="CUDA not available"
)
class TestGPU:
    """GPU-specific tests (skipped without CUDA)."""

    def test_cpu_gpu_equivalence(self):
        np.random.seed(42)
        arrays = [np.random.randn(5000).astype(np.float64) for _ in range(10)]

        cpu_result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')
        gpu_result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cuda')

        np.testing.assert_array_equal(cpu_result.counts, gpu_result.counts)

        for i in range(len(arrays)):
            count = cpu_result.counts[i]
            np.testing.assert_allclose(
                cpu_result.means[i, :count],
                gpu_result.means[i, :count],
                rtol=1e-10,
                atol=1e-10
            )

    def test_large_batch(self):
        np.random.seed(42)
        arrays = [np.random.randn(1000).astype(np.float64) for _ in range(1000)]

        result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cuda')

        assert len(result.counts) == 1000
        assert all(count > 0 for count in result.counts)
