"""
PyTorch-based T-Digest implementation for GPU acceleration.

This module provides a GPU-accelerated alternative to the Rust implementation.
Can run on CPU for development/testing or GPU for production.

Note: This module requires PyTorch. Install with:
    pip install tdigest-rs[gpu]
    or
    uv pip install tdigest-rs[gpu]
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass


def _check_torch_available():
    """Raise helpful error if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch backend requires torch to be installed. "
            "Install with: pip install tdigest-rs[gpu] or uv pip install tdigest-rs[gpu]"
        )


@dataclass
class TDigestResult:
    """Result from batch T-Digest computation."""
    means: np.ndarray      # [batch_size, max_centroids]
    weights: np.ndarray    # [batch_size, max_centroids]
    counts: np.ndarray     # [batch_size] - number of valid centroids per digest


class TDigestTorch:
    """
    PyTorch implementation of T-Digest for batch processing.

    Designed for processing large batches (1000s) of arrays on GPU.
    Falls back to CPU when GPU unavailable.
    """

    @staticmethod
    def _compute_q_limit(q: float, delta: float, n: int) -> float:
        """
        Compute the q_limit threshold for clustering.

        This matches the scale function from the Rust implementation:
        log_q_limit(q, delta, n) via log_scale and inverse_log_scale.
        """
        if q <= 0 or q >= 1:
            return 0.0 if q <= 0 else 1.0

        n_over_delta = n / delta
        base_factor = np.log(n_over_delta) * 4.0 + 24.0
        scale_factor = delta / base_factor

        k = scale_factor * np.log(q / (1.0 - q)) + 1.0
        q_limit = 1.0 / (1.0 + np.exp(-k * base_factor / delta))

        return q_limit

    @staticmethod
    def _cluster_single_digest(
        sorted_data: torch.Tensor,
        delta: float,
        max_centroids: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Cluster a single sorted array into centroids.

        This is the core T-Digest algorithm, inherently sequential.
        Matches the compute() function from Rust core.rs.

        Returns:
            means: [max_centroids] padded tensor
            weights: [max_centroids] padded tensor
            count: actual number of centroids
        """
        n = len(sorted_data)
        device = sorted_data.device
        dtype = sorted_data.dtype

        means = torch.zeros(max_centroids, device=device, dtype=dtype)
        weights = torch.zeros(max_centroids, device=device, dtype=dtype)

        if n == 0:
            return means, weights, 0

        # Handle infinities
        start = 0
        end = n
        while start < n and sorted_data[start] == float('-inf'):
            start += 1
        while end > start and sorted_data[end - 1] == float('inf'):
            end -= 1

        centroid_idx = 0

        if start > 0:
            # Aggregate all negative infinities
            means[centroid_idx] = float('-inf')
            weights[centroid_idx] = float(start)
            centroid_idx += 1

        inf_weight = float(n - end)

        # Main clustering
        if end > start:
            data_slice = sorted_data[start:end]
            slice_len = end - start
            total_weight = float(slice_len)

            cumulative_weight = 0.0
            sigma_mean = data_slice[0].item()
            sigma_weight = 1.0

            q_limit = TDigestTorch._compute_q_limit(0.0, delta, slice_len)

            for i in range(1, slice_len):
                mu = data_slice[i].item()

                if np.isnan(mu):
                    continue

                q = (cumulative_weight + sigma_weight + 1.0) / total_weight

                if q <= q_limit:
                    # Merge into current centroid
                    sigma_mean = (sigma_mean * sigma_weight + mu) / (sigma_weight + 1.0)
                    sigma_weight += 1.0
                else:
                    # Save current centroid
                    means[centroid_idx] = sigma_mean
                    weights[centroid_idx] = sigma_weight
                    centroid_idx += 1

                    # Start new centroid
                    cumulative_weight += sigma_weight
                    q_limit = TDigestTorch._compute_q_limit(
                        cumulative_weight / total_weight, delta, slice_len
                    )
                    sigma_mean = mu
                    sigma_weight = 1.0

            # Save final centroid
            if not np.isnan(sigma_mean):
                means[centroid_idx] = sigma_mean
                weights[centroid_idx] = sigma_weight
                centroid_idx += 1

        # Handle positive infinity
        if inf_weight > 0:
            means[centroid_idx] = float('inf')
            weights[centroid_idx] = inf_weight
            centroid_idx += 1

        return means, weights, centroid_idx

    @classmethod
    def batch_from_arrays(
        cls,
        arrays: List[np.ndarray],
        delta: float = 0.01,
        device: str = 'cpu',
        max_centroids: int = 500
    ) -> TDigestResult:
        """
        Create T-Digests for a batch of arrays.

        Args:
            arrays: List of 1D numpy arrays to digest
            delta: Compression parameter (default: 0.01)
            device: 'cpu', 'cuda', or 'cuda:0', etc.
            max_centroids: Maximum centroids per digest (default: 500)

        Returns:
            TDigestResult with means, weights, and counts for each digest

        Raises:
            ImportError: If torch is not installed
        """
        _check_torch_available()

        if not arrays:
            return TDigestResult(
                means=np.array([]),
                weights=np.array([]),
                counts=np.array([])
            )

        batch_size = len(arrays)

        # Convert to tensor and move to device
        # Use float64 for numerical precision to match Rust
        data = torch.tensor(
            np.array(arrays),
            dtype=torch.float64,
            device=device
        )

        # Sort each array
        sorted_data, _ = torch.sort(data, dim=1)

        # Process each digest independently
        # This loop is sequential across batch but each iteration is independent
        # On GPU, we'd parallelize this, but for CPU it's fine
        all_means = []
        all_weights = []
        all_counts = []

        for i in range(batch_size):
            means, weights, count = cls._cluster_single_digest(
                sorted_data[i],
                delta,
                max_centroids
            )
            all_means.append(means)
            all_weights.append(weights)
            all_counts.append(count)

        # Stack results
        means_batch = torch.stack(all_means)
        weights_batch = torch.stack(all_weights)
        counts_batch = torch.tensor(all_counts, dtype=torch.int32)

        # Convert back to numpy
        return TDigestResult(
            means=means_batch.cpu().numpy(),
            weights=weights_batch.cpu().numpy(),
            counts=counts_batch.cpu().numpy()
        )

    @staticmethod
    def is_available() -> bool:
        """Check if GPU acceleration is available."""
        if not TORCH_AVAILABLE:
            return False
        return torch.cuda.is_available()

    @staticmethod
    def get_device() -> str:
        """Get the best available device."""
        if not TORCH_AVAILABLE:
            return 'cpu'

        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'


def batch_from_arrays_torch(
    arrays: List[np.ndarray],
    delta: float = 0.01,
    device: Optional[str] = None
) -> TDigestResult:
    """
    Convenience function for batch T-Digest creation with PyTorch.

    Args:
        arrays: List of 1D numpy arrays
        delta: Compression parameter
        device: Device to use ('cpu', 'cuda', or None for auto)

    Returns:
        TDigestResult with means, weights, and counts

    Raises:
        ImportError: If torch is not installed
    """
    _check_torch_available()

    if device is None:
        device = TDigestTorch.get_device()

    return TDigestTorch.batch_from_arrays(arrays, delta, device)
