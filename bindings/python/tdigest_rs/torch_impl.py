"""
PyTorch-based T-Digest implementation for GPU acceleration.

Provides a GPU-accelerated alternative to the Rust implementation.
Works on CPU for development/testing or GPU for production.

Requires PyTorch: pip install tdigest-rs[gpu]
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

# T-Digest algorithm constants from the paper
SCALE_BASE_MULTIPLIER = 4.0
SCALE_BASE_OFFSET = 24.0


def _check_torch_available():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required. Install with: pip install tdigest-rs[gpu]"
        )


@dataclass
class TDigestResult:
    """Result from batch T-Digest computation."""
    means: np.ndarray      # [batch_size, max_centroids]
    weights: np.ndarray    # [batch_size, max_centroids]
    counts: np.ndarray     # [batch_size]


class TDigestTorch:
    """
    PyTorch implementation of T-Digest for batch processing on GPU.

    Designed for processing large batches (1000s) of arrays.
    Falls back to CPU when GPU unavailable.
    """

    @staticmethod
    def _compute_q_limit(q: float, delta: float, n: int) -> float:
        """
        Compute clustering threshold matching Rust's log_q_limit function.
        """
        if q <= 0:
            return 0.0
        if q >= 1:
            return 1.0

        n_over_delta = n / delta
        base_factor = np.log(n_over_delta) * SCALE_BASE_MULTIPLIER + SCALE_BASE_OFFSET
        scale_factor = delta / base_factor

        k = scale_factor * np.log(q / (1.0 - q)) + 1.0
        q_limit = 1.0 / (1.0 + np.exp(-k * base_factor / delta))

        return q_limit

    @staticmethod
    def _cluster_single_digest(
        sorted_data: torch.Tensor,
        delta: float,
        max_centroids: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Core T-Digest clustering algorithm matching Rust's compute() function.

        Returns:
            means, weights (padded to max_centroids), actual count
        """
        n = len(sorted_data)
        device = sorted_data.device
        dtype = sorted_data.dtype

        means = torch.zeros(max_centroids, device=device, dtype=dtype)
        weights = torch.zeros(max_centroids, device=device, dtype=dtype)

        if n == 0:
            return means, weights, 0

        start = 0
        end = n
        while start < n and sorted_data[start] == float('-inf'):
            start += 1
        while end > start and sorted_data[end - 1] == float('inf'):
            end -= 1

        centroid_idx = 0

        if start > 0:
            means[centroid_idx] = float('-inf')
            weights[centroid_idx] = float(start)
            centroid_idx += 1

        inf_count = n - end

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
                    sigma_mean = (sigma_mean * sigma_weight + mu) / (sigma_weight + 1.0)
                    sigma_weight += 1.0
                else:
                    if centroid_idx >= max_centroids:
                        raise ValueError(
                            f"Exceeded max_centroids={max_centroids}. "
                            "Increase max_centroids or use larger delta."
                        )

                    means[centroid_idx] = sigma_mean
                    weights[centroid_idx] = sigma_weight
                    centroid_idx += 1

                    cumulative_weight += sigma_weight
                    q_limit = TDigestTorch._compute_q_limit(
                        cumulative_weight / total_weight, delta, slice_len
                    )
                    sigma_mean = mu
                    sigma_weight = 1.0

            if not np.isnan(sigma_mean):
                if centroid_idx >= max_centroids:
                    raise ValueError(
                        f"Exceeded max_centroids={max_centroids}. "
                        "Increase max_centroids or use larger delta."
                    )
                means[centroid_idx] = sigma_mean
                weights[centroid_idx] = sigma_weight
                centroid_idx += 1

        if inf_count > 0:
            if centroid_idx >= max_centroids:
                raise ValueError(
                    f"Exceeded max_centroids={max_centroids}. "
                    "Increase max_centroids or use larger delta."
                )
            means[centroid_idx] = float('inf')
            weights[centroid_idx] = float(inf_count)
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
            arrays: List of 1D numpy arrays (must all be same length)
            delta: Compression parameter (0 < delta <= 1)
            device: 'cpu', 'cuda', 'cuda:0', etc.
            max_centroids: Maximum centroids per digest

        Returns:
            TDigestResult with means, weights, and counts

        Raises:
            ImportError: If torch not installed
            ValueError: If arrays have different lengths or invalid delta
        """
        _check_torch_available()

        if not arrays:
            return TDigestResult(
                means=np.array([]),
                weights=np.array([]),
                counts=np.array([])
            )

        if delta <= 0 or delta > 1:
            raise ValueError(f"delta must be in (0, 1], got {delta}")

        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All arrays must have same length. Got lengths: {set(lengths)}"
            )

        batch_size = len(arrays)

        data = torch.tensor(
            np.array(arrays),
            dtype=torch.float64,
            device=device
        )

        sorted_data, _ = torch.sort(data, dim=1)

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

        means_batch = torch.stack(all_means)
        weights_batch = torch.stack(all_weights)
        counts_batch = torch.tensor(all_counts, dtype=torch.int64)

        return TDigestResult(
            means=means_batch.cpu().numpy(),
            weights=weights_batch.cpu().numpy(),
            counts=counts_batch.cpu().numpy()
        )

    @staticmethod
    def is_available() -> bool:
        """Check if CUDA GPU acceleration is available."""
        return TORCH_AVAILABLE and torch.cuda.is_available()

    @staticmethod
    def get_device() -> str:
        """Get the best available device: cuda > mps > cpu."""
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
    Convenience function with automatic device selection.

    Args:
        arrays: List of 1D numpy arrays
        delta: Compression parameter
        device: Device ('cpu', 'cuda', or None for auto)

    Returns:
        TDigestResult
    """
    _check_torch_available()

    if device is None:
        device = TDigestTorch.get_device()

    return TDigestTorch.batch_from_arrays(arrays, delta, device)
