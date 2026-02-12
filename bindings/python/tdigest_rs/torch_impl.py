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
import math
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
        (Scalar version for CPU path)
        """
        if q <= 0:
            return 0.0
        if q >= 1:
            return 1.0

        n_over_delta = n / delta
        base_factor = math.log(n_over_delta) * SCALE_BASE_MULTIPLIER + SCALE_BASE_OFFSET
        scale_factor = delta / base_factor

        k = scale_factor * math.log(q / (1.0 - q)) + 1.0
        q_limit = 1.0 / (1.0 + math.exp(-k * base_factor / delta))

        return q_limit

    @staticmethod
    def _compute_q_limit_torch(q: torch.Tensor, delta: float, n: int) -> torch.Tensor:
        """
        Compute clustering threshold using torch operations (for GPU).
        """
        if q <= 0:
            return torch.tensor(0.0, dtype=q.dtype, device=q.device)
        if q >= 1:
            return torch.tensor(1.0, dtype=q.dtype, device=q.device)

        n_over_delta = n / delta
        base_factor = torch.log(torch.tensor(n_over_delta, dtype=q.dtype, device=q.device)) * SCALE_BASE_MULTIPLIER + SCALE_BASE_OFFSET
        scale_factor = delta / base_factor

        k = scale_factor * torch.log(q / (1.0 - q)) + 1.0
        q_limit = 1.0 / (1.0 + torch.exp(-k * base_factor / delta))

        return q_limit

    @staticmethod
    @torch.compile
    def _cluster_single_digest_compiled(
        sorted_data: torch.Tensor,
        delta: float,
        max_centroids: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPU-optimized clustering using torch.compile.
        Keeps all data on GPU, uses pure torch operations.

        Returns:
            means, weights (padded to max_centroids), count (as tensor)
        """
        n = sorted_data.shape[0]
        device = sorted_data.device
        dtype = sorted_data.dtype

        means = torch.zeros(max_centroids, device=device, dtype=dtype)
        weights = torch.zeros(max_centroids, device=device, dtype=dtype)

        if n == 0:
            return means, weights, torch.tensor(0, device=device, dtype=torch.int64)

        # Find infinity boundaries using vectorized operations
        is_neg_inf = sorted_data == float('-inf')
        is_pos_inf = sorted_data == float('inf')

        start = int(is_neg_inf.sum()) if is_neg_inf.any() else 0
        end = n - int(is_pos_inf.sum()) if is_pos_inf.any() else n

        centroid_idx = 0

        # Handle negative infinities
        if start > 0:
            means[centroid_idx] = float('-inf')
            weights[centroid_idx] = float(start)
            centroid_idx += 1

        inf_count = n - end

        # Process finite values
        if end > start:
            data_slice = sorted_data[start:end]
            slice_len = end - start
            total_weight = float(slice_len)

            cumulative_weight = torch.tensor(0.0, dtype=dtype, device=device)
            sigma_mean = data_slice[0]
            sigma_weight = torch.tensor(1.0, dtype=dtype, device=device)

            q = torch.tensor(0.0, dtype=dtype, device=device)
            q_limit = TDigestTorch._compute_q_limit_torch(q, delta, slice_len)

            for i in range(1, slice_len):
                mu = data_slice[i]

                # Skip NaN values
                if torch.isnan(mu):
                    continue

                q = (cumulative_weight + sigma_weight + 1.0) / total_weight

                if q <= q_limit:
                    # Merge with current centroid
                    sigma_mean = (sigma_mean * sigma_weight + mu) / (sigma_weight + 1.0)
                    sigma_weight = sigma_weight + 1.0
                else:
                    # Create new centroid
                    means[centroid_idx] = sigma_mean
                    weights[centroid_idx] = sigma_weight
                    centroid_idx += 1

                    cumulative_weight = cumulative_weight + sigma_weight
                    q_limit = TDigestTorch._compute_q_limit_torch(
                        cumulative_weight / total_weight, delta, slice_len
                    )
                    sigma_mean = mu
                    sigma_weight = torch.tensor(1.0, dtype=dtype, device=device)

            # Flush final centroid
            if not torch.isnan(sigma_mean):
                means[centroid_idx] = sigma_mean
                weights[centroid_idx] = sigma_weight
                centroid_idx += 1

        # Handle positive infinities
        if inf_count > 0:
            means[centroid_idx] = float('inf')
            weights[centroid_idx] = float(inf_count)
            centroid_idx += 1

        return means, weights, torch.tensor(centroid_idx, device=device, dtype=torch.int64)

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

        is_neg_inf = sorted_data == float('-inf')
        is_pos_inf = sorted_data == float('inf')

        start = int(is_neg_inf.sum().item()) if is_neg_inf.any().item() else 0
        end = n - int(is_pos_inf.sum().item()) if is_pos_inf.any().item() else n

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

            data_cpu = data_slice.cpu().numpy()

            cumulative_weight = 0.0
            sigma_mean = float(data_cpu[0])
            sigma_weight = 1.0

            q_limit = TDigestTorch._compute_q_limit(0.0, delta, slice_len)

            for i in range(1, slice_len):
                mu = float(data_cpu[i])

                if math.isnan(mu):
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

            if not math.isnan(sigma_mean):
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
    def batch_from_tensor(
        cls,
        data: torch.Tensor,
        delta: float = 0.01,
        max_centroids: int = 500,
        use_compile: bool = True
    ) -> TDigestResult:
        """
        Create T-Digests for a batch of data.

        Args:
            data: 2D tensor of shape [batch_size, n]
            delta: Compression parameter (0 < delta <= 1)
            max_centroids: Maximum centroids per digest
            use_compile: Use torch.compile + vmap for GPU acceleration (default True)

        Returns:
            TDigestResult with means, weights, and counts as numpy arrays

        Raises:
            ImportError: If torch not installed
            ValueError: If data is not 2D or invalid delta
        """
        _check_torch_available()

        if data.ndim != 2:
            raise ValueError(f"data must be 2D tensor, got shape {data.shape}")

        if delta <= 0 or delta > 1:
            raise ValueError(f"delta must be in (0, 1], got {delta}")

        batch_size = data.shape[0]
        device = data.device

        sorted_data, _ = torch.sort(data, dim=1)

        if use_compile and device.type in ['cuda', 'mps']:
            # GPU path: use vmap + torch.compile for parallelization
            from torch import vmap

            # Create a closure with fixed delta and max_centroids
            def cluster_fn(sorted_row):
                return cls._cluster_single_digest_compiled(sorted_row, delta, max_centroids)

            # vmap over batch dimension (in_dims=0 means first dim is batch)
            means_batch, weights_batch, counts_batch = vmap(cluster_fn)(sorted_data)

        else:
            # CPU path: sequential processing (compile doesn't help much on CPU)
            means_batch = torch.zeros(
                (batch_size, max_centroids),
                dtype=torch.float64,
                device=device
            )
            weights_batch = torch.zeros(
                (batch_size, max_centroids),
                dtype=torch.float64,
                device=device
            )
            counts_batch = torch.zeros(batch_size, dtype=torch.int64, device=device)

            for i in range(batch_size):
                means, weights, count = cls._cluster_single_digest(
                    sorted_data[i],
                    delta,
                    max_centroids
                )
                means_batch[i] = means
                weights_batch[i] = weights
                counts_batch[i] = count

        return TDigestResult(
            means=means_batch.cpu().numpy(),
            weights=weights_batch.cpu().numpy(),
            counts=counts_batch.cpu().numpy()
        )

    @classmethod
    def batch_from_arrays(
        cls,
        arrays: List[np.ndarray],
        delta: float = 0.01,
        device: str = 'cpu',
        max_centroids: int = 500
    ) -> TDigestResult:
        """
        Create T-Digests for a batch of arrays (convenience wrapper).

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

        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All arrays must have same length. Got lengths: {set(lengths)}"
            )

        data = torch.tensor(
            np.array(arrays),
            dtype=torch.float64,
            device=device
        )

        return cls.batch_from_tensor(data, delta, max_centroids)

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


def batch_from_tensor_torch(
    data: torch.Tensor,
    delta: float = 0.01,
) -> TDigestResult:
    """
    Convenience function for batch T-Digest computation.

    Args:
        data: 2D tensor of shape [batch_size, n]
        delta: Compression parameter

    Returns:
        TDigestResult
    """
    _check_torch_available()
    return TDigestTorch.batch_from_tensor(data, delta)


def batch_from_arrays_torch(
    arrays: List[np.ndarray],
    delta: float = 0.01,
    device: Optional[str] = None
) -> TDigestResult:
    """
    Convenience function with automatic device selection (legacy API).

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


__all__ = [
    'TDigestTorch',
    'TDigestResult',
    'batch_from_tensor_torch',
    'batch_from_arrays_torch',
]
