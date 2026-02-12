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
    def _compute_q_limits_vectorized(q_values: torch.Tensor, delta: float, n: torch.Tensor) -> torch.Tensor:
        """Vectorized q_limit computation for multiple q values."""
        device = q_values.device
        dtype = q_values.dtype

        # Handle edge cases with masking
        result = torch.zeros_like(q_values)

        # Mask for valid range (0 < q < 1)
        valid_mask = (q_values > 0) & (q_values < 1)

        # Work on full array with masking (no boolean indexing - vmap compatible)
        n_float = n.float() if n.dtype != dtype else n
        n_over_delta = n_float / delta
        base_factor = torch.log(n_over_delta) * SCALE_BASE_MULTIPLIER + SCALE_BASE_OFFSET
        scale_factor = delta / base_factor

        # Compute for all values, will mask later
        # Clamp q_values to avoid log(0) or log(negative)
        q_safe = torch.clamp(q_values, 1e-10, 1.0 - 1e-10)
        k = scale_factor * torch.log(q_safe / (1.0 - q_safe)) + 1.0
        q_limit_all = 1.0 / (1.0 + torch.exp(-k * base_factor / delta))

        # Apply masks
        result = torch.where(valid_mask, q_limit_all, result)
        result = torch.where(q_values >= 1.0, torch.ones_like(result), result)

        return result

    @staticmethod
    def _cluster_single_digest_vmapped(
        sorted_data: torch.Tensor,
        delta: float,
        max_centroids: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized clustering without data-dependent control flow (vmap-compatible).

        Strategy:
        1. Use masks instead of extracting values (fixed shapes)
        2. Compute all merge decisions using vectorized operations
        3. Use scatter operations for aggregation
        4. All operations work on full-length arrays with masks

        Returns:
            means, weights (padded to max_centroids), count (as tensor)
        """
        n = sorted_data.shape[0]
        device = sorted_data.device
        dtype = sorted_data.dtype

        means = torch.zeros(max_centroids, device=device, dtype=dtype)
        weights = torch.zeros(max_centroids, device=device, dtype=dtype)

        # Classify values using masks (keep fixed shapes)
        is_neg_inf = sorted_data == float('-inf')
        is_pos_inf = sorted_data == float('inf')
        is_nan = torch.isnan(sorted_data)
        is_finite = ~(is_neg_inf | is_pos_inf | is_nan)

        # Count special values
        n_neg_inf = is_neg_inf.sum().long()
        n_pos_inf = is_pos_inf.sum().long()
        n_finite = is_finite.sum().long()

        # Create position indices for finite values only
        # Use cumsum to assign positions to finite values
        finite_positions = is_finite.cumsum(0) - 1  # 0-indexed positions
        finite_positions = torch.where(is_finite, finite_positions, torch.tensor(0, device=device, dtype=torch.long))

        # Compute approximate q values for finite positions
        # Map position -> q value
        positions = (finite_positions + 1).float()
        n_finite_float = n_finite.float().clamp(min=1.0)  # Avoid division by zero
        approx_q = positions / n_finite_float

        # Compute q_limits vectorized (for all positions, but only finite ones will be used)
        # Pass n_finite as tensor, not .item()
        q_limits = TDigestTorch._compute_q_limits_vectorized(approx_q, delta, n_finite)

        # Compute merge decisions for finite values
        # First finite value always starts new centroid
        is_first_finite = is_finite & (finite_positions == 0)
        should_merge = (approx_q <= q_limits) & ~is_first_finite & is_finite

        # Assign centroid IDs: cumsum of "starts new centroid"
        starts_new_centroid = is_finite & ~should_merge
        centroid_ids = starts_new_centroid.cumsum(0) - 1

        # Adjust centroid IDs to account for -inf centroid if present
        has_neg_inf = n_neg_inf > 0
        centroid_ids = torch.where(is_finite,
                                   centroid_ids + has_neg_inf.long(),
                                   torch.tensor(0, device=device, dtype=torch.long))

        # Count total finite centroids
        n_finite_centroids = torch.where(is_finite, starts_new_centroid, torch.zeros_like(starts_new_centroid)).sum()

        # Aggregate finite values into centroids using scatter
        # We need to scatter over all data, but only finite values contribute
        total_centroids = n_finite_centroids + has_neg_inf.long() + n_pos_inf.long()
        total_centroids = torch.min(total_centroids, torch.tensor(max_centroids, device=device, dtype=torch.long))

        # Create temporary buffers for all possible centroids
        all_means = torch.zeros(max_centroids, dtype=dtype, device=device)
        all_weights = torch.zeros(max_centroids, dtype=dtype, device=device)

        # Scatter finite values
        # Only scatter where is_finite is True
        masked_data = torch.where(is_finite, sorted_data, torch.tensor(0.0, dtype=dtype, device=device))
        masked_ones = torch.where(is_finite, torch.tensor(1.0, dtype=dtype, device=device), torch.tensor(0.0, dtype=dtype, device=device))

        all_weights.scatter_add_(0, centroid_ids, masked_ones)
        all_means.scatter_add_(0, centroid_ids, masked_data)

        # Compute means (avoid division by zero)
        all_means = torch.where(all_weights > 0, all_means / all_weights, torch.tensor(0.0, dtype=dtype, device=device))

        # Add -inf centroid if present
        all_means[0] = torch.where(has_neg_inf,
                                   torch.tensor(float('-inf'), dtype=dtype, device=device),
                                   all_means[0])
        all_weights[0] = torch.where(has_neg_inf,
                                     n_neg_inf.float() + all_weights[0],
                                     all_weights[0])

        # Add +inf centroid if present
        inf_idx = total_centroids - 1
        inf_idx = torch.clamp(inf_idx, 0, max_centroids - 1)
        all_means[inf_idx] = torch.where(n_pos_inf > 0,
                                         torch.tensor(float('inf'), dtype=dtype, device=device),
                                         all_means[inf_idx])
        all_weights[inf_idx] = torch.where(n_pos_inf > 0,
                                           n_pos_inf.float() + all_weights[inf_idx],
                                           all_weights[inf_idx])

        return all_means, all_weights, total_centroids

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
            # GPU path: Use vmap with vectorized implementation for batch parallelism
            from torch.func import vmap

            # Create wrapper function for vmap
            def cluster_fn(data_row):
                return cls._cluster_single_digest_vmapped(data_row, delta, max_centroids)

            # Process entire batch in parallel with vmap
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
