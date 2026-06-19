# PyTorch T-Digest API Update

## Summary

The PyTorch implementation now uses a cleaner, more PyTorch-native API that takes 2D tensors instead of lists of numpy arrays.

## New API (Recommended)

### Basic Usage

```python
import torch
from tdigest_rs.torch_impl import TDigestTorch

# Create a 2D tensor [batch_size, n]
data = torch.randn(20000, 32000, dtype=torch.float64)

# Process on CPU
result = TDigestTorch.batch_from_tensor(data, delta=0.01)

# Process on GPU (device is inferred from input tensor)
data_gpu = data.cuda()
result = TDigestTorch.batch_from_tensor(data_gpu, delta=0.01)

# Access results (returned as numpy arrays for compatibility)
for i in range(len(result.counts)):
    count = result.counts[i]
    means = result.means[i, :count]
    weights = result.weights[i, :count]
```

### Convenience Function

```python
from tdigest_rs.torch_impl import batch_from_tensor_torch

result = batch_from_tensor_torch(data, delta=0.01)
```

## Old API (Still Supported)

The legacy list-of-arrays API is still available for backwards compatibility:

```python
import numpy as np
from tdigest_rs.torch_impl import TDigestTorch

# List of 1D numpy arrays
arrays = [np.random.randn(32000).astype(np.float64) for _ in range(20000)]

# Must specify device explicitly
result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cpu')
```

## Key Differences

| Feature | New API (`batch_from_tensor`) | Old API (`batch_from_arrays`) |
|---------|------------------------------|-------------------------------|
| **Input** | 2D `torch.Tensor` | `List[np.ndarray]` |
| **Device** | Inferred from input tensor | Must specify explicitly |
| **PyTorch-native** | ✅ Yes | ❌ No (converts internally) |
| **Overhead** | Lower (no list→tensor conversion) | Higher (conversion needed) |
| **Recommended** | ✅ Yes | For backwards compatibility only |

## Migration Guide

### Before
```python
import numpy as np
from tdigest_rs.torch_impl import TDigestTorch

arrays = [np.random.randn(32000).astype(np.float64) for _ in range(20000)]
result = TDigestTorch.batch_from_arrays(arrays, delta=0.01, device='cuda')
```

### After
```python
import torch
from tdigest_rs.torch_impl import TDigestTorch

data = torch.randn(20000, 32000, dtype=torch.float64, device='cuda')
result = TDigestTorch.batch_from_tensor(data, delta=0.01)
```

## Benefits of New API

1. **More PyTorch-native**: Works directly with tensors
2. **Automatic device handling**: Device is inferred from input
3. **Less overhead**: No list→tensor conversion needed
4. **Cleaner code**: Single tensor instead of list of arrays
5. **Type safety**: Static analyzers can validate tensor shapes

## Return Value

Both APIs return the same `TDigestResult` dataclass:
```python
@dataclass
class TDigestResult:
    means: np.ndarray      # [batch_size, max_centroids]
    weights: np.ndarray    # [batch_size, max_centroids]
    counts: np.ndarray     # [batch_size]
```

Results are returned as numpy arrays for compatibility with the Rust TDigest API.

## Examples

See `bindings/python/examples/torch_example.py` for complete examples using the new API.

## Tests

All existing tests pass with both APIs:
- ✅ 36 passed, 2 skipped
- ✅ New tensor API tests added
- ✅ Backwards compatibility maintained
