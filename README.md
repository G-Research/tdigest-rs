# TDigest-rs

<a href="https://pypi.org/project/tdigest-rs/">
  <img src="https://img.shields.io/pypi/v/tdigest-rs.svg" alt="PyPi Latest Release"/>
</a>

Simple Python package to compute TDigests, implemented in Rust.

## Introduction

TDigest-rs is a Python library with a Rust backend that implements the T-Digest algorithm, enhancing the estimation of quantiles in streaming data. For an in-depth exploration of the T-Digest algorithm, refer to [Ted Dunning and Otmar Ertl's paper](https://arxiv.org/abs/1902.04023) and the [G-Research blog post](https://www.gresearch.com/blog/article/approximate-percentiles-with-t-digests/).


## Usage

```shell
pip install tdigest-rs
```

The library contains a single ``TDigest`` class.

### Creating a TDigest object

```python

from tdigest_rs import TDigest

# Fit a TDigest from a numpy array (float32 or float64)
arr = np.random.randn(1000)
tdigest = TDigest.from_array(arr=arr, delta=100.0)  # delta is optional and defaults to 300.0
print(tdigest.means, tdigest.weights)

# Create directly from means and weights arrays
vals = np.random.randn(1000).astype(np.float32)
weights = np.ones(1000).astype(np.uint32)
tdigest = TDigest.from_means_weights(arr=vals, weights=weights)
```

### Computing quantiles

```python

# Compute a quantile
tdigest.quantile(0.1)

# Compute median
tdigest.median()

# Compute trimmed mean
tdigest.trimmed_mean(lower=0.05, upper=0.95)
```

### Merging TDigests

```python

arr1 = np.random.randn(1000)
arr2 = np.ones(1000)
digest1 = TDigest.from_array(arr=arr1)
digest2 = TDigest.from_array(arr=arr2)

merged_digest = digest1.merge(digest2, delta=100.0)  # delta again defaults to 300.0
```

### Serialising TDigests

The ``TDigest`` object can be converted to a dictionary and JSON-serialised and is also pickleable.

```python

# Convert and load to/from a python dict
d = tdigest.to_dict()
loaded_digest = TDigest.from_dict(d)

# Pickle a digest
import pickle

pickle.dumps(tdigest)
```


## Development workflow

```bash
pip install hatch

cd bindings/python

# Run linters
hatch run dev:lint

# Run tests
hatch run dev:test

# Run benchmark
hatch run dev:benchmark

# Format code
hatch run dev:format
```

## Contributing

Please read our [contributing](https://github.com/G-Research/tdigest-rs/blob/main/CONTRIBUTING.md) guide and [code of conduct](https://github.com/G-Research/tdigest-rs/blob/main/CODE_OF_CONDUCT.md) if you'd like to contribute to the project.

## Community Guidelines

Please read our [code of conduct](https://github.com/G-Research/tdigest-rs/blob/main/CODE_OF_CONDUCT.md) before participating in or contributing to this project.

## Security

Please see our [security policy](https://github.com/G-Research/tdigest-rs/blob/main/SECURITY.md) for details on reporting security vulnerabilities.

## License

TDigest-rs is licensed under the [Apache Software License 2.0 (Apache-2.0)](https://github.com/G-Research/tdigest-rs/blob/main/LICENSE)
