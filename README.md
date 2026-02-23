# tdigest-rs

`tdigest-rs` provides a Rust TDigest core with Python bindings.

## Features
- Mergeable digest with quantile, CDF, median, and trimmed mean.
- Scale families: `Quad`, `K1`, `K2`, `K3`.
- Singleton policies: `off`, `use`, `edges`.
- Weighted ingest (`add_weighted`, `from_means_weights`).
- Wire format support with explicit encode version (`v1|v2|v3`) and precision inspection.
- Strict validation for non-finite training/probe inputs.

## Python example
```python
import tdigest_rs as td

# Build
d = td.TDigest.from_array([0.0, 1.0, 2.0, 3.0], max_size=1000, scale="k2")

# Query
print(d.quantile(0.5))
print(d.cdf(1.5))
print(d.trimmed_mean(0.05, 0.95))

# Compatibility helpers
m = td.TDigest.from_means_weights([0.0, 1.0], [1.0, 2.0], max_size=200, scale="k2")
m2 = m.update([2.0, 3.0])      # returns a new digest
m3 = m.merge(m2)                # returns a new digest
print(len(m3), m3.means, m3.weights)
```

## Migration notes (0.x -> 2.0.0)
- Constructor compatibility accepts either `max_size` or legacy `delta`.
  - Do not pass both in the same call.
  - If neither is set, Python constructors default to `max_size=100`.
  - `delta` runs a dedicated legacy mode that mirrors the old tdigest-rs K2-style merge rule.
  - In `delta` mode, only `scale='k2'` and `singleton_policy='off'` are supported.
- Python keeps compatibility methods:
  - `from_means_weights(...)`
  - `update(...)` (returns new digest)
  - `merge(...)` (returns new digest)
  - `to_dict()` / `from_dict()` (new schema + legacy dict support)
  - `means`, `weights`, `__len__`

## Development
```bash
make setup
make build
make test
```

## Changelog
See `CHANGELOG.md`.

## License
Apache-2.0
