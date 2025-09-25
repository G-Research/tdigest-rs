# Build Optimization Guide

This document explains the simplified build configurations available for TDigest-rs.

## Quick Start

### Standard Build (Recommended)
```bash
cargo build --release
```

This provides excellent performance for most use cases with standard compiler optimizations.

## Build Profiles

### Release (Default)
```bash
cargo build --release
```
- **Optimization**: Maximum (opt-level = 3)
- **LTO**: Thin (faster builds)
- **Use case**: Production deployments
- **Performance**: High

### Release-Fast (Maximum Performance)
```bash
cargo build --profile release-fast
```
- **Optimization**: Maximum with full LTO
- **LTO**: Fat (slower builds, maximum performance)
- **Use case**: Performance-critical applications
- **Performance**: Highest

### Release-Size (Smallest Binary)
```bash
cargo build --profile release-size
```
- **Optimization**: Size-optimized (opt-level = "s")
- **LTO**: Fat with size optimization
- **Use case**: Embedded systems, minimizing binary size
- **Performance**: Good, optimized for size

### Dev-Fast (Development)
```bash
cargo build --profile dev-fast
```
- **Optimization**: Basic (opt-level = 1)
- **Debug info**: Enabled
- **Use case**: Development with some optimizations
- **Performance**: Moderate, fast compile times

## Platform-Specific Optimizations

### Target CPU Optimization
For maximum performance on your specific hardware:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Conservative Compatibility
For broader hardware compatibility:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release
```

## Testing

### Standard Tests
```bash
cargo test --release
```

### Performance Tests
```bash
cargo test --release test_performance_regression_large_dataset -- --nocapture
```

### All Tests with Logging
```bash
RUST_LOG=debug cargo test --release
```

## Build Automation

The included build script supports multiple variants:

```bash
./scripts/build-all.sh fast modern size
```

Available variants:
- **fast**: Standard release build
- **modern**: Release-fast profile for maximum performance
- **size**: Size-optimized build
- **native**: Native CPU optimization
- **dev-fast**: Development build with optimizations

## Python Bindings

To build the Python extension:

```bash
cd bindings/python
maturin develop --release
```

For distribution wheels:
```bash
maturin build --release
```

## Performance Characteristics

All build configurations provide excellent T-Digest performance:

- **Compression**: Efficiently reduces large datasets to ~100-500 centroids
- **Quantile Accuracy**: High accuracy for extreme quantiles (p99, p99.9)
- **Memory Efficiency**: Minimal memory overhead
- **Streaming**: Supports incremental updates via merge operations

The performance differences between build configurations are primarily in compilation speed vs runtime optimization level, not in algorithmic complexity.

## Benchmarking

To benchmark your specific configuration:

```bash
cargo bench
```

For custom performance testing:
```bash
time cargo run --release -- --benchmark
```

## Cross-Compilation

The simplified build system supports easy cross-compilation:

```bash
# For ARM64
cargo build --release --target aarch64-unknown-linux-gnu

# For Windows
cargo build --release --target x86_64-pc-windows-gnu
```

## Summary

The TDigest-rs library now uses a simplified build system focused on:

1. **Reliability**: Standard library functions with excellent performance
2. **Portability**: Works consistently across all platforms and architectures
3. **Simplicity**: Easy to build, deploy, and debug
4. **Performance**: Excellent quantile computation performance without complexity

Choose the build profile that matches your deployment requirements and hardware constraints.