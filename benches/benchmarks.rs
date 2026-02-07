use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tdigest_core::TDigest;

fn generate_data(size: usize) -> (Vec<f64>, Vec<u32>) {
    let mut rng_state = 12345u64;
    let means: Vec<f64> = (0..size)
        .map(|_| {
            // Simple LCG random number generator for reproducibility
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f64 / u64::MAX as f64) * 100.0
        })
        .collect();
    let weights = vec![1u32; size];
    (means, weights)
}

fn bench_create_digest(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_digest");

    for size in [100, 1_000, 10_000, 32_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let (means, _) = generate_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                TDigest::from_array(
                    black_box(&means),
                    black_box(100.0),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_merge_digests(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_digests");

    for size in [100, 1_000, 10_000].iter() {
        let (means1, _) = generate_data(*size);
        let (means2, _) = generate_data(*size);

        // Create digests first
        let digest1 = TDigest::from_array(&means1, 100.0).unwrap();
        let digest2 = TDigest::from_array(&means2, 100.0).unwrap();

        group.throughput(Throughput::Elements((*size * 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(&digest1).merge(
                    black_box(&digest2),
                    black_box(100.0),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_quantile");

    for size in [100, 1_000, 10_000].iter() {
        let (means, _) = generate_data(*size);
        let digest = TDigest::from_array(&means, 100.0).unwrap();

        let quantiles = vec![0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                for &q in &quantiles {
                    black_box(
                        black_box(&digest).quantile(black_box(q))
                        .unwrap()
                    );
                }
            });
        });
    }

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");

    for size in [100, 1_000, 10_000, 32_000].iter() {
        let (means1, _) = generate_data(1000);
        let digest = TDigest::from_array(&means1, 100.0).unwrap();

        let (buffer_means, _) = generate_data(*size);
        let buffer_digest = TDigest::from_array(&buffer_means, 100.0).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(&digest).merge(
                    black_box(&buffer_digest),
                    black_box(100.0),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_large_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_batch");
    group.sample_size(20);

    // Simulate production scenario: 32k element batch
    let (means, _) = generate_data(32_000);

    group.throughput(Throughput::Elements(32_000));
    group.bench_function("create_32k", |b| {
        b.iter(|| {
            TDigest::from_array(
                black_box(&means),
                black_box(100.0),
            )
            .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_create_digest,
    bench_merge_digests,
    bench_quantile,
    bench_update,
    bench_large_batch
);
criterion_main!(benches);
