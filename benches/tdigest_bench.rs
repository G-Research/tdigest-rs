use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use tdigest_core::TDigest;

const DELTA: f64 = 100.0;
const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000];

fn generate_data(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| StandardNormal.sample(&mut rng)).collect()
}

fn bench_from_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_array");

    for &size in SIZES {
        let data = generate_data(size, 42);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &data, |b, data| {
            b.iter(|| TDigest::from_array(black_box(data), black_box(DELTA)).unwrap());
        });
    }

    // f32 comparison at 10k
    let data_f32: Vec<f32> = generate_data(10_000, 42)
        .into_iter()
        .map(|x| x as f32)
        .collect();
    group.throughput(Throughput::Elements(10_000));
    group.bench_with_input(BenchmarkId::new("f32", 10_000), &data_f32, |b, data| {
        b.iter(|| TDigest::from_array(black_box(data), black_box(100.0f32)).unwrap());
    });

    group.finish();
}

fn bench_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile");

    for &size in SIZES {
        let data = generate_data(size, 42);
        let digest = TDigest::from_array(&data, DELTA).unwrap();

        group.bench_with_input(BenchmarkId::new("p01", size), &digest, |b, digest| {
            b.iter(|| digest.quantile(black_box(0.01)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("p50", size), &digest, |b, digest| {
            b.iter(|| digest.quantile(black_box(0.50)).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("p99", size), &digest, |b, digest| {
            b.iter(|| digest.quantile(black_box(0.99)).unwrap());
        });
    }

    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge");

    for &size in SIZES {
        let d1 = TDigest::from_array(&generate_data(size, 42), DELTA).unwrap();
        let d2 = TDigest::from_array(&generate_data(size, 99), DELTA).unwrap();

        group.bench_with_input(
            BenchmarkId::new("symmetric", size),
            &(&d1, &d2),
            |b, (d1, d2)| {
                b.iter(|| d1.merge(black_box(d2), black_box(DELTA)).unwrap());
            },
        );
    }

    // Asymmetric: 100k + 100
    let d_large = TDigest::from_array(&generate_data(100_000, 42), DELTA).unwrap();
    let d_small = TDigest::from_array(&generate_data(100, 99), DELTA).unwrap();
    group.bench_function("asymmetric_100k_100", |b| {
        b.iter(|| d_large.merge(black_box(&d_small), black_box(DELTA)).unwrap());
    });

    group.finish();
}

fn bench_trimmed_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("trimmed_mean");

    for &size in SIZES {
        let data = generate_data(size, 42);
        let digest = TDigest::from_array(&data, DELTA).unwrap();

        group.bench_with_input(
            BenchmarkId::new("trim_5_95", size),
            &digest,
            |b, digest| {
                b.iter(|| digest.trimmed_mean(black_box(0.05), black_box(0.95)).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("trim_25_75", size),
            &digest,
            |b, digest| {
                b.iter(|| digest.trimmed_mean(black_box(0.25), black_box(0.75)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_from_means_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_means_weights");

    for &size in SIZES {
        let data = generate_data(size, 42);
        let digest = TDigest::from_array(&data, DELTA).unwrap();

        group.bench_with_input(
            BenchmarkId::new("recompress", size),
            &digest,
            |b, digest| {
                b.iter(|| {
                    TDigest::from_means_weights(
                        black_box(&digest.means),
                        black_box(&digest.weights),
                        black_box(DELTA),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_delta_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_sensitivity");
    let data = generate_data(100_000, 42);

    for delta in [10.0, 50.0, 100.0, 500.0, 1000.0, 10000.0] {
        group.bench_with_input(
            BenchmarkId::new("from_array_100k", delta as u64),
            &delta,
            |b, &delta| {
                b.iter(|| TDigest::from_array(black_box(&data), black_box(delta)).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_from_array,
    bench_quantile,
    bench_merge,
    bench_trimmed_mean,
    bench_from_means_weights,
    bench_delta_sensitivity,
);
criterion_main!(benches);
