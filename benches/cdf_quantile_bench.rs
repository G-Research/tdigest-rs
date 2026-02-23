use std::hint::black_box;
use std::time::Duration;

use bytesize::ByteSize;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jemalloc_ctl::{epoch, stats};
use tdigest_rs::tdigest::{ScaleFamily, TDigest};
use testdata::{gen_dataset, DistKind};

/* ------------------------ helpers ------------------------ */

fn build_digest(
    kind: DistKind,
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    seed: u64,
) -> TDigest<f64> {
    let mut data = gen_dataset(kind, n, seed);
    data.sort_by(|a, b| a.total_cmp(b));
    TDigest::builder()
        .max_size(max_size)
        .scale(scale)
        .build()
        .merge_sorted(data)
        .expect("valid benchmark dataset")
}

/// Allocate inside `f`, keep the object alive until after reading jemalloc counters.
/// Returns `(object, delta_bytes)`.
fn alloc_bytes_hold<T, F: FnOnce() -> T>(f: F) -> (T, usize) {
    epoch::advance().unwrap();
    let before = stats::allocated::read().unwrap();
    let obj = f();
    epoch::advance().unwrap();
    let after = stats::allocated::read().unwrap();
    (obj, after.saturating_sub(before))
}

/* ------------------------ main benchmark ------------------------ */

fn bench_cdf_light(c: &mut Criterion) {
    let sizes = [1usize, 1_000, 100_000, 5_000_000];

    for &m in &sizes {
        let mut group = c.benchmark_group(format!("cdf_light/size={m}"));
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(m as u64));

        let values: Vec<f64> = (0..m).map(|i| i as f64 / m as f64).collect();

        // Base digest for this size (Mixture is a nice stressor; K2 is your current fave)
        let td = build_digest(DistKind::Mixture, m, 1_000, ScaleFamily::K2, 4242);

        // light
        group.bench_with_input(BenchmarkId::new("cdf", m), &values, |b, values| {
            b.iter(|| {
                let out = td.cdf(black_box(values));
                black_box(out[out.len() / 2])
            });
        });

        group.finish();

        // ---- memory footprint (one line per size) ----
        // 1) Digest construction footprint
        let (_td_hold, digest_bytes) =
            alloc_bytes_hold(|| build_digest(DistKind::Mixture, m, 1_000, ScaleFamily::K2, 4242));

        // 2) One evaluation footprint (keep both digest and output Vec alive)
        let ((_td_eval_hold, _out_hold), eval_bytes) = alloc_bytes_hold(|| {
            let td2 = build_digest(DistKind::Mixture, m, 1_000, ScaleFamily::K2, 4242);
            let out = td2.cdf(&values);
            (td2, out)
        });

        println!(
            "[memory] size={:<8} digest={}, eval_once={}",
            m,
            ByteSize(digest_bytes as u64),
            ByteSize(eval_bytes as u64),
        );
    }
}

/* ------------------------ registration ------------------------ */

fn configure() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30)
        .without_plots()
}

criterion_group!(
    name = tdigest_benches;
    config = configure();
    targets = bench_cdf_light
);
criterion_main!(tdigest_benches);
