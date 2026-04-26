// benches/codecs_bench.rs
use std::hint::black_box;
use std::time::{Duration, Instant};
use tdigest_rs::tdigest::{Centroid, TDigest};

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};

const DIGESTS: usize = 100; // number of TDigest objects (rows)
const CENTROIDS: usize = 1000; // centroids per digest
const MAX_SIZE: usize = 1000; // digest max capacity (k)

fn synth_digest(centroids: usize, shift: f64) -> TDigest {
    let cs: Vec<Centroid> = (0..centroids)
        .map(|i| {
            let mean = (i as f64 + 0.5 + shift) * 1e-3;
            let weight = 1.0 + ((i as f64 + shift) % 3.0);
            Centroid::new(mean, weight)
        })
        .collect();

    let sum = cs.iter().map(|c| c.mean() * c.weight()).sum::<f64>();
    let count = cs.iter().map(|c| c.weight()).sum::<f64>();
    let min = cs.first().map(|c| c.mean()).unwrap_or(0.0);
    let max = cs.last().map(|c| c.mean()).unwrap_or(0.0);

    // NOTE: with_centroids is a *builder* method; include the max_size override.
    TDigest::builder()
        .with_centroids(
            cs, sum, count, /*max:*/ max, /*min:*/ min, /*override:*/ MAX_SIZE,
        )
        .build()
}

fn codec_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_quick");

    // --- steadier settings for ms-level work ---
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);
    group.noise_threshold(0.02);
    group.sampling_mode(SamplingMode::Flat); // keep per-sample iters steady

    // Pre-build some digests and their Series encodings once per benchmark group
    let digests: Vec<TDigest> = (0..DIGESTS)
        .map(|r| synth_digest(CENTROIDS, r as f64 * 0.123))
        .collect();

    let ser_f64 = {
        let mut acc = digests[0].try_to_series("td").expect("to_series f64");
        for td in &digests[1..] {
            let part = td.try_to_series("td").expect("to_series f64");
            acc.append(&part).expect("append f64");
        }
        acc
    };

    let ser_f32 = {
        let mut acc = digests[0]
            .try_to_series_compact("td")
            .expect("to_series f32");
        for td in &digests[1..] {
            let part = td.try_to_series_compact("td").expect("to_series f32");
            acc.append(&part).expect("append f32");
        }
        acc
    };

    // Accumulators so our footer reports avg ms/op across all samples (not just the last one)
    let (mut w64_secs, mut w64_iters) = (0.0_f64, 0_u64);
    let (mut w32_secs, mut w32_iters) = (0.0_f64, 0_u64);
    let (mut r64_secs, mut r64_iters) = (0.0_f64, 0_u64);
    let (mut r32_secs, mut r32_iters) = (0.0_f64, 0_u64);

    // ---------------- WRITE: TDigest -> Series ----------------
    group.bench_function("write_f64", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut acc = digests[0].try_to_series("td").unwrap();
                for td in &digests[1..] {
                    let part = td.try_to_series("td").unwrap();
                    acc.append(&part).unwrap();
                }
                black_box(acc.len());
            }
            let dur = start.elapsed();
            w64_secs += dur.as_secs_f64();
            w64_iters += iters;
            dur
        });
    });

    group.bench_function("write_f32", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut acc = digests[0].try_to_series_compact("td").unwrap();
                for td in &digests[1..] {
                    let part = td.try_to_series_compact("td").unwrap();
                    acc.append(&part).unwrap();
                }
                black_box(acc.len());
            }
            let dur = start.elapsed();
            w32_secs += dur.as_secs_f64();
            w32_iters += iters;
            dur
        });
    });

    // ---------------- READ: Series -> Vec<TDigest> ----------------
    group.bench_function("read_f64", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let v = TDigest::try_from_series(black_box(&ser_f64)).unwrap();
                black_box(v.len());
            }
            let dur = start.elapsed();
            r64_secs += dur.as_secs_f64();
            r64_iters += iters;
            dur
        });
    });

    group.bench_function("read_f32", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let v = TDigest::try_from_series_compact(black_box(&ser_f32)).unwrap();
                black_box(v.len());
            }
            let dur = start.elapsed();
            r32_secs += dur.as_secs_f64();
            r32_iters += iters;
            dur
        });
    });

    group.finish();

    // ---------------- summary (averaged across all samples) ----------------
    let secs_per_iter = |secs: f64, iters: u64| {
        if iters > 0 {
            secs / iters as f64
        } else {
            f64::NAN
        }
    };
    let to_ms = |s: f64| s * 1e3;

    let write_f64_ms = to_ms(secs_per_iter(w64_secs, w64_iters));
    let write_f32_ms = to_ms(secs_per_iter(w32_secs, w32_iters));
    let read_f64_ms = to_ms(secs_per_iter(r64_secs, r64_iters));
    let read_f32_ms = to_ms(secs_per_iter(r32_secs, r32_iters));

    let pct = |b: f64, a: f64| 100.0 * (b / a - 1.0); // % change vs f64

    println!(
        "\n== Codec quick sanity (DIGESTS={DIGESTS}, CENTROIDS={CENTROIDS}, MAX_SIZE={MAX_SIZE}) =="
    );
    println!(
        "WRITE  f64: {:6.3} ms | f32: {:6.3} ms  => f32 {:+.1}%",
        write_f64_ms,
        write_f32_ms,
        pct(write_f32_ms, write_f64_ms)
    );
    println!(
        "READ   f64: {:6.3} ms | f32: {:6.3} ms  => f32 {:+.1}%",
        read_f64_ms,
        read_f32_ms,
        pct(read_f32_ms, read_f64_ms)
    );
}

criterion_group!(benches, codec_bench);
criterion_main!(benches);
