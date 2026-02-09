use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tdigest_core::TDigest;
use std::time::Instant;

fn profile_create_digest(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_digest_profile");

    for size in [1_000, 10_000, 32_000] {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                TDigest::from_array(black_box(&data), black_box(0.01))
            })
        });
    }

    group.finish();
}

fn profile_merge_digests(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_profile");

    let data: Vec<f64> = (0..32_000).map(|i| i as f64 * 0.1).collect();
    let digest1 = TDigest::from_array(&data, 0.01).unwrap();
    let digest2 = TDigest::from_array(&data, 0.01).unwrap();

    group.bench_function("merge_32k", |b| {
        b.iter(|| {
            black_box(&digest1).merge(black_box(&digest2), black_box(0.01))
        })
    });

    group.finish();
}

fn profile_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_profile");

    // Create 100 arrays of 32k elements each
    let arrays: Vec<Vec<f64>> = (0..100)
        .map(|_| (0..32_000).map(|i| i as f64 * 0.1).collect())
        .collect();

    let array_refs: Vec<&[f64]> = arrays.iter().map(|v| v.as_slice()).collect();

    group.bench_function("batch_from_arrays_100x32k", |b| {
        b.iter(|| {
            TDigest::batch_from_arrays(black_box(&array_refs), black_box(0.01))
        })
    });

    group.finish();
}

fn manual_profiling() {
    println!("\n=== DETAILED PROFILING ===\n");

    // Profile individual digest creation
    let sizes = [1_000, 10_000, 32_000];
    for &size in &sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();

        let start = Instant::now();
        for _ in 0..100 {
            let _ = TDigest::from_array(&data, 0.01).unwrap();
        }
        let elapsed = start.elapsed();

        println!("Create digest (size={}): {:?} per operation", size, elapsed / 100);
        println!("  Throughput: {:.2} M elements/sec",
                 (size as f64 * 100.0) / elapsed.as_secs_f64() / 1_000_000.0);
    }

    println!("\n--- Merge Operations ---");
    let data: Vec<f64> = (0..32_000).map(|i| i as f64 * 0.1).collect();
    let digest1 = TDigest::from_array(&data, 0.01).unwrap();
    let digest2 = TDigest::from_array(&data, 0.01).unwrap();

    let start = Instant::now();
    for _ in 0..100 {
        let _ = digest1.merge(&digest2, 0.01).unwrap();
    }
    let elapsed = start.elapsed();
    println!("Merge (32k centroids): {:?} per operation", elapsed / 100);

    println!("\n--- Batch Operations ---");
    let arrays: Vec<Vec<f64>> = (0..100)
        .map(|_| (0..32_000).map(|i| i as f64 * 0.1).collect())
        .collect();
    let array_refs: Vec<&[f64]> = arrays.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let _ = TDigest::batch_from_arrays(&array_refs, 0.01).unwrap();
    let elapsed = start.elapsed();
    println!("Batch create (100 x 32k): {:?} total", elapsed);
    println!("  Per digest: {:?}", elapsed / 100);
    println!("  Throughput: {:.2} M elements/sec",
             (100.0 * 32_000.0) / elapsed.as_secs_f64() / 1_000_000.0);

    println!("\n=== END PROFILING ===\n");
}

criterion_group!(benches, profile_create_digest, profile_merge_digests, profile_batch_operations);
criterion_main!(benches);
