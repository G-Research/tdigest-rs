// Detailed profiling to identify bottlenecks
// Run with: cargo bench --bench detailed_profile

use std::time::Instant;

// Import the internal functions we want to profile
use tdigest_core::TDigest;

fn profile_component_times() {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║         DETAILED COMPONENT PROFILING                  ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Test with realistic production data
    let sizes = [1_000, 10_000, 32_000];
    let iterations = 1000;

    for &size in &sizes {
        println!("\n━━━ Dataset Size: {} elements ━━━", size);

        // Create realistic data
        let data: Vec<f64> = (0..size)
            .map(|i| (i as f64 * 0.1) + (i as f64 % 100.0))
            .collect();

        // Profile: Full digest creation
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = TDigest::from_array(&data, 0.01).unwrap();
        }
        let total_time = start.elapsed();
        let per_op = total_time / iterations;

        println!("\n  Full Digest Creation:");
        println!("    Total time: {:?}", total_time);
        println!("    Per operation: {:?}", per_op);
        println!("    Throughput: {:.2} M elements/sec",
                 (size as f64 * iterations as f64) / total_time.as_secs_f64() / 1_000_000.0);

        // Profile: Sorting overhead
        let mut data_copy = data.clone();
        let start = Instant::now();
        for _ in 0..iterations {
            data_copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let sort_time = start.elapsed();
        println!("\n  Sorting Only:");
        println!("    Per operation: {:?}", sort_time / iterations);
        println!("    % of total: {:.1}%",
                 (sort_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);

        // Profile: Type conversions
        let weights: Vec<u32> = vec![1; size];
        let start = Instant::now();
        for _ in 0..iterations {
            let _: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
        }
        let convert_time = start.elapsed();
        println!("\n  Type Conversion (u32->f64):");
        println!("    Per operation: {:?}", convert_time / iterations);
        println!("    % of total: {:.1}%",
                 (convert_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);

        // Profile: Transcendental functions (estimate)
        let start = Instant::now();
        for _ in 0..iterations {
            for i in 0..100 {
                let q = (i as f64) / 100.0;
                let _ = (q / (1.0 - q)).ln();
                let _ = (-q).exp();
            }
        }
        let trans_time = start.elapsed();
        println!("\n  Transcendental Functions (100 log+exp):");
        println!("    Per 100 calls: {:?}", trans_time / iterations);
        println!("    Single call: {:?}", trans_time / (iterations * 100));
    }

    println!("\n━━━ Batch Operations ━━━");
    let batch_size = 100;
    let element_size = 32_000;

    let arrays: Vec<Vec<f64>> = (0..batch_size)
        .map(|_| (0..element_size).map(|i| i as f64 * 0.1).collect())
        .collect();
    let array_refs: Vec<&[f64]> = arrays.iter().map(|v| v.as_slice()).collect();

    // Sequential (baseline)
    let start = Instant::now();
    for arr in &array_refs {
        let _ = TDigest::from_array(arr, 0.01).unwrap();
    }
    let sequential_time = start.elapsed();

    // Parallel (batch API)
    let start = Instant::now();
    let _ = TDigest::batch_from_arrays(&array_refs, 0.01).unwrap();
    let parallel_time = start.elapsed();

    println!("\n  Sequential: {:?}", sequential_time);
    println!("  Parallel (batch): {:?}", parallel_time);
    println!("  Speedup: {:.2}x", sequential_time.as_secs_f64() / parallel_time.as_secs_f64());
    println!("  Parallel efficiency: {:.1}%",
             (sequential_time.as_secs_f64() / parallel_time.as_secs_f64()) / num_cpus::get() as f64 * 100.0);

    println!("\n━━━ Memory Allocation Patterns ━━━");

    // Estimate allocation overhead
    let data: Vec<f64> = (0..32_000).map(|i| i as f64 * 0.1).collect();

    let start = Instant::now();
    for _ in 0..1000 {
        let _v1: Vec<f64> = Vec::with_capacity(32_000);
        let _v2: Vec<u32> = Vec::with_capacity(32_000);
        let _v3: Vec<bool> = Vec::with_capacity(32_000);
    }
    let alloc_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = TDigest::from_array(&data, 0.01).unwrap();
    }
    let digest_time = start.elapsed();

    println!("\n  Empty vector allocation (3 vecs): {:?}", alloc_time / 1000);
    println!("  Full digest creation: {:?}", digest_time / 1000);
    println!("  Allocation overhead: {:.1}%",
             (alloc_time.as_secs_f64() / digest_time.as_secs_f64()) * 100.0);

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║         END PROFILING                                  ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
}

fn main() {
    profile_component_times();
}
