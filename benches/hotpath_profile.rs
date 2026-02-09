// Profile the actual hot path functions with instrumentation
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

static SCALE_TIME: AtomicU64 = AtomicU64::new(0);
static SCALE_CALLS: AtomicU64 = AtomicU64::new(0);
static CONVERT_TIME: AtomicU64 = AtomicU64::new(0);
static MERGE_TIME: AtomicU64 = AtomicU64::new(0);
static SORT_TIME: AtomicU64 = AtomicU64::new(0);

// We'll manually instrument by measuring key operations
use tdigest_core::TDigest;

fn profile_with_instrumentation() {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║         HOT PATH PROFILING                            ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let iterations = 1000;
    let size = 32_000;

    // Create realistic data - unsorted to trigger all paths
    let mut data: Vec<f64> = (0..size)
        .map(|i| ((i * 7919) % size) as f64 * 0.1) // Pseudo-random but deterministic
        .collect();

    println!("Testing with {} elements, {} iterations\n", size, iterations);

    // Measure: Complete digest creation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = TDigest::from_array(&data, 0.01).unwrap();
    }
    let total_time = start.elapsed();

    println!("━━━ Overall Performance ━━━");
    println!("  Total time: {:?}", total_time);
    println!("  Per digest: {:?}", total_time / iterations);
    println!("  Throughput: {:.2} M elements/sec\n",
             (size as f64 * iterations as f64) / total_time.as_secs_f64() / 1_000_000.0);

    // Measure: Sorting (argsort)
    println!("━━━ Breaking Down Components ━━━\n");

    let mut indices: Vec<usize> = (0..size).collect();
    let start = Instant::now();
    for _ in 0..iterations {
        indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap());
    }
    let sort_time = start.elapsed();

    println!("1. Sorting (argsort):");
    println!("   Time: {:?} ({:.1}% of total)",
             sort_time, (sort_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("   Per operation: {:?}\n", sort_time / iterations);

    // Measure: Weight conversion
    let weights: Vec<u32> = vec![1; size];
    let start = Instant::now();
    for _ in 0..iterations {
        let _: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
    }
    let convert_time = start.elapsed();

    println!("2. Weight Conversion (u32→f64):");
    println!("   Time: {:?} ({:.1}% of total)",
             convert_time, (convert_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("   Per operation: {:?}\n", convert_time / iterations);

    // Measure: log_q_limit approximation
    // Simulate typical usage pattern
    let n = 100; // Typical number of centroids
    let delta = 0.01;
    let test_q_values: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();

    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        for &q in &test_q_values {
            if q > 0.0 && q < 1.0 {
                // Simulate log_q_limit computation
                let k = (delta / ((n as f64 / delta).ln() * 4.0 + 24.0))
                    * (q / (1.0 - q)).ln() + 1.0;
                let result = 1.0 / (1.0 + (-k * ((n as f64 / delta).ln() * 4.0 + 24.0) / delta).exp());
                sum += result;
            }
        }
    }
    let scale_time = start.elapsed();
    // Use sum to prevent optimization
    if sum == 0.0 { println!("unreachable"); }

    println!("3. log_q_limit Calls ({} per digest):", n);
    println!("   Time: {:?} ({:.1}% of total)",
             scale_time, (scale_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("   Per digest: {:?}", scale_time / iterations);
    println!("   Per call: {:?}\n", scale_time / (iterations * n as u32));

    // Measure: Hot loop (clustering) - rough estimate
    let clustering_time = total_time
        .saturating_sub(sort_time)
        .saturating_sub(convert_time)
        .saturating_sub(scale_time);

    println!("4. Clustering Hot Loop (remainder):");
    println!("   Time: {:?} ({:.1}% of total)",
             clustering_time, (clustering_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("   Per digest: {:?}\n", clustering_time / iterations);

    // Detailed breakdown
    println!("━━━ Component Breakdown ━━━\n");
    let components = vec![
        ("Sorting", sort_time),
        ("Weight Conversion", convert_time),
        ("log_q_limit calls", scale_time),
        ("Clustering Loop", clustering_time),
    ];

    for (name, time) in components {
        let pct = (time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  {:20} {:6.1}% {}", name, pct, bar);
    }

    println!("\n━━━ Optimization Opportunities ━━━\n");

    // Estimate: What if we cache log_q_limit?
    let cached_scale_time = scale_time / 10; // Assume 10x speedup with lookup table
    let potential_savings = scale_time.saturating_sub(cached_scale_time);
    let new_total = total_time.saturating_sub(potential_savings);
    let speedup = total_time.as_secs_f64() / new_total.as_secs_f64();

    println!("  Opportunity 1: Lookup Table for log_q_limit");
    println!("    Current: {:?} ({:.1}% of total)", scale_time,
             (scale_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("    Cached:  {:?} (estimated 10x faster)", cached_scale_time);
    println!("    Potential speedup: {:.2}x overall\n", speedup);

    // Estimate: What if we eliminate type conversions?
    let new_total2 = total_time.saturating_sub(convert_time);
    let speedup2 = total_time.as_secs_f64() / new_total2.as_secs_f64();

    println!("  Opportunity 2: Eliminate Weight Conversions");
    println!("    Current: {:?} ({:.1}% of total)", convert_time,
             (convert_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0);
    println!("    Potential speedup: {:.2}x overall\n", speedup2);

    // Combined estimate
    let combined_savings = potential_savings + convert_time;
    let new_total_combined = total_time.saturating_sub(combined_savings);
    let combined_speedup = total_time.as_secs_f64() / new_total_combined.as_secs_f64();

    println!("  Combined (Both Optimizations):");
    println!("    Potential speedup: {:.2}x overall", combined_speedup);

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║         END HOT PATH PROFILING                        ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}

fn main() {
    profile_with_instrumentation();
}
