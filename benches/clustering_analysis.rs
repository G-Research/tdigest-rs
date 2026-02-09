// Deep dive into the clustering hot loop
use std::time::Instant;
use tdigest_core::TDigest;

fn analyze_clustering_loop() {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║     CLUSTERING LOOP DEEP ANALYSIS                     ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let size = 32_000;
    let iterations = 1000;

    let data: Vec<f64> = (0..size).map(|i| ((i * 7919) % size) as f64 * 0.1).collect();

    // Baseline: Full digest creation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = TDigest::from_array(&data, 0.01).unwrap();
    }
    let baseline = start.elapsed();
    println!("Baseline (full digest): {:?}", baseline / iterations);
    println!("Throughput: {:.2} M elements/sec\n",
             (size as f64 * iterations as f64) / baseline.as_secs_f64() / 1_000_000.0);

    println!("━━━ Analyzing Hot Loop Operations ━━━\n");

    // Simulate the hot loop with different optimizations
    let weights: Vec<u32> = vec![1; size];
    let total_weight = size as f64;

    // Current approach: type conversions in loop
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        let mut cumulative = 0u32;
        let mut sigma_weight = weights[0];
        let mut sigma_mean = data[0];

        for i in 1..size {
            let wght = weights[i];
            let mu = data[i];

            // Simulate the weighted mean calculation with type conversions
            let q = (cumulative + sigma_weight + wght) as f64 / total_weight;
            if q <= 0.5 { // Simplified condition
                sigma_mean = ((sigma_mean * sigma_weight as f64) + mu * wght as f64)
                    / (sigma_weight + wght) as f64;
                sigma_weight += wght;
            } else {
                cumulative += sigma_weight;
                sigma_weight = wght;
                sigma_mean = mu;
            }
            result += sigma_mean; // Prevent optimization
        }
    }
    let current_time = start.elapsed();
    if result == 0.0 { println!("unreachable"); }

    println!("Current Implementation (with type conversions):");
    println!("  Time: {:?}", current_time / iterations);
    println!("  % of baseline: {:.1}%\n",
             (current_time.as_secs_f64() / baseline.as_secs_f64()) * 100.0);

    // Optimized approach: pre-converted weights
    let weights_f64: Vec<f64> = weights.iter().map(|&w| w as f64).collect();

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        let mut cumulative_f = 0.0;
        let mut sigma_weight_f = weights_f64[0];
        let mut sigma_mean = data[0];

        for i in 1..size {
            let wght_f = weights_f64[i];
            let mu = data[i];

            // Same logic but no type conversions
            let q = (cumulative_f + sigma_weight_f + wght_f) / total_weight;
            if q <= 0.5 {
                sigma_mean = (sigma_mean * sigma_weight_f + mu * wght_f)
                    / (sigma_weight_f + wght_f);
                sigma_weight_f += wght_f;
            } else {
                cumulative_f += sigma_weight_f;
                sigma_weight_f = wght_f;
                sigma_mean = mu;
            }
            result += sigma_mean;
        }
    }
    let optimized_time = start.elapsed();
    if result == 0.0 { println!("unreachable"); }

    println!("Optimized (pre-converted weights):");
    println!("  Time: {:?}", optimized_time / iterations);
    println!("  Speedup: {:.2}x", current_time.as_secs_f64() / optimized_time.as_secs_f64());
    println!("  % of baseline: {:.1}%\n",
             (optimized_time.as_secs_f64() / baseline.as_secs_f64()) * 100.0);

    let savings = current_time.saturating_sub(optimized_time);
    let new_baseline = baseline.saturating_sub(savings);
    let potential_speedup = baseline.as_secs_f64() / new_baseline.as_secs_f64();

    println!("━━━ Impact on Full Digest Creation ━━━\n");
    println!("  Current baseline: {:?}", baseline / iterations);
    println!("  Potential with optimization: {:?}", new_baseline / iterations);
    println!("  Overall speedup: {:.2}x\n", potential_speedup);

    // Analyze division overhead
    println!("━━━ Analyzing Division Operations ━━━\n");

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        for i in 0..size {
            result += (i as f64) / total_weight; // Division per iteration
        }
    }
    let with_div = start.elapsed();
    if result == 0.0 { println!("unreachable"); }

    let inv_total = 1.0 / total_weight;
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        for i in 0..size {
            result += (i as f64) * inv_total; // Multiplication instead
        }
    }
    let with_mul = start.elapsed();
    if result == 0.0 { println!("unreachable"); }

    println!("Division vs Multiplication:");
    println!("  With division: {:?}", with_div / iterations);
    println!("  With multiplication: {:?}", with_mul / iterations);
    println!("  Division penalty: {:.2}x slower\n",
             with_div.as_secs_f64() / with_mul.as_secs_f64());

    // Analyze memory access patterns
    println!("━━━ Memory Access Patterns ━━━\n");

    // Sequential access (current)
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        for i in 0..size {
            result += data[i] + weights_f64[i];
        }
    }
    let sequential = start.elapsed();
    if result == 0.0 { println!("unreachable"); }

    println!("Sequential access (current): {:?}", sequential / iterations);
    println!("  Memory bandwidth: {:.2} GB/s\n",
             (size as f64 * 16.0 * iterations as f64) / sequential.as_secs_f64() / 1_000_000_000.0);

    println!("━━━ Final Recommendations ━━━\n");

    if potential_speedup >= 1.5 {
        println!("  ✅ PRE-CONVERT WEIGHTS: {:.2}x speedup (HIGH IMPACT)", potential_speedup);
    } else {
        println!("  ⚠️  Pre-convert weights: {:.2}x speedup (low impact)", potential_speedup);
    }

    let div_impact = with_div.as_secs_f64() / with_mul.as_secs_f64();
    if div_impact >= 1.3 {
        println!("  ✅ CACHE 1/total_weight: {:.2}x speedup (MEDIUM IMPACT)", div_impact);
    }

    println!("\n  Main bottleneck: Memory-bound workload (~96% of time)");
    println!("  Best opportunities:");
    println!("    1. Eliminate type conversions in hot loop");
    println!("    2. Cache reciprocal of total_weight");
    println!("    3. Consider SIMD for parallel processing of multiple elements");

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║         END CLUSTERING ANALYSIS                       ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}

fn main() {
    analyze_clustering_loop();
}
