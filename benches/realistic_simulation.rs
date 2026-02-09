// Simulate the exact pattern of the clustering algorithm
use std::time::Instant;
use tdigest_core::TDigest;

fn simulate_clustering_pattern() {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║     REALISTIC CLUSTERING PATTERN SIMULATION           ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let size = 32_000;
    let iterations = 1000;

    // Get actual digest to see compression ratio
    let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
    let digest = TDigest::from_array(&data, 0.01).unwrap();
    let final_size = digest.means.len();

    println!("Compression ratio: {} → {} centroids ({:.1}x compression)\n",
             size, final_size, size as f64 / final_size as f64);

    println!("━━━ Baseline Performance ━━━\n");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = TDigest::from_array(&data, 0.01).unwrap();
    }
    let baseline = start.elapsed();
    println!("Full digest creation: {:?}", baseline / iterations);
    println!("Throughput: {:.2} M elements/sec\n",
             (size as f64 * iterations as f64) / baseline.as_secs_f64() / 1_000_000.0);

    println!("━━━ Simulating Different Vector Strategies ━━━\n");

    // Strategy 1: Current approach - Vec::with_capacity(n) then push
    let start = Instant::now();
    for _ in 0..iterations {
        let mut means = Vec::with_capacity(size);
        let mut weights = Vec::with_capacity(size);
        let mut masks = Vec::with_capacity(size);

        // Simulate: process all elements, but only keep ~final_size
        for i in 0..size {
            if i % (size / final_size) == 0 {  // Realistic push rate
                means.push(i as f64);
                weights.push(i as u32);
                masks.push(i % 2 == 0);
            }
        }

        std::hint::black_box((means, weights, masks));
    }
    let current_strategy = start.elapsed();
    println!("Strategy 1: with_capacity(n=32k) + conditional push");
    println!("   Time: {:?}", current_strategy / iterations);
    println!("   % of baseline: {:.1}%\n",
             (current_strategy.as_secs_f64() / baseline.as_secs_f64()) * 100.0);

    // Strategy 2: Pre-allocate to expected size
    let expected_size = size / 100; // Typical compression
    let start = Instant::now();
    for _ in 0..iterations {
        let mut means = Vec::with_capacity(expected_size);
        let mut weights = Vec::with_capacity(expected_size);
        let mut masks = Vec::with_capacity(expected_size);

        for i in 0..size {
            if i % (size / final_size) == 0 {
                means.push(i as f64);
                weights.push(i as u32);
                masks.push(i % 2 == 0);
            }
        }

        std::hint::black_box((means, weights, masks));
    }
    let optimized_prealloc = start.elapsed();
    println!("Strategy 2: with_capacity(expected=320) + conditional push");
    println!("   Time: {:?}", optimized_prealloc / iterations);
    println!("   Speedup: {:.2}x", current_strategy.as_secs_f64() / optimized_prealloc.as_secs_f64());
    println!("   % of baseline: {:.1}%\n",
             (optimized_prealloc.as_secs_f64() / baseline.as_secs_f64()) * 100.0);

    // Strategy 3: Pre-initialize to max size, then use index
    let start = Instant::now();
    for _ in 0..iterations {
        let mut means = vec![0.0; size];
        let mut weights = vec![0u32; size];
        let mut masks = vec![false; size];
        let mut count = 0;

        for i in 0..size {
            if i % (size / final_size) == 0 {
                means[count] = i as f64;
                weights[count] = i as u32;
                masks[count] = i % 2 == 0;
                count += 1;
            }
        }

        // Truncate to actual size
        means.truncate(count);
        weights.truncate(count);
        masks.truncate(count);

        std::hint::black_box((means, weights, masks));
    }
    let preinit_strategy = start.elapsed();
    println!("Strategy 3: pre-initialize to max + index + truncate");
    println!("   Time: {:?}", preinit_strategy / iterations);
    println!("   Speedup: {:.2}x", current_strategy.as_secs_f64() / preinit_strategy.as_secs_f64());
    println!("   % of baseline: {:.1}%\n",
             (preinit_strategy.as_secs_f64() / baseline.as_secs_f64()) * 100.0);

    println!("━━━ Testing with Realistic Workload ━━━\n");

    // Simulate actual clustering behavior: iterate, calculate, conditionally push
    let weights: Vec<u32> = vec![1; size];
    let total_weight = size as f64;
    let delta = 0.01;

    let start = Instant::now();
    for _ in 0..iterations {
        let mut new_means = Vec::with_capacity(size);
        let mut new_weights = Vec::with_capacity(size);

        let mut cumulative = 0u32;
        let mut sigma_weight = weights[0];
        let mut sigma_mean = data[0];

        for i in 1..size {
            let wght = weights[i];
            let mu = data[i];

            let q = (cumulative + sigma_weight + wght) as f64 / total_weight;
            let q_limit = delta; // Simplified

            if q <= q_limit {
                // Merge
                sigma_mean = ((sigma_mean * sigma_weight as f64) + mu * wght as f64)
                    / (sigma_weight + wght) as f64;
                sigma_weight += wght;
            } else {
                // Push current cluster
                new_means.push(sigma_mean);
                new_weights.push(sigma_weight);

                cumulative += sigma_weight;
                sigma_weight = wght;
                sigma_mean = mu;
            }
        }

        // Push final cluster
        new_means.push(sigma_mean);
        new_weights.push(sigma_weight);

        std::hint::black_box((new_means, new_weights));
    }
    let realistic_current = start.elapsed();
    println!("Realistic simulation (current approach):");
    println!("   Time: {:?}", realistic_current / iterations);
    println!("   % of baseline: {:.1}%\n",
             (realistic_current.as_secs_f64() / baseline.as_secs_f64()) * 100.0);

    // Now with pre-converted weights
    let weights_f: Vec<f64> = weights.iter().map(|&w| w as f64).collect();

    let start = Instant::now();
    for _ in 0..iterations {
        let mut new_means = Vec::with_capacity(expected_size);
        let mut new_weights = Vec::with_capacity(expected_size);

        let mut cumulative_f = 0.0;
        let mut sigma_weight_f = weights_f[0];
        let mut sigma_mean = data[0];

        for i in 1..size {
            let wght_f = weights_f[i];
            let mu = data[i];

            let q = (cumulative_f + sigma_weight_f + wght_f) / total_weight;
            let q_limit = delta;

            if q <= q_limit {
                sigma_mean = (sigma_mean * sigma_weight_f + mu * wght_f)
                    / (sigma_weight_f + wght_f);
                sigma_weight_f += wght_f;
            } else {
                new_means.push(sigma_mean);
                new_weights.push(sigma_weight_f as u32);

                cumulative_f += sigma_weight_f;
                sigma_weight_f = wght_f;
                sigma_mean = mu;
            }
        }

        new_means.push(sigma_mean);
        new_weights.push(sigma_weight_f as u32);

        std::hint::black_box((new_means, new_weights));
    }
    let realistic_optimized = start.elapsed();
    println!("Realistic simulation (optimized: pre-converted + smaller capacity):");
    println!("   Time: {:?}", realistic_optimized / iterations);
    println!("   Speedup vs current: {:.2}x",
             realistic_current.as_secs_f64() / realistic_optimized.as_secs_f64());
    println!("   % of baseline: {:.1}%\n",
             (realistic_optimized.as_secs_f64() / baseline.as_secs_f64()) * 100.0);

    // Calculate potential overall speedup
    let savings = realistic_current.saturating_sub(realistic_optimized);
    let new_baseline = baseline.saturating_sub(savings);
    let overall_speedup = baseline.as_secs_f64() / new_baseline.as_secs_f64();

    println!("━━━ Projected Impact ━━━\n");
    println!("  Current realistic loop: {:?} ({:.1}% of total)",
             realistic_current / iterations,
             (realistic_current.as_secs_f64() / baseline.as_secs_f64()) * 100.0);
    println!("  Optimized realistic loop: {:?} ({:.1}% of total)",
             realistic_optimized / iterations,
             (realistic_optimized.as_secs_f64() / baseline.as_secs_f64()) * 100.0);
    println!("  Time saved: {:?}", savings / iterations);
    println!("\n  Projected overall speedup: {:.2}x", overall_speedup);

    if overall_speedup >= 1.5 {
        println!("\n  ✅ RECOMMENDED: This optimization gives 1.5x+ speedup!");
    } else if overall_speedup >= 1.2 {
        println!("\n  ✅ Worthwhile: This optimization gives {:.1}%+ improvement", (overall_speedup - 1.0) * 100.0);
    } else {
        println!("\n  ⚠️  Limited impact: Only {:.1}% improvement", (overall_speedup - 1.0) * 100.0);
    }

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║         END REALISTIC SIMULATION                      ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}

fn main() {
    simulate_clustering_pattern();
}
