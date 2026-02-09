// Profile vector operations and allocations in the hot path
use std::time::Instant;

fn profile_vector_operations() {
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║         VECTOR OPERATIONS PROFILING                   ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let size = 32_000;
    let iterations = 1000;

    println!("━━━ Vector Allocation Strategies ━━━\n");

    // Strategy 1: No pre-allocation (naive)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut vec = Vec::new();
        for i in 0..size {
            vec.push(i as f64);
        }
        std::hint::black_box(vec);
    }
    let no_prealloc = start.elapsed();
    println!("1. No pre-allocation:");
    println!("   Time: {:?}", no_prealloc / iterations);

    // Strategy 2: Pre-allocated to exact size
    let start = Instant::now();
    for _ in 0..iterations {
        let mut vec = Vec::with_capacity(size);
        for i in 0..size {
            vec.push(i as f64);
        }
        std::hint::black_box(vec);
    }
    let with_prealloc = start.elapsed();
    println!("\n2. Pre-allocated (with_capacity):");
    println!("   Time: {:?}", with_prealloc / iterations);
    println!("   Speedup: {:.2}x", no_prealloc.as_secs_f64() / with_prealloc.as_secs_f64());

    // Strategy 3: Pre-allocated but conditional push
    let start = Instant::now();
    for _ in 0..iterations {
        let mut vec = Vec::with_capacity(size / 10); // Realistic for digest
        for i in 0..size {
            if i % 10 == 0 { // ~10% push rate
                vec.push(i as f64);
            }
        }
        std::hint::black_box(vec);
    }
    let conditional_push = start.elapsed();
    println!("\n3. Conditional push (~10% push rate):");
    println!("   Time: {:?}", conditional_push / iterations);

    println!("\n━━━ Push vs Pre-initialized ━━━\n");

    // Approach A: Push during iteration
    let start = Instant::now();
    for _ in 0..iterations {
        let mut means = Vec::with_capacity(100);
        let mut weights = Vec::with_capacity(100);
        let mut masks = Vec::with_capacity(100);

        for i in 0..100 {
            means.push(i as f64);
            weights.push(i as u32);
            masks.push(i % 2 == 0);
        }

        std::hint::black_box((means, weights, masks));
    }
    let push_approach = start.elapsed();
    println!("Push approach (3 vectors, 100 elements each):");
    println!("   Time: {:?}", push_approach / iterations);

    // Approach B: Pre-initialize with default values
    let start = Instant::now();
    for _ in 0..iterations {
        let mut means = vec![0.0; 100];
        let mut weights = vec![0u32; 100];
        let mut masks = vec![false; 100];

        for i in 0..100 {
            means[i] = i as f64;
            weights[i] = i as u32;
            masks[i] = i % 2 == 0;
        }

        std::hint::black_box((means, weights, masks));
    }
    let preinit_approach = start.elapsed();
    println!("\nPre-initialized approach:");
    println!("   Time: {:?}", preinit_approach / iterations);
    println!("   Speedup: {:.2}x", push_approach.as_secs_f64() / preinit_approach.as_secs_f64());

    println!("\n━━━ Iterator Overhead ━━━\n");

    let data1: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let data2: Vec<u32> = (0..size).map(|i| i as u32).collect();
    let data3: Vec<bool> = (0..size).map(|i| i % 2 == 0).collect();

    // Manual indexing
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        for i in 0..size {
            sum += data1[i] + data2[i] as f64;
            if data3[i] { sum += 1.0; }
        }
    }
    let manual_index = start.elapsed();
    std::hint::black_box(sum);
    println!("Manual indexing:");
    println!("   Time: {:?}", manual_index / iterations);

    // zip iterator
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        for ((&v1, &v2), &v3) in data1.iter().zip(data2.iter()).zip(data3.iter()) {
            sum += v1 + v2 as f64;
            if v3 { sum += 1.0; }
        }
    }
    let zip_iter = start.elapsed();
    std::hint::black_box(sum);
    println!("\nZip iterator:");
    println!("   Time: {:?}", zip_iter / iterations);
    println!("   Overhead: {:.2}x", zip_iter.as_secs_f64() / manual_index.as_secs_f64());

    // izip! macro (from itertools)
    use itertools::izip;
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        for (&v1, &v2, &v3) in izip!(&data1, &data2, &data3) {
            sum += v1 + v2 as f64;
            if v3 { sum += 1.0; }
        }
    }
    let izip_macro = start.elapsed();
    std::hint::black_box(sum);
    println!("\nizip! macro:");
    println!("   Time: {:?}", izip_macro / iterations);
    println!("   Overhead: {:.2}x", izip_macro.as_secs_f64() / manual_index.as_secs_f64());

    println!("\n━━━ Branch Prediction ━━━\n");

    // Predictable branches
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        for i in 0..size {
            if i % 2 == 0 {  // Predictable pattern
                sum += i as f64;
            }
        }
    }
    let predictable = start.elapsed();
    std::hint::black_box(sum);
    println!("Predictable branches (i % 2 == 0):");
    println!("   Time: {:?}", predictable / iterations);

    // Unpredictable branches
    let random_data: Vec<bool> = (0..size).map(|i| ((i * 7919) % size) % 2 == 0).collect();
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        for i in 0..size {
            if random_data[i] {  // Unpredictable pattern
                sum += i as f64;
            }
        }
    }
    let unpredictable = start.elapsed();
    std::hint::black_box(sum);
    println!("\nUnpredictable branches (pseudo-random):");
    println!("   Time: {:?}", unpredictable / iterations);
    println!("   Branch misprediction penalty: {:.2}x",
             unpredictable.as_secs_f64() / predictable.as_secs_f64());

    println!("\n━━━ Key Findings ━━━\n");

    if no_prealloc.as_secs_f64() / with_prealloc.as_secs_f64() > 1.5 {
        println!("  ✅ Pre-allocation: {:.2}x speedup (HIGH IMPACT)",
                 no_prealloc.as_secs_f64() / with_prealloc.as_secs_f64());
    }

    if push_approach.as_secs_f64() / preinit_approach.as_secs_f64() > 1.2 {
        println!("  ✅ Pre-initialize vs Push: {:.2}x speedup (MEDIUM IMPACT)",
                 push_approach.as_secs_f64() / preinit_approach.as_secs_f64());
    }

    if zip_iter.as_secs_f64() / manual_index.as_secs_f64() > 1.1 {
        println!("  ⚠️  Iterator overhead: {:.2}x slower",
                 zip_iter.as_secs_f64() / manual_index.as_secs_f64());
    }

    if unpredictable.as_secs_f64() / predictable.as_secs_f64() > 1.5 {
        println!("  ⚠️  Branch misprediction: {:.2}x slower",
                 unpredictable.as_secs_f64() / predictable.as_secs_f64());
    }

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║         END VECTOR OPERATIONS PROFILING               ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}

fn main() {
    profile_vector_operations();
}
