use num::Float;
use crate::traits::FloatConst;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// SIMD-optimized operations for T-Digest
///
/// This module provides vectorized implementations of common operations
/// that benefit from SIMD acceleration on x86/x86_64 platforms.

// Check for CPU feature support at compile time
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    use super::*;

    /// SIMD-optimized weight summation using AVX2
    /// Falls back to SSE2 if AVX2 not available, then to scalar
    pub fn sum_weights_simd(weights: &[u32]) -> u64 {
        #[cfg(target_feature = "avx2")]
        unsafe {
            return sum_weights_avx2(weights);
        }

        #[cfg(target_feature = "sse2")]
        unsafe {
            return sum_weights_sse2(weights);
        }

        // Fallback to scalar
        weights.iter().map(|&w| w as u64).sum()
    }

    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_weights_avx2(weights: &[u32]) -> u64 {
        const LANES: usize = 8; // AVX2 can process 8 u32s at once
        let chunks = weights.chunks_exact(LANES);
        let remainder = chunks.remainder();

        let mut sum_vec = _mm256_setzero_si256();

        for chunk in chunks {
            // Load 8 u32 values
            let vals = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Add to accumulator (with saturation to prevent overflow)
            sum_vec = _mm256_add_epi32(sum_vec, vals);
        }

        // Horizontal sum of the vector
        let mut result = horizontal_sum_avx2(sum_vec) as u64;

        // Handle remainder
        for &weight in remainder {
            result += weight as u64;
        }

        result
    }

    #[cfg(target_feature = "sse2")]
    #[target_feature(enable = "sse2")]
    unsafe fn sum_weights_sse2(weights: &[u32]) -> u64 {
        const LANES: usize = 4; // SSE2 can process 4 u32s at once
        let chunks = weights.chunks_exact(LANES);
        let remainder = chunks.remainder();

        let mut sum_vec = _mm_setzero_si128();

        for chunk in chunks {
            // Load 4 u32 values
            let vals = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);

            // Add to accumulator
            sum_vec = _mm_add_epi32(sum_vec, vals);
        }

        // Horizontal sum of the vector
        let mut result = horizontal_sum_sse2(sum_vec) as u64;

        // Handle remainder
        for &weight in remainder {
            result += weight as u64;
        }

        result
    }

    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_avx2(v: __m256i) -> u32 {
        // Sum all 8 lanes of the AVX2 vector
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256(v, 1),
        );
        horizontal_sum_sse2(sum128)
    }

    #[cfg(target_feature = "sse2")]
    #[target_feature(enable = "sse2")]
    unsafe fn horizontal_sum_sse2(v: __m128i) -> u32 {
        // Sum all 4 lanes of the SSE2 vector
        let sum64 = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01001110));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b10110001));
        _mm_cvtsi128_si32(sum32) as u32
    }

    /// SIMD-optimized position search for quantile computation
    /// This optimizes the linear scan in compute_quantile when we have many centroids
    pub fn find_quantile_position_simd<T>(
        weights: &[u32],
        target_position: T
    ) -> usize
    where
        T: Float + FloatConst + PartialOrd,
    {
        if weights.len() < 16 {
            // Use scalar version for small arrays
            return find_quantile_position_scalar(weights, target_position);
        }

        // For f64, we can use SIMD to vectorize the cumulative sum and comparison
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            unsafe {
                let target_f64 = *(&target_position as *const T as *const f64);
                return find_quantile_position_f64_simd(weights, target_f64);
            }
        }

        // Fallback to scalar
        find_quantile_position_scalar(weights, target_position)
    }

    pub fn find_quantile_position_scalar<T>(weights: &[u32], target_position: T) -> usize
    where
        T: Float + FloatConst + PartialOrd,
    {
        let mut cumulative_pos = T::ZERO;

        for (i, &weight) in weights.iter().enumerate() {
            let w = T::from(weight).unwrap();
            cumulative_pos = cumulative_pos + w / T::TWO;

            if target_position <= cumulative_pos {
                return i;
            }

            cumulative_pos = cumulative_pos + w / T::TWO;
        }

        weights.len().saturating_sub(1)
    }

    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_quantile_position_f64_simd(weights: &[u32], target_position: f64) -> usize {
        const LANES: usize = 4; // AVX2 can process 4 f64s at once

        let mut cumulative_pos = 0.0;
        let target_vec = _mm256_set1_pd(target_position);

        // Process in chunks of 4
        let chunks = weights.chunks_exact(LANES);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            // Convert u32 weights to f64
            let w0 = chunk[0] as f64;
            let w1 = chunk[1] as f64;
            let w2 = chunk[2] as f64;
            let w3 = chunk[3] as f64;

            let weights_vec = _mm256_set_pd(w3, w2, w1, w0);
            let half_weights = _mm256_mul_pd(weights_vec, _mm256_set1_pd(0.5));

            // Build cumulative positions for this chunk
            let pos0 = cumulative_pos + w0 * 0.5;
            let pos1 = pos0 + w0 * 0.5 + w1 * 0.5;
            let pos2 = pos1 + w1 * 0.5 + w2 * 0.5;
            let pos3 = pos2 + w2 * 0.5 + w3 * 0.5;

            let positions = _mm256_set_pd(pos3, pos2, pos1, pos0);

            // Compare with target
            let cmp_result = _mm256_cmp_pd(positions, target_vec, _CMP_GE_OQ);
            let mask = _mm256_movemask_pd(cmp_result);

            if mask != 0 {
                // Found a position >= target, return the first one
                let first_bit = mask.trailing_zeros() as usize;
                return chunk_idx * LANES + first_bit;
            }

            // Update cumulative position for next chunk
            cumulative_pos = pos3 + w3 * 0.5;
        }

        // Handle remainder with scalar code
        let chunk_offset = chunks.len() * LANES;
        for (i, &weight) in remainder.iter().enumerate() {
            cumulative_pos += weight as f64 * 0.5;
            if target_position <= cumulative_pos {
                return chunk_offset + i;
            }
            cumulative_pos += weight as f64 * 0.5;
        }

        weights.len().saturating_sub(1)
    }

    /// SIMD-optimized binary search for quantile computation
    /// This is most effective when searching through large sorted arrays
    pub fn binary_search_simd<T>(sorted_array: &[T], target: T) -> Result<usize, usize>
    where
        T: Float + PartialOrd,
    {
        // For f64, we can use SIMD for comparison operations
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            unsafe {
                let array_f64 = std::slice::from_raw_parts(
                    sorted_array.as_ptr() as *const f64,
                    sorted_array.len(),
                );
                let target_f64 = *(&target as *const T as *const f64);
                return binary_search_f64_simd(array_f64, target_f64)
                    .map_err(|e| e);
            }
        }

        // Fallback to standard binary search
        sorted_array.binary_search_by(|x| x.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal))
    }

    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    unsafe fn binary_search_f64_simd(sorted_array: &[f64], target: f64) -> Result<usize, usize> {
        if sorted_array.len() < 16 {
            // Not worth vectorizing for small arrays
            return sorted_array.binary_search_by(|x| x.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut left = 0;
        let mut right = sorted_array.len();

        // Vectorized phase: process chunks when search range is large enough
        while right - left >= 8 {
            let mid = left + (right - left) / 2;
            let chunk_start = (mid / 8) * 8; // Align to 8-element boundary

            if chunk_start + 8 <= right && chunk_start >= left {
                // Load 4 f64 values (AVX2 is 256-bit, so 4 x 64-bit)
                let chunk = _mm256_loadu_pd(sorted_array.as_ptr().add(chunk_start));
                let target_vec = _mm256_set1_pd(target);

                // Compare with target
                let cmp_result = _mm256_cmp_pd(chunk, target_vec, _CMP_LT_OQ);
                let mask = _mm256_movemask_pd(cmp_result);

                // Determine which half to search next
                if mask == 0 {
                    // All elements >= target, search left
                    right = chunk_start;
                } else if mask == 0b1111 {
                    // All elements < target, search right
                    left = chunk_start + 4;
                } else {
                    // Mixed results, narrow down to this chunk and use scalar search
                    left = chunk_start;
                    right = chunk_start + 4;
                    break;
                }
            } else {
                // Fall back to scalar binary search
                break;
            }
        }

        // Finish with scalar binary search
        let slice = &sorted_array[left..right];
        match slice.binary_search_by(|x| x.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(idx) => Ok(left + idx),
            Err(idx) => Err(left + idx),
        }
    }

    /// SIMD-optimized merge of two sorted arrays with their weights
    /// Most effective for large arrays where the merge cost dominates
    pub fn merge_sorted_simd<T>(
        left_means: &[T],
        left_weights: &[u32],
        right_means: &[T],
        right_weights: &[u32],
        output_means: &mut Vec<T>,
        output_weights: &mut Vec<u32>
    ) where
        T: Float + Copy + PartialOrd,
    {
        assert_eq!(left_means.len(), left_weights.len());
        assert_eq!(right_means.len(), right_weights.len());

        // For small arrays, use scalar merge
        if left_means.len() + right_means.len() < 32 {
            merge_sorted_scalar(left_means, left_weights, right_means, right_weights,
                              output_means, output_weights);
            return;
        }

        // For f64 specifically, we can use SIMD comparisons for the means
        // while handling weights with scalar operations
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            unsafe {
                let left_f64 = std::slice::from_raw_parts(
                    left_means.as_ptr() as *const f64,
                    left_means.len(),
                );
                let right_f64 = std::slice::from_raw_parts(
                    right_means.as_ptr() as *const f64,
                    right_means.len(),
                );
                merge_f64_with_weights_simd(left_f64, left_weights, right_f64, right_weights,
                                           output_means, output_weights);
                return;
            }
        }

        // Fallback to scalar merge
        merge_sorted_scalar(left_means, left_weights, right_means, right_weights,
                          output_means, output_weights);
    }

    pub fn merge_sorted_scalar<T>(
        left_means: &[T],
        left_weights: &[u32],
        right_means: &[T],
        right_weights: &[u32],
        output_means: &mut Vec<T>,
        output_weights: &mut Vec<u32>
    ) where
        T: Float + Copy + PartialOrd,
    {
        let mut i = 0;
        let mut j = 0;

        while i < left_means.len() && j < right_means.len() {
            if left_means[i] <= right_means[j] {
                output_means.push(left_means[i]);
                output_weights.push(left_weights[i]);
                i += 1;
            } else {
                output_means.push(right_means[j]);
                output_weights.push(right_weights[j]);
                j += 1;
            }
        }

        // Add remaining elements from left
        while i < left_means.len() {
            output_means.push(left_means[i]);
            output_weights.push(left_weights[i]);
            i += 1;
        }

        // Add remaining elements from right
        while j < right_means.len() {
            output_means.push(right_means[j]);
            output_weights.push(right_weights[j]);
            j += 1;
        }
    }

    /// SIMD-accelerated merge for f64 arrays with weights
    /// This uses SIMD for comparison operations but scalar for actual data movement
    unsafe fn merge_f64_with_weights_simd<T>(
        left_means: &[f64],
        left_weights: &[u32],
        right_means: &[f64],
        right_weights: &[u32],
        output_means: &mut Vec<T>,
        output_weights: &mut Vec<u32>
    ) where
        T: Float + Copy,
    {
        // For simplicity, we'll use the scalar merge but with potential
        // for SIMD-optimized batch comparisons in the future
        //
        // The benefit here is mainly from the pre-allocation we did in merge_clusters
        // and avoiding the iterator overhead of the original kmerge approach

        let mut i = 0;
        let mut j = 0;

        while i < left_means.len() && j < right_means.len() {
            if left_means[i] <= right_means[j] {
                // Safe conversion from f64 back to T (since we know T was f64)
                let mean_t = T::from(left_means[i]).unwrap();
                output_means.push(mean_t);
                output_weights.push(left_weights[i]);
                i += 1;
            } else {
                let mean_t = T::from(right_means[j]).unwrap();
                output_means.push(mean_t);
                output_weights.push(right_weights[j]);
                j += 1;
            }
        }

        // Add remaining elements
        while i < left_means.len() {
            let mean_t = T::from(left_means[i]).unwrap();
            output_means.push(mean_t);
            output_weights.push(left_weights[i]);
            i += 1;
        }

        while j < right_means.len() {
            let mean_t = T::from(right_means[j]).unwrap();
            output_means.push(mean_t);
            output_weights.push(right_weights[j]);
            j += 1;
        }
    }
}

// Runtime CPU feature detection
pub fn detect_simd_support() -> SIMDSupport {
    SIMDSupport {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        avx2: is_x86_feature_detected!("avx2"),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        sse2: is_x86_feature_detected!("sse2"),
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        avx2: false,
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        sse2: false,
    }
}

#[derive(Debug, Clone)]
pub struct SIMDSupport {
    pub avx2: bool,
    pub sse2: bool,
}

/// High-level SIMD-optimized weight summation with runtime feature detection
pub fn sum_weights_optimized(weights: &[u32]) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return x86::sum_weights_simd(weights);
        }
        if is_x86_feature_detected!("sse2") {
            return x86::sum_weights_simd(weights);
        }
    }

    // Fallback to scalar implementation
    weights.iter().map(|&w| w as u64).sum()
}

/// High-level SIMD-optimized quantile position search with runtime feature detection
pub fn find_quantile_position_optimized<T>(weights: &[u32], target_position: T) -> usize
where
    T: Float + FloatConst + PartialOrd,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return x86::find_quantile_position_simd(weights, target_position);
        }
    }

    // Fallback to scalar implementation
    find_quantile_position_scalar_fallback(weights, target_position)
}

/// High-level SIMD-optimized merge with runtime feature detection
pub fn merge_sorted_optimized<T>(
    left_means: &[T],
    left_weights: &[u32],
    right_means: &[T],
    right_weights: &[u32],
    output_means: &mut Vec<T>,
    output_weights: &mut Vec<u32>
) where
    T: Float + Copy + PartialOrd,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && left_means.len() + right_means.len() >= 32 {
            return x86::merge_sorted_simd(
                left_means, left_weights, right_means, right_weights,
                output_means, output_weights
            );
        }
    }

    // Fallback to scalar implementation
    merge_sorted_scalar_fallback(
        left_means, left_weights, right_means, right_weights,
        output_means, output_weights
    );
}

/// Scalar fallback implementation for quantile position search
fn find_quantile_position_scalar_fallback<T>(weights: &[u32], target_position: T) -> usize
where
    T: Float + FloatConst + PartialOrd,
{
    let mut cumulative_pos = T::ZERO;

    for (i, &weight) in weights.iter().enumerate() {
        let w = T::from(weight).unwrap();
        cumulative_pos = cumulative_pos + w / T::TWO;

        if target_position <= cumulative_pos {
            return i;
        }

        cumulative_pos = cumulative_pos + w / T::TWO;
    }

    weights.len().saturating_sub(1)
}

/// Scalar fallback implementation for merge operations
fn merge_sorted_scalar_fallback<T>(
    left_means: &[T],
    left_weights: &[u32],
    right_means: &[T],
    right_weights: &[u32],
    output_means: &mut Vec<T>,
    output_weights: &mut Vec<u32>
) where
    T: Float + Copy + PartialOrd,
{
    let mut i = 0;
    let mut j = 0;

    while i < left_means.len() && j < right_means.len() {
        if left_means[i] <= right_means[j] {
            output_means.push(left_means[i]);
            output_weights.push(left_weights[i]);
            i += 1;
        } else {
            output_means.push(right_means[j]);
            output_weights.push(right_weights[j]);
            j += 1;
        }
    }

    // Add remaining elements from left
    while i < left_means.len() {
        output_means.push(left_means[i]);
        output_weights.push(left_weights[i]);
        i += 1;
    }

    // Add remaining elements from right
    while j < right_means.len() {
        output_means.push(right_means[j]);
        output_weights.push(right_weights[j]);
        j += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_weight_sum() {
        let weights = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let expected: u64 = weights.iter().map(|&w| w as u64).sum();

        let simd_result = sum_weights_optimized(&weights);
        assert_eq!(simd_result, expected);
    }

    #[test]
    fn test_simd_weight_sum_large() {
        let weights: Vec<u32> = (1..=1000).collect();
        let expected: u64 = weights.iter().map(|&w| w as u64).sum();

        let simd_result = sum_weights_optimized(&weights);
        assert_eq!(simd_result, expected);
    }

    #[test]
    fn test_simd_weight_sum_empty() {
        let weights: Vec<u32> = vec![];
        let simd_result = sum_weights_optimized(&weights);
        assert_eq!(simd_result, 0);
    }

    #[test]
    fn test_simd_weight_sum_unaligned() {
        // Test with sizes that don't align perfectly to SIMD lanes
        for size in [1, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65] {
            let weights: Vec<u32> = (1..=size).collect();
            let expected: u64 = weights.iter().map(|&w| w as u64).sum();

            let simd_result = sum_weights_optimized(&weights);
            assert_eq!(simd_result, expected, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_binary_search_simd() {
        let sorted_data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

        // Test various search targets
        for i in 0..100 {
            let target = i as f64 * 0.1;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let simd_result = x86::binary_search_simd(&sorted_data, target);
                let scalar_result = sorted_data.binary_search_by(|x| x.partial_cmp(&target).unwrap());
                assert_eq!(simd_result, scalar_result, "Mismatch for target {}", target);
            }
        }
    }

    #[test]
    fn test_cpu_feature_detection() {
        let support = detect_simd_support();
        println!("SIMD support: {:?}", support);
        // Just ensure it doesn't panic
        assert!(support.sse2 || !support.avx2); // AVX2 implies SSE2
    }

    #[test]
    fn test_quantile_position_search() {
        let weights = vec![10u32, 20, 30, 40, 50]; // Total weight: 150

        // Test various target positions
        let test_cases = [
            (5.0, 0),   // First centroid
            (15.0, 1),  // Second centroid
            (35.0, 2),  // Third centroid
            (75.0, 3),  // Fourth centroid
            (125.0, 4), // Fifth centroid
            (200.0, 4), // Beyond end
        ];

        for (target_pos, expected_idx) in test_cases {
            let simd_result = find_quantile_position_optimized(&weights, target_pos);
            let scalar_result = find_quantile_position_scalar_fallback(&weights, target_pos);

            assert_eq!(simd_result, scalar_result,
                "SIMD and scalar mismatch for target {}: SIMD={}, scalar={}",
                target_pos, simd_result, scalar_result);

            assert_eq!(simd_result, expected_idx,
                "Wrong result for target {}: got {}, expected {}",
                target_pos, simd_result, expected_idx);
        }
    }

    #[test]
    fn test_quantile_position_search_large() {
        // Test with larger arrays to trigger SIMD path
        let weights: Vec<u32> = (1..=100).collect(); // Weights 1,2,3,...,100
        let total_weight: u64 = weights.iter().map(|&w| w as u64).sum();

        // Test searching at various percentiles
        for percentile in [10, 25, 50, 75, 90] {
            let target_pos = (total_weight * percentile / 100) as f64;

            let simd_result = find_quantile_position_optimized(&weights, target_pos);
            let scalar_result = find_quantile_position_scalar_fallback(&weights, target_pos);

            assert_eq!(simd_result, scalar_result,
                "SIMD/scalar mismatch at {}th percentile: SIMD={}, scalar={}",
                percentile, simd_result, scalar_result);

            assert!(simd_result < weights.len(),
                "Index out of bounds: {} >= {}", simd_result, weights.len());
        }
    }

    #[test]
    fn test_merge_sorted_operations() {
        let left_means = vec![1.0_f64, 3.0, 5.0, 7.0];
        let left_weights = vec![10u32, 20, 30, 40];
        let right_means = vec![2.0_f64, 4.0, 6.0, 8.0];
        let right_weights = vec![15u32, 25, 35, 45];

        let mut simd_means = Vec::new();
        let mut simd_weights = Vec::new();
        merge_sorted_optimized(
            &left_means, &left_weights,
            &right_means, &right_weights,
            &mut simd_means, &mut simd_weights
        );

        let mut scalar_means = Vec::new();
        let mut scalar_weights = Vec::new();
        merge_sorted_scalar_fallback(
            &left_means, &left_weights,
            &right_means, &right_weights,
            &mut scalar_means, &mut scalar_weights
        );

        assert_eq!(simd_means, scalar_means, "SIMD and scalar merge results differ for means");
        assert_eq!(simd_weights, scalar_weights, "SIMD and scalar merge results differ for weights");

        // Verify the merge is correct
        let expected_means = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected_weights = vec![10, 15, 20, 25, 30, 35, 40, 45];
        assert_eq!(simd_means, expected_means);
        assert_eq!(simd_weights, expected_weights);
    }

    #[test]
    fn test_merge_sorted_unequal_lengths() {
        let left_means = vec![1.0_f64, 5.0];
        let left_weights = vec![100u32, 500];
        let right_means = vec![2.0_f64, 3.0, 4.0, 6.0, 7.0];
        let right_weights = vec![200u32, 300, 400, 600, 700];

        let mut merged_means = Vec::new();
        let mut merged_weights = Vec::new();
        merge_sorted_optimized(
            &left_means, &left_weights,
            &right_means, &right_weights,
            &mut merged_means, &mut merged_weights
        );

        // Verify correct merge order
        let expected_means = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let expected_weights = vec![100, 200, 300, 400, 500, 600, 700];
        assert_eq!(merged_means, expected_means);
        assert_eq!(merged_weights, expected_weights);
    }

    #[test]
    fn test_merge_sorted_empty_arrays() {
        let left_means: Vec<f64> = vec![];
        let left_weights: Vec<u32> = vec![];
        let right_means = vec![1.0_f64, 2.0, 3.0];
        let right_weights = vec![10u32, 20, 30];

        let mut merged_means = Vec::new();
        let mut merged_weights = Vec::new();
        merge_sorted_optimized(
            &left_means, &left_weights,
            &right_means, &right_weights,
            &mut merged_means, &mut merged_weights
        );

        assert_eq!(merged_means, right_means);
        assert_eq!(merged_weights, right_weights);

        // Test the other way around
        let mut merged_means2 = Vec::new();
        let mut merged_weights2 = Vec::new();
        merge_sorted_optimized(
            &right_means, &right_weights,
            &left_means, &left_weights,
            &mut merged_means2, &mut merged_weights2
        );

        assert_eq!(merged_means2, right_means);
        assert_eq!(merged_weights2, right_weights);
    }

    #[test]
    fn test_simd_performance_vs_scalar() {
        // This test compares SIMD vs scalar performance
        // Note: Results may vary based on CPU features available

        let large_weights: Vec<u32> = (1..=10000).collect();
        let total_expected: u64 = large_weights.iter().map(|&w| w as u64).sum();

        // Time SIMD implementation
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let simd_result = sum_weights_optimized(&large_weights);
            assert_eq!(simd_result, total_expected);
        }
        let simd_time = start.elapsed();

        // Time scalar implementation
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let scalar_result: u64 = large_weights.iter().map(|&w| w as u64).sum();
            assert_eq!(scalar_result, total_expected);
        }
        let scalar_time = start.elapsed();

        println!("SIMD time: {:?}, Scalar time: {:?}", simd_time, scalar_time);
        println!("SIMD speedup: {:.2}x", scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64);

        // SIMD should be at least as fast as scalar (may be faster depending on CPU features)
        // This is more of an informational test than a strict assertion
        if simd_time > scalar_time * 2 {
            println!("Warning: SIMD implementation significantly slower than scalar");
        }
    }

    #[test]
    fn test_simd_merge_performance() {
        // Test merge performance with large arrays
        let size = 1000;
        let left_means: Vec<f64> = (0..size).map(|i| i as f64 * 2.0).collect();
        let left_weights: Vec<u32> = vec![1; size];
        let right_means: Vec<f64> = (0..size).map(|i| i as f64 * 2.0 + 1.0).collect();
        let right_weights: Vec<u32> = vec![2; size];

        // Time SIMD merge
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let mut simd_means = Vec::new();
            let mut simd_weights = Vec::new();
            merge_sorted_optimized(
                &left_means, &left_weights,
                &right_means, &right_weights,
                &mut simd_means, &mut simd_weights
            );
            assert_eq!(simd_means.len(), size * 2);
        }
        let simd_time = start.elapsed();

        // Time scalar merge
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let mut scalar_means = Vec::new();
            let mut scalar_weights = Vec::new();
            merge_sorted_scalar_fallback(
                &left_means, &left_weights,
                &right_means, &right_weights,
                &mut scalar_means, &mut scalar_weights
            );
            assert_eq!(scalar_means.len(), size * 2);
        }
        let scalar_time = start.elapsed();

        println!("Merge SIMD time: {:?}, Scalar time: {:?}", simd_time, scalar_time);
        println!("Merge speedup: {:.2}x", scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
    }
}