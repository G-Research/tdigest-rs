use crate::tdigest::{singleton_policy::SingletonPolicy, Precision, ScaleFamily, TDigest};

pub use testdata::{gen_dataset, DistKind};

#[derive(Debug, Clone, Copy)]
pub struct QualityReport {
    pub n: usize,
    /// KS-like error (max absolute error on the chosen grid).
    pub ks: f64,
    /// Mean absolute error on the grid.
    pub mae: f64,
    /// A single scalar for rough comparison (higher is better).
    pub score: f64,
}

impl QualityReport {
    #[inline]
    pub fn from_metrics(n: usize, ks: f64, mae: f64) -> Self {
        // Same heuristic everywhere so numbers are comparable.
        let score = (-((1200.0 * mae) + (18.0 * ks))).exp();
        QualityReport { n, ks, mae, score }
    }
}

/// Pretty banner for section headings in story-style tests.
pub fn print_banner(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("{title}");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
}

/// Subsection header (indented a touch) for size/scale groups.
pub fn print_section(title: &str) {
    println!("  ── {title} ────────────────────────────────────────────");
}

/// Small, shared print helper used in tests/benches.
pub fn print_report(tag: &str, r: QualityReport) {
    println!(
        "{} -> QualityReport(n={}, KS={:.6e}, MAE={:.6e}, score={:.3})",
        tag, r.n, r.ks, r.mae, r.score
    );
}

/// Interpolate between order statistics to get an expected value at quantile `q`.
pub fn expected_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        return sorted[0];
    }
    if q >= 1.0 {
        return sorted[n - 1];
    }
    let t = q * (n as f64 - 1.0);
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let alpha = t - lo as f64;
        (1.0 - alpha) * sorted[lo] + alpha * sorted[hi]
    }
}

/// Exact ECDF on a sorted sample using the midpoint convention on ties.
pub fn exact_ecdf_for_sorted(sorted: &[f64]) -> Vec<f64> {
    let n = sorted.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }
        let mid = (i + j) as f64 / 2.0;
        let f = mid / nf;
        for _ in i..j {
            out.push(f);
        }
        i = j;
    }
    out
}

pub fn build_digest_sorted(
    mut data: Vec<f64>,
    max_size: usize,
    scale: ScaleFamily,
    precision: Precision,
) -> TDigest<f64> {
    // Apply input “precision” policy (affects only inputs; TDigest math still in f64).
    if let Precision::F32 = precision {
        for x in &mut data {
            *x = (*x as f32) as f64;
        }
    }

    TDigest::<f64>::builder()
        .max_size(max_size)
        .scale(scale)
        .singleton_policy(SingletonPolicy::Use)
        .build()
        .merge_sorted(data)
        .expect("no NaNs")
}
