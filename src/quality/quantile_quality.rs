//! Quality checks for `TDigest::quantile(q)`.
//!
//! Two styles:
//! 1) A pinned **regression** test at max_size=1000 that FAILS on any change
//!    (even improvements). You update the baseline constants to bless changes.
//! 2) A **story** printer comparing max_size=100 vs 1000 across all scale
//!    families and Precision {F64, F32} so you can *see* what moved.

use super::quality_base::{
    build_digest_sorted, expected_quantile, gen_dataset, DistKind, QualityReport,
};
use crate::tdigest::{Precision, ScaleFamily, TDigest};

/// Evaluate by sampling a quantile grid and comparing to order-stat interpolation.
/// Returns (ks_like, mae).
fn quantile_grid_errors(td: &TDigest<f64>, sorted: &[f64]) -> (f64, f64) {
    let steps = 1000usize;
    let mut ks_like = 0.0f64;
    let mut mae = 0.0f64;

    // Exclude exact 0/1 to avoid edge clamping biases; 1..(steps-1) gives (0,1).
    for i in 1..steps {
        let q = (i as f64) / (steps as f64);
        let est = td.quantile(q);
        let exp = expected_quantile(sorted, q);
        let err = (est - exp).abs();
        mae += err;
        if err > ks_like {
            ks_like = err;
        }
    }
    mae /= (steps - 1) as f64;
    (ks_like, mae)
}

pub fn assess_quantiles_with(
    kind: DistKind,
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    precision: Precision,
    seed: u64,
) -> QualityReport {
    let mut data = gen_dataset(kind, n, seed);
    data.sort_by(|a, b| a.total_cmp(b));
    let td: TDigest<f64> = build_digest_sorted(data.clone(), max_size, scale, precision);
    let (ks, mae) = quantile_grid_errors(&td, &data);
    QualityReport::from_metrics(n, ks, mae)
}

#[cfg(test)]
mod tests {
    use super::super::quality_base::print_report;
    use super::*;

    /// ========= 1) REGRESSION: pin a single “truthy” scenario =========
    ///
    /// We freeze baseline error metrics for a representative hard case:
    ///   Dist = Mixture, n=100_000, max_size=1000, scale=Quad, precision=F64
    /// Any change (better or worse) FAILS. To bless a change, replace the
    /// constants below with the newly printed numbers and keep a small tol.
    #[test]
    fn quantile_regression_max1000_quad_f64() {
        const SEED: u64 = 4242;
        const N: usize = 100_000;
        const MAX_SIZE: usize = 1000;
        const SCALE: ScaleFamily = ScaleFamily::Quad;
        const PREC: Precision = Precision::F64;

        // ---- UPDATE THESE THREE WHEN YOU INTENTIONALLY CHANGE THE ALGO ----
        const BASE_KS: f64 = 7.318705e-4; // <- pinned from 20:52 run
        const BASE_MAE: f64 = 3.616685e-5; // <- pinned from 20:52 run
        const BASE_SCORE: f64 = 0.942988915480563; // <- pinned from 20:52 run
        const TOL: f64 = 5e-4; // strict-ish; adjust if you see flakiness

        let r = assess_quantiles_with(DistKind::Mixture, N, MAX_SIZE, SCALE, PREC, SEED);
        print_report("REG/Q[Mixture, k=1000, Quad, F64]", r);

        // symmetric “pin”: any drift beyond TOL fails—even if score improves.
        assert!(
            (r.ks - BASE_KS).abs() <= TOL,
            "KS changed: {} vs {}",
            r.ks,
            BASE_KS
        );
        assert!(
            (r.mae - BASE_MAE).abs() <= TOL,
            "MAE changed: {} vs {}",
            r.mae,
            BASE_MAE
        );
        assert!(
            (r.score - BASE_SCORE).abs() <= TOL,
            "score changed: {} vs {}",
            r.score,
            BASE_SCORE
        );
    }

    /// ========= 2) STORY: print a readable matrix you can eyeball =========
    ///
    /// Runs all 4 distributions [Uniform, Normal, LogNormal(σ=1), Mixture].
    /// For each dist: max_size = [100,1000] × scales [Quad,K1,K2,K3] × precision [F64,F32].
    /// Prints clear separators so you can scroll the output easily.
    /// No assertions — this is diagnostic and narrative only.
    #[test]
    #[ignore = "slow"]
    fn quantile_story_matrix() {
        use DistKind::*;

        const SEED: u64 = 4242;
        const N: usize = 100_000;
        let sizes = [100usize, 1000usize];
        let scales = [
            ScaleFamily::Quad,
            ScaleFamily::K1,
            ScaleFamily::K2,
            ScaleFamily::K3,
        ];
        let precs = [Precision::F64, Precision::F32];
        let dists = [Uniform, Normal, LogNormal { sigma: 1.0 }, Mixture];

        println!();
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("QUANTILE STORY MATRIX — full diagnostic sweep (n={N}, seed={SEED})");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        for &dist in &dists {
            println!("═══════════════════════════════════════════════════════════════════════════");
            println!("DIST: {:?}\n", dist);

            for &k in &sizes {
                println!("  ── max_size = {k} ────────────────────────────────────────────");
                for &scale in &scales {
                    println!("    SCALE: {:?} ▼", scale);
                    for &prec in &precs {
                        let tag = format!("      [prec={:?}] →", prec);
                        let r = assess_quantiles_with(dist, N, k, scale, prec, SEED);
                        print_report(&tag, r);
                    }
                    println!(); // spacing between scales
                }
                println!(); // spacing between sizes
            }
            println!(
                "═══════════════════════════════════════════════════════════════════════════\n"
            );
        }

        println!("(Hint: For same scale+precision, score(1000) should usually ≥ score(100).)");
        println!("═══════════════════════════════════════════════════════════════════════════");
    }
}
