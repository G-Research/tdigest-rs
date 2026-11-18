//! Quality checks for `TDigest::quantile(q)`.
//!

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

    #[test]
    fn quantile_regression_max1000_quad_f64() {
        const SEED: u64 = 4242;
        const N: usize = 100_000;
        const MAX_SIZE: usize = 1000;
        const SCALE: ScaleFamily = ScaleFamily::Quad;
        const PREC: Precision = Precision::F64;

        const BASE_KS: f64 = 7.261420e-4;
        const BASE_MAE: f64 = 3.605881e-5;
        const BASE_SCORE: f64 = 0.9452166400891764;
        const TOL: f64 = 5e-4;

        let r = assess_quantiles_with(DistKind::Mixture, N, MAX_SIZE, SCALE, PREC, SEED);
        print_report("REG/Q[Mixture, k=1000, Quad, F64]", r);

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

    #[test]
    #[ignore = "slow"]
    fn quantile_accuracy_matrix() {
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

        for &dist in &dists {
            for &k in &sizes {
                for &scale in &scales {
                    for &prec in &precs {
                        let tag =
                            format!("Quantile {dist:?}, max_size={k}, {scale:?}, input={prec:?}");
                        let report = assess_quantiles_with(dist, N, k, scale, prec, SEED);
                        print_report(&tag, report);
                    }
                }
            }
        }
    }
}
