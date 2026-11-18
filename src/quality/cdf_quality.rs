use super::quality_base::{
    build_digest_sorted, empirical_cdf, gen_dataset, DistKind, QualityReport,
};
use crate::tdigest::{Precision, ScaleFamily, TDigest};

/// Returns (ks_like, mae) between digest-estimated CDF and empirical CDF on a dense grid.
fn cdf_grid_errors(td: &TDigest<f64>, sorted: &[f64]) -> (f64, f64) {
    let n = sorted.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }

    // Sample a linear grid across the observed data range.
    let steps = 1000usize;
    let (xmin, xmax) = (sorted[0], sorted[n - 1]);
    let span = xmax - xmin;
    let values: Vec<f64> = if span == 0.0 {
        vec![xmin; steps + 1]
    } else {
        (0..=steps)
            .map(|i| xmin + (i as f64) * span / (steps as f64))
            .collect()
    };

    let est: Vec<f64> = td.cdf(&values);
    let exp = values.iter().map(|&x| empirical_cdf(sorted, x));

    let mut ks_like: f64 = 0.0;
    let mut mae: f64 = 0.0;
    for (e, a) in est.iter().zip(exp) {
        let err = (*e - a).abs();
        mae += err;
        if err > ks_like {
            ks_like = err;
        }
    }
    mae /= (steps + 1) as f64;
    (ks_like, mae)
}

/// Build, score, and wrap as `QualityReport` with configurable scale/precision.
pub fn assess_cdf_with(
    kind: DistKind,
    n: usize,
    max_size: usize,
    scale: ScaleFamily,
    precision: Precision,
    seed: u64,
) -> QualityReport {
    let mut data: Vec<f64> = gen_dataset(kind, n, seed);
    data.sort_by(|a: &f64, b: &f64| a.total_cmp(b));

    let td: TDigest<f64> = build_digest_sorted(data.clone(), max_size, scale, precision);
    let (ks, mae) = cdf_grid_errors(&td, &data);
    QualityReport::from_metrics(n, ks, mae)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::quality_base::print_report;

    #[test]
    fn empirical_rank_at_ties_and_between_observations() {
        let values = [1.0, 1.0, 3.0, 4.0];
        for (x, expected) in [
            (0.0, 0.0),
            (1.0, 0.25),
            (2.0, 0.5),
            (3.0, 0.625),
            (5.0, 1.0),
        ] {
            assert_eq!(empirical_cdf(&values, x), expected);
        }
    }

    #[test]
    fn cdf_regression_max1000_quad_f64() {
        const SEED: u64 = 4242;
        const N: usize = 100_000;
        const MAX_SIZE: usize = 1000;
        const SCALE: ScaleFamily = ScaleFamily::Quad;
        const PREC: Precision = Precision::F64;

        const BASE_KS: f64 = 1.678400e-3;
        const BASE_MAE: f64 = 4.239504e-5;
        const BASE_SCORE: f64 = 0.922115080;

        const KS_TOL: f64 = 5e-4;
        const MAE_TOL: f64 = 5e-6;
        const SCORE_TOL: f64 = 1e-3;

        let r = assess_cdf_with(DistKind::Mixture, N, MAX_SIZE, SCALE, PREC, SEED);
        print_report("REG/F[Mixture, k=1000, Quad, F64]", r);

        assert!(
            (r.ks - BASE_KS).abs() <= KS_TOL,
            "KS changed: {} vs {}",
            r.ks,
            BASE_KS
        );
        assert!(
            (r.mae - BASE_MAE).abs() <= MAE_TOL,
            "MAE changed: {} vs {}",
            r.mae,
            BASE_MAE
        );
        assert!(
            (r.score - BASE_SCORE).abs() <= SCORE_TOL,
            "score changed: {} vs {}",
            r.score,
            BASE_SCORE
        );
    }

    #[test]
    #[ignore = "slow"]
    fn cdf_accuracy_matrix() {
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
                        let tag = format!("CDF {dist:?}, max_size={k}, {scale:?}, input={prec:?}");
                        let report = assess_cdf_with(dist, N, k, scale, prec, SEED);
                        print_report(&tag, report);
                    }
                }
            }
        }
    }
}
