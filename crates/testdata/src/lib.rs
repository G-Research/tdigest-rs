//! tdigest-testdata
//! Synthetic data generators shared by benches and tests.
//! Values are squashed into \[0,1] so shapes are comparable.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Available synthetic distributions.
#[derive(Clone, Copy, Debug)]
pub enum DistKind {
    /// Uniform in \[0,1)
    Uniform,
    /// Gaussian mapped to \[0,1]
    Normal,
    /// Log-normal-ish: exp(N(0, σ²)) squashed to \[0,1]
    LogNormal { sigma: f64 },
    /// Mixed: clumps, broad region, and heavy tails
    Mixture,
}

/// Generate `n` samples for the chosen distribution, squashed into \[0,1].
pub fn gen_dataset(kind: DistKind, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);

    match kind {
        DistKind::Uniform => {
            for _ in 0..n {
                out.push(rng.random::<f64>());
            }
        }
        DistKind::Normal => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            for _ in 0..n {
                let z: f64 = normal.sample(&mut rng);
                out.push((0.5 + 0.2 * z).clamp(0.0, 1.0));
            }
        }
        DistKind::LogNormal { sigma } => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            for _ in 0..n {
                let z: f64 = normal.sample(&mut rng);
                let x = (sigma * z).exp();
                let y = x / (1.0 + x);
                out.push(y.clamp(0.0, 1.0));
            }
        }
        DistKind::Mixture => {
            for _ in 0..n {
                let bucket: u32 = rng.random_range(0..100);
                let v = match bucket {
                    // Clumps around 0.1, 0.5, 0.9 with micro-noise
                    0..=29 => {
                        let center = match rng.random_range(0..3) {
                            0 => 0.10,
                            1 => 0.50,
                            _ => 0.90,
                        };
                        center + rng.random_range(-1.0..1.0) * 1e-3
                    }
                    // Broad uniform region
                    30..=69 => rng.random::<f64>(),
                    // Heavier tails near 0 and 1
                    _ => {
                        let exp = rng.random_range(3.0..9.0);
                        if rng.random_bool(0.5) {
                            rng.random::<f64>().clamp(1e-12, 1.0).powf(exp)
                        } else {
                            1.0 - rng.random::<f64>().clamp(1e-12, 1.0).powf(exp)
                        }
                    }
                };
                out.push(v.clamp(0.0, 1.0));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_gen() {
        for kind in [
            DistKind::Uniform,
            DistKind::Normal,
            DistKind::LogNormal { sigma: 1.0 },
            DistKind::Mixture,
        ] {
            let values = gen_dataset(kind, 10_000, 123);
            assert_eq!(values.len(), 10_000);
            assert!(values.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }
}
