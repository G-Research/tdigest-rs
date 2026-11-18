//! Compensated f64 summation for digest mass and weighted sums.

#[derive(Clone, Copy, Debug)]
pub(crate) struct Sum {
    main: f64,
    correction: f64,
    scale: f64,
}

impl Default for Sum {
    fn default() -> Self {
        Self {
            main: 0.0,
            correction: 0.0,
            scale: 1.0,
        }
    }
}

impl Sum {
    pub(crate) fn add(&mut self, value: f64) {
        let mut value = value * self.scale;
        // A finite exact final sum can follow intermediate overflow when
        // sorted negative values precede positive values. Rescale by an exact
        // power of two only on that exceptional path, retaining compensation.
        while self.main.is_finite() && value.is_finite() && !(self.main + value).is_finite() {
            self.main *= 0.25;
            self.correction *= 0.25;
            self.scale *= 0.25;
            value *= 0.25;
        }
        let next = self.main + value;
        // Keep overflow observable; inf-inf compensation would turn it into NaN.
        if next.is_finite() {
            self.correction += if self.main.abs() >= value.abs() {
                (self.main - next) + value
            } else {
                (value - next) + self.main
            };
        }
        self.main = next;
    }

    pub(crate) fn total(self) -> f64 {
        (self.main + self.correction) / self.scale
    }
}

pub(crate) fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut result = Sum::default();
    for value in values {
        result.add(value);
    }
    result.total()
}

#[cfg(test)]
mod tests {
    #[test]
    fn retains_finite_cancellation_and_exposes_overflow() {
        assert_eq!(super::sum([-1e16, 1.0, 1e16]), 1.0);
        assert!(super::sum([f64::MAX, f64::MAX]).is_infinite());
        assert_eq!(
            super::sum([-f64::MAX, -f64::MAX, 1.0, f64::MAX, f64::MAX]),
            1.0
        );
    }
}
