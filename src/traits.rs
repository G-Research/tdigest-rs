pub trait FloatConst {
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const FOUR: Self;
    const TWENTYFOUR: Self;

    const E: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
}

impl FloatConst for f32 {
    const ZERO: Self = 0_f32;
    const ONE: Self = 1_f32;
    const TWO: Self = 2_f32;
    const FOUR: Self = 4_f32;
    const TWENTYFOUR: Self = 24_f32;

    const E: Self = std::f32::consts::E;
    const INFINITY: Self = std::f32::INFINITY;
    const NEG_INFINITY: Self = std::f32::NEG_INFINITY;
}

impl FloatConst for f64 {
    const ZERO: Self = 0_f64;
    const ONE: Self = 1_f64;
    const TWO: Self = 2_f64;
    const FOUR: Self = 4_f64;
    const TWENTYFOUR: Self = 24_f64;

    const E: Self = std::f64::consts::E;
    const INFINITY: Self = std::f64::INFINITY;
    const NEG_INFINITY: Self = std::f64::NEG_INFINITY;
}
