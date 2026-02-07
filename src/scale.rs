use crate::traits::FloatConst;
use num::Float;

pub struct ScaleParams<T> {
    factor: T,
    inv_factor: T,
}

impl<T: Float + FloatConst> ScaleParams<T> {
    #[inline]
    pub fn new(delta: T, n: usize) -> Self {
        let f = (T::from(n).unwrap() / delta)
            .log(T::E)
            .mul_add(T::FOUR, T::TWENTYFOUR)
            / delta;
        ScaleParams {
            factor: T::ONE / f,
            inv_factor: f,
        }
    }
}

#[inline]
pub fn log_q_limit<T>(q0: T, params: &ScaleParams<T>) -> T
where
    T: Float + FloatConst,
{
    inverse_log_scale(log_scale(q0, params) + T::ONE, params)
}

#[inline]
fn inverse_log_scale<T>(k: T, params: &ScaleParams<T>) -> T
where
    T: Float + FloatConst,
{
    (T::ONE + (-k * params.inv_factor).exp()).recip()
}

#[inline]
fn log_scale<T>(q: T, params: &ScaleParams<T>) -> T
where
    T: Float + FloatConst,
{
    params.factor * (q / (T::ONE - q)).log(T::E)
}
