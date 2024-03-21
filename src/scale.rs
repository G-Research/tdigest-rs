use crate::traits::FloatConst;
use anyhow::Result;
use num::Float;

pub fn log_q_limit<T>(q0: T, delta: T, n: usize) -> Result<T>
where
    T: Float + FloatConst,
{
    inverse_log_scale(log_scale(q0, delta, n)? + T::ONE, delta, n)
}

pub fn inverse_log_scale<T>(k: T, delta: T, n: usize) -> Result<T>
where
    T: Float + FloatConst,
{
    let factor = (T::from(n).unwrap() / delta)
        .log(T::E)
        .mul_add(T::FOUR, T::TWENTYFOUR)
        / delta;
    Ok((T::ONE + (-k * factor).exp()).recip())
}

pub fn log_scale<T>(q: T, delta: T, n: usize) -> Result<T>
where
    T: Float + FloatConst,
{
    let factor = delta
        / (T::from(n).unwrap() / delta)
            .log(T::E)
            .mul_add(T::FOUR, T::TWENTYFOUR);
    Ok(factor * (q / (T::ONE - q)).log(T::E))
}
