use std::sync::Arc;

use pumpkin_util::random::{legacy_rand::LegacyRand, RandomImpl};

use crate::{
    generation::noise::{
        clamped_map, perlin::DoublePerlinNoiseSampler, simplex::SimplexNoiseSampler,
    },
    noise_router::density_function_ast::{
        ClampData, ClampedYGradientData, RangeChoiceData, WeirdScaledData,
    },
};

use super::{ChunkNoiseFunctionRange, NoisePos, StaticIndependentChunkNoiseFunctionComponentImpl};

#[derive(Clone)]
pub struct EndIsland {
    sampler: Arc<SimplexNoiseSampler>,
}

impl EndIsland {
    pub fn new(seed: u64) -> Self {
        let mut rand = LegacyRand::from_seed(seed);
        rand.skip(17292);
        Self {
            sampler: Arc::new(SimplexNoiseSampler::new(&mut rand)),
        }
    }

    fn sample_2d(sampler: &SimplexNoiseSampler, x: i32, z: i32) -> f32 {
        let i = x / 2;
        let j = z / 2;
        let k = x % 2;
        let l = z % 2;

        let f = ((x * x + z * z) as f32).sqrt().mul_add(-8f32, 100f32);
        let mut f = f.clamp(-100f32, 80f32);

        for m in -12..=12 {
            for n in -12..=12 {
                let o = (i + m) as i64;
                let p = (j + n) as i64;

                if (o * o + p * p) > 4096i64
                    && sampler.sample_2d(o as f64, p as f64) < -0.9f32 as f64
                {
                    let g =
                        (o as f32).abs().mul_add(3439f32, (p as f32).abs() * 147f32) % 13f32 + 9f32;
                    let h = (k - m * 2) as f32;
                    let q = (l - n * 2) as f32;
                    let r = h.hypot(q).mul_add(-g, 100f32);
                    let s = r.clamp(-100f32, 80f32);

                    f = f.max(s);
                }
            }
        }

        f
    }
}

// These values are hardcoded from java
impl ChunkNoiseFunctionRange for EndIsland {
    #[inline]
    fn min(&self) -> f64 {
        -0.84375
    }

    #[inline]
    fn max(&self) -> f64 {
        0.5625
    }
}

impl StaticIndependentChunkNoiseFunctionComponentImpl for EndIsland {
    fn sample(&self, pos: &impl NoisePos) -> f64 {
        (Self::sample_2d(&self.sampler, pos.x() / 8, pos.z() / 8) as f64 - 8f64) / 128f64
    }
}

#[derive(Clone)]
pub struct WeirdScaled<'a> {
    pub(crate) input_index: usize,
    pub(crate) sampler: Arc<DoublePerlinNoiseSampler>,
    pub(crate) data: &'a WeirdScaledData,
}

impl<'a> WeirdScaled<'a> {
    pub fn new(
        input_index: usize,
        sampler: Arc<DoublePerlinNoiseSampler>,
        data: &'a WeirdScaledData,
    ) -> Self {
        Self {
            input_index,
            sampler,
            data,
        }
    }
}

impl ChunkNoiseFunctionRange for WeirdScaled<'_> {
    #[inline]
    fn min(&self) -> f64 {
        -self.max()
    }

    #[inline]
    fn max(&self) -> f64 {
        self.sampler.max_value() * self.data.mapper().max_multiplier()
    }
}

#[derive(Clone)]
pub struct ClampedYGradient<'a> {
    data: &'a ClampedYGradientData,
}

impl<'a> ClampedYGradient<'a> {
    pub fn new(data: &'a ClampedYGradientData) -> Self {
        Self { data }
    }
}

impl ChunkNoiseFunctionRange for ClampedYGradient<'_> {
    #[inline]
    fn min(&self) -> f64 {
        self.data.from_value().min(*self.data.to_value())
    }

    #[inline]
    fn max(&self) -> f64 {
        self.data.from_value().max(*self.data.to_value())
    }
}

impl StaticIndependentChunkNoiseFunctionComponentImpl for ClampedYGradient<'_> {
    fn sample(&self, pos: &impl NoisePos) -> f64 {
        clamped_map(
            pos.y() as f64,
            *self.data.from_y() as f64,
            *self.data.to_y() as f64,
            *self.data.from_value(),
            *self.data.to_value(),
        )
    }
}

#[derive(Clone)]
pub struct Clamp<'a> {
    pub(crate) input_index: usize,
    pub(crate) data: &'a ClampData,
}

impl<'a> Clamp<'a> {
    pub fn new(input_index: usize, data: &'a ClampData) -> Self {
        Self { input_index, data }
    }
}

impl ChunkNoiseFunctionRange for Clamp<'_> {
    #[inline]
    fn min(&self) -> f64 {
        *self.data.min_value()
    }

    #[inline]
    fn max(&self) -> f64 {
        *self.data.max_value()
    }
}

#[derive(Clone)]
pub struct RangeChoice<'a> {
    pub(crate) input_index: usize,
    pub(crate) when_in_index: usize,
    pub(crate) when_out_index: usize,
    pub(crate) data: &'a RangeChoiceData,
    min_value: f64,
    max_value: f64,
}

impl<'a> RangeChoice<'a> {
    pub fn new(
        input_index: usize,
        when_in_index: usize,
        when_out_index: usize,
        min_value: f64,
        max_value: f64,
        data: &'a RangeChoiceData,
    ) -> Self {
        Self {
            input_index,
            when_in_index,
            when_out_index,
            min_value,
            max_value,
            data,
        }
    }
}

impl ChunkNoiseFunctionRange for RangeChoice<'_> {
    #[inline]
    fn min(&self) -> f64 {
        self.min_value
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max_value
    }
}
