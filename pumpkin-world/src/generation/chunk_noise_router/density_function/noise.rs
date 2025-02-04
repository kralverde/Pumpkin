use std::sync::Arc;

use pumpkin_util::random::RandomGenerator;

use crate::{
    generation::noise::{
        clamped_lerp,
        perlin::{DoublePerlinNoiseSampler, OctavePerlinNoiseSampler},
    },
    noise_router::density_function_ast::{
        InterpolatedNoiseSamplerData, NoiseData, ShiftedNoiseData,
    },
};

use super::{ChunkNoiseFunctionRange, NoisePos, StaticIndependentChunkNoiseFunctionComponentImpl};

pub struct Noise<'a> {
    sampler: Arc<DoublePerlinNoiseSampler>,
    data: &'a NoiseData,
}

impl<'a> Noise<'a> {
    pub fn new(sampler: Arc<DoublePerlinNoiseSampler>, data: &'a NoiseData) -> Self {
        Self { sampler, data }
    }
}

impl ChunkNoiseFunctionRange for Noise<'_> {
    #[inline]
    fn min(&self) -> f64 {
        -self.max()
    }

    #[inline]
    fn max(&self) -> f64 {
        self.sampler.max_value()
    }
}

impl StaticIndependentChunkNoiseFunctionComponentImpl for Noise<'_> {
    fn sample(&self, pos: &impl NoisePos) -> f64 {
        self.sampler.sample(
            pos.x() as f64 * self.data.xz_scale(),
            pos.y() as f64 * self.data.y_scale(),
            pos.z() as f64 * self.data.xz_scale(),
        )
    }
}

#[inline]
fn shift_sample_3d(sampler: &DoublePerlinNoiseSampler, x: f64, y: f64, z: f64) -> f64 {
    sampler.sample(x * 0.25f64, y * 0.25f64, z * 0.25f64) * 4f64
}

pub struct ShiftA {
    sampler: Arc<DoublePerlinNoiseSampler>,
}

impl ShiftA {
    pub fn new(sampler: Arc<DoublePerlinNoiseSampler>) -> Self {
        Self { sampler }
    }
}

impl ChunkNoiseFunctionRange for ShiftA {
    #[inline]
    fn min(&self) -> f64 {
        -self.max()
    }

    #[inline]
    fn max(&self) -> f64 {
        self.sampler.max_value() * 4.0
    }
}

impl StaticIndependentChunkNoiseFunctionComponentImpl for ShiftA {
    fn sample(&self, pos: &impl NoisePos) -> f64 {
        shift_sample_3d(&self.sampler, pos.x() as f64, 0.0, pos.z() as f64)
    }
}

pub struct ShiftB {
    sampler: Arc<DoublePerlinNoiseSampler>,
}

impl ShiftB {
    pub fn new(sampler: Arc<DoublePerlinNoiseSampler>) -> Self {
        Self { sampler }
    }
}

impl ChunkNoiseFunctionRange for ShiftB {
    #[inline]
    fn min(&self) -> f64 {
        -self.max()
    }

    #[inline]
    fn max(&self) -> f64 {
        self.sampler.max_value() * 4.0
    }
}

impl StaticIndependentChunkNoiseFunctionComponentImpl for ShiftB {
    fn sample(&self, pos: &impl NoisePos) -> f64 {
        shift_sample_3d(&self.sampler, pos.z() as f64, pos.x() as f64, 0.0)
    }
}

pub struct ShiftedNoise<'a> {
    pub(crate) x_index: usize,
    pub(crate) y_index: usize,
    pub(crate) z_index: usize,
    pub(crate) sampler: Arc<DoublePerlinNoiseSampler>,
    pub(crate) data: &'a ShiftedNoiseData,
}

impl ChunkNoiseFunctionRange for ShiftedNoise<'_> {
    #[inline]
    fn min(&self) -> f64 {
        -self.max()
    }

    #[inline]
    fn max(&self) -> f64 {
        self.sampler.max_value()
    }
}

impl<'a> ShiftedNoise<'a> {
    pub fn new(
        x_index: usize,
        y_index: usize,
        z_index: usize,
        sampler: Arc<DoublePerlinNoiseSampler>,
        data: &'a ShiftedNoiseData,
    ) -> Self {
        Self {
            x_index,
            y_index,
            z_index,
            sampler,
            data,
        }
    }
}

pub struct InterpolatedNoiseSampler<'a> {
    lower_noise: Box<OctavePerlinNoiseSampler>,
    upper_noise: Box<OctavePerlinNoiseSampler>,
    noise: Box<OctavePerlinNoiseSampler>,
    data: &'a InterpolatedNoiseSamplerData,
}

impl<'a> InterpolatedNoiseSampler<'a> {
    pub fn new(data: &'a InterpolatedNoiseSamplerData, random: &mut RandomGenerator) -> Self {
        let big_start = -15;
        let big_amplitudes = [1.0; 16];

        let little_start = -7;
        let little_amplitudes = [1.0; 8];

        let lower_noise = Box::new(OctavePerlinNoiseSampler::new(
            random,
            big_start,
            &big_amplitudes,
            true,
        ));
        let upper_noise = Box::new(OctavePerlinNoiseSampler::new(
            random,
            big_start,
            &big_amplitudes,
            true,
        ));
        let noise = Box::new(OctavePerlinNoiseSampler::new(
            random,
            little_start,
            &little_amplitudes,
            true,
        ));

        Self {
            lower_noise,
            upper_noise,
            noise,
            data,
        }
    }
}

impl ChunkNoiseFunctionRange for InterpolatedNoiseSampler<'_> {
    #[inline]
    fn min(&self) -> f64 {
        -self.max()
    }

    #[inline]
    fn max(&self) -> f64 {
        *self.data.max_value()
    }
}

impl StaticIndependentChunkNoiseFunctionComponentImpl for InterpolatedNoiseSampler<'_> {
    fn sample(&self, pos: &impl NoisePos) -> f64 {
        let d = pos.x() as f64 * self.data.scaled_xz_scale();
        let e = pos.y() as f64 * self.data.scaled_y_scale();
        let f = pos.z() as f64 * self.data.scaled_xz_scale();

        let g = d / self.data.xz_factor();
        let h = e / self.data.y_factor();
        let i = f / self.data.xz_factor();

        let j = self.data.scaled_y_scale() * self.data.smear_scale_multiplier();
        let k = j / self.data.y_factor();

        let mut n = 0f64;
        let mut o = 1f64;

        for p in 0..8 {
            let sampler = self.noise.get_octave(p);
            if let Some(sampler) = sampler {
                n += sampler.sample_no_fade(
                    OctavePerlinNoiseSampler::maintain_precision(g * o),
                    OctavePerlinNoiseSampler::maintain_precision(h * o),
                    OctavePerlinNoiseSampler::maintain_precision(i * o),
                    k * o,
                    h * o,
                ) / o;
            }

            o /= 2f64;
        }

        let q = (n / 10f64 + 1f64) / 2f64;
        let bl2 = q >= 1f64;
        let bl3 = q <= 0f64;
        let mut o = 1f64;
        let mut l = 0f64;
        let mut m = 0f64;

        for r in 0..16 {
            let s = OctavePerlinNoiseSampler::maintain_precision(d * o);
            let t = OctavePerlinNoiseSampler::maintain_precision(e * o);
            let u = OctavePerlinNoiseSampler::maintain_precision(f * o);
            let v = j * o;

            if !bl2 {
                let sampler = self.lower_noise.get_octave(r);
                if let Some(sampler) = sampler {
                    l += sampler.sample_no_fade(s, t, u, v, e * o) / o;
                }
            }

            if !bl3 {
                let sampler = self.upper_noise.get_octave(r);
                if let Some(sampler) = sampler {
                    m += sampler.sample_no_fade(s, t, u, v, e * o) / o;
                }
            }

            o /= 2f64;
        }

        clamped_lerp(l / 512f64, m / 512f64, q) / 128f64
    }
}
