use density_function::{
    spline::{Range, Spline, SplineValue},
    ChunkNoiseFunctionRange, NoisePos, ProtoChunkNoiseFunction,
    StaticDependentChunkNoiseFunctionComponent, StaticIndependentChunkNoiseFunctionComponent,
    StaticIndependentChunkNoiseFunctionComponentImpl, UniversalChunkNoiseFunctionComponent,
};

use crate::noise_router::density_function_ast::{
    BinaryData, BinaryOperation, RangeChoiceData, ShiftedNoiseData, WeirdScaledData,
};

use super::noise::{lerp, perlin::DoublePerlinNoiseSampler};

mod density_function;

// Implementations of referenced data to handle mutable borrow stuff with `ChunkNoiseFunction`
impl BinaryData {
    fn sample(
        &self,
        pos: &impl NoisePos,
        arg1_index: usize,
        arg2_index: usize,
        parent: &mut ChunkNoiseFunction,
    ) -> f64 {
        let density_1 = parent.sample_index(arg1_index, pos);

        match self.operation() {
            BinaryOperation::Add => density_1 + parent.sample_index(arg2_index, pos),
            BinaryOperation::Mul => {
                if density_1 == 0.0 {
                    0.0
                } else {
                    density_1 * parent.sample_index(arg2_index, pos)
                }
            }
            BinaryOperation::Min => {
                let min_2 = parent.min_index(arg2_index);
                if density_1 < min_2 {
                    density_1
                } else {
                    density_1.min(parent.sample_index(arg2_index, pos))
                }
            }
            BinaryOperation::Max => {
                let max_2 = parent.max_index(arg2_index);
                if density_1 > max_2 {
                    density_1
                } else {
                    density_1.max(parent.sample_index(arg2_index, pos))
                }
            }
        }
    }
}

impl ShiftedNoiseData {
    fn sample(
        &self,
        pos: &impl NoisePos,
        x_index: usize,
        y_index: usize,
        z_index: usize,
        sampler: &DoublePerlinNoiseSampler,
        parent: &mut ChunkNoiseFunction,
    ) -> f64 {
        let translated_x = pos.x() as f64 * self.xz_scale() + parent.sample_index(x_index, pos);
        let translated_y = pos.y() as f64 * self.y_scale() + parent.sample_index(y_index, pos);
        let translated_z = pos.z() as f64 * self.xz_scale() + parent.sample_index(z_index, pos);

        sampler.sample(translated_x, translated_y, translated_z)
    }
}

impl WeirdScaledData {
    fn sample(
        &self,
        pos: &impl NoisePos,
        input_index: usize,
        sampler: &DoublePerlinNoiseSampler,
        parent: &mut ChunkNoiseFunction,
    ) -> f64 {
        let density = parent.sample_index(input_index, pos);
        let scaled_density = self.mapper().scale(density);
        scaled_density
            * sampler
                .sample(
                    pos.x() as f64 / scaled_density,
                    pos.y() as f64 / scaled_density,
                    pos.z() as f64 / scaled_density,
                )
                .abs()
    }
}

impl RangeChoiceData {
    fn sample(
        &self,
        pos: &impl NoisePos,
        input_index: usize,
        when_in_index: usize,
        when_out_index: usize,
        parent: &mut ChunkNoiseFunction,
    ) -> f64 {
        let density = parent.sample_index(input_index, pos);
        if density >= *self.min_inclusive() && density < *self.max_exclusive() {
            parent.sample_index(when_in_index, pos)
        } else {
            parent.sample_index(when_out_index, pos)
        }
    }
}

impl SplineValue {
    fn sample(&self, pos: &impl NoisePos, parent: &mut ChunkNoiseFunction) -> f32 {
        match self {
            Self::Fixed(value) => *value,
            Self::Spline(spline) => spline.sample_internal(pos, parent),
        }
    }
}

impl Spline {
    fn sample(&self, pos: &impl NoisePos, parent: &mut ChunkNoiseFunction) -> f64 {
        self.sample_internal(pos, parent) as f64
    }

    fn sample_internal(&self, pos: &impl NoisePos, parent: &mut ChunkNoiseFunction) -> f32 {
        let location = parent.sample_index(self.input_index, pos) as f32;

        match self.find_index_for_location(location) {
            Range::In(index) => {
                if index == self.points.len() - 1 {
                    let last_known_sample = self.points[index].value.sample(pos, parent);
                    self.points[index].sample_outside_range(location, last_known_sample)
                } else {
                    let lower_point = &self.points[index];
                    let upper_point = &self.points[index + 1];

                    let lower_value = lower_point.value.sample(pos, parent);
                    let upper_value = upper_point.value.sample(pos, parent);

                    let x_scale = (location - lower_point.location)
                        / (upper_point.location - lower_point.location);
                    let extrapolated_lower_value = lower_point.derivative
                        * (upper_point.location - lower_point.location)
                        - (upper_value - lower_value);
                    let extrapolated_upper_value = -upper_point.derivative
                        * (upper_point.location - lower_point.location)
                        + (upper_value - lower_value);

                    (x_scale * (1f32 - x_scale))
                        * lerp(x_scale, extrapolated_lower_value, extrapolated_upper_value)
                        + lerp(x_scale, lower_value, upper_value)
                }
            }
            Range::Below => {
                let last_known_sample = self.points[0].value.sample(pos, parent);
                self.points[0].sample_outside_range(location, last_known_sample)
            }
        }
    }
}

/// A complete chunk-specific density function that is able to be sampled.
/// Uses a stack to be able to mutate chunk-specific components as well as
/// all chunk-specific componenets to reference top-level data
pub struct ChunkNoiseFunction<'a> {
    function_components: Box<[ChunkNoiseFunctionComponent<'a>]>,
}

pub enum ChunkNoiseFunctionWrapperHandler {
    Standard,
    #[cfg(test)]
    TestNoiseConfig,
}

#[cfg(test)]
struct PassThrough {
    input_index: usize,
    min_value: f64,
    max_value: f64,
}

#[cfg(test)]
impl ChunkNoiseFunctionRange for PassThrough {
    fn min(&self) -> f64 {
        self.min_value
    }

    fn max(&self) -> f64 {
        self.max_value
    }
}

impl<'a> ChunkNoiseFunction<'a> {
    pub fn new(
        base: &'a ProtoChunkNoiseFunction,
        wrapper_handler: ChunkNoiseFunctionWrapperHandler,
    ) -> Self {
        let mut components = Vec::with_capacity(base.components().len());
        base.components()
            .iter()
            .for_each(|component| match component {
                UniversalChunkNoiseFunctionComponent::StaticIndependent(independent) => {
                    components.push(ChunkNoiseFunctionComponent::StaticIndependent(independent));
                }
                UniversalChunkNoiseFunctionComponent::StaticDependent(dependent) => {
                    components.push(ChunkNoiseFunctionComponent::StaticDependent(dependent));
                }
                // This case handles chunk specific components
                UniversalChunkNoiseFunctionComponent::Wrapped(wrapped) => match wrapper_handler {
                    #[cfg(test)]
                    ChunkNoiseFunctionWrapperHandler::TestNoiseConfig => {
                        // Due to our previous invariant with the proto-function, it is guaranteed
                        // that the wrapped function is already on the stack
                        let min_value = components[wrapped.input_index()].min();
                        let max_value = components[wrapped.input_index()].max();

                        components.push(ChunkNoiseFunctionComponent::PassThrough(PassThrough {
                            input_index: wrapped.input_index(),
                            min_value,
                            max_value,
                        }));
                    }
                    ChunkNoiseFunctionWrapperHandler::Standard => {
                        let _wrapped = wrapped;
                        todo!()
                    }
                },
            });

        Self {
            function_components: components.into_boxed_slice(),
        }
    }

    /// Returns the min value of the component at `index`
    fn min_index(&self, index: usize) -> f64 {
        self.function_components[index].min()
    }

    /// Returns the max value of the component at `index`
    fn max_index(&self, index: usize) -> f64 {
        self.function_components[index].max()
    }

    fn sample_index(&mut self, index: usize, pos: &impl NoisePos) -> f64 {
        let component = &self.function_components[index];
        match component {
            ChunkNoiseFunctionComponent::StaticIndependent(static_independent) => {
                static_independent.sample(pos)
            }
            // The following must be computed here so we can access the over-all function list mutably and for lifetime/mutable borrowing stuff
            ChunkNoiseFunctionComponent::StaticDependent(static_dependent) => {
                match static_dependent {
                    StaticDependentChunkNoiseFunctionComponent::Linear(linear) => {
                        let input_index = linear.arg_index;
                        let data = linear.data;
                        let input_density = self.sample_index(input_index, pos);
                        data.apply_density(input_density)
                    }
                    StaticDependentChunkNoiseFunctionComponent::Binary(binary) => {
                        let arg1_index = binary.arg1_index;
                        let arg2_index = binary.arg2_index;
                        binary.data.sample(pos, arg1_index, arg2_index, self)
                    }
                    StaticDependentChunkNoiseFunctionComponent::Unary(unary) => {
                        let input_index = unary.arg_index;
                        let data = unary.data;
                        let input_density = self.sample_index(input_index, pos);
                        data.apply_density(input_density)
                    }
                    StaticDependentChunkNoiseFunctionComponent::ShiftedNoise(shifted_noise) => {
                        let x_index = shifted_noise.x_index;
                        let y_index = shifted_noise.y_index;
                        let z_index = shifted_noise.z_index;
                        let sampler = &shifted_noise.sampler;
                        shifted_noise
                            .data
                            .sample(pos, x_index, y_index, z_index, sampler, self)
                    }
                    StaticDependentChunkNoiseFunctionComponent::WeirdScaled(weird_scaled) => {
                        let input_index = weird_scaled.input_index;
                        let sampler = &weird_scaled.sampler;
                        weird_scaled.data.sample(pos, input_index, sampler, self)
                    }
                    StaticDependentChunkNoiseFunctionComponent::Clamp(clamp) => {
                        let input_index = clamp.input_index;
                        let input_density = self.sample_index(input_index, pos);
                        clamp.data.apply_density(input_density)
                    }
                    StaticDependentChunkNoiseFunctionComponent::RangeChoice(range_choice) => {
                        let input_index = range_choice.input_index;
                        let when_in_index = range_choice.when_in_index;
                        let when_out_index = range_choice.when_out_index;
                        range_choice.data.sample(
                            pos,
                            input_index,
                            when_in_index,
                            when_out_index,
                            self,
                        )
                    }
                    StaticDependentChunkNoiseFunctionComponent::Spline(spline_function) => {
                        spline_function.spline.sample(pos, self)
                    }
                }
            }
            #[cfg(test)]
            ChunkNoiseFunctionComponent::PassThrough(pass_through) => {
                self.sample_index(pass_through.input_index, pos)
            }
        }
    }

    pub fn sample(&mut self, pos: &impl NoisePos) -> f64 {
        let last_index = self.function_components.len() - 1;
        self.sample_index(last_index, pos)
    }
}

enum ChunkNoiseFunctionComponent<'a> {
    StaticIndependent(&'a StaticIndependentChunkNoiseFunctionComponent<'a>),
    StaticDependent(&'a StaticDependentChunkNoiseFunctionComponent<'a>),
    #[cfg(test)]
    PassThrough(PassThrough),
}

impl ChunkNoiseFunctionRange for ChunkNoiseFunctionComponent<'_> {
    #[inline]
    fn min(&self) -> f64 {
        match self {
            Self::StaticDependent(dependent) => dependent.min(),
            Self::StaticIndependent(independent) => independent.min(),
            #[cfg(test)]
            Self::PassThrough(pass_through) => pass_through.min(),
        }
    }

    #[inline]
    fn max(&self) -> f64 {
        match self {
            Self::StaticDependent(dependent) => dependent.max(),
            Self::StaticIndependent(independent) => independent.max(),
            #[cfg(test)]
            Self::PassThrough(pass_through) => pass_through.max(),
        }
    }
}
