use std::sync::Arc;

use enum_dispatch::enum_dispatch;
use math::{Binary, Constant, Linear, Unary};
use misc::{Clamp, ClampedYGradient, EndIsland, RangeChoice, WeirdScaled};
use noise::{InterpolatedNoiseSampler, Noise, ShiftA, ShiftB, ShiftedNoise};
use pumpkin_data::chunk::DoublePerlinNoiseParameters;
use pumpkin_util::random::{xoroshiro128::Xoroshiro, RandomDeriver, RandomImpl};
use spline::{Range, Spline, SplineFunction, SplinePoint, SplineValue};

use crate::{
    generation::noise::{lerp, perlin::DoublePerlinNoiseSampler},
    noise_router::density_function_ast::{
        BinaryData, BinaryOperation, DensityFunctionRepr, RangeChoiceData, ShiftedNoiseData,
        SplineRepr, WeirdScaledData, WrapperType,
    },
};

mod math;
mod misc;
mod noise;
mod spline;

#[cfg(test)]
mod test;

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

pub trait NoisePos {
    fn x(&self) -> i32;
    fn y(&self) -> i32;
    fn z(&self) -> i32;
}

pub trait IndexToNoisePos {
    fn at(&self, index: usize) -> impl NoisePos;
}

struct ProtoChunkNoiseFunctionBuilderData<'a> {
    seed: u64,
    rand: RandomDeriver,
    id_to_sampler_map: Vec<(&'a str, Arc<DoublePerlinNoiseSampler>)>,
}

impl<'a> ProtoChunkNoiseFunctionBuilderData<'a> {
    fn new(seed: u64) -> Self {
        let random_deriver = RandomDeriver::Xoroshiro(Xoroshiro::from_seed(seed).next_splitter());
        Self {
            seed,
            rand: random_deriver,
            id_to_sampler_map: Vec::new(),
        }
    }

    fn get_noise_sampler_for_id(&mut self, id: &'a str) -> Arc<DoublePerlinNoiseSampler> {
        self.id_to_sampler_map
            .iter()
            .find_map(|ele| {
                if ele.0.eq(id) {
                    Some(ele.1.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                let parameters = DoublePerlinNoiseParameters::id_to_parameters(id)
                    .unwrap_or_else(|| panic!("Unknown noise id: {}", id));

                // Note that the parameters' id is differenent than `id`
                let mut random = self.rand.split_string(parameters.id());
                let sampler = DoublePerlinNoiseSampler::new(&mut random, parameters, false);
                let wrapped = Arc::new(sampler);
                self.id_to_sampler_map.push((id, wrapped.clone()));
                wrapped
            })
    }
}

/// A proto-noise function that initializes everything specific to the world seed.
/// This function cannot make any samples because all `Wrapped` components have a
/// chunk-specific initialization that must take place first
pub struct ProtoChunkNoiseFunction<'a> {
    function_components: Box<[UniversalChunkNoiseFunctionComponent<'a>]>,
}

impl<'a> ProtoChunkNoiseFunction<'a> {
    fn recursive_generate_spline(
        spline_ast: &'a SplineRepr,
        function_vec: &mut Vec<UniversalChunkNoiseFunctionComponent<'a>>,
        build_data: &mut ProtoChunkNoiseFunctionBuilderData<'a>,
    ) -> SplineValue {
        match spline_ast {
            SplineRepr::Standard {
                location_function,
                locations,
                values,
                derivatives,
            } => {
                let input_index = Self::recursive_generate_chunk_noise_function(
                    location_function,
                    function_vec,
                    build_data,
                );

                let points: Vec<_> = locations
                    .iter()
                    .zip(values)
                    .zip(derivatives)
                    .map(|((l, v), d)| {
                        let value = Self::recursive_generate_spline(v, function_vec, build_data);
                        SplinePoint::new(*l, value, *d)
                    })
                    .collect();

                SplineValue::Spline(Spline::new(input_index, points.into_boxed_slice()))
            }
            SplineRepr::Fixed { value } => SplineValue::Fixed(*value),
        }
    }

    fn recursive_generate_chunk_noise_function(
        function_ast: &'a DensityFunctionRepr,
        function_vec: &mut Vec<UniversalChunkNoiseFunctionComponent<'a>>,
        build_data: &mut ProtoChunkNoiseFunctionBuilderData<'a>,
    ) -> usize {
        //println!("Generating: {}", function_ast.as_str());
        match function_ast {
            DensityFunctionRepr::BlendAlpha => {
                // TODO: Replace this with the cache when the blender is implemented
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::Constant(Constant::new(1.0)),
                ));
            }
            DensityFunctionRepr::BlendOffset => {
                // TODO: Replace this with the cache when the blender is implemented
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::Constant(Constant::new(0.0)),
                ));
            }
            DensityFunctionRepr::BlendDensity { input } => {
                // TODO: Replace this when the blender is implemented
                return Self::recursive_generate_chunk_noise_function(
                    input,
                    function_vec,
                    build_data,
                );
            }
            DensityFunctionRepr::Constant { value } => {
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::Constant(Constant::new(*value)),
                ));
            }
            DensityFunctionRepr::Linear { input, data } => {
                let input_index =
                    Self::recursive_generate_chunk_noise_function(input, function_vec, build_data);
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::Linear(Linear::new(
                        input_index,
                        data,
                    )),
                ));
            }
            DensityFunctionRepr::Unary { input, data } => {
                let input_index =
                    Self::recursive_generate_chunk_noise_function(input, function_vec, build_data);
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::Unary(Unary::new(
                        input_index,
                        data,
                    )),
                ));
            }
            DensityFunctionRepr::Binary {
                argument1,
                argument2,
                data,
            } => {
                let arg1_index = Self::recursive_generate_chunk_noise_function(
                    argument1,
                    function_vec,
                    build_data,
                );
                let arg2_index = Self::recursive_generate_chunk_noise_function(
                    argument2,
                    function_vec,
                    build_data,
                );
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::Binary(Binary::new(
                        arg1_index, arg2_index, data,
                    )),
                ));
            }
            DensityFunctionRepr::EndIslands => {
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::EndIsland(EndIsland::new(
                        build_data.seed,
                    )),
                ));
            }
            DensityFunctionRepr::Noise { data } => {
                let sampler = build_data.get_noise_sampler_for_id(data.noise_id());
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::Noise(Noise::new(sampler, data)),
                ));
            }
            DensityFunctionRepr::ShiftA { noise_id } => {
                let sampler = build_data.get_noise_sampler_for_id(noise_id);
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::ShiftA(ShiftA::new(sampler)),
                ));
            }
            DensityFunctionRepr::ShiftB { noise_id } => {
                let sampler = build_data.get_noise_sampler_for_id(noise_id);
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::ShiftB(ShiftB::new(sampler)),
                ));
            }
            DensityFunctionRepr::ShiftedNoise {
                shift_x,
                shift_y,
                shift_z,
                data,
            } => {
                let x_index = Self::recursive_generate_chunk_noise_function(
                    shift_x,
                    function_vec,
                    build_data,
                );
                let y_index = Self::recursive_generate_chunk_noise_function(
                    shift_y,
                    function_vec,
                    build_data,
                );
                let z_index = Self::recursive_generate_chunk_noise_function(
                    shift_z,
                    function_vec,
                    build_data,
                );
                let sampler = build_data.get_noise_sampler_for_id(data.noise_id());
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::ShiftedNoise(ShiftedNoise::new(
                        x_index, y_index, z_index, sampler, data,
                    )),
                ));
            }
            DensityFunctionRepr::InterpolatedNoiseSampler { data } => {
                let mut random = build_data.rand.split_string("minecraft:terrain");
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::InterpolatedNoise(
                        InterpolatedNoiseSampler::new(data, &mut random),
                    ),
                ));
            }
            DensityFunctionRepr::WeirdScaled { input, data } => {
                let input_index =
                    Self::recursive_generate_chunk_noise_function(input, function_vec, build_data);
                let sampler = build_data.get_noise_sampler_for_id(data.noise_id());
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::WeirdScaled(WeirdScaled::new(
                        input_index,
                        sampler,
                        data,
                    )),
                ));
            }
            DensityFunctionRepr::Wrapper { input, wrapper } => {
                let input_index =
                    Self::recursive_generate_chunk_noise_function(input, function_vec, build_data);
                let min_value = function_vec[input_index].min();
                let max_value = function_vec[input_index].max();

                function_vec.push(UniversalChunkNoiseFunctionComponent::Wrapped(Wrapper {
                    input_index,
                    wrapper_type: *wrapper,
                    min_value,
                    max_value,
                }))
            }
            DensityFunctionRepr::ClampedYGradient { data } => {
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::ClampedYGradient(
                        ClampedYGradient::new(data),
                    ),
                ));
            }
            DensityFunctionRepr::Clamp { input, data } => {
                let input_index =
                    Self::recursive_generate_chunk_noise_function(input, function_vec, build_data);
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::Clamp(Clamp::new(
                        input_index,
                        data,
                    )),
                ));
            }
            DensityFunctionRepr::RangeChoice {
                input,
                when_in_range,
                when_out_range,
                data,
            } => {
                let input_index =
                    Self::recursive_generate_chunk_noise_function(input, function_vec, build_data);
                let when_in_index = Self::recursive_generate_chunk_noise_function(
                    when_in_range,
                    function_vec,
                    build_data,
                );
                let when_out_index = Self::recursive_generate_chunk_noise_function(
                    when_out_range,
                    function_vec,
                    build_data,
                );
                let min_value = function_vec[when_in_index]
                    .min()
                    .min(function_vec[when_out_index].min());
                let max_value = function_vec[when_in_index]
                    .max()
                    .max(function_vec[when_out_index].max());

                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::RangeChoice(RangeChoice::new(
                        input_index,
                        when_in_index,
                        when_out_index,
                        min_value,
                        max_value,
                        data,
                    )),
                ));
            }
            DensityFunctionRepr::Spline { spline, data } => {
                let spline = match Self::recursive_generate_spline(spline, function_vec, build_data)
                {
                    SplineValue::Spline(spline) => spline,
                    _ => unreachable!(),
                };

                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticDependent(
                    StaticDependentChunkNoiseFunctionComponent::Spline(SplineFunction::new(
                        spline, data,
                    )),
                ));
            }
        };

        // This component is always the last thing pushed to the stack
        function_vec.len() - 1
    }

    pub fn generate(function_ast: &'a DensityFunctionRepr, seed: u64) -> Self {
        let mut function_vec = Vec::new();
        let mut build_data = ProtoChunkNoiseFunctionBuilderData::new(seed);
        let _top_level = Self::recursive_generate_chunk_noise_function(
            function_ast,
            &mut function_vec,
            &mut build_data,
        );

        Self {
            function_components: function_vec.into_boxed_slice(),
        }
    }
}

#[enum_dispatch]
pub trait ChunkNoiseFunctionRange {
    fn min(&self) -> f64;
    fn max(&self) -> f64;
}

#[enum_dispatch]
pub trait StaticIndependentChunkNoiseFunctionComponentImpl: ChunkNoiseFunctionRange {
    fn sample(&self, pos: &impl NoisePos) -> f64;
    fn fill(&self, array: &mut [f64], mapper: &impl IndexToNoisePos) {
        array.iter_mut().enumerate().for_each(|(index, value)| {
            let pos = mapper.at(index);
            *value = self.sample(&pos);
        });
    }
}

#[enum_dispatch(
    StaticIndependentChunkNoiseFunctionComponentImpl,
    ChunkNoiseFunctionRange
)]
enum StaticIndependentChunkNoiseFunctionComponent<'a> {
    Constant(Constant),
    EndIsland(EndIsland),
    Noise(Noise<'a>),
    ShiftA(ShiftA),
    ShiftB(ShiftB),
    InterpolatedNoise(InterpolatedNoiseSampler<'a>),
    ClampedYGradient(ClampedYGradient<'a>),
}

#[enum_dispatch(ChunkNoiseFunctionRange)]
enum StaticDependentChunkNoiseFunctionComponent<'a> {
    Linear(Linear<'a>),
    Unary(Unary<'a>),
    Binary(Binary<'a>),
    ShiftedNoise(ShiftedNoise<'a>),
    WeirdScaled(WeirdScaled<'a>),
    Clamp(Clamp<'a>),
    RangeChoice(RangeChoice<'a>),
    Spline(SplineFunction<'a>),
}

struct Wrapper {
    input_index: usize,
    wrapper_type: WrapperType,
    min_value: f64,
    max_value: f64,
}

impl ChunkNoiseFunctionRange for Wrapper {
    fn min(&self) -> f64 {
        self.min_value
    }

    fn max(&self) -> f64 {
        self.max_value
    }
}

#[enum_dispatch(ChunkNoiseFunctionRange)]
enum UniversalChunkNoiseFunctionComponent<'a> {
    StaticIndependent(StaticIndependentChunkNoiseFunctionComponent<'a>),
    StaticDependent(StaticDependentChunkNoiseFunctionComponent<'a>),
    Wrapped(Wrapper),
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
        let mut components = Vec::with_capacity(base.function_components.len());
        base.function_components
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
                        let min_value = components[wrapped.input_index].min();
                        let max_value = components[wrapped.input_index].max();

                        components.push(ChunkNoiseFunctionComponent::PassThrough(PassThrough {
                            input_index: wrapped.input_index,
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
