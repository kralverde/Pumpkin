use std::sync::Arc;

use enum_dispatch::enum_dispatch;
use math::{Binary, Constant, Linear, Unary};
use misc::{Clamp, ClampedYGradient, EndIsland, RangeChoice, WeirdScaled};
use noise::{InterpolatedNoiseSampler, Noise, ShiftA, ShiftB, ShiftedNoise};
use pumpkin_data::chunk::DoublePerlinNoiseParameters;
use spline::{Spline, SplineFunction, SplinePoint, SplineValue};

use crate::{
    generation::{noise::perlin::DoublePerlinNoiseSampler, GlobalRandomConfig},
    noise_router::density_function_ast::{DensityFunctionRepr, SplineRepr, WrapperType},
};

// These are for enum_dispatch
use super::chunk_density_function::{
    Cache2D, CacheOnce, CellCache, ChunkSpecificNoiseFunctionComponent, DensityInterpolator,
    FlatCache, WrapperData,
};

mod math;
mod misc;
mod noise;
pub mod spline;

#[cfg(test)]
mod test;

pub trait NoisePos {
    fn x(&self) -> i32;
    fn y(&self) -> i32;
    fn z(&self) -> i32;
}

pub struct UnblendedNoisePos {
    x: i32,
    y: i32,
    z: i32,
}

impl UnblendedNoisePos {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl NoisePos for UnblendedNoisePos {
    #[inline]
    fn x(&self) -> i32 {
        self.x
    }

    #[inline]
    fn y(&self) -> i32 {
        self.y
    }

    #[inline]
    fn z(&self) -> i32 {
        self.z
    }
}

pub trait IndexToNoisePos {
    fn at(&self, index: usize, wrapper_inputs: Option<&mut WrapperData>)
        -> impl NoisePos + 'static;
}

struct ProtoChunkNoiseFunctionBuilderData<'a, 'b> {
    random_config: &'b GlobalRandomConfig,
    id_to_sampler_map: Vec<(&'a str, Arc<DoublePerlinNoiseSampler>)>,
}

impl<'a, 'b> ProtoChunkNoiseFunctionBuilderData<'a, 'b> {
    fn new(rand: &'b GlobalRandomConfig) -> Self {
        Self {
            random_config: rand,
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
                let mut random = self
                    .random_config
                    .base_random_deriver
                    .split_string(parameters.id());
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
    pub fn components(&self) -> &[UniversalChunkNoiseFunctionComponent<'a>] {
        &self.function_components
    }

    fn recursive_generate_spline(
        spline_ast: &'a SplineRepr,
        function_vec: &mut Vec<UniversalChunkNoiseFunctionComponent<'a>>,
        build_data: &mut ProtoChunkNoiseFunctionBuilderData<'a, '_>,
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
        build_data: &mut ProtoChunkNoiseFunctionBuilderData<'a, '_>,
    ) -> usize {
        //println!("Generating: {}", function_ast.as_str());
        match function_ast {
            DensityFunctionRepr::Beardifier => {
                // TODO: Replace this when world structures are implemented
                function_vec.push(UniversalChunkNoiseFunctionComponent::StaticIndependent(
                    StaticIndependentChunkNoiseFunctionComponent::Constant(Constant::new(0.0)),
                ));
            }

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
                        build_data.random_config.seed,
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
                let mut random = build_data
                    .random_config
                    .base_random_deriver
                    .split_string("minecraft:terrain");
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

    pub fn generate(
        function_ast: &'a DensityFunctionRepr,
        random_config: &GlobalRandomConfig,
    ) -> Self {
        let mut function_vec = Vec::new();
        let mut build_data = ProtoChunkNoiseFunctionBuilderData::new(random_config);
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
            let pos = mapper.at(index, None);
            *value = self.sample(&pos);
        });
    }
}

#[enum_dispatch(
    StaticIndependentChunkNoiseFunctionComponentImpl,
    ChunkNoiseFunctionRange
)]
pub enum StaticIndependentChunkNoiseFunctionComponent<'a> {
    Constant(Constant),
    EndIsland(EndIsland),
    Noise(Noise<'a>),
    ShiftA(ShiftA),
    ShiftB(ShiftB),
    InterpolatedNoise(InterpolatedNoiseSampler<'a>),
    ClampedYGradient(ClampedYGradient<'a>),
}

#[enum_dispatch(ChunkNoiseFunctionRange)]
pub enum StaticDependentChunkNoiseFunctionComponent<'a> {
    Linear(Linear<'a>),
    Unary(Unary<'a>),
    Binary(Binary<'a>),
    ShiftedNoise(ShiftedNoise<'a>),
    WeirdScaled(WeirdScaled<'a>),
    Clamp(Clamp<'a>),
    RangeChoice(RangeChoice<'a>),
    Spline(SplineFunction<'a>),
}

pub struct Wrapper {
    input_index: usize,
    wrapper_type: WrapperType,
    min_value: f64,
    max_value: f64,
}

impl Wrapper {
    pub fn new(
        input_index: usize,
        wrapper_type: WrapperType,
        min_value: f64,
        max_value: f64,
    ) -> Self {
        Self {
            input_index,
            wrapper_type,
            min_value,
            max_value,
        }
    }

    pub fn input_index(&self) -> usize {
        self.input_index
    }

    pub fn wrapper_type(&self) -> WrapperType {
        self.wrapper_type
    }
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
pub enum UniversalChunkNoiseFunctionComponent<'a> {
    StaticIndependent(StaticIndependentChunkNoiseFunctionComponent<'a>),
    StaticDependent(StaticDependentChunkNoiseFunctionComponent<'a>),
    Wrapped(Wrapper),
}
