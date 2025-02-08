use std::mem;

use super::density_function::{
    spline::{Range, Spline, SplineValue},
    ChunkNoiseFunctionRange, IndexToNoisePos, NoisePos, PassThrough, ProtoChunkNoiseFunction,
    StaticDependentChunkNoiseFunctionComponent, StaticIndependentChunkNoiseFunctionComponent,
    StaticIndependentChunkNoiseFunctionComponentImpl, UnblendedNoisePos,
    UniversalChunkNoiseFunctionComponent,
};
use enum_dispatch::enum_dispatch;
use pumpkin_util::math::vector2::Vector2;

use crate::{
    generation::{
        aquifer_sampler::AquiferSampler,
        biome_coords,
        chunk_noise::{BlockStateSampler, ChainedBlockStateSampler},
        noise::{lerp, lerp3, perlin::DoublePerlinNoiseSampler},
        ore_sampler::OreVeinSampler,
        positions::chunk_pos,
    },
    noise_router::density_function_ast::{
        BinaryData, BinaryOperation, RangeChoiceData, ShiftedNoiseData, WeirdScaledData,
        WrapperType,
    },
};

fn sample_component_stack(
    component_stack: &mut [ChunkNoiseFunctionComponent],
    index: usize,
    pos: &impl NoisePos,
    sample_options: &ChunkNoiseFunctionSampleOptions,
) -> f64 {
    let component = &mut component_stack[index];
    match component {
        ChunkNoiseFunctionComponent::StaticIndependent(static_independent) => {
            static_independent.sample(pos)
        }
        // The following must be computed here so we can access the over-all function list mutably and for lifetime/mutable borrowing stuff
        ChunkNoiseFunctionComponent::StaticDependent(static_dependent) => match static_dependent {
            StaticDependentChunkNoiseFunctionComponent::Linear(linear) => {
                let input_index = linear.arg_index;
                let data = linear.data;
                let input_density =
                    sample_component_stack(component_stack, input_index, pos, sample_options);
                data.apply_density(input_density)
            }
            StaticDependentChunkNoiseFunctionComponent::Binary(binary) => {
                let arg1_index = binary.arg1_index;
                let arg2_index = binary.arg2_index;
                binary
                    .data
                    .sample(pos, arg1_index, arg2_index, component_stack, sample_options)
            }
            StaticDependentChunkNoiseFunctionComponent::Unary(unary) => {
                let input_index = unary.arg_index;
                let data = unary.data;
                let input_density =
                    sample_component_stack(component_stack, input_index, pos, sample_options);
                data.apply_density(input_density)
            }
            StaticDependentChunkNoiseFunctionComponent::ShiftedNoise(shifted_noise) => {
                let x_index = shifted_noise.x_index;
                let y_index = shifted_noise.y_index;
                let z_index = shifted_noise.z_index;
                let sampler = &shifted_noise.sampler;
                shifted_noise.data.sample(
                    pos,
                    x_index,
                    y_index,
                    z_index,
                    sampler,
                    component_stack,
                    sample_options,
                )
            }
            StaticDependentChunkNoiseFunctionComponent::WeirdScaled(weird_scaled) => {
                let input_index = weird_scaled.input_index;
                let sampler = &weird_scaled.sampler;
                weird_scaled
                    .data
                    .sample(pos, input_index, sampler, component_stack, sample_options)
            }
            StaticDependentChunkNoiseFunctionComponent::Clamp(clamp) => {
                let input_index = clamp.input_index;
                let input_density =
                    sample_component_stack(component_stack, input_index, pos, sample_options);
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
                    component_stack,
                    sample_options,
                )
            }
            StaticDependentChunkNoiseFunctionComponent::Spline(spline_function) => spline_function
                .spline
                .sample(pos, component_stack, sample_options),
            StaticDependentChunkNoiseFunctionComponent::PassThrough(pass_through) => {
                let defered_index = pass_through.input_index;
                sample_component_stack(component_stack, defered_index, pos, sample_options)
            }
        },
        ChunkNoiseFunctionComponent::ChunkSpecific(chunk_specific) => {
            match chunk_specific {
                ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) => {
                    match sample_options.action {
                        SampleAction::Wrappers(WrapperData {
                            cell_x_block_position,
                            cell_y_block_position,
                            cell_z_block_position,
                            horizontal_cell_block_count,
                            vertical_cell_block_count,
                        }) => {
                            if sample_options.populating_caches {
                                lerp3(
                                    cell_x_block_position as f64
                                        / horizontal_cell_block_count as f64,
                                    cell_y_block_position as f64 / vertical_cell_block_count as f64,
                                    cell_z_block_position as f64
                                        / horizontal_cell_block_count as f64,
                                    density_interpolator.first_pass[0],
                                    density_interpolator.first_pass[4],
                                    density_interpolator.first_pass[2],
                                    density_interpolator.first_pass[6],
                                    density_interpolator.first_pass[1],
                                    density_interpolator.first_pass[5],
                                    density_interpolator.first_pass[3],
                                    density_interpolator.first_pass[7],
                                )
                            } else {
                                density_interpolator.result
                            }
                        }
                        SampleAction::SkipWrappers => {
                            let input_index = density_interpolator.input_index;
                            sample_component_stack(
                                component_stack,
                                input_index,
                                pos,
                                sample_options,
                            )
                        }
                    }
                }
                ChunkSpecificNoiseFunctionComponent::FlatCache(flat_cache) => {
                    let absolute_biome_x_position = biome_coords::from_block(pos.x());
                    let absolute_biome_z_position = biome_coords::from_block(pos.z());

                    let relative_biome_x_position =
                        absolute_biome_x_position - flat_cache.start_biome_x;
                    let relative_biome_z_position =
                        absolute_biome_z_position - flat_cache.start_biome_z;

                    if relative_biome_x_position >= 0
                        && relative_biome_z_position >= 0
                        && relative_biome_x_position <= flat_cache.horizontal_biome_end as i32
                        && relative_biome_z_position <= flat_cache.horizontal_biome_end as i32
                    {
                        let cache_index = flat_cache.xz_to_index_const(
                            relative_biome_x_position as usize,
                            relative_biome_z_position as usize,
                        );
                        flat_cache.cache[cache_index]
                    } else {
                        let input_index = flat_cache.input_index;
                        sample_component_stack(component_stack, input_index, pos, sample_options)
                    }
                }
                ChunkSpecificNoiseFunctionComponent::Cache2D(cache_2d) => {
                    let packed_column = chunk_pos::packed(&Vector2::new(pos.x(), pos.z()));
                    if packed_column == cache_2d.last_sample_column {
                        cache_2d.last_sample_result
                    } else {
                        let mut cache_2d = cache_2d.clone();
                        let result = sample_component_stack(
                            component_stack,
                            cache_2d.input_index,
                            pos,
                            sample_options,
                        );
                        cache_2d.last_sample_column = packed_column;
                        cache_2d.last_sample_result = result;

                        // We need to re-write the struct instead of a mutable reference to parameters because of mutability
                        // rules
                        component_stack[index] = ChunkNoiseFunctionComponent::ChunkSpecific(
                            ChunkSpecificNoiseFunctionComponent::Cache2D(cache_2d),
                        );

                        result
                    }
                }
                ChunkSpecificNoiseFunctionComponent::CacheOnce(cache_once) => {
                    match sample_options.action {
                        SampleAction::Wrappers(_) => {
                            if cache_once.cache_fill_unique_id
                                == sample_options.cache_fill_unique_id
                                && !cache_once.cache.is_empty()
                            {
                                cache_once.cache[sample_options.fill_index]
                            } else if cache_once.cache_result_unique_id
                                == sample_options.cache_result_unique_id
                            {
                                cache_once.last_sample_result
                            } else {
                                // Effectively take whats on the stack, leaving an invalid state remaining
                                let mut cache_once = cache_once.take_cache_clone();
                                // This is safe because one of our invariants is no cyclic graphs on the map; the
                                // input index will never reference this
                                let result = sample_component_stack(
                                    component_stack,
                                    cache_once.input_index,
                                    pos,
                                    sample_options,
                                );
                                cache_once.cache_result_unique_id =
                                    sample_options.cache_result_unique_id;
                                cache_once.last_sample_result = result;
                                // We need to re-write the struct instead of a mutable reference to parameters because of mutability
                                // rules
                                component_stack[index] = ChunkNoiseFunctionComponent::ChunkSpecific(
                                    ChunkSpecificNoiseFunctionComponent::CacheOnce(cache_once),
                                );
                                result
                            }
                        }
                        SampleAction::SkipWrappers => {
                            let input_index = cache_once.input_index;
                            sample_component_stack(
                                component_stack,
                                input_index,
                                pos,
                                sample_options,
                            )
                        }
                    }
                }
                ChunkSpecificNoiseFunctionComponent::CellCache(cell_cache) => {
                    match sample_options.action {
                        SampleAction::Wrappers(WrapperData {
                            cell_x_block_position,
                            cell_y_block_position,
                            cell_z_block_position,
                            horizontal_cell_block_count,
                            vertical_cell_block_count,
                        }) => {
                            let cache_index =
                                ((vertical_cell_block_count - 1 - cell_y_block_position)
                                    * horizontal_cell_block_count
                                    + cell_x_block_position)
                                    * horizontal_cell_block_count
                                    + cell_z_block_position;

                            cell_cache.cache[cache_index]
                        }
                        SampleAction::SkipWrappers => {
                            let input_index = cell_cache.input_index;
                            sample_component_stack(
                                component_stack,
                                input_index,
                                pos,
                                sample_options,
                            )
                        }
                    }
                }
            }
        }
        ChunkNoiseFunctionComponent::PassThrough(pass_through) => {
            let defered_index = pass_through.input_index;
            sample_component_stack(component_stack, defered_index, pos, sample_options)
        }
    }
}

// Implementations of referenced data to handle mutable borrow stuff with `ChunkNoiseFunction`
impl BinaryData {
    fn sample(
        &self,
        pos: &impl NoisePos,
        arg1_index: usize,
        arg2_index: usize,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        let density_1 = sample_component_stack(component_stack, arg1_index, pos, sample_options);

        match self.operation() {
            BinaryOperation::Add => {
                density_1 + sample_component_stack(component_stack, arg2_index, pos, sample_options)
            }
            BinaryOperation::Mul => {
                if density_1 == 0.0 {
                    0.0
                } else {
                    density_1
                        * sample_component_stack(component_stack, arg2_index, pos, sample_options)
                }
            }
            BinaryOperation::Min => {
                let min_2 = component_stack[arg2_index].min();
                if density_1 < min_2 {
                    density_1
                } else {
                    density_1.min(sample_component_stack(
                        component_stack,
                        arg2_index,
                        pos,
                        sample_options,
                    ))
                }
            }
            BinaryOperation::Max => {
                let max_2 = component_stack[arg2_index].max();
                if density_1 > max_2 {
                    density_1
                } else {
                    density_1.max(sample_component_stack(
                        component_stack,
                        arg2_index,
                        pos,
                        sample_options,
                    ))
                }
            }
        }
    }
}

impl ShiftedNoiseData {
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        pos: &impl NoisePos,
        x_index: usize,
        y_index: usize,
        z_index: usize,
        sampler: &DoublePerlinNoiseSampler,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        let translated_x = pos.x() as f64 * self.xz_scale()
            + sample_component_stack(component_stack, x_index, pos, sample_options);
        let translated_y = pos.y() as f64 * self.y_scale()
            + sample_component_stack(component_stack, y_index, pos, sample_options);
        let translated_z = pos.z() as f64 * self.xz_scale()
            + sample_component_stack(component_stack, z_index, pos, sample_options);

        sampler.sample(translated_x, translated_y, translated_z)
    }
}

impl WeirdScaledData {
    fn sample(
        &self,
        pos: &impl NoisePos,
        input_index: usize,
        sampler: &DoublePerlinNoiseSampler,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        let density = sample_component_stack(component_stack, input_index, pos, sample_options);
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
        component_stack: &mut [ChunkNoiseFunctionComponent],
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        let density = sample_component_stack(component_stack, input_index, pos, sample_options);
        if density >= *self.min_inclusive() && density < *self.max_exclusive() {
            sample_component_stack(component_stack, when_in_index, pos, sample_options)
        } else {
            sample_component_stack(component_stack, when_out_index, pos, sample_options)
        }
    }
}

impl SplineValue {
    fn sample(
        &self,
        pos: &impl NoisePos,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f32 {
        match self {
            Self::Fixed(value) => *value,
            Self::Spline(spline) => spline.sample_internal(pos, component_stack, sample_options),
        }
    }
}

impl Spline {
    #[inline]
    fn sample(
        &self,
        pos: &impl NoisePos,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        self.sample_internal(pos, component_stack, sample_options) as f64
    }

    fn sample_internal(
        &self,
        pos: &impl NoisePos,
        component_stack: &mut [ChunkNoiseFunctionComponent],
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f32 {
        let location =
            sample_component_stack(component_stack, self.input_index, pos, sample_options) as f32;

        match self.find_index_for_location(location) {
            Range::In(index) => {
                if index == self.points.len() - 1 {
                    let last_known_sample =
                        self.points[index]
                            .value
                            .sample(pos, component_stack, sample_options);
                    self.points[index].sample_outside_range(location, last_known_sample)
                } else {
                    let lower_point = &self.points[index];
                    let upper_point = &self.points[index + 1];

                    let lower_value =
                        lower_point
                            .value
                            .sample(pos, component_stack, sample_options);
                    let upper_value =
                        upper_point
                            .value
                            .sample(pos, component_stack, sample_options);

                    // Use linear interpolation (-ish cuz of derivatives) to derivate a point between two points
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
                let last_known_sample =
                    self.points[0]
                        .value
                        .sample(pos, component_stack, sample_options);
                self.points[0].sample_outside_range(location, last_known_sample)
            }
        }
    }
}

pub struct WrapperData {
    // Our relative position within the cell
    pub(crate) cell_x_block_position: usize,
    pub(crate) cell_y_block_position: usize,
    pub(crate) cell_z_block_position: usize,

    // Number of blocks per cell per axis
    pub(crate) horizontal_cell_block_count: usize,
    pub(crate) vertical_cell_block_count: usize,
}

pub enum SampleAction {
    SkipWrappers,
    Wrappers(WrapperData),
}

pub struct ChunkNoiseFunctionSampleOptions {
    populating_caches: bool,
    pub(crate) action: SampleAction,

    // Global IDs for the `CacheOnce` wrapper
    pub(crate) cache_result_unique_id: u64,
    pub(crate) cache_fill_unique_id: u64,

    // The current index of a slice being filled by the `fill` function
    pub(crate) fill_index: usize,
}

impl ChunkNoiseFunctionSampleOptions {
    pub const fn new(
        populating_caches: bool,
        action: SampleAction,
        cache_result_unique_id: u64,
        cache_fill_unique_id: u64,
        fill_index: usize,
    ) -> Self {
        Self {
            populating_caches,
            action,
            cache_result_unique_id,
            cache_fill_unique_id,
            fill_index,
        }
    }
}

pub struct ChunkNoiseFunctionBuilderOptions {
    // Number of blocks per cell per axis
    horizontal_cell_block_count: usize,
    vertical_cell_block_count: usize,

    // Number of cells per chunk per axis
    vertical_cell_count: usize,
    horizontal_cell_count: usize,

    // The biome coords of this chunk
    start_biome_x: i32,
    start_biome_z: i32,

    // Number of biome regions per chunk per axis
    horizontal_biome_end: usize,
}

impl ChunkNoiseFunctionBuilderOptions {
    pub const fn new(
        horizontal_cell_block_count: usize,
        vertical_cell_block_count: usize,
        vertical_cell_count: usize,
        horizontal_cell_count: usize,
        start_biome_x: i32,
        start_biome_z: i32,
        horizontal_biome_end: usize,
    ) -> Self {
        Self {
            horizontal_cell_block_count,
            vertical_cell_block_count,
            vertical_cell_count,
            horizontal_cell_count,
            start_biome_x,
            start_biome_z,
            horizontal_biome_end,
        }
    }
}

// These are chunk specific function components that are picked based on the wrapper type
pub struct DensityInterpolator {
    // What we are interpolating
    input_index: usize,

    // y-z plane buffers to be interpolated together, each of these values is that of the cell, not
    // the block
    start_buffer: Box<[f64]>,
    end_buffer: Box<[f64]>,

    first_pass: [f64; 8],
    second_pass: [f64; 4],
    third_pass: [f64; 2],
    result: f64,

    vertical_cell_count: usize,
    min_value: f64,
    max_value: f64,
}

impl ChunkNoiseFunctionRange for DensityInterpolator {
    #[inline]
    fn min(&self) -> f64 {
        self.min_value
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max_value
    }
}

impl DensityInterpolator {
    fn new(
        input_index: usize,
        min_value: f64,
        max_value: f64,
        builder_options: &ChunkNoiseFunctionBuilderOptions,
    ) -> Self {
        // These are all dummy values to be populated when sampling values
        Self {
            input_index,
            start_buffer: vec![
                0.0;
                (builder_options.vertical_cell_count + 1)
                    * (builder_options.horizontal_cell_count + 1)
            ]
            .into_boxed_slice(),
            end_buffer: vec![
                0.0;
                (builder_options.vertical_cell_count + 1)
                    * (builder_options.horizontal_cell_count + 1)
            ]
            .into_boxed_slice(),
            first_pass: Default::default(),
            second_pass: Default::default(),
            third_pass: Default::default(),
            result: Default::default(),
            vertical_cell_count: builder_options.vertical_cell_count,
            min_value,
            max_value,
        }
    }

    #[inline]
    fn yz_to_buf_index(&self, cell_y_position: usize, cell_z_position: usize) -> usize {
        cell_z_position * (self.vertical_cell_count + 1) + cell_y_position
    }

    fn on_sampled_cell_corners(&mut self, cell_y_position: usize, cell_z_position: usize) {
        self.first_pass[0] =
            self.start_buffer[self.yz_to_buf_index(cell_y_position, cell_z_position)];
        self.first_pass[1] =
            self.start_buffer[self.yz_to_buf_index(cell_y_position, cell_z_position + 1)];
        self.first_pass[4] =
            self.end_buffer[self.yz_to_buf_index(cell_y_position, cell_z_position)];
        self.first_pass[5] =
            self.end_buffer[self.yz_to_buf_index(cell_y_position, cell_z_position + 1)];
        self.first_pass[2] =
            self.start_buffer[self.yz_to_buf_index(cell_y_position + 1, cell_z_position)];
        self.first_pass[3] =
            self.start_buffer[self.yz_to_buf_index(cell_y_position + 1, cell_z_position + 1)];
        self.first_pass[6] =
            self.end_buffer[self.yz_to_buf_index(cell_y_position + 1, cell_z_position)];
        self.first_pass[7] =
            self.end_buffer[self.yz_to_buf_index(cell_y_position + 1, cell_z_position + 1)];
    }

    fn interpolate_y(&mut self, delta: f64) {
        self.second_pass[0] = lerp(delta, self.first_pass[0], self.first_pass[2]);
        self.second_pass[2] = lerp(delta, self.first_pass[4], self.first_pass[6]);
        self.second_pass[1] = lerp(delta, self.first_pass[1], self.first_pass[3]);
        self.second_pass[3] = lerp(delta, self.first_pass[5], self.first_pass[7]);
    }

    #[inline]
    fn interpolate_x(&mut self, delta: f64) {
        self.third_pass[0] = lerp(delta, self.second_pass[0], self.second_pass[2]);
        self.third_pass[1] = lerp(delta, self.second_pass[1], self.second_pass[3]);
    }

    #[inline]
    fn interpolate_z(&mut self, delta: f64) {
        self.result = lerp(delta, self.third_pass[0], self.third_pass[1]);
    }

    #[inline]
    fn swap_buffers(&mut self) {
        #[cfg(debug_assertions)]
        let test = self.start_buffer[0];
        mem::swap(&mut self.start_buffer, &mut self.end_buffer);
        #[cfg(debug_assertions)]
        assert_eq!(test, self.end_buffer[0]);
    }

    /// Clones this instance, creating a new struct taking ownership of the cache and replacing the
    /// original with a dummy
    fn take_cache_clone(&mut self) -> Self {
        let mut start_buffer: Box<[f64]> = Box::new([]);
        mem::swap(&mut start_buffer, &mut self.start_buffer);
        let mut end_buffer: Box<[f64]> = Box::new([]);
        mem::swap(&mut end_buffer, &mut self.end_buffer);
        Self {
            input_index: self.input_index,
            start_buffer,
            end_buffer,
            first_pass: self.first_pass,
            second_pass: self.second_pass,
            third_pass: self.third_pass,
            result: self.result,
            vertical_cell_count: self.vertical_cell_count,
            min_value: self.min_value,
            max_value: self.max_value,
        }
    }
}

pub struct FlatCache {
    input_index: usize,

    cache: Box<[f64]>,
    start_biome_x: i32,
    start_biome_z: i32,
    horizontal_biome_end: usize,

    min_value: f64,
    max_value: f64,
}

impl ChunkNoiseFunctionRange for FlatCache {
    #[inline]
    fn min(&self) -> f64 {
        self.min_value
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max_value
    }
}

impl FlatCache {
    fn new(
        input_index: usize,
        min_value: f64,
        max_value: f64,
        builder_options: &ChunkNoiseFunctionBuilderOptions,
    ) -> Self {
        Self {
            input_index,
            cache: vec![
                0.0;
                (builder_options.horizontal_biome_end + 1)
                    * (builder_options.horizontal_biome_end + 1)
            ]
            .into_boxed_slice(),
            start_biome_x: builder_options.start_biome_x,
            start_biome_z: builder_options.start_biome_z,
            horizontal_biome_end: builder_options.horizontal_biome_end,
            min_value,
            max_value,
        }
    }

    #[inline]
    fn xz_to_index_const(&self, biome_x_position: usize, biome_z_position: usize) -> usize {
        biome_x_position * (self.horizontal_biome_end + 1) + biome_z_position
    }
}

#[derive(Clone)]
pub struct Cache2D {
    input_index: usize,
    last_sample_column: u64,
    last_sample_result: f64,

    min_value: f64,
    max_value: f64,
}

impl ChunkNoiseFunctionRange for Cache2D {
    #[inline]
    fn min(&self) -> f64 {
        self.min_value
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max_value
    }
}

impl Cache2D {
    fn new(input_index: usize, min_value: f64, max_value: f64) -> Self {
        Self {
            input_index,
            // I know this is because theres is definately world coords that are this marker, but this
            // is how vanilla does it, so I'm going to for pairity
            last_sample_column: chunk_pos::MARKER,
            last_sample_result: Default::default(),
            min_value,
            max_value,
        }
    }
}

pub struct CacheOnce {
    input_index: usize,
    cache_result_unique_id: u64,
    cache_fill_unique_id: u64,
    last_sample_result: f64,

    cache: Box<[f64]>,

    min_value: f64,
    max_value: f64,
}

impl ChunkNoiseFunctionRange for CacheOnce {
    #[inline]
    fn min(&self) -> f64 {
        self.min_value
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max_value
    }
}

impl CacheOnce {
    fn new(input_index: usize, min_value: f64, max_value: f64) -> Self {
        Self {
            input_index,
            // Make these max, just to be different from the overall default of 0
            cache_result_unique_id: 0,
            cache_fill_unique_id: 0,
            last_sample_result: Default::default(),
            cache: Box::new([]),
            min_value,
            max_value,
        }
    }

    /// Clones this instance, creating a new struct taking ownership of the cache and replacing the
    /// original with a dummy
    fn take_cache_clone(&mut self) -> Self {
        let mut cache: Box<[f64]> = Box::new([]);
        mem::swap(&mut cache, &mut self.cache);
        Self {
            input_index: self.input_index,
            cache_result_unique_id: self.cache_result_unique_id,
            cache_fill_unique_id: self.cache_fill_unique_id,
            last_sample_result: self.last_sample_result,
            cache,
            min_value: self.min_value,
            max_value: self.max_value,
        }
    }
}

pub struct CellCache {
    input_index: usize,
    pub(crate) cache: Box<[f64]>,

    min_value: f64,
    max_value: f64,
}

impl ChunkNoiseFunctionRange for CellCache {
    #[inline]
    fn min(&self) -> f64 {
        self.min_value
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max_value
    }
}

impl CellCache {
    fn new(
        input_index: usize,
        min_value: f64,
        max_value: f64,
        build_options: &ChunkNoiseFunctionBuilderOptions,
    ) -> Self {
        Self {
            input_index,
            cache: vec![
                0.0;
                build_options.horizontal_cell_block_count
                    * build_options.horizontal_cell_block_count
                    * build_options.vertical_cell_block_count
            ]
            .into_boxed_slice(),
            min_value,
            max_value,
        }
    }

    /// Clones this instance, creating a new struct taking ownership of the cache and replacing the
    /// original with a dummy
    fn take_cache_clone(&mut self) -> Self {
        let mut cache: Box<[f64]> = Box::new([]);
        mem::swap(&mut cache, &mut self.cache);
        Self {
            input_index: self.input_index,
            cache,
            min_value: self.min_value,
            max_value: self.max_value,
        }
    }
}

/// A complete chunk-specific density function that is able to be sampled.
/// Uses a stack to be able to mutate chunk-specific components as well as
/// all chunk-specific componenets to reference top-level data
pub struct ChunkNoiseFunction<'a> {
    pub(crate) function_components: Box<[ChunkNoiseFunctionComponent<'a>]>,
    pub(crate) cell_cache_indices: Box<[usize]>,
    interpolator_indices: Box<[usize]>,
}

pub enum ChunkNoiseFunctionWrapperHandler {
    PopulateNoise,
    MultiNoiseConfig,
    #[cfg(test)]
    TestNoiseConfig,
}

impl<'a> ChunkNoiseFunction<'a> {
    pub fn new(
        base: &'a ProtoChunkNoiseFunction<'a>,
        wrapper_handler: ChunkNoiseFunctionWrapperHandler,
        build_options: &ChunkNoiseFunctionBuilderOptions,
    ) -> Self {
        let mut components = Vec::with_capacity(base.components().len());
        let mut cell_cache_indices = Vec::new();
        let mut interpolator_indices = Vec::new();
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
                UniversalChunkNoiseFunctionComponent::Wrapped(wrapped) => {
                    // Due to our previous invariant with the proto-function, it is guaranteed
                    // that the wrapped function is already on the stack
                    let min_value = components[wrapped.input_index()].min();
                    let max_value = components[wrapped.input_index()].max();

                    match &wrapper_handler {
                        #[cfg(test)]
                        ChunkNoiseFunctionWrapperHandler::TestNoiseConfig => {
                            components.push(ChunkNoiseFunctionComponent::PassThrough(
                                PassThrough {
                                    input_index: wrapped.input_index(),
                                    min_value,
                                    max_value,
                                },
                            ));
                        }
                        _ => match wrapped.wrapper_type() {
                            WrapperType::Interpolated => {
                                // Interpolation only occurs within the chunk noise sampler. The
                                // multi-noise sampler does not interpolate
                                if matches!(
                                    wrapper_handler,
                                    ChunkNoiseFunctionWrapperHandler::MultiNoiseConfig
                                ) {
                                    components.push(ChunkNoiseFunctionComponent::PassThrough(
                                        PassThrough {
                                            input_index: wrapped.input_index(),
                                            min_value,
                                            max_value,
                                        },
                                    ));
                                } else {
                                    components.push(ChunkNoiseFunctionComponent::ChunkSpecific(
                                        ChunkSpecificNoiseFunctionComponent::DensityInterpolator(
                                            DensityInterpolator::new(
                                                wrapped.input_index(),
                                                min_value,
                                                max_value,
                                                build_options,
                                            ),
                                        ),
                                    ));
                                    interpolator_indices.push(components.len() - 1);
                                }
                            }
                            WrapperType::CacheFlat => {
                                let mut flat_cache = FlatCache::new(
                                    wrapped.input_index(),
                                    min_value,
                                    max_value,
                                    build_options,
                                );
                                let sample_options = ChunkNoiseFunctionSampleOptions::new(
                                    false,
                                    SampleAction::SkipWrappers,
                                    0,
                                    0,
                                    0,
                                );

                                for biome_x_position in 0..=build_options.horizontal_biome_end {
                                    let absolute_biome_x_position =
                                        build_options.start_biome_x + biome_x_position as i32;
                                    let block_x_position =
                                        biome_coords::to_block(absolute_biome_x_position);

                                    for biome_z_position in 0..=build_options.horizontal_biome_end {
                                        let absolute_biome_z_position =
                                            build_options.start_biome_z + biome_z_position as i32;
                                        let block_z_position =
                                            biome_coords::to_block(absolute_biome_z_position);

                                        let pos = UnblendedNoisePos::new(
                                            block_x_position,
                                            0,
                                            block_z_position,
                                        );

                                        // Due to our stack invariant, what is on the stack is a
                                        // valid density function
                                        let sample = sample_component_stack(
                                            &mut components,
                                            wrapped.input_index(),
                                            &pos,
                                            &sample_options,
                                        );
                                        let cache_index = flat_cache
                                            .xz_to_index_const(biome_x_position, biome_z_position);
                                        flat_cache.cache[cache_index] = sample;
                                    }
                                }

                                components.push(ChunkNoiseFunctionComponent::ChunkSpecific(
                                    ChunkSpecificNoiseFunctionComponent::FlatCache(flat_cache),
                                ));
                            }
                            WrapperType::Cache2D => {
                                components.push(ChunkNoiseFunctionComponent::ChunkSpecific(
                                    ChunkSpecificNoiseFunctionComponent::Cache2D(Cache2D::new(
                                        wrapped.input_index(),
                                        min_value,
                                        max_value,
                                    )),
                                ));
                            }
                            WrapperType::CacheOnce => {
                                components.push(ChunkNoiseFunctionComponent::ChunkSpecific(
                                    ChunkSpecificNoiseFunctionComponent::CacheOnce(CacheOnce::new(
                                        wrapped.input_index(),
                                        min_value,
                                        max_value,
                                    )),
                                ));
                            }
                            WrapperType::CellCache => {
                                components.push(ChunkNoiseFunctionComponent::ChunkSpecific(
                                    ChunkSpecificNoiseFunctionComponent::CellCache(CellCache::new(
                                        wrapped.input_index(),
                                        min_value,
                                        max_value,
                                        build_options,
                                    )),
                                ));
                                cell_cache_indices.push(components.len() - 1);
                            }
                        },
                    }
                }
            });

        Self {
            function_components: components.into_boxed_slice(),
            cell_cache_indices: cell_cache_indices.into_boxed_slice(),
            interpolator_indices: interpolator_indices.into_boxed_slice(),
        }
    }

    #[inline]
    fn sample_index(
        &mut self,
        index: usize,
        pos: &impl NoisePos,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        sample_component_stack(&mut self.function_components, index, pos, sample_options)
    }

    #[inline]
    pub fn sample(
        &mut self,
        pos: &impl NoisePos,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> f64 {
        let last_index = self.function_components.len() - 1;
        self.sample_index(last_index, pos, sample_options)
    }

    #[cfg(test)]
    pub fn sample_test(&mut self, pos: &impl NoisePos) -> f64 {
        // These options are not actually used because we never build chunk-specific components in
        // the config tests
        let dummy_options =
            ChunkNoiseFunctionSampleOptions::new(false, SampleAction::SkipWrappers, 0, 0, 0);
        self.sample(pos, &dummy_options)
    }

    fn recursive_fill(
        &mut self,
        component_index: usize,
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        let component = &mut self.function_components[component_index];
        match component {
            ChunkNoiseFunctionComponent::StaticIndependent(static_independent) => {
                static_independent.fill(array, mapper);
            }
            ChunkNoiseFunctionComponent::StaticDependent(
                StaticDependentChunkNoiseFunctionComponent::RangeChoice(range_choice),
            ) => {
                self.recursive_fill(range_choice.input_index, array, mapper, sample_options);
                array.iter_mut().enumerate().for_each(|(index, value)| {
                    let pos = mapper.at(index, Some(sample_options));
                    *value = if *value >= *range_choice.data.min_inclusive()
                        && *value < *range_choice.data.max_exclusive()
                    {
                        sample_component_stack(
                            &mut self.function_components,
                            range_choice.when_in_index,
                            &pos,
                            sample_options,
                        )
                    } else {
                        sample_component_stack(
                            &mut self.function_components,
                            range_choice.when_out_index,
                            &pos,
                            sample_options,
                        )
                    };
                });
            }
            ChunkNoiseFunctionComponent::StaticDependent(
                StaticDependentChunkNoiseFunctionComponent::Linear(linear),
            ) => {
                self.recursive_fill(linear.arg_index, array, mapper, sample_options);
                array.iter_mut().for_each(|value| {
                    *value = linear.data.apply_density(*value);
                });
            }
            ChunkNoiseFunctionComponent::StaticDependent(
                StaticDependentChunkNoiseFunctionComponent::Unary(unary),
            ) => {
                self.recursive_fill(unary.arg_index, array, mapper, sample_options);
                array.iter_mut().for_each(|value| {
                    *value = unary.data.apply_density(*value);
                });
            }
            ChunkNoiseFunctionComponent::StaticDependent(
                StaticDependentChunkNoiseFunctionComponent::Clamp(clamp),
            ) => {
                self.recursive_fill(clamp.input_index, array, mapper, sample_options);
                array.iter_mut().for_each(|value| {
                    *value = clamp.data.apply_density(*value);
                });
            }
            ChunkNoiseFunctionComponent::StaticDependent(
                StaticDependentChunkNoiseFunctionComponent::Binary(binary),
            ) => {
                self.recursive_fill(binary.arg1_index, array, mapper, sample_options);
                match binary.data.operation() {
                    BinaryOperation::Add => {
                        let mut temp_array = vec![0.0; array.len()];
                        self.recursive_fill(
                            binary.arg2_index,
                            &mut temp_array,
                            mapper,
                            sample_options,
                        );
                        array
                            .iter_mut()
                            .zip(temp_array)
                            .for_each(|(value, temp)| *value += temp);
                    }
                    BinaryOperation::Mul => {
                        array.iter_mut().enumerate().for_each(|(index, value)| {
                            if *value != 0.0 {
                                let pos = mapper.at(index, Some(sample_options));
                                *value *= sample_component_stack(
                                    &mut self.function_components,
                                    binary.arg2_index,
                                    &pos,
                                    sample_options,
                                );
                            }
                        });
                    }
                    BinaryOperation::Min => {
                        let min_2 = self.function_components[binary.arg2_index].min();
                        array.iter_mut().enumerate().for_each(|(index, value)| {
                            if *value > min_2 {
                                let pos = mapper.at(index, Some(sample_options));
                                *value = value.min(sample_component_stack(
                                    &mut self.function_components,
                                    binary.arg2_index,
                                    &pos,
                                    sample_options,
                                ));
                            }
                        });
                    }
                    BinaryOperation::Max => {
                        let max_2 = self.function_components[binary.arg2_index].max();
                        array.iter_mut().enumerate().for_each(|(index, value)| {
                            if *value < max_2 {
                                let pos = mapper.at(index, Some(sample_options));
                                *value = value.max(sample_component_stack(
                                    &mut self.function_components,
                                    binary.arg2_index,
                                    &pos,
                                    sample_options,
                                ));
                            }
                        });
                    }
                }
            }
            ChunkNoiseFunctionComponent::StaticDependent(
                StaticDependentChunkNoiseFunctionComponent::WeirdScaled(weird_scaled),
            ) => {
                self.recursive_fill(weird_scaled.input_index, array, mapper, sample_options);
                array.iter_mut().enumerate().for_each(|(index, value)| {
                    let pos = mapper.at(index, Some(sample_options));
                    let scaled_density = weird_scaled.data.mapper().scale(*value);
                    *value = scaled_density
                        * weird_scaled
                            .sampler
                            .sample(
                                pos.x() as f64 / scaled_density,
                                pos.y() as f64 / scaled_density,
                                pos.z() as f64 / scaled_density,
                            )
                            .abs();
                });
            }
            ChunkNoiseFunctionComponent::ChunkSpecific(
                ChunkSpecificNoiseFunctionComponent::CacheOnce(cache_once),
            ) => {
                if cache_once.cache_fill_unique_id == sample_options.cache_fill_unique_id
                    && !cache_once.cache.is_empty()
                {
                    array.copy_from_slice(&cache_once.cache);
                    return;
                }

                // Effectively take whats on the stack, leaving an invalid state remaining
                let mut cache_once = cache_once.take_cache_clone();
                #[cfg(debug_assertions)]
                assert!(cache_once.cache.len() > 0);

                // This is safe because one of our invariants is no cyclic graphs on the map; the
                // input index will never reference this
                self.recursive_fill(cache_once.input_index, array, mapper, sample_options);

                // We need to make a new cache
                if cache_once.cache.len() != array.len() {
                    cache_once.cache = vec![0.0; array.len()].into_boxed_slice();
                }

                // Set values and replace in stack
                cache_once.cache.copy_from_slice(array);
                cache_once.cache_fill_unique_id = sample_options.cache_fill_unique_id;
                self.function_components[component_index] =
                    ChunkNoiseFunctionComponent::ChunkSpecific(
                        ChunkSpecificNoiseFunctionComponent::CacheOnce(cache_once),
                    );
            }
            ChunkNoiseFunctionComponent::ChunkSpecific(
                ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
            ) => {
                if sample_options.populating_caches {
                    array.iter_mut().enumerate().for_each(|(index, value)| {
                        let pos = mapper.at(index, Some(sample_options));
                        let result = sample_component_stack(
                            &mut self.function_components,
                            component_index,
                            &pos,
                            sample_options,
                        );
                        *value = result;
                    });
                } else {
                    let input_index = density_interpolator.input_index;
                    self.recursive_fill(input_index, array, mapper, sample_options);
                }
            }
            // The default
            _ => {
                array.iter_mut().enumerate().for_each(|(index, value)| {
                    let pos = mapper.at(index, Some(sample_options));
                    let result = sample_component_stack(
                        &mut self.function_components,
                        component_index,
                        &pos,
                        sample_options,
                    );
                    *value = result;
                });
            }
        }
    }

    #[inline]
    pub fn fill(
        &mut self,
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        let index = self.function_components.len() - 1;
        self.recursive_fill(index, array, mapper, sample_options);
    }
}

#[enum_dispatch]
pub trait ChunkDensityFunctionOwner {
    fn fill_cell_caches(
        &mut self,
        mapper: &impl IndexToNoisePos,
        options: &mut ChunkNoiseFunctionSampleOptions,
    );

    fn fill_interpolator_buffers(
        &mut self,
        start: bool,
        cell_z: usize,
        mapper: &impl IndexToNoisePos,
        options: &mut ChunkNoiseFunctionSampleOptions,
    );

    fn on_sampled_cell_corners(&mut self, cell_y_position: usize, cell_z_position: usize);
    fn interpolate_x(&mut self, delta: f64);
    fn interpolate_y(&mut self, delta: f64);
    fn interpolate_z(&mut self, delta: f64);
    fn swap_buffers(&mut self);
}

impl ChunkDensityFunctionOwner for ChunkNoiseFunction<'_> {
    fn fill_cell_caches(
        &mut self,
        mapper: &impl IndexToNoisePos,
        options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        let indices = self.cell_cache_indices.clone();
        for cell_cache_index in indices {
            let cell_cache = match &mut self.function_components[cell_cache_index] {
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::CellCache(cell_cache),
                ) => cell_cache,
                _ => unreachable!(),
            };
            // Effectively take whats on the stack, leaving an invalid state remaining
            let mut cell_cache = cell_cache.take_cache_clone();

            #[cfg(debug_assertions)]
            assert!(cell_cache.cache.len() > 0);

            // This is safe because one of our invariants is no cyclic graphs on the map; the
            // input index will never reference this
            self.recursive_fill(
                cell_cache.input_index,
                &mut cell_cache.cache,
                mapper,
                options,
            );

            // Replace version on the stack
            self.function_components[cell_cache_index] = ChunkNoiseFunctionComponent::ChunkSpecific(
                ChunkSpecificNoiseFunctionComponent::CellCache(cell_cache),
            );
        }
    }

    fn fill_interpolator_buffers(
        &mut self,
        start: bool,
        cell_z: usize,
        mapper: &impl IndexToNoisePos,
        options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        let indices = self.interpolator_indices.clone();
        for interpolator_index in indices {
            let density_interpolator = match &mut self.function_components[interpolator_index] {
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
                ) => density_interpolator,
                _ => unreachable!(),
            };
            // Effectively take whats on the stack, leaving an invalid state remaining
            let mut density_interpolator = density_interpolator.take_cache_clone();
            let start_index = density_interpolator.yz_to_buf_index(0, cell_z);
            let buf = if start {
                &mut density_interpolator.start_buffer
                    [start_index..=start_index + density_interpolator.vertical_cell_count]
            } else {
                &mut density_interpolator.end_buffer
                    [start_index..=start_index + density_interpolator.vertical_cell_count]
            };

            // This is safe because one of our invariants is no cyclic graphs on the map; the
            // input index will never reference this
            self.recursive_fill(density_interpolator.input_index, buf, mapper, options);

            // Replace version on the stack
            self.function_components[interpolator_index] =
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
                );
        }
    }

    fn interpolate_x(&mut self, delta: f64) {
        let indices = self.interpolator_indices.clone();
        for interpolator_index in indices {
            let density_interpolator = match &mut self.function_components[interpolator_index] {
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
                ) => density_interpolator,
                _ => unreachable!(),
            };
            density_interpolator.interpolate_x(delta);
        }
    }

    fn interpolate_y(&mut self, delta: f64) {
        let indices = self.interpolator_indices.clone();
        for interpolator_index in indices {
            let density_interpolator = match &mut self.function_components[interpolator_index] {
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
                ) => density_interpolator,
                _ => unreachable!(),
            };
            density_interpolator.interpolate_y(delta);
        }
    }

    fn interpolate_z(&mut self, delta: f64) {
        let indices = self.interpolator_indices.clone();
        for interpolator_index in indices {
            let density_interpolator = match &mut self.function_components[interpolator_index] {
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
                ) => density_interpolator,
                _ => unreachable!(),
            };
            density_interpolator.interpolate_z(delta);
        }
    }

    fn on_sampled_cell_corners(&mut self, cell_y_position: usize, cell_z_position: usize) {
        let indices = self.interpolator_indices.clone();
        for interpolator_index in indices {
            let density_interpolator = match &mut self.function_components[interpolator_index] {
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
                ) => density_interpolator,
                _ => unreachable!(),
            };
            density_interpolator.on_sampled_cell_corners(cell_y_position, cell_z_position);
        }
    }

    fn swap_buffers(&mut self) {
        let indices = self.interpolator_indices.clone();
        for interpolator_index in indices {
            let density_interpolator = match &mut self.function_components[interpolator_index] {
                ChunkNoiseFunctionComponent::ChunkSpecific(
                    ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator),
                ) => density_interpolator,
                _ => unreachable!(),
            };
            density_interpolator.swap_buffers();
        }
    }
}

#[enum_dispatch(ChunkNoiseFunctionRange)]
pub enum ChunkSpecificNoiseFunctionComponent {
    DensityInterpolator(DensityInterpolator),
    FlatCache(FlatCache),
    Cache2D(Cache2D),
    CacheOnce(CacheOnce),
    CellCache(CellCache),
}

pub enum ChunkNoiseFunctionComponent<'a> {
    StaticIndependent(&'a StaticIndependentChunkNoiseFunctionComponent<'a>),
    StaticDependent(&'a StaticDependentChunkNoiseFunctionComponent<'a>),
    ChunkSpecific(ChunkSpecificNoiseFunctionComponent),
    PassThrough(PassThrough),
}

impl ChunkNoiseFunctionComponent<'_> {
    fn name(&self) -> &str {
        match self {
            Self::StaticIndependent(independent) => match independent {
                StaticIndependentChunkNoiseFunctionComponent::ShiftA(_) => "ShiftA",
                StaticIndependentChunkNoiseFunctionComponent::Noise(_) => "Noise",
                StaticIndependentChunkNoiseFunctionComponent::ClampedYGradient(_) => {
                    "ClampedYGradient"
                }
                StaticIndependentChunkNoiseFunctionComponent::EndIsland(_) => "EndIsland",
                StaticIndependentChunkNoiseFunctionComponent::Constant(_) => "Constant",
                StaticIndependentChunkNoiseFunctionComponent::InterpolatedNoise(_) => {
                    "InterpolatedNoise"
                }
                StaticIndependentChunkNoiseFunctionComponent::ShiftB(_) => "ShiftB",
            },
            Self::StaticDependent(dependent) => match dependent {
                StaticDependentChunkNoiseFunctionComponent::PassThrough(_) => "PassThrough",
                StaticDependentChunkNoiseFunctionComponent::Binary(_) => "Binary",
                StaticDependentChunkNoiseFunctionComponent::Unary(_) => "Unary",
                StaticDependentChunkNoiseFunctionComponent::Linear(_) => "Linear",
                StaticDependentChunkNoiseFunctionComponent::ShiftedNoise(_) => "ShiftedNoise",
                StaticDependentChunkNoiseFunctionComponent::Spline(_) => "Spline",
                StaticDependentChunkNoiseFunctionComponent::Clamp(_) => "Clamp",
                StaticDependentChunkNoiseFunctionComponent::RangeChoice(_) => "RangeChoice",
                StaticDependentChunkNoiseFunctionComponent::WeirdScaled(_) => "WeirdScaled",
            },
            Self::ChunkSpecific(chunk_specific) => match chunk_specific {
                ChunkSpecificNoiseFunctionComponent::CellCache(_) => "CellCache",
                ChunkSpecificNoiseFunctionComponent::DensityInterpolator(_) => {
                    "DensityInterpolator"
                }
                ChunkSpecificNoiseFunctionComponent::FlatCache(_) => "FlatCache",
                ChunkSpecificNoiseFunctionComponent::Cache2D(_) => "Cache2D",
                ChunkSpecificNoiseFunctionComponent::CacheOnce(_) => "CacheOnce",
            },
            Self::PassThrough(_) => "PassThrough",
        }
    }
}

impl ChunkNoiseFunctionRange for ChunkNoiseFunctionComponent<'_> {
    #[inline]
    fn min(&self) -> f64 {
        match self {
            Self::StaticDependent(dependent) => dependent.min(),
            Self::StaticIndependent(independent) => independent.min(),
            Self::ChunkSpecific(chunk_specific) => chunk_specific.min(),
            Self::PassThrough(pass_through) => pass_through.min(),
        }
    }

    #[inline]
    fn max(&self) -> f64 {
        match self {
            Self::StaticDependent(dependent) => dependent.max(),
            Self::StaticIndependent(independent) => independent.max(),
            Self::ChunkSpecific(chunk_specific) => chunk_specific.max(),
            Self::PassThrough(pass_through) => pass_through.max(),
        }
    }
}
