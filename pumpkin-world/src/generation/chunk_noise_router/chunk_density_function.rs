use std::mem;

use super::density_function::{
    spline::{Range, Spline, SplineValue},
    ChunkNoiseFunctionRange, IndexToNoisePos, NoisePos, ProtoChunkNoiseFunction,
    StaticDependentChunkNoiseFunctionComponent, StaticIndependentChunkNoiseFunctionComponent,
    StaticIndependentChunkNoiseFunctionComponentImpl, UnblendedNoisePos,
    UniversalChunkNoiseFunctionComponent,
};
use enum_dispatch::enum_dispatch;
use pumpkin_util::math::vector2::Vector2;

use crate::{
    generation::{
        biome_coords,
        noise::{lerp, lerp3, perlin::DoublePerlinNoiseSampler},
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
        },
        ChunkNoiseFunctionComponent::ChunkSpecific(chunk_specific) => match chunk_specific {
            ChunkSpecificNoiseFunctionComponent::DensityInterpolator(density_interpolator) => {
                match sample_options.action {
                    SampleAction::Wrappers {
                        cell_x_block_position,
                        cell_y_block_position,
                        cell_z_block_position,
                        horizontal_cell_block_count,
                        vertical_cell_block_count,
                        cache_result_unique_id: _,
                        cache_fill_unique_id: _,
                        fill_index: _,
                    } => {
                        #[cfg(debug_assertions)]
                        assert!(sample_options.interpolating);

                        if sample_options.populating_caches {
                            lerp3(
                                cell_x_block_position as f64 / horizontal_cell_block_count as f64,
                                cell_y_block_position as f64 / vertical_cell_block_count as f64,
                                cell_z_block_position as f64 / horizontal_cell_block_count as f64,
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
                        sample_component_stack(component_stack, input_index, pos, sample_options)
                    }
                }
            }
            ChunkSpecificNoiseFunctionComponent::FlatCache(flat_cache) => {
                let absolute_biome_x_position = biome_coords::from_block(pos.x());
                let absolute_biome_z_position = biome_coords::from_block(pos.z());

                let relative_biome_x_position =
                    absolute_biome_x_position - sample_options.start_biome_x;
                let relative_biome_z_position =
                    absolute_biome_z_position - sample_options.start_biome_z;

                let sample_index = flat_cache.xz_to_index_const(
                    relative_biome_x_position as usize,
                    relative_biome_z_position as usize,
                );

                flat_cache.cache[sample_index]
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
                    SampleAction::Wrappers {
                        cell_x_block_position: _,
                        cell_y_block_position: _,
                        cell_z_block_position: _,
                        horizontal_cell_block_count: _,
                        vertical_cell_block_count: _,
                        cache_result_unique_id,
                        cache_fill_unique_id,
                        fill_index,
                    } => {
                        if cache_once.cache_fill_unique_id == cache_fill_unique_id {
                            return cache_once.cache[fill_index];
                        }

                        if cache_once.cache_result_unique_id == cache_result_unique_id {
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
                            cache_once.cache_result_unique_id = cache_result_unique_id;
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
                        sample_component_stack(component_stack, input_index, pos, sample_options)
                    }
                }
            }
            ChunkSpecificNoiseFunctionComponent::CellCache(cell_cache) => {
                match sample_options.action {
                    SampleAction::Wrappers {
                        cell_x_block_position,
                        cell_y_block_position,
                        cell_z_block_position,
                        horizontal_cell_block_count,
                        vertical_cell_block_count,
                        cache_result_unique_id: _,
                        cache_fill_unique_id: _,
                        fill_index: _,
                    } => {
                        #[cfg(debug_assertions)]
                        assert!(sample_options.interpolating);

                        let cache_index = ((vertical_cell_block_count - 1 - cell_y_block_position)
                            * horizontal_cell_block_count
                            + cell_x_block_position)
                            * horizontal_cell_block_count
                            + cell_z_block_position;

                        cell_cache.cache[cache_index]
                    }
                    SampleAction::SkipWrappers => {
                        let input_index = cell_cache.input_index;
                        sample_component_stack(component_stack, input_index, pos, sample_options)
                    }
                }
            }
        },
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

pub enum SampleAction {
    SkipWrappers,
    Wrappers {
        // Our relative position within the cell
        cell_x_block_position: usize,
        cell_y_block_position: usize,
        cell_z_block_position: usize,

        // Number of blocks per cell per axis
        horizontal_cell_block_count: usize,
        vertical_cell_block_count: usize,

        // Global IDs for the `CacheOnce` wrapper
        cache_result_unique_id: u64,
        cache_fill_unique_id: u64,

        // The current index of a slice being filled by the `fill` function
        fill_index: usize,
    },
}

pub struct ChunkNoiseFunctionSampleOptions {
    interpolating: bool,
    populating_caches: bool,
    action: SampleAction,

    // The biome coords of this chunk
    start_biome_x: i32,
    start_biome_z: i32,
}

impl ChunkNoiseFunctionSampleOptions {
    pub fn new(
        interpolating: bool,
        populating_caches: bool,
        action: SampleAction,
        start_biome_x: i32,
        start_biome_z: i32,
    ) -> Self {
        Self {
            interpolating,
            populating_caches,
            action,
            start_biome_x,
            start_biome_z,
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
            self.end_buffer[self.yz_to_buf_index(cell_y_position, cell_z_position)];
        self.first_pass[2] =
            self.start_buffer[self.yz_to_buf_index(cell_y_position, cell_z_position)];
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
        mem::swap(&mut self.start_buffer, &mut self.end_buffer);
    }
}

pub struct FlatCache {
    input_index: usize,

    cache: Box<[f64]>,
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
            cache_result_unique_id: u64::MAX,
            cache_fill_unique_id: u64::MAX,
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
    cache: Box<[f64]>,

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
}

/// A complete chunk-specific density function that is able to be sampled.
/// Uses a stack to be able to mutate chunk-specific components as well as
/// all chunk-specific componenets to reference top-level data
pub struct ChunkNoiseFunction<'a> {
    function_components: Box<[ChunkNoiseFunctionComponent<'a>]>,
    build_options: &'a ChunkNoiseFunctionBuilderOptions,
}

pub enum ChunkNoiseFunctionWrapperHandler {
    PopulateNoise,
    MultiNoiseConfig,
    #[cfg(test)]
    TestNoiseConfig,
}

struct PassThrough {
    input_index: usize,
    min_value: f64,
    max_value: f64,
}

impl ChunkNoiseFunctionRange for PassThrough {
    #[inline]
    fn min(&self) -> f64 {
        self.min_value
    }

    #[inline]
    fn max(&self) -> f64 {
        self.max_value
    }
}

impl<'a> ChunkNoiseFunction<'a> {
    pub fn new(
        base: &'a ProtoChunkNoiseFunction,
        wrapper_handler: ChunkNoiseFunctionWrapperHandler,
        options: &'a ChunkNoiseFunctionBuilderOptions,
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
                                                options,
                                            ),
                                        ),
                                    ));
                                }
                            }
                            WrapperType::CacheFlat => {
                                let mut flat_cache = FlatCache::new(
                                    wrapped.input_index(),
                                    min_value,
                                    max_value,
                                    options,
                                );

                                for biome_x_position in 0..options.horizontal_biome_end {
                                    let absolute_biome_x_position =
                                        options.start_biome_x + biome_x_position as i32;
                                    let block_x_position =
                                        biome_coords::to_block(absolute_biome_x_position);

                                    for biome_z_position in 0..options.horizontal_biome_end {
                                        let absolute_biome_z_position =
                                            options.start_biome_z + biome_z_position as i32;
                                        let block_z_position =
                                            biome_coords::to_block(absolute_biome_z_position);

                                        let pos = UnblendedNoisePos::new(
                                            block_x_position,
                                            0,
                                            block_z_position,
                                        );
                                        let options = ChunkNoiseFunctionSampleOptions::new(
                                            false,
                                            false,
                                            SampleAction::SkipWrappers,
                                            options.start_biome_x,
                                            options.start_biome_z,
                                        );

                                        // Due to our stack invariant, what is on the stack is a
                                        // valid density function
                                        let sample = sample_component_stack(
                                            &mut components,
                                            wrapped.input_index(),
                                            &pos,
                                            &options,
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
                                        options,
                                    )),
                                ));
                            }
                        },
                    }
                }
            });

        Self {
            function_components: components.into_boxed_slice(),
            build_options: options,
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
            ChunkNoiseFunctionSampleOptions::new(false, false, SampleAction::SkipWrappers, 0, 0);
        self.sample(pos, &dummy_options)
    }

    fn recursive_fill(
        &mut self,
        index: usize,
        array: &mut [f64],
        mapper: &impl IndexToNoisePos,
        sample_options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        let component = &mut self.function_components[index];
        match component {
            ChunkNoiseFunctionComponent::StaticIndependent(static_independent) => {
                static_independent.fill(array, mapper);
            }
            ChunkNoiseFunctionComponent::StaticDependent(
                StaticDependentChunkNoiseFunctionComponent::RangeChoice(range_choice),
            ) => {
                self.recursive_fill(range_choice.input_index, array, mapper, sample_options);
                array.iter_mut().enumerate().for_each(|(index, value)| {
                    if let SampleAction::Wrappers {
                        cell_x_block_position: _,
                        cell_y_block_position: _,
                        cell_z_block_position: _,
                        horizontal_cell_block_count: _,
                        vertical_cell_block_count: _,
                        cache_result_unique_id: _,
                        cache_fill_unique_id: _,
                        fill_index,
                    } = &mut sample_options.action
                    {
                        *fill_index = index;
                    }

                    let pos = mapper.at(index);
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
                        let temp_array = vec![0.0; array.len()];
                        self.recursive_fill(binary.arg2_index, array, mapper, sample_options);
                        array
                            .iter_mut()
                            .zip(temp_array)
                            .for_each(|(value, temp)| *value += temp);
                    }
                    BinaryOperation::Mul => {
                        array.iter_mut().enumerate().for_each(|(index, value)| {
                            if let SampleAction::Wrappers {
                                cell_x_block_position: _,
                                cell_y_block_position: _,
                                cell_z_block_position: _,
                                horizontal_cell_block_count: _,
                                vertical_cell_block_count: _,
                                cache_result_unique_id: _,
                                cache_fill_unique_id: _,
                                fill_index,
                            } = &mut sample_options.action
                            {
                                *fill_index = index;
                            }

                            if *value != 0.0 {
                                let pos = mapper.at(index);
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
                            if let SampleAction::Wrappers {
                                cell_x_block_position: _,
                                cell_y_block_position: _,
                                cell_z_block_position: _,
                                horizontal_cell_block_count: _,
                                vertical_cell_block_count: _,
                                cache_result_unique_id: _,
                                cache_fill_unique_id: _,
                                fill_index,
                            } = &mut sample_options.action
                            {
                                *fill_index = index;
                            }

                            if *value > min_2 {
                                let pos = mapper.at(index);
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
                            if let SampleAction::Wrappers {
                                cell_x_block_position: _,
                                cell_y_block_position: _,
                                cell_z_block_position: _,
                                horizontal_cell_block_count: _,
                                vertical_cell_block_count: _,
                                cache_result_unique_id: _,
                                cache_fill_unique_id: _,
                                fill_index,
                            } = &mut sample_options.action
                            {
                                *fill_index = index;
                            }

                            if *value < max_2 {
                                let pos = mapper.at(index);
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
                array.iter_mut().for_each(|value| {
                    let pos = mapper.at(index);
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
                match sample_options.action {
                    SampleAction::Wrappers {
                        cell_x_block_position: _,
                        cell_y_block_position: _,
                        cell_z_block_position: _,
                        horizontal_cell_block_count: _,
                        vertical_cell_block_count: _,
                        cache_result_unique_id: _,
                        cache_fill_unique_id,
                        fill_index: _,
                    } => {
                        if cache_once.cache_fill_unique_id == cache_fill_unique_id {
                            array.copy_from_slice(&cache_once.cache);
                            return;
                        }

                        // Effectively take whats on the stack, leaving an invalid state remaining
                        let mut cache_once = cache_once.take_cache_clone();
                        // This is safe because one of our invariants is no cyclic graphs on the map; the
                        // input index will never reference this
                        self.recursive_fill(cache_once.input_index, array, mapper, sample_options);

                        // We need to make a new cache
                        if cache_once.cache.len() != array.len() {
                            cache_once.cache = vec![0.0; array.len()].into_boxed_slice();
                        }

                        // Set values and replace in stack
                        cache_once.cache.copy_from_slice(array);
                        cache_once.cache_fill_unique_id = cache_fill_unique_id;
                        self.function_components[index] =
                            ChunkNoiseFunctionComponent::ChunkSpecific(
                                ChunkSpecificNoiseFunctionComponent::CacheOnce(cache_once),
                            );
                    }
                    SampleAction::SkipWrappers => {
                        let input_index = cache_once.input_index;
                        self.recursive_fill(input_index, array, mapper, sample_options);
                    }
                }
            }
            // The default
            _ => {
                array.iter_mut().enumerate().for_each(|(index, value)| {
                    if let SampleAction::Wrappers {
                        cell_x_block_position: _,
                        cell_y_block_position: _,
                        cell_z_block_position: _,
                        horizontal_cell_block_count: _,
                        vertical_cell_block_count: _,
                        cache_result_unique_id: _,
                        cache_fill_unique_id: _,
                        fill_index,
                    } = &mut sample_options.action
                    {
                        *fill_index = index;
                    }

                    let pos = mapper.at(index);
                    *value = sample_component_stack(
                        &mut self.function_components,
                        index,
                        &pos,
                        sample_options,
                    );
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

#[enum_dispatch(ChunkNoiseFunctionRange)]
pub enum ChunkSpecificNoiseFunctionComponent {
    DensityInterpolator(DensityInterpolator),
    FlatCache(FlatCache),
    Cache2D(Cache2D),
    CacheOnce(CacheOnce),
    CellCache(CellCache),
}

enum ChunkNoiseFunctionComponent<'a> {
    StaticIndependent(&'a StaticIndependentChunkNoiseFunctionComponent<'a>),
    StaticDependent(&'a StaticDependentChunkNoiseFunctionComponent<'a>),
    ChunkSpecific(ChunkSpecificNoiseFunctionComponent),
    PassThrough(PassThrough),
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
