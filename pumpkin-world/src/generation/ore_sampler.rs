use pumpkin_util::random::RandomDeriver;

use crate::block::BlockState;

use super::{
    chunk_noise_router::{
        chunk_density_function::{
            ChunkDensityFunctionOwner, ChunkNoiseFunction, ChunkNoiseFunctionSampleOptions,
        },
        density_function::{IndexToNoisePos, NoisePos},
    },
    noise::clamped_map,
};

pub struct OreVeinSampler<'a> {
    vein_toggle: ChunkNoiseFunction<'a>,
    vein_ridged: ChunkNoiseFunction<'a>,
    vein_gap: ChunkNoiseFunction<'a>,
    random_deriver: RandomDeriver,
}

impl<'a> OreVeinSampler<'a> {
    pub fn new(
        random_deriver: RandomDeriver,
        vein_toggle: ChunkNoiseFunction<'a>,
        vein_ridged: ChunkNoiseFunction<'a>,
        vein_gap: ChunkNoiseFunction<'a>,
    ) -> Self {
        Self {
            vein_toggle,
            vein_ridged,
            vein_gap,
            random_deriver,
        }
    }

    #[inline]
    fn density_functions(&mut self) -> [&mut ChunkNoiseFunction<'a>; 3] {
        [
            &mut self.vein_toggle,
            &mut self.vein_ridged,
            &mut self.vein_gap,
        ]
    }
}

impl ChunkDensityFunctionOwner for OreVeinSampler<'_> {
    #[inline]
    fn fill_cell_caches(
        &mut self,
        mapper: &impl IndexToNoisePos,
        options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        self.density_functions()
            .into_iter()
            .for_each(|function| function.fill_cell_caches(mapper, options));
    }

    #[inline]
    fn fill_interpolator_buffers(
        &mut self,
        start: bool,
        cell_z: usize,
        mapper: &impl IndexToNoisePos,
        options: &mut ChunkNoiseFunctionSampleOptions,
    ) {
        self.density_functions().into_iter().for_each(|function| {
            function.fill_interpolator_buffers(start, cell_z, mapper, options)
        });
    }

    #[inline]
    fn interpolate_x(&mut self, delta: f64) {
        self.density_functions()
            .into_iter()
            .for_each(|function| function.interpolate_x(delta));
    }

    #[inline]
    fn interpolate_y(&mut self, delta: f64) {
        self.density_functions()
            .into_iter()
            .for_each(|function| function.interpolate_y(delta));
    }

    #[inline]
    fn interpolate_z(&mut self, delta: f64) {
        self.density_functions()
            .into_iter()
            .for_each(|function| function.interpolate_z(delta));
    }

    #[inline]
    fn swap_buffers(&mut self) {
        self.density_functions()
            .into_iter()
            .for_each(|function| function.swap_buffers());
    }

    #[inline]
    fn on_sampled_cell_corners(&mut self, cell_y_position: usize, cell_z_position: usize) {
        self.density_functions().into_iter().for_each(|function| {
            function.on_sampled_cell_corners(cell_y_position, cell_z_position)
        });
    }
}

impl OreVeinSampler<'_> {
    pub fn sample(
        &mut self,
        pos: &impl NoisePos,
        sample_options: &ChunkNoiseFunctionSampleOptions,
    ) -> Option<BlockState> {
        let vein_sample = self.vein_toggle.sample(pos, sample_options);
        let vein_type: &VeinType = if vein_sample > 0f64 {
            &vein_type::COPPER
        } else {
            &vein_type::IRON
        };

        let block_y = pos.y();
        let max_to_y = vein_type.max_y - block_y;
        let y_to_min = block_y - vein_type.min_y;
        if (max_to_y >= 0) && (y_to_min >= 0) {
            let closest_to_bound = max_to_y.min(y_to_min);
            let mapped_diff = clamped_map(closest_to_bound as f64, 0f64, 20f64, -0.2f64, 0f64);
            let abs_sample = vein_sample.abs();
            if abs_sample + mapped_diff >= 0.4f32 as f64 {
                let mut random = self.random_deriver.split_pos(pos.x(), block_y, pos.z());
                if random.next_f32() <= 0.7f32
                    && self.vein_ridged.sample(pos, sample_options) < 0f64
                {
                    let clamped_sample = clamped_map(
                        abs_sample,
                        0.4f32 as f64,
                        0.6f32 as f64,
                        0.1f32 as f64,
                        0.3f32 as f64,
                    );

                    return if (random.next_f32() as f64) < clamped_sample
                        && self.vein_gap.sample(pos, sample_options) > (-0.3f32 as f64)
                    {
                        Some(if random.next_f32() < 0.02f32 {
                            vein_type.raw_ore
                        } else {
                            vein_type.ore
                        })
                    } else {
                        Some(vein_type.stone)
                    };
                }
            }
        }
        None
    }
}

pub struct VeinType {
    ore: BlockState,
    raw_ore: BlockState,
    stone: BlockState,
    min_y: i32,
    max_y: i32,
}

// One of the victims of removing compile time blocks
pub mod vein_type {
    use pumpkin_macros::block_state;

    use super::*;

    pub const COPPER: VeinType = VeinType {
        ore: block_state!("copper_ore"),
        raw_ore: block_state!("raw_copper_block"),
        stone: block_state!("granite"),
        min_y: 0,
        max_y: 50,
    };
    pub const IRON: VeinType = VeinType {
        ore: block_state!("deepslate_iron_ore"),
        raw_ore: block_state!("raw_iron_block"),
        stone: block_state!("tuff"),
        min_y: -60,
        max_y: -8,
    };
    pub const MIN_Y: i32 = IRON.min_y;
    pub const MAX_Y: i32 = COPPER.max_y;
}
