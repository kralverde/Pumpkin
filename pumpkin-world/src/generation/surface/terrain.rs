use pumpkin_macros::block_state;
use pumpkin_util::{
    math::vector3::Vector3,
    random::{RandomDeriver, RandomGenerator},
};

use crate::{
    ProtoChunk,
    block::ChunkBlockState,
    generation::{
        chunk_noise::WATER_BLOCK, height_limit::HeightLimitView,
        noise::perlin::DoublePerlinNoiseSampler,
        noise_router::proto_noise_router::DoublePerlinNoiseBuilder,
    },
};

pub struct SurfaceTerrainBuilder {
    // Badlands stuff
    terracotta_bands: Box<[ChunkBlockState]>,
    terracotta_bands_offset_noise: DoublePerlinNoiseSampler,
    badlands_pillar_noise: DoublePerlinNoiseSampler,
    badlands_surface_noise: DoublePerlinNoiseSampler,
    badlands_pillar_roof_noise: DoublePerlinNoiseSampler,
    // Iceberg stuff
    iceberg_pillar_noise: DoublePerlinNoiseSampler,
    iceberg_pillar_roof_noise: DoublePerlinNoiseSampler,
    iceberg_surface_noise: DoublePerlinNoiseSampler,
}

impl SurfaceTerrainBuilder {
    pub fn new(
        noise_builder: &mut DoublePerlinNoiseBuilder,
        random_deriver: &RandomDeriver,
    ) -> Self {
        Self {
            terracotta_bands: Self::create_terracotta_bands(
                random_deriver.split_string("minecraft:clay_bands"),
            ),
            terracotta_bands_offset_noise: noise_builder
                .get_noise_sampler_for_id("clay_bands_offset"),
            badlands_pillar_noise: noise_builder.get_noise_sampler_for_id("badlands_pillar"),
            badlands_surface_noise: noise_builder.get_noise_sampler_for_id("badlands_surface"),
            badlands_pillar_roof_noise: noise_builder
                .get_noise_sampler_for_id("badlands_pillar_roof"),
            iceberg_pillar_noise: noise_builder.get_noise_sampler_for_id("iceberg_pillar"),
            iceberg_pillar_roof_noise: noise_builder
                .get_noise_sampler_for_id("iceberg_pillar_roof"),
            iceberg_surface_noise: noise_builder.get_noise_sampler_for_id("iceberg_surface"),
        }
    }

    const ORANGE_TERRACOTTA: ChunkBlockState = block_state!("orange_terracotta");
    const YELLOW_TERRACOTTA: ChunkBlockState = block_state!("yellow_terracotta");
    const BROWN_TERRACOTTA: ChunkBlockState = block_state!("brown_terracotta");
    const RED_TERRACOTTA: ChunkBlockState = block_state!("red_terracotta");
    const WHITE_TERRACOTTA: ChunkBlockState = block_state!("white_terracotta");
    const LIGHT_GRAY_TERRACOTTA: ChunkBlockState = block_state!("light_gray_terracotta");
    const TERRACOTTA: ChunkBlockState = block_state!("terracotta");

    fn create_terracotta_bands(mut random: RandomGenerator) -> Box<[ChunkBlockState]> {
        let mut block_states = [Self::TERRACOTTA; 192];

        let mut i = 0;
        while i < block_states.len() {
            i += random.next_bounded_i32(5) as usize + 1;
            if i >= block_states.len() {
                break;
            }
            block_states[i] = Self::ORANGE_TERRACOTTA;
            i += 1;
        }

        Self::add_terracotta_bands(&mut random, &mut block_states, 1, Self::YELLOW_TERRACOTTA);
        Self::add_terracotta_bands(&mut random, &mut block_states, 2, Self::BROWN_TERRACOTTA);
        Self::add_terracotta_bands(&mut random, &mut block_states, 1, Self::RED_TERRACOTTA);

        let band_count = random.next_inbetween_i32(9, 15);
        let mut current_band = 0;
        let mut index = 0;

        while current_band < band_count && index < block_states.len() {
            block_states[index] = Self::WHITE_TERRACOTTA;

            if index > 1 && random.next_bool() {
                block_states[index - 1] = Self::LIGHT_GRAY_TERRACOTTA;
            }

            if index + 1 < block_states.len() && random.next_bool() {
                block_states[index + 1] = Self::LIGHT_GRAY_TERRACOTTA;
            }

            index += random.next_bounded_i32(16) as usize + 4;
            current_band += 1;
        }

        Box::new(block_states)
    }

    fn add_terracotta_bands(
        random: &mut RandomGenerator,
        terracotta_bands: &mut [ChunkBlockState],
        min_band_size: i32,
        state: ChunkBlockState,
    ) {
        let band_count = random.next_inbetween_i32(6, 15);

        for _ in 0..band_count {
            let band_width = min_band_size + random.next_bounded_i32(3);
            let start_index = random.next_bounded_i32(terracotta_bands.len() as i32);

            for m in 0..band_width {
                if (start_index + m < terracotta_bands.len() as i32) && (m < band_width) {
                    terracotta_bands[(start_index + m) as usize] = state;
                } else {
                    break; // Stop if we reach the end of the array
                }
            }
        }
    }

    pub fn place_badlands_pillar(
        &self,
        chunk: &mut ProtoChunk,
        global_x: i32,
        global_z: i32,
        surface_y: i32,
        default_state: ChunkBlockState,
    ) {
        let surface_noise =
            (self
                .badlands_surface_noise
                .sample(global_x as f64, 0.0, global_z as f64)
                * 8.25)
                .abs();
        let pillar_noise =
            self.badlands_pillar_noise
                .sample(global_x as f64 * 0.2, 0.0, global_z as f64 * 0.2)
                * 15.0;

        let threshold = surface_noise.min(pillar_noise);

        if threshold > 0.0 {
            let pillar_roof_noise = (self.badlands_pillar_roof_noise.sample(
                global_x as f64 * 0.75,
                0.0,
                global_z as f64 * 0.75,
            ) * 1.5)
                .abs();

            let scaled_threshold = threshold * threshold * 2.5;
            let transformed_roof = (pillar_roof_noise * 50.0).ceil() + 24.0;
            let elevation = 64.0 + scaled_threshold.min(transformed_roof);
            let elevation_y = elevation.floor() as i32;
            if surface_y <= elevation_y {
                for y in (chunk.bottom_y() as i32..=elevation_y).rev() {
                    let pos = Vector3::new(global_x, y, global_z);
                    let block_state = chunk.get_block_state(&pos);
                    if block_state.of_block(default_state.block_id) {
                        break;
                    }

                    if block_state.of_block(WATER_BLOCK.block_id) {
                        return;
                    }
                }

                for y in (chunk.bottom_y() as i32..=elevation_y).rev() {
                    let pos = Vector3::new(global_x, y, global_z);
                    let block_state = chunk.get_block_state(&pos);
                    if !block_state.is_air() {
                        break;
                    }

                    chunk.set_block_state(&pos, default_state);
                }
            }
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub fn place_iceberg(
        &self,
        chunk: &mut ProtoChunk,
        min_y: i32,
        x: i32,
        z: i32,
        surface_y: i32,
        sea_level: i32,
        random_deriver: &RandomDeriver,
    ) {
        let d = 1.28;
        let e = f64::min(
            self.iceberg_surface_noise
                .sample(x as f64, 0.0, z as f64)
                .abs()
                * 8.25,
            self.iceberg_pillar_noise
                .sample(x as f64 * d, 0.0, z as f64 * d)
                * 15.0,
        );
        if e <= 1.8 {
            return;
        }
        let f = 1.17;
        let g = 1.5;
        let h = self
            .iceberg_pillar_roof_noise
            .sample(x as f64 * f, 0.0, z as f64 * f)
            .abs()
            * g;
        let i = f64::min(e * e * 1.2, h.ceil() * 40.0 + 14.0);

        // TODO
        // if biome.should_generate_lower_frozen_ocean_surface(mutable_pos, sea_level) {
        //     i -= 2.0;
        // }

        let (k, j) = if i > 2.0 {
            let j = sea_level as f64 - i - 7.0;
            (i + sea_level as f64, j)
        } else {
            (0.0, 0.0)
        };

        let mut random = random_deriver.split_pos(x, 0, z);
        let l = 2 + random.next_bounded_i32(4);
        let m = sea_level + 18 + random.next_bounded_i32(10);
        let mut n = 0;

        for o in (i32::max(surface_y, k as i32 + 1)..=min_y).rev() {
            let block_state = chunk.get_block_state(&Vector3::new(x, o, z));

            if !(block_state.is_air() && o < k as i32 && random.next_f64() > 0.01)
                && !(block_state == WATER_BLOCK
                    && o <= j as i32
                    && o >= sea_level
                    && j == 0.0
                    && random.next_f64() > 0.15)
            {
                continue;
            }

            if n <= l && o > m {
                chunk.set_block_state(&Vector3::new(x, o, z), block_state!("snow_block"));
                n += 1;
                continue;
            }
            chunk.set_block_state(&Vector3::new(x, o, z), block_state!("packed_ice"));
        }
    }

    pub fn get_terracotta_block(&self, pos: &Vector3<i32>) -> ChunkBlockState {
        let offset = (self
            .terracotta_bands_offset_noise
            .sample(pos.x as f64, 0.0, pos.z as f64)
            * 4.0)
            .round() as i32;
        let offset = pos.y + offset;
        self.terracotta_bands
            [(offset as usize + self.terracotta_bands.len()) % self.terracotta_bands.len()]
    }
}
