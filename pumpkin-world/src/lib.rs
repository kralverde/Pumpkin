use generation::{
    aquifer_sampler::{FluidLevel, FluidLevelSampler},
    chunk_noise::{ChunkNoiseGenerator, LAVA_BLOCK, WATER_BLOCK},
    generation_shapes::GenerationShape,
    proto_chunk::{ProtoChunk, StandardChunkFluidLevelSampler},
};
use pumpkin_util::math::vector2::Vector2;

pub mod biome;
pub mod block;
pub mod chunk;
pub mod coordinates;
pub mod cylindrical_chunk_iterator;
pub mod dimension;
pub mod entity;
mod generation;
pub mod item;
pub mod level;
mod lock;
mod noise_router;
pub mod world_info;
pub const WORLD_HEIGHT: usize = 384;
pub const WORLD_LOWEST_Y: i16 = -64;
pub const WORLD_MAX_Y: i16 = WORLD_HEIGHT as i16 - WORLD_LOWEST_Y.abs();
pub const DIRECT_PALETTE_BITS: u32 = 15;

#[macro_export]
macro_rules! read_data_from_file {
    ($path:expr) => {
        serde_json::from_str(
            &fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join(file!())
                    .parent()
                    .unwrap()
                    .join($path),
            )
            .expect("no data file"),
        )
        .expect("failed to decode data")
    };
}

// TODO: is there a way to do in-file benches?
pub fn bench_create_and_populate_noise() {
    let mut chunk = ProtoChunk::new(Vector2::new(0, 0), 0);
    chunk.populate_noise();
}
