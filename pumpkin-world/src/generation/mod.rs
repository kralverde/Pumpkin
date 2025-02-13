#![allow(dead_code)]

pub mod aquifer_sampler;
mod blender;
pub mod chunk_noise;
pub mod generation_shapes;
mod generator;
mod generic_generator;
pub mod height_limit;
mod implementation;
pub mod noise;
pub mod noise_router;
pub mod ore_sampler;
mod positions;
pub mod proto_chunk;
mod seed;

use derive_getters::Getters;
pub use generator::WorldGenerator;
use implementation::{
    //overworld::biome::plains::PlainsGenerator,
    test::TestGenerator,
};
use num_traits::Float;
use pumpkin_util::random::{xoroshiro128::Xoroshiro, RandomDeriver, RandomImpl};
pub use seed::Seed;

use generator::GeneratorInit;

pub fn get_world_gen(seed: Seed) -> Box<dyn WorldGenerator> {
    // TODO decide which WorldGenerator to pick based on config.
    //Box::new(PlainsGenerator::new(seed))
    Box::new(TestGenerator::new(seed))
}

// FMA's mul-add has a speed up over seperate multiply and adds, and has more precision.
// We don't want to `mul_add` on non-FMA cpus because it is a lot slower due to the need for the
// increased accuracy. We don't care about the acurracy, just the speed up.
#[inline]
pub fn multiply_add<T: Float>(base: T, mul: T, add: T) -> T {
    #[cfg(target_feature = "fma")]
    {
        base.mul_add(mul, add)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        base * mul + add
    }
}

#[derive(Getters)]
pub struct GlobalRandomConfig {
    seed: u64,
    base_random_deriver: RandomDeriver,
    aquifier_random_deriver: RandomDeriver,
    ore_random_deriver: RandomDeriver,
}

impl GlobalRandomConfig {
    pub fn new(seed: u64) -> Self {
        let random_deriver = RandomDeriver::Xoroshiro(Xoroshiro::from_seed(seed).next_splitter());
        let aquifer_deriver = random_deriver
            .split_string("minecraft:aquifer")
            .next_splitter();
        let ore_deriver = random_deriver.split_string("minecraft:ore").next_splitter();
        Self {
            seed,
            base_random_deriver: random_deriver,
            aquifier_random_deriver: aquifer_deriver,
            ore_random_deriver: ore_deriver,
        }
    }
}

pub mod section_coords {
    use num_traits::PrimInt;

    #[inline]
    pub fn block_to_section<T>(coord: T) -> T
    where
        T: PrimInt,
    {
        coord >> 4
    }

    #[inline]
    pub fn section_to_block<T>(coord: T) -> T
    where
        T: PrimInt,
    {
        coord << 4
    }
}

pub mod biome_coords {
    use num_traits::PrimInt;

    #[inline]
    pub fn from_block<T>(coord: T) -> T
    where
        T: PrimInt,
    {
        coord >> 2
    }

    #[inline]
    pub fn to_block<T>(coord: T) -> T
    where
        T: PrimInt,
    {
        coord << 2
    }

    #[inline]
    pub fn from_chunk<T>(coord: T) -> T
    where
        T: PrimInt,
    {
        coord << 2
    }

    #[inline]
    pub fn to_chunk<T>(coord: T) -> T
    where
        T: PrimInt,
    {
        coord >> 2
    }
}

#[derive(PartialEq)]
pub enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}
