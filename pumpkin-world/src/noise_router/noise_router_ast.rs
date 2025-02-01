use derive_getters::Getters;
use serde::{Deserialize, Deserializer};

use super::density_function_ast::DensityFunctionRepr;

#[derive(Deserialize, Getters)]
pub struct NoiseRouterReprs {
    overworld: NoiseRouterRepr,
    #[serde(rename(deserialize = "large_biomes"))]
    overworld_large_biomes: NoiseRouterRepr,
    #[serde(rename(deserialize = "amplified"))]
    overworld_amplified: NoiseRouterRepr,
    nether: NoiseRouterRepr,
    end: NoiseRouterRepr,
    #[serde(rename(deserialize = "floating_islands"))]
    end_islands: NoiseRouterRepr,
}

#[derive(Deserialize, Getters)]
pub struct NoiseRouterRepr {
    #[serde(rename(deserialize = "barrierNoise"))]
    barrier_noise: DensityFunctionRepr,
    #[serde(rename(deserialize = "fluidLevelFloodednessNoise"))]
    fluid_level_floodedness_noise: DensityFunctionRepr,
    #[serde(rename(deserialize = "fluidLevelSpreadNoise"))]
    fluid_level_spread_noise: DensityFunctionRepr,
    #[serde(rename(deserialize = "lavaNoise"))]
    lava_noise: DensityFunctionRepr,
    temperature: DensityFunctionRepr,
    vegetation: DensityFunctionRepr,
    continents: DensityFunctionRepr,
    erosion: DensityFunctionRepr,
    depth: DensityFunctionRepr,
    ridges: DensityFunctionRepr,
    #[serde(rename(deserialize = "initialDensityWithoutJaggedness"))]
    initial_density_without_jaggedness: DensityFunctionRepr,
    #[serde(rename(deserialize = "finalDensity"))]
    final_density: DensityFunctionRepr,
    #[serde(rename(deserialize = "veinToggle"))]
    vein_toggle: DensityFunctionRepr,
    #[serde(rename(deserialize = "veinRidged"))]
    vein_ridged: DensityFunctionRepr,
    #[serde(rename(deserialize = "veinGap"))]
    vein_gap: DensityFunctionRepr,
}
