use density_function::ProtoChunkNoiseFunction;
use derive_getters::Getters;

use crate::noise_router::noise_router_ast::NoiseRouterRepr;

use super::GlobalRandomConfig;

pub mod chunk_density_function;
pub mod density_function;

#[derive(Getters, Clone)]
pub struct GlobalProtoNoiseRouter<'a> {
    barrier_noise: ProtoChunkNoiseFunction<'a>,
    fluid_level_floodedness_noise: ProtoChunkNoiseFunction<'a>,
    fluid_level_spread_noise: ProtoChunkNoiseFunction<'a>,
    lava_noise: ProtoChunkNoiseFunction<'a>,
    temperature: ProtoChunkNoiseFunction<'a>,
    vegetation: ProtoChunkNoiseFunction<'a>,
    continents: ProtoChunkNoiseFunction<'a>,
    erosion: ProtoChunkNoiseFunction<'a>,
    depth: ProtoChunkNoiseFunction<'a>,
    ridges: ProtoChunkNoiseFunction<'a>,
    initial_density_without_jaggedness: ProtoChunkNoiseFunction<'a>,
    final_density: ProtoChunkNoiseFunction<'a>,
    vein_toggle: ProtoChunkNoiseFunction<'a>,
    vein_ridged: ProtoChunkNoiseFunction<'a>,
    vein_gap: ProtoChunkNoiseFunction<'a>,
}

impl<'a> GlobalProtoNoiseRouter<'a> {
    pub fn generate(ast: &'a NoiseRouterRepr, random_config: &GlobalRandomConfig) -> Self {
        Self {
            barrier_noise: ProtoChunkNoiseFunction::generate(ast.barrier_noise(), random_config),
            fluid_level_floodedness_noise: ProtoChunkNoiseFunction::generate(
                ast.fluid_level_floodedness_noise(),
                random_config,
            ),
            fluid_level_spread_noise: ProtoChunkNoiseFunction::generate(
                ast.fluid_level_spread_noise(),
                random_config,
            ),
            lava_noise: ProtoChunkNoiseFunction::generate(ast.lava_noise(), random_config),
            temperature: ProtoChunkNoiseFunction::generate(ast.temperature(), random_config),
            vegetation: ProtoChunkNoiseFunction::generate(ast.vegetation(), random_config),
            continents: ProtoChunkNoiseFunction::generate(ast.continents(), random_config),
            erosion: ProtoChunkNoiseFunction::generate(ast.erosion(), random_config),
            depth: ProtoChunkNoiseFunction::generate(ast.depth(), random_config),
            ridges: ProtoChunkNoiseFunction::generate(ast.ridges(), random_config),
            initial_density_without_jaggedness: ProtoChunkNoiseFunction::generate(
                ast.initial_density_without_jaggedness(),
                random_config,
            ),
            final_density: ProtoChunkNoiseFunction::generate(ast.final_density(), random_config),
            vein_toggle: ProtoChunkNoiseFunction::generate(ast.vein_toggle(), random_config),
            vein_ridged: ProtoChunkNoiseFunction::generate(ast.vein_ridged(), random_config),
            vein_gap: ProtoChunkNoiseFunction::generate(ast.vein_gap(), random_config),
        }
    }
}

impl<'a> GlobalProtoNoiseRouter<'a> {
    pub fn iter_functions(&mut self) -> impl Iterator<Item = &mut ProtoChunkNoiseFunction<'a>> {
        [
            &mut self.barrier_noise,
            &mut self.fluid_level_floodedness_noise,
            &mut self.fluid_level_spread_noise,
            &mut self.lava_noise,
            &mut self.temperature,
            &mut self.vegetation,
            &mut self.continents,
            &mut self.erosion,
            &mut self.depth,
            &mut self.ridges,
            &mut self.initial_density_without_jaggedness,
            &mut self.final_density,
            &mut self.vein_toggle,
            &mut self.vein_ridged,
            &mut self.vein_gap,
        ]
        .into_iter()
    }
}
