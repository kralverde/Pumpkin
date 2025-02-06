use density_function::ProtoChunkNoiseFunction;
use derive_getters::Getters;

use crate::noise_router::noise_router_ast::NoiseRouterRepr;

mod chunk_density_function;
mod density_function;

#[derive(Getters)]
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
    pub fn generate(ast: &'a NoiseRouterRepr, seed: u64) -> Self {
        Self {
            barrier_noise: ProtoChunkNoiseFunction::generate(ast.barrier_noise(), seed),
            fluid_level_floodedness_noise: ProtoChunkNoiseFunction::generate(
                ast.fluid_level_floodedness_noise(),
                seed,
            ),
            fluid_level_spread_noise: ProtoChunkNoiseFunction::generate(
                ast.fluid_level_spread_noise(),
                seed,
            ),
            lava_noise: ProtoChunkNoiseFunction::generate(ast.lava_noise(), seed),
            temperature: ProtoChunkNoiseFunction::generate(ast.temperature(), seed),
            vegetation: ProtoChunkNoiseFunction::generate(ast.vegetation(), seed),
            continents: ProtoChunkNoiseFunction::generate(ast.continents(), seed),
            erosion: ProtoChunkNoiseFunction::generate(ast.erosion(), seed),
            depth: ProtoChunkNoiseFunction::generate(ast.depth(), seed),
            ridges: ProtoChunkNoiseFunction::generate(ast.ridges(), seed),
            initial_density_without_jaggedness: ProtoChunkNoiseFunction::generate(
                ast.initial_density_without_jaggedness(),
                seed,
            ),
            final_density: ProtoChunkNoiseFunction::generate(ast.final_density(), seed),
            vein_toggle: ProtoChunkNoiseFunction::generate(ast.vein_toggle(), seed),
            vein_ridged: ProtoChunkNoiseFunction::generate(ast.vein_ridged(), seed),
            vein_gap: ProtoChunkNoiseFunction::generate(ast.vein_gap(), seed),
        }
    }
}
