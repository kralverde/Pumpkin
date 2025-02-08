use std::sync::LazyLock;

use density_function_ast::{BinaryData, BinaryOperation, DensityFunctionRepr, WrapperType};
use noise_router_ast::NoiseRouterReprs;

use crate::generation::{
    chunk_noise_router::density_function::{ChunkNoiseFunctionRange, ProtoChunkNoiseFunction},
    GlobalRandomConfig,
};

pub mod density_function_ast;
pub mod noise_router_ast;

macro_rules! fix_final_density {
    ($router:expr) => {{
        // This is just to get the min and max value of the original final density
        let dummy_config = GlobalRandomConfig::new(0);
        let dummy_function =
            ProtoChunkNoiseFunction::generate(&$router.final_density, &dummy_config);
        let min_value = dummy_function.function_components.last().unwrap().min();
        let max_value = dummy_function.function_components.last().unwrap().max();

        $router.final_density = DensityFunctionRepr::Wrapper {
            input: Box::new(DensityFunctionRepr::Binary {
                argument1: Box::new($router.final_density),
                argument2: Box::new(DensityFunctionRepr::Beardifier),
                data: BinaryData {
                    operation: BinaryOperation::Add,
                    max_value,
                    min_value,
                },
            }),
            wrapper: WrapperType::CellCache,
        };
    }};
}

pub static NOISE_ROUTER_ASTS: LazyLock<NoiseRouterReprs> = LazyLock::new(|| {
    // JSON5 is needed because of NaN, Inf, and -Inf
    let mut reprs: NoiseRouterReprs =
        serde_json5::from_str(include_str!("../../../assets/density_function.json"))
            .expect("could not deserialize density_function.json");

    // The `final_density` function is mutated at runtime for the aquifer generator.
    fix_final_density!(reprs.overworld);
    fix_final_density!(reprs.overworld_amplified);
    fix_final_density!(reprs.overworld_large_biomes);
    fix_final_density!(reprs.nether);

    reprs
});
