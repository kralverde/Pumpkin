use std::sync::LazyLock;

use noise_router_ast::NoiseRouterReprs;

pub mod density_function_ast;
mod noise_router_ast;

pub static NOISE_ROUTER_ASTS: LazyLock<NoiseRouterReprs> = LazyLock::new(|| {
    // JSON5 is needed because of NaN, Inf, and -Inf
    serde_json5::from_str(include_str!("../../../assets/density_function.json"))
        .expect("could not deserialize density_function.json")
});
