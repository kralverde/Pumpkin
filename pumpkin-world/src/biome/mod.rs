use std::{cell::RefCell, collections::HashMap, sync::LazyLock};

use enum_dispatch::enum_dispatch;
use multi_noise::{NoiseHypercube, SearchTree, TreeLeafNode};
use pumpkin_data::chunk::Biome;
use pumpkin_util::math::vector3::Vector3;
use serde::Deserialize;

use crate::{
    dimension::Dimension, generation::noise_router::multi_noise_sampler::MultiNoiseSampler,
};
pub mod multi_noise;

#[derive(Deserialize)]
pub struct BiomeEntries {
    biomes: Vec<BiomeEntry>,
}

#[derive(Deserialize)]
pub struct BiomeEntry {
    parameters: NoiseHypercube,
    biome: Biome,
}

pub static BIOME_ENTRIES: LazyLock<SearchTree<Biome>> = LazyLock::new(|| {
    let data: HashMap<Dimension, BiomeEntries> =
        serde_json::from_str(include_str!("../../../assets/multi_noise.json"))
            .expect("Could not parse multi_noise.json.");

    // TODO: support non overworld biomes
    let overworld_data = data
        .get(&Dimension::Overworld)
        .expect("Overworld dimension not found");

    let entries: Vec<(Biome, &NoiseHypercube)> = overworld_data
        .biomes
        .iter()
        .map(|entry| (entry.biome, &entry.parameters))
        .collect();

    SearchTree::create(entries)
});

thread_local! {
    static LAST_RESULT_NODE: RefCell<Option<TreeLeafNode<Biome>>> = const {RefCell::new(None) };
}

#[enum_dispatch]
pub trait BiomeSupplier {
    fn biome(at: &Vector3<i32>, noise: &mut MultiNoiseSampler<'_>) -> Biome;
}

pub struct MultiNoiseBiomeSupplier;

// TODO: Add End supplier

impl BiomeSupplier for MultiNoiseBiomeSupplier {
    fn biome(at: &Vector3<i32>, noise: &mut MultiNoiseSampler<'_>) -> Biome {
        //panic!("{}:{}:{}", at.x, at.y, at.z);
        let point = noise.sample(at.x, at.y, at.z);
        LAST_RESULT_NODE.with_borrow_mut(|last_result| {
            BIOME_ENTRIES
                .get(&point, last_result)
                .expect("failed to get biome entry")
        })
    }
}

#[cfg(test)]
mod test {
    use pumpkin_data::chunk::Biome;
    use pumpkin_util::math::{vector2::Vector2, vector3::Vector3};
    use serde::Deserialize;

    use crate::{
        GENERATION_SETTINGS, GeneratorSetting, GlobalProtoNoiseRouter, GlobalRandomConfig,
        NOISE_ROUTER_ASTS, ProtoChunk,
        generation::{
            biome_coords,
            height_limit::HeightLimitView,
            noise_router::multi_noise_sampler::{
                MultiNoiseSampler, MultiNoiseSamplerBuilderOptions,
            },
            positions::chunk_pos,
        },
        read_data_from_file,
    };

    use super::{BiomeSupplier, MultiNoiseBiomeSupplier};

    #[test]
    fn test_biome_desert() {
        let seed = 13579;
        let random_config = GlobalRandomConfig::new(seed, false);
        let noise_rounter =
            GlobalProtoNoiseRouter::generate(&NOISE_ROUTER_ASTS.overworld, &random_config);
        let multi_noise_config = MultiNoiseSamplerBuilderOptions::new(1, 1, 1);
        let mut sampler = MultiNoiseSampler::generate(&noise_rounter, &multi_noise_config);
        let biome = MultiNoiseBiomeSupplier::biome(
            &pumpkin_util::math::vector3::Vector3 { x: -24, y: 1, z: 8 },
            &mut sampler,
        );
        assert_eq!(biome, Biome::Desert)
    }

    #[test]
    fn test_wide_area_surface() {
        #[derive(Deserialize)]
        struct BiomeData {
            x: i32,
            z: i32,
            data: Vec<u16>,
        }

        let expected_data: Vec<BiomeData> =
            read_data_from_file!("../../assets/biome_no_blend_no_beard_0.json");

        let seed = 0;
        let random_config = GlobalRandomConfig::new(seed, false);
        let noise_rounter =
            GlobalProtoNoiseRouter::generate(&NOISE_ROUTER_ASTS.overworld, &random_config);

        let surface_config = GENERATION_SETTINGS
            .get(&GeneratorSetting::Overworld)
            .unwrap();

        for data in expected_data.into_iter() {
            let chunk_pos = Vector2::new(data.x, data.z);
            let mut chunk =
                ProtoChunk::new(chunk_pos, &noise_rounter, &random_config, surface_config);
            chunk.populate_biomes();

            for x in 0..16 {
                for y in 0..chunk.height() as usize {
                    for z in 0..16 {
                        let global_block_x = chunk_pos::start_block_x(&chunk_pos) + x as i32;
                        let global_block_z = chunk_pos::start_block_z(&chunk_pos) + z as i32;
                        let global_block_y = chunk.bottom_y() as i32 + y as i32;

                        let global_biome_x = biome_coords::from_block(global_block_x);
                        let global_biome_y = biome_coords::from_block(global_block_y);
                        let global_biome_z = biome_coords::from_block(global_block_z);

                        let calculated_biome = chunk.get_biome(&Vector3::new(
                            global_biome_x,
                            global_biome_y,
                            global_biome_z,
                        ));

                        let local_biome_x = biome_coords::from_block(x);
                        let local_biome_y = biome_coords::from_block(y);
                        let local_biome_z = biome_coords::from_block(z);

                        let index = (((local_biome_y << 2) | local_biome_z) << 2) | local_biome_x;

                        let expected_biome_id = data.data[index];

                        assert_eq!(
                            expected_biome_id,
                            calculated_biome.to_id(),
                            "Expected {} ({:?}) was {} ({:?}) at {},{},{} ({},{})",
                            expected_biome_id,
                            Biome::from_id(expected_biome_id),
                            calculated_biome.to_id(),
                            calculated_biome,
                            x,
                            y,
                            z,
                            chunk_pos.x,
                            chunk_pos.z
                        );
                    }
                }
            }
        }
    }
}
