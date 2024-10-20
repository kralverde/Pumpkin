use std::{path::PathBuf, sync::Arc};

use dashmap::{DashMap, Entry};
use pumpkin_core::math::vector2::Vector2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tokio::sync::mpsc;

use crate::{
    chunk::{
        anvil::AnvilChunkReader, ChunkData, ChunkParsingError, ChunkReader, ChunkReadingError,
    },
    world_gen::{get_world_gen, Seed, WorldGenerator},
};

/// The `Level` module provides functionality for working with chunks within or outside a Minecraft world.
///
/// Key features include:
///
/// - **Chunk Loading:** Efficiently loads chunks from disk.
/// - **Chunk Caching:** Stores accessed chunks in memory for faster access.
/// - **Chunk Generation:** Generates new chunks on-demand using a specified `WorldGenerator`.
///
/// For more details on world generation, refer to the `WorldGenerator` module.
pub struct Level {
    save_file: Option<SaveFile>,
    loaded_chunks: Arc<DashMap<Vector2<i32>, Arc<ChunkData>>>,
    // NOTE: I dont imagine more that 65565 or whatever people viewing 1 chunk
    chunk_watchers: Arc<DashMap<Vector2<i32>, u16>>,
    chunk_reader: Arc<dyn ChunkReader>,
    world_gen: Arc<dyn WorldGenerator>,
}

#[derive(Clone)]
pub struct SaveFile {
    #[expect(dead_code)]
    root_folder: PathBuf,
    pub region_folder: PathBuf,
}

impl Level {
    pub fn from_root_folder(root_folder: PathBuf) -> Self {
        let world_gen = get_world_gen(Seed(0)); // TODO Read Seed from config.

        if root_folder.exists() {
            let region_folder = root_folder.join("region");
            assert!(
                region_folder.exists(),
                "World region folder does not exist, despite there being a root folder."
            );

            Self {
                world_gen: world_gen.into(),
                save_file: Some(SaveFile {
                    root_folder,
                    region_folder,
                }),
                chunk_reader: Arc::new(AnvilChunkReader::new()),
                loaded_chunks: Arc::new(DashMap::new()),
                chunk_watchers: Arc::new(DashMap::new()),
            }
        } else {
            log::warn!(
                "Pumpkin currently only supports Superflat World generation. Use a vanilla ./world folder to play in a normal world."
            );

            Self {
                world_gen: world_gen.into(),
                save_file: None,
                chunk_reader: Arc::new(AnvilChunkReader::new()),
                loaded_chunks: Arc::new(DashMap::new()),
                chunk_watchers: Arc::new(DashMap::new()),
            }
        }
    }

    pub fn get_block() {}

    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded_chunks.len()
    }

    /// Marks chunks as "watched" by a unique player. When no players are watching a chunk,
    /// it is removed from memory. Should only be called on chunks the player was not watching
    /// before
    pub fn mark_chunk_as_newly_watched(&self, chunks: &[Vector2<i32>]) {
        chunks.par_iter().for_each(|chunk| {
            match self.chunk_watchers.entry(*chunk) {
                Entry::Occupied(mut occupied) => {
                    let value = occupied.get_mut();
                    if let Some(new_value) = value.checked_add(1) {
                        *value = new_value;
                        //log::debug!("Watch value for {:?}: {}", chunk, value);
                    } else {
                        log::error!("Watching overflow on chunk {:?}", chunk);
                    }
                }
                Entry::Vacant(vacant) => {
                    vacant.insert(1);
                }
            }
        });
    }

    /// Marks chunks no longer "watched" by a unique player. When no players are watching a chunk,
    /// it is removed from memory. Should only be called on chunks the player was watching before
    pub fn mark_chunk_as_not_watched_and_clean(&self, chunks: &[Vector2<i32>]) {
        let dropped_chunks = {
            chunks
                .par_iter()
                .filter(|chunk| match self.chunk_watchers.entry(**chunk) {
                    Entry::Occupied(mut occupied) => {
                        let value = occupied.get_mut();
                        *value = value.saturating_sub(1);
                        if *value == 0 {
                            occupied.remove_entry();
                            true
                        } else {
                            false
                        }
                    }
                    Entry::Vacant(_) => {
                        log::error!(
                            "Marking a chunk as not watched, but was vacant! ({:?})",
                            chunk
                        );
                        false
                    }
                })
                .collect::<Vec<_>>()
        };
        let dropped_chunk_data = dropped_chunks
            .par_iter()
            .filter_map(|chunk| {
                //log::debug!("Unloading chunk {:?}", chunk);
                self.loaded_chunks.remove(chunk)
            })
            .collect();

        self.write_chunks(dropped_chunk_data);
    }

    pub fn write_chunks(&self, _chunks_to_write: Vec<(Vector2<i32>, Arc<ChunkData>)>) {
        //TODO
    }

    /// Reads/Generates many chunks in a world
    /// MUST be called from a tokio runtime thread
    ///
    /// Note: The order of the output chunks will almost never be in the same order as the order of input chunks

    pub async fn fetch_chunks(
        &self,
        chunks: &[Vector2<i32>],
        channel: mpsc::Sender<Arc<ChunkData>>,
    ) {
        chunks.iter().for_each(|at| {
            let channel = channel.clone();
            let loaded_chunks = self.loaded_chunks.clone();
            let chunk_reader = self.chunk_reader.clone();
            let save_file = self.save_file.clone();
            let world_gen = self.world_gen.clone();
            let chunk_pos = *at;

            tokio::spawn(async move {
                let maybe_chunk = {
                    loaded_chunks
                        .get(&chunk_pos)
                        .map(|entry| entry.value().clone())
                }
                .or_else(|| {
                    let chunk_data = match save_file {
                        Some(save_file) => {
                            match chunk_reader.read_chunk(&save_file, &chunk_pos) {
                                Ok(data) => Ok(Arc::new(data)),
                                Err(
                                    ChunkReadingError::ChunkNotExist
                                    | ChunkReadingError::ParsingError(
                                        ChunkParsingError::ChunkNotGenerated,
                                    ),
                                ) => {
                                    // This chunk was not generated yet.
                                    let chunk = Arc::new(world_gen.generate_chunk(chunk_pos));
                                    Ok(chunk)
                                }
                                Err(err) => Err(err),
                            }
                        }
                        None => {
                            // There is no savefile yet -> generate the chunks
                            let chunk = Arc::new(world_gen.generate_chunk(chunk_pos));
                            Ok(chunk)
                        }
                    };
                    match chunk_data {
                        Ok(data) => {
                            if let Some(data) = loaded_chunks.get(&chunk_pos) {
                                // Another thread populated in between the previous check and now
                                // We did work, but this is basically like a cache miss, not much we
                                // can do about it
                                Some(data.value().clone())
                            } else {
                                // TODO: What to do about caching
                                //self.loaded_chunks.insert(*at, data.clone());
                                Some(data)
                            }
                        }
                        Err(err) => {
                            // TODO: Panic here?
                            log::warn!("Failed to read chunk {:?}: {:?}", chunk_pos, err);
                            None
                        }
                    }
                });
                match maybe_chunk {
                    Some(chunk) => {
                        let _ = channel.send(chunk).await.inspect_err(|err| {
                            log::error!("unable to send chunk to channel: {}", err)
                        });
                    }
                    None => {
                        log::error!("Unable to send chunk {:?}!", chunk_pos);
                    }
                };
            });
        })
    }
}
