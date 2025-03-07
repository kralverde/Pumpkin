use async_trait::async_trait;
use pumpkin_util::math::vector2::Vector2;
use std::{
    collections::{BTreeMap, VecDeque},
    io::{ErrorKind, SeekFrom},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWrite, AsyncWriteExt},
    sync::{Mutex, RwLock},
};

use crate::{
    chunk::{
        ChunkData, ChunkReadingError, ChunkWritingError,
        format::anvil::{SECTOR_BYTES, get_region_coords},
        io::{ChunkSerializer, LoadedData},
    },
    level::SyncChunk,
};

use super::anvil::{AnvilChunkData, CHUNK_COUNT, get_chunk_index};

struct FileData {
    file: Option<File>,
    location_table: [u32; CHUNK_COUNT],
}

pub struct EfficientAnvilChunkFile {
    file_data: Mutex<FileData>,
    write_cache: RwLock<VecDeque<(Vector2<i32>, SyncChunk, AnvilChunkData)>>,
}

impl Default for EfficientAnvilChunkFile {
    fn default() -> Self {
        Self {
            file_data: Mutex::new(FileData {
                file: None,
                location_table: [0; CHUNK_COUNT],
            }),
            write_cache: RwLock::new(VecDeque::new()),
        }
    }
}

#[async_trait]
impl ChunkSerializer for EfficientAnvilChunkFile {
    type Data = SyncChunk;

    fn should_write(_is_watched: bool) -> bool {
        true
    }

    fn get_chunk_key(chunk: &Vector2<i32>) -> String {
        let (region_x, region_z) = get_region_coords(chunk);
        format!("./r.{}.{}.mca", region_x, region_z)
    }

    async fn write(
        &self,
        write: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<(), std::io::Error> {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        let mut write_cache = self.write_cache.write().await;
        let mut chunks = Vec::with_capacity(write_cache.len());
        while let Some((position, _, anvil)) = write_cache.pop_front() {
            chunks.push((position, anvil));
        }
        drop(write_cache);

        chunks.sort_unstable_by_key(|(pos, _)| get_chunk_index(pos));
        let mut file_data = self.file_data.lock().await;
        // If the chunk never existed in the file -> append to the end
        // If the chunk can fit in the same location -> put back
        // If chunk can fit in another location and the other can fit in this -> swap
        // If chunk cannot fit in any other spot and the end can fit in the current spot -> swap
        // with the end
        // Otherwise, rewrite everything

        if let Some(file) = &mut file_data.file {
            let new_pos = file.seek(SeekFrom::Start(0)).await?;
            debug_assert_eq!(new_pos, 0);

            let mut header = [0u8; SECTOR_BYTES * 2];
            file.read_exact(&mut header).await?;
            let Some((locations, timestamps)) = header.split_at_mut_checked(SECTOR_BYTES) else {
                return Err(std::io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "Failed to read file header before write",
                ));
            };

            let mut betwixt = Vec::new();
            let mut end = Vec::new();
            // How much to offset the offset in the location header past a given offset (lots of
            // offset in this comment)
            // Maps the offset to how much to offset an offset past this offset
            let mut offsets = BTreeMap::<u32, i16>::new();

            for (pos, chunk) in chunks.into_iter() {
                let index = get_chunk_index(&pos);
                let offset = index * 4;
                timestamps[offset..offset + 4].copy_from_slice(&epoch.to_be_bytes());

                let new_sector_count = chunk.raw_write_size().div_ceil(SECTOR_BYTES) as u8;

                let current_location_bytes = <[u8; 4]>::try_from(&locations[offset..offset + 4])
                    .expect("This is 4 bytes for a u32");
                let current_location = u32::from_be_bytes(current_location_bytes);

                let current_sector_count = (current_location & 0xFF) as u32;
                let current_sector_offset = (current_location >> 8) as u32;

                if current_sector_count == 0 || current_sector_offset == 0 {
                    // Append to EOF
                    log::trace!("Appending new chunk {:?} to EOF", pos);
                    end.push((index, chunk));
                } else {
                    // If our new sector count is greater than our old sector count, then we want to
                    // shift all locations that come after us in the file to the right in the
                    // header, vice versa for if its smaller

                    // Sector count is max 0xff, but we need it negative too
                    let internal_offset = new_sector_count as i16 - current_sector_count as i16;
                    if internal_offset != 0 {
                        log::trace!(
                            "Resizing chunk {:?}: was {} now {}",
                            pos,
                            current_sector_count,
                            new_sector_count
                        );

                        let mut cumulative_previous_offset = 0;
                        for (key, offset) in offsets.iter_mut() {
                            #[allow(clippy::comparison_chain)]
                            if *key < current_sector_offset {
                                // We take all previous offsets and accumulate them
                                cumulative_previous_offset += *offset;
                            } else if *key > current_sector_offset {
                                // Otherwise, we add our local offset to all offsets that come after us
                                *offset += internal_offset;
                            }
                        }

                        // Then add ours
                        offsets.insert(
                            current_sector_offset,
                            internal_offset + cumulative_previous_offset,
                        );
                        betwixt.push((current_sector_offset, current_sector_count, chunk));

                        // And update our sector count
                        let packed_location =
                            (current_sector_offset << 8) | new_sector_count as u32;
                        locations[offset..offset + 4]
                            .copy_from_slice(&packed_location.to_be_bytes());
                    }
                }
            }

            // Update locations in the header for existing chunks
            let mut last_sector = 0;
            for location in locations.chunks_mut(4) {
                let current_location_bytes =
                    <[u8; 4]>::try_from(&*location).expect("This is 4 bytes for a u32");
                let current_location = u32::from_be_bytes(current_location_bytes);

                let current_sector_count = current_location & 0xFF;
                let current_sector_offset = current_location >> 8;

                // Get the first offset-offset who's offset is smaller than ours
                if let Some(offset) = offsets.iter().rev().find_map(|(k, v)| {
                    if *k < current_sector_offset {
                        Some(v)
                    } else {
                        None
                    }
                }) {
                    let new_sector_offset = if *offset < 0 {
                        current_sector_offset - offset.unsigned_abs() as u32
                    } else {
                        current_sector_offset + *offset as u32
                    };
                    let packed_location = (new_sector_offset << 8) | current_sector_count;
                    location.copy_from_slice(&packed_location.to_be_bytes());

                    let final_sector = new_sector_offset + current_sector_count;
                    if last_sector < final_sector {
                        last_sector = final_sector;
                    }
                } else {
                    let final_sector = current_sector_offset + current_sector_count;
                    if last_sector < final_sector {
                        last_sector = final_sector;
                    }
                }
            }

            // Update locations for new chunks
            for (index, chunk) in end.iter() {
                let new_sector_count = chunk.raw_write_size().div_ceil(SECTOR_BYTES) as u8;

                let packed_location = (last_sector << 8) | new_sector_count as u32;
                let offset = *index;
                locations[offset..offset + 4].copy_from_slice(&packed_location.to_be_bytes());

                last_sector += new_sector_count as u32;
            }

            // Finally, lets write everything
            write.write_all(locations).await?;
            write.write_all(timestamps).await?;
            for (read_until, skip, chunk) in betwixt {
                let current_position = file.stream_position().await?;
                let amount_to_read = read_until as u64 * SECTOR_BYTES as u64 - current_position;
                let mut reader = file.take(amount_to_read);
                tokio::io::copy(&mut reader, write).await?;
                chunk.write(write).await?;

                file.seek(SeekFrom::Current(skip as i64)).await?;
            }
            for (_, chunk) in end {
                chunk.write(write).await?;
            }
        } else {
            // Create a new file from scratch

            let mut locations = [0u32; CHUNK_COUNT];
            let mut timestamps = [0u32; CHUNK_COUNT];

            let mut sector_offset = 2;
            for (pos, chunk) in chunks.iter() {
                let index = get_chunk_index(pos);
                let sectors = chunk.raw_write_size().div_ceil(SECTOR_BYTES);

                let current_sector = sector_offset as u32;
                let sector_count = sectors as u32;
                timestamps[index] = epoch;
                locations[index] = (current_sector << 8) | sector_count;

                sector_offset += sectors;
            }

            for location in locations {
                write.write_u32(location).await?;
            }

            for timestamp in timestamps {
                write.write_u32(timestamp).await?;
            }

            for (_, chunk) in chunks.iter() {
                chunk.write(write).await?;
            }
        }

        Ok(())
    }

    async fn read(path: &Path) -> Result<Self, ChunkReadingError> {
        let mut file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .truncate(false)
            .open(path)
            .await
            .map_err(|err| match err.kind() {
                ErrorKind::NotFound => ChunkReadingError::ChunkNotExist,
                kind => ChunkReadingError::IoError(kind),
            })?;

        let mut positions_bytes = [0u8; SECTOR_BYTES];
        file.read_exact(&mut positions_bytes)
            .await
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;
        let mut positions = [0; CHUNK_COUNT];
        positions
            .iter_mut()
            .enumerate()
            .for_each(|(index, position)| {
                // Looking forward to destructive array chunks
                let offset = index * 4;
                let bytes = <[u8; 4]>::try_from(&positions_bytes[offset..offset + 4])
                    .expect("We have an array of aligned bytes");
                *position = u32::from_be_bytes(bytes);
            });

        Ok(Self {
            file_data: Mutex::new(FileData {
                file: Some(file),
                location_table: positions,
            }),
            write_cache: RwLock::new(VecDeque::new()),
        })
    }

    async fn update_chunks(&mut self, chunks_data: &[Self::Data]) -> Result<(), ChunkWritingError> {
        let mut queue = self.write_cache.write().await;
        for chunk in chunks_data {
            let chunk_guard = chunk.read().await;
            let anvil_chunk = AnvilChunkData::from_chunk(&chunk_guard)?;
            queue.push_back((chunk_guard.position, chunk.clone(), anvil_chunk));
        }
        Ok(())
    }

    async fn get_chunks(
        &self,
        chunks: &[Vector2<i32>],
        stream: tokio::sync::mpsc::Sender<LoadedData<SyncChunk, ChunkReadingError>>,
    ) {
        let mut chunks_to_read = chunks.iter().collect::<Vec<_>>();

        let cache = self.write_cache.read().await;
        for (chunk_pos, chunk, _) in cache.iter() {
            if let Some(index) = chunks_to_read.iter().position(|pos| *pos == chunk_pos) {
                chunks_to_read.swap_remove(index);
                stream
                    .send(LoadedData::Loaded(chunk.clone()))
                    .await
                    .expect("Failed to send data from chunk write cache");
            }
        }
        drop(cache);

        // Seek the file in order
        chunks_to_read.sort_unstable_by_key(|pos| get_chunk_index(pos));

        let mut file_data = self.file_data.lock().await;
        for position in chunks_to_read {
            let index = get_chunk_index(position);
            let location = file_data.location_table[index];

            let sector_count = (location & 0xFF) as u64;
            let sector_offset = (location >> 8) as u64;

            // If the sector offset or count is 0, the chunk is not present (we should not parse empty chunks)
            if sector_offset == 0 || sector_count == 0 {
                stream
                    .send(LoadedData::Missing(*position))
                    .await
                    .expect("Failed to send missing chunk");
            } else {
                let file = file_data
                    .file
                    .as_mut()
                    .expect("We have sector data, so must have read from the file");

                let read_from_file = async move || {
                    let mut raw_data = Vec::with_capacity(sector_count as usize * SECTOR_BYTES);

                    let byte_offset = sector_offset * SECTOR_BYTES as u64;
                    let new_pos = file
                        .seek(SeekFrom::Start(byte_offset))
                        .await
                        .map_err(|err| ChunkReadingError::IoError(err.kind()))?;

                    debug_assert_eq!(new_pos, byte_offset);

                    file.take(sector_count * SECTOR_BYTES as u64)
                        .read_to_end(&mut raw_data)
                        .await
                        .map_err(|err| ChunkReadingError::IoError(err.kind()))?;

                    let chunk_data = AnvilChunkData::from_bytes(raw_data.into())?;
                    let chunk = chunk_data.to_chunk(*position)?;

                    Result::<ChunkData, ChunkReadingError>::Ok(chunk)
                };

                let mut read_from_file = read_from_file;
                let chunk_data = read_from_file().await;
                match chunk_data {
                    Ok(chunk) => {
                        stream
                            .send(LoadedData::Loaded(Arc::new(RwLock::new(chunk))))
                            .await
                            .expect("Failed to send chunk");
                    }
                    Err(err) => {
                        stream
                            .send(LoadedData::Error((*position, err)))
                            .await
                            .expect("Failed to send chunk");
                    }
                }
            }
        }
    }
}
