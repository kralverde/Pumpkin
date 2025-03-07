use async_trait::async_trait;
use bytes::*;
use flate2::read::{GzDecoder, GzEncoder, ZlibDecoder, ZlibEncoder};
use indexmap::IndexMap;
use pumpkin_config::ADVANCED_CONFIG;
use pumpkin_data::chunk::ChunkStatus;
use pumpkin_nbt::serializer::to_bytes;
use pumpkin_util::math::ceil_log2;
use pumpkin_util::math::vector2::Vector2;
use std::{
    collections::HashSet,
    io::{ErrorKind, Read, Write},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncWrite, AsyncWriteExt},
    sync::RwLock,
};

use crate::{
    block::registry::STATE_ID_TO_REGISTRY_ID,
    chunk::{
        ChunkData, ChunkReadingError, ChunkSerializingError, ChunkWritingError, CompressionError,
        io::{ChunkSerializer, LoadedData},
    },
    level::SyncChunk,
};

use super::{ChunkNbt, ChunkSection, ChunkSectionBlockStates, PaletteEntry};

/// The side size of a region in chunks (one region is 32x32 chunks)
pub const REGION_SIZE: usize = 32;

/// The number of bits that identify two chunks in the same region
pub const SUBREGION_BITS: u8 = pumpkin_util::math::ceil_log2(REGION_SIZE as u32);

pub const SUBREGION_AND: i32 = i32::pow(2, SUBREGION_BITS as u32) - 1;

/// The number of chunks in a region
pub const CHUNK_COUNT: usize = REGION_SIZE * REGION_SIZE;

/// The number of bytes in a sector (4 KiB)
pub const SECTOR_BYTES: usize = 4096;

// 1.21.4
const WORLD_DATA_VERSION: i32 = 4189;

pub const fn get_region_coords(at: &Vector2<i32>) -> (i32, i32) {
    // Divide by 32 for the region coordinates
    (at.x >> SUBREGION_BITS, at.z >> SUBREGION_BITS)
}

pub const fn get_chunk_index(pos: &Vector2<i32>) -> usize {
    let local_x = pos.x & SUBREGION_AND;
    let local_z = pos.z & SUBREGION_AND;
    let index = (local_z << SUBREGION_BITS) + local_x;
    index as usize
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Compression {
    /// GZip Compression
    GZip = Self::GZIP_ID,
    /// ZLib Compression
    ZLib = Self::ZLIB_ID,
    /// LZ4 Compression (since 24w04a)
    LZ4 = Self::LZ4_ID,
    /// Custom compression algorithm (since 24w05a)
    Custom = Self::CUSTOM_ID,
}

pub enum CompressionRead<R: Read> {
    GZip(GzDecoder<R>),
    ZLib(ZlibDecoder<R>),
    LZ4(lz4::Decoder<R>),
}

impl<R: Read> Read for CompressionRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::GZip(gzip) => gzip.read(buf),
            Self::ZLib(zlib) => zlib.read(buf),
            Self::LZ4(lz4) => lz4.read(buf),
        }
    }
}

impl Compression {
    const GZIP_ID: u8 = 1;
    const ZLIB_ID: u8 = 2;
    const NO_COMPRESSION_ID: u8 = 3;
    const LZ4_ID: u8 = 4;
    const CUSTOM_ID: u8 = 127;

    fn decompress_data(&self, compressed_data: &[u8]) -> Result<Box<[u8]>, CompressionError> {
        match self {
            Compression::GZip => {
                let mut decoder = GzDecoder::new(compressed_data);
                let mut chunk_data = Vec::new();
                decoder
                    .read_to_end(&mut chunk_data)
                    .map_err(CompressionError::GZipError)?;
                Ok(chunk_data.into_boxed_slice())
            }
            Compression::ZLib => {
                let mut decoder = ZlibDecoder::new(compressed_data);
                let mut chunk_data = Vec::new();
                decoder
                    .read_to_end(&mut chunk_data)
                    .map_err(CompressionError::ZlibError)?;
                Ok(chunk_data.into_boxed_slice())
            }
            Compression::LZ4 => {
                let mut decoder =
                    lz4::Decoder::new(compressed_data).map_err(CompressionError::LZ4Error)?;
                let mut decompressed_data = Vec::new();
                decoder
                    .read_to_end(&mut decompressed_data)
                    .map_err(CompressionError::LZ4Error)?;
                Ok(decompressed_data.into_boxed_slice())
            }
            Compression::Custom => todo!(),
        }
    }

    fn compress_data(
        &self,
        uncompressed_data: &[u8],
        compression_level: u32,
    ) -> Result<Vec<u8>, CompressionError> {
        match self {
            Compression::GZip => {
                let mut encoder = GzEncoder::new(
                    uncompressed_data,
                    flate2::Compression::new(compression_level),
                );
                let mut chunk_data = Vec::new();
                encoder
                    .read_to_end(&mut chunk_data)
                    .map_err(CompressionError::GZipError)?;
                Ok(chunk_data)
            }
            Compression::ZLib => {
                let mut encoder = ZlibEncoder::new(
                    uncompressed_data,
                    flate2::Compression::new(compression_level),
                );
                let mut chunk_data = Vec::new();
                encoder
                    .read_to_end(&mut chunk_data)
                    .map_err(CompressionError::ZlibError)?;
                Ok(chunk_data)
            }

            Compression::LZ4 => {
                let mut compressed_data = Vec::new();
                let mut encoder = lz4::EncoderBuilder::new()
                    .level(compression_level)
                    .build(&mut compressed_data)
                    .map_err(CompressionError::LZ4Error)?;
                if let Err(err) = encoder.write_all(uncompressed_data) {
                    return Err(CompressionError::LZ4Error(err));
                }
                if let (_output, Err(err)) = encoder.finish() {
                    return Err(CompressionError::LZ4Error(err));
                }
                Ok(compressed_data)
            }
            Compression::Custom => todo!(),
        }
    }

    /// Returns Ok when a compression is found otherwise an Err
    #[allow(clippy::result_unit_err)]
    pub fn from_byte(byte: u8) -> Result<Option<Self>, ()> {
        match byte {
            Self::GZIP_ID => Ok(Some(Self::GZip)),
            Self::ZLIB_ID => Ok(Some(Self::ZLib)),
            // Uncompressed (since a version before 1.15.1)
            Self::NO_COMPRESSION_ID => Ok(None),
            Self::LZ4_ID => Ok(Some(Self::LZ4)),
            Self::CUSTOM_ID => Ok(Some(Self::Custom)),
            // Unknown format
            _ => Err(()),
        }
    }
}

impl From<pumpkin_config::chunk::Compression> for Compression {
    fn from(value: pumpkin_config::chunk::Compression) -> Self {
        // :c
        match value {
            pumpkin_config::chunk::Compression::GZip => Self::GZip,
            pumpkin_config::chunk::Compression::ZLib => Self::ZLib,
            pumpkin_config::chunk::Compression::LZ4 => Self::LZ4,
            pumpkin_config::chunk::Compression::Custom => Self::Custom,
        }
    }
}

#[derive(Default, Clone)]
pub struct AnvilChunkData {
    compression: Option<Compression>,
    // Length is always the length of this + compression byte (1) so we dont need to save a length
    compressed_data: Bytes,
}

impl AnvilChunkData {
    /// Raw size of serialized chunk
    #[inline]
    pub fn raw_write_size(&self) -> usize {
        // 4 bytes for the *length* and 1 byte for the *compression* method
        self.compressed_data.remaining() + 4 + 1
    }

    /// Size of serialized chunk with padding
    #[inline]
    fn padded_size(&self) -> usize {
        let total_size = self.raw_write_size();
        let sector_count = total_size.div_ceil(SECTOR_BYTES);
        sector_count * SECTOR_BYTES
    }

    pub fn from_bytes(bytes: Bytes) -> Result<Self, ChunkReadingError> {
        let mut bytes = bytes;
        // Minus one for the compression byte
        let length = bytes.get_u32() as usize - 1;

        let compression_method = bytes.get_u8();
        let compression = Compression::from_byte(compression_method)
            .map_err(|_| ChunkReadingError::Compression(CompressionError::UnknownCompression))?;

        Ok(AnvilChunkData {
            compression,
            // If this has padding, we need to trim it
            compressed_data: bytes.slice(..length),
        })
    }

    pub async fn write(
        &self,
        w: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<(), std::io::Error> {
        let padded_size = self.padded_size();

        w.write_u32((self.compressed_data.remaining() + 1) as u32)
            .await?;
        w.write_u8(
            self.compression
                .map_or(Compression::NO_COMPRESSION_ID, |c| c as u8),
        )
        .await?;

        w.write_all(&self.compressed_data).await?;
        for _ in 0..(padded_size - self.raw_write_size()) {
            w.write_u8(0).await?;
        }

        Ok(())
    }

    pub fn to_chunk(&self, pos: Vector2<i32>) -> Result<ChunkData, ChunkReadingError> {
        let chunk = if let Some(compression) = self.compression {
            let decompress_bytes = compression
                .decompress_data(&self.compressed_data)
                .map_err(ChunkReadingError::Compression)?;

            ChunkData::from_bytes(&decompress_bytes, pos)
        } else {
            ChunkData::from_bytes(&self.compressed_data, pos)
        }
        .map_err(ChunkReadingError::ParsingError)?;

        Ok(chunk)
    }

    pub fn from_chunk(chunk: &ChunkData) -> Result<Self, ChunkWritingError> {
        let raw_bytes = chunk_to_bytes(chunk)
            .map_err(|err| ChunkWritingError::ChunkSerializingError(err.to_string()))?;

        let compression: Compression = ADVANCED_CONFIG.chunk.compression.algorithm.clone().into();
        // We need to buffer here anyway so theres no use in making an impl Write for this
        let compressed_data = compression
            .compress_data(&raw_bytes, ADVANCED_CONFIG.chunk.compression.level)
            .map_err(ChunkWritingError::Compression)?;

        Ok(AnvilChunkData {
            compression: Some(compression),
            compressed_data: compressed_data.into(),
        })
    }
}

pub struct FastAnvilChunkFile {
    timestamp_table: [u32; CHUNK_COUNT],
    chunks_data: [Option<AnvilChunkData>; CHUNK_COUNT],
}

impl Default for FastAnvilChunkFile {
    fn default() -> Self {
        Self {
            timestamp_table: [0; CHUNK_COUNT],
            chunks_data: [const { None }; CHUNK_COUNT],
        }
    }
}

#[async_trait]
impl ChunkSerializer for FastAnvilChunkFile {
    type Data = SyncChunk;

    fn should_write(is_watched: bool) -> bool {
        !is_watched
    }

    fn get_chunk_key(chunk: &Vector2<i32>) -> String {
        let (region_x, region_z) = get_region_coords(chunk);
        format!("./r.{}.{}.mca", region_x, region_z)
    }

    async fn write(
        &self,
        write: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<(), std::io::Error> {
        // The first two sectors are reserved for the location table
        let mut current_sector: u32 = 2;
        for i in 0..CHUNK_COUNT {
            if let Some(chunk) = &self.chunks_data[i] {
                let chunk_bytes = chunk.padded_size();
                let sector_count = (chunk_bytes / SECTOR_BYTES) as u32;
                write
                    .write_u32((current_sector << 8) | sector_count)
                    .await?;
                current_sector += sector_count;
            } else {
                // If the chunk is not present, we write 0 to the location and timestamp tables
                write.write_u32(0).await?;
            };
        }

        for timestamp in self.timestamp_table {
            write.write_u32(timestamp).await?;
        }

        for chunk in self.chunks_data.iter().flatten() {
            chunk.write(write).await?;
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

        let capacity = match file.metadata().await {
            Ok(metadata) => metadata.len() as usize,
            Err(_) => 4096, // A sane default
        };

        let mut file_bytes = Vec::with_capacity(capacity);
        file.read_to_end(&mut file_bytes)
            .await
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;
        let mut raw_file_bytes: Bytes = file_bytes.into();

        if raw_file_bytes.len() < SECTOR_BYTES * 2 {
            return Err(ChunkReadingError::InvalidHeader);
        }

        let headers = raw_file_bytes.split_to(SECTOR_BYTES * 2);
        let (mut location_bytes, mut timestamp_bytes) = headers.split_at(SECTOR_BYTES);

        let mut chunk_file = FastAnvilChunkFile::default();

        for i in 0..CHUNK_COUNT {
            chunk_file.timestamp_table[i] = timestamp_bytes.get_u32();
            let location = location_bytes.get_u32();

            let sector_count = (location & 0xFF) as usize;
            let sector_offset = (location >> 8) as usize;

            // If the sector offset or count is 0, the chunk is not present (we should not parse empty chunks)
            if sector_offset == 0 || sector_count == 0 {
                continue;
            }

            // We always subtract 2 for the first two sectors for the timestamp and location tables
            // that we walked earlier
            let bytes_offset = (sector_offset - 2) * SECTOR_BYTES;
            let bytes_count = sector_count * SECTOR_BYTES;

            chunk_file.chunks_data[i] = Some(AnvilChunkData::from_bytes(
                raw_file_bytes.slice(bytes_offset..bytes_offset + bytes_count),
            )?);
        }

        Ok(chunk_file)
    }

    async fn update_chunks(&mut self, chunks_data: &[Self::Data]) -> Result<(), ChunkWritingError> {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        for chunk in chunks_data {
            let chunk = chunk.read().await;
            let index = get_chunk_index(&chunk.position);
            self.chunks_data[index] = Some(AnvilChunkData::from_chunk(&chunk)?);
            self.timestamp_table[index] = epoch;
        }

        Ok(())
    }

    async fn get_chunks(
        &self,
        chunks: &[Vector2<i32>],
        stream: tokio::sync::mpsc::Sender<LoadedData<SyncChunk, ChunkReadingError>>,
    ) {
        // Create an unbounded buffer so we don't block the rayon thread pool
        let (bridge_send, mut bridge_recv) = tokio::sync::mpsc::unbounded_channel();

        // Don't par iter here so we can prevent backpressure with the await in the async
        // runtime
        for chunk in chunks.iter().cloned() {
            let index = get_chunk_index(&chunk);
            let anvil_chunk = self.chunks_data[index].clone();

            let send = bridge_send.clone();
            rayon::spawn(move || {
                let result = if let Some(data) = anvil_chunk {
                    match data.to_chunk(chunk) {
                        Ok(chunk) => LoadedData::Loaded(Arc::new(RwLock::new(chunk))),
                        Err(err) => LoadedData::Error((chunk, err)),
                    }
                } else {
                    LoadedData::Missing(chunk)
                };

                send.send(result)
                    .expect("Failed to send anvil chunks from rayon thread");
            });
        }
        // Drop the original so streams clean-up
        drop(bridge_send);

        // We don't want to waste work, so recv unbounded from the rayon thread pool, then re-send
        // to the channel

        while let Some(data) = bridge_recv.recv().await {
            stream
                .send(data)
                .await
                .expect("Failed to send anvil chunks from bridge");
        }
    }
}

pub fn chunk_to_bytes(chunk_data: &ChunkData) -> Result<Vec<u8>, ChunkSerializingError> {
    let mut sections = Vec::new();

    for (i, blocks) in chunk_data.subchunks.array_iter().enumerate() {
        // get unique blocks
        let unique_blocks: HashSet<_> = blocks.iter().collect();

        let palette: IndexMap<_, _> = unique_blocks
            .into_iter()
            .enumerate()
            .map(|(i, block)| {
                let name = STATE_ID_TO_REGISTRY_ID.get(block).unwrap();
                (block, (name, i))
            })
            .collect();

        // Determine the number of bits needed to represent the largest index in the palette
        let block_bit_size = if palette.len() < 16 {
            4
        } else {
            ceil_log2(palette.len() as u32).max(4)
        };

        let mut section_longs = Vec::new();
        let mut current_pack_long: i64 = 0;
        let mut bits_used_in_pack: u32 = 0;

        // Empty data if the palette only contains one index https://minecraft.fandom.com/wiki/Chunk_format
        // if palette.len() > 1 {}
        // TODO: Update to write empty data. Rn or read does not handle this elegantly
        for block in blocks.iter() {
            // Push if next bit does not fit
            if bits_used_in_pack + block_bit_size as u32 > 64 {
                section_longs.push(current_pack_long);
                current_pack_long = 0;
                bits_used_in_pack = 0;
            }
            let index = palette.get(block).expect("Just added all unique").1;
            current_pack_long |= (index as i64) << bits_used_in_pack;
            bits_used_in_pack += block_bit_size as u32;

            assert!(bits_used_in_pack <= 64);

            // If the current 64-bit integer is full, push it to the section_longs and start a new one
            if bits_used_in_pack >= 64 {
                section_longs.push(current_pack_long);
                current_pack_long = 0;
                bits_used_in_pack = 0;
            }
        }

        // Push the last 64-bit integer if it contains any data
        if bits_used_in_pack > 0 {
            section_longs.push(current_pack_long);
        }

        sections.push(ChunkSection {
            y: i as i8 - 4,
            block_states: Some(ChunkSectionBlockStates {
                data: Some(section_longs.into_boxed_slice()),
                palette: palette
                    .into_iter()
                    .map(|entry| PaletteEntry {
                        name: entry.1.0.to_string(),
                        properties: {
                            /*
                            let properties = &get_block(entry.1 .0).unwrap().properties;
                            let mut map = HashMap::new();
                            for property in properties {
                                map.insert(property.name.to_string(), property.values.clone());
                            }
                            Some(map)
                            */
                            None
                        },
                    })
                    .collect(),
            }),
        });
    }

    let nbt = ChunkNbt {
        data_version: WORLD_DATA_VERSION,
        x_pos: chunk_data.position.x,
        z_pos: chunk_data.position.z,
        status: ChunkStatus::Full,
        heightmaps: chunk_data.heightmap.clone(),
        sections,
    };

    let mut result = Vec::new();
    to_bytes(&nbt, &mut result).map_err(ChunkSerializingError::ErrorSerializingChunk)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use pumpkin_util::math::vector2::Vector2;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use temp_dir::TempDir;
    use tokio::sync::RwLock;

    use crate::chunk::format::anvil::FastAnvilChunkFile;
    use crate::chunk::io::chunk_file_manager::ChunkFileManager;
    use crate::chunk::io::{ChunkIO, LoadedData};
    use crate::generation::{Seed, get_world_gen};
    use crate::level::LevelFolder;

    #[tokio::test(flavor = "multi_thread")]
    async fn not_existing() {
        let region_path = PathBuf::from("not_existing");
        let chunk_saver = ChunkFileManager::<FastAnvilChunkFile>::default();

        let mut chunks = Vec::new();
        let (send, mut recv) = tokio::sync::mpsc::channel(1);

        chunk_saver
            .fetch_chunks(
                &LevelFolder {
                    root_folder: PathBuf::from(""),
                    region_folder: region_path,
                },
                &[Vector2::new(0, 0)],
                send,
            )
            .await;

        while let Some(data) = recv.recv().await {
            chunks.push(data);
        }

        assert!(chunks.len() == 1 && matches!(chunks[0], LoadedData::Missing(_)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_writing() {
        let generator = get_world_gen(Seed(0));

        let temp_dir = TempDir::new().unwrap();
        let level_folder = LevelFolder {
            root_folder: temp_dir.path().to_path_buf(),
            region_folder: temp_dir.path().join("region"),
        };
        fs::create_dir(&level_folder.region_folder).expect("couldn't create region folder");
        let chunk_saver = ChunkFileManager::<FastAnvilChunkFile>::default();

        // Generate chunks
        let mut chunks = vec![];
        for x in -5..5 {
            for y in -5..5 {
                let position = Vector2::new(x, y);
                let chunk = generator.generate_chunk(position);
                chunks.push((position, Arc::new(RwLock::new(chunk))));
            }
        }

        for i in 0..5 {
            println!("Iteration {}", i + 1);
            chunk_saver
                .save_chunks(&level_folder, chunks.clone())
                .await
                .expect("Failed to write chunk");

            let mut read_chunks = Vec::new();
            let (send, mut recv) = tokio::sync::mpsc::channel(1);

            let chunk_pos = chunks.iter().map(|(at, _)| *at).collect::<Vec<_>>();
            let spawn = chunk_saver.fetch_chunks(&level_folder, &chunk_pos, send);
            let collect = async {
                while let Some(data) = recv.recv().await {
                    read_chunks.push(data);
                }
            };

            tokio::join!(spawn, collect);

            let read_chunks = read_chunks
                .into_iter()
                .map(|chunk| match chunk {
                    LoadedData::Loaded(chunk) => chunk,
                    LoadedData::Missing(_) => panic!("Missing chunk"),
                    LoadedData::Error((position, error)) => {
                        panic!("Error reading chunk at {:?} | Error: {:?}", position, error)
                    }
                })
                .collect::<Vec<_>>();

            for (_, chunk) in &chunks {
                let chunk = chunk.read().await;
                for read_chunk in read_chunks.iter() {
                    let read_chunk = read_chunk.read().await;
                    if read_chunk.position == chunk.position {
                        assert_eq!(chunk.subchunks, read_chunk.subchunks, "Chunks don't match");
                        break;
                    }
                }
            }
        }

        println!("Checked chunks successfully");
    }

    // TODO
    /*
    #[test]
    fn test_load_java_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let level_folder = LevelFolder {
            root_folder: temp_dir.path().to_path_buf(),
            region_folder: temp_dir.path().join("region"),
        };

        fs::create_dir(&level_folder.region_folder).unwrap();
        fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join(file!())
                .parent()
                .unwrap()
                .join("../../assets/r.0.0.mca"),
            level_folder.region_folder.join("r.0.0.mca"),
        )
        .unwrap();

        let mut actually_tested = false;
        for x in 0..(1 << 5) {
            for z in 0..(1 << 5) {
                let result = AnvilChunkFormat {}.read_chunk(&level_folder, &Vector2 { x, z });

                match result {
                    Ok(_) => actually_tested = true,
                    Err(ChunkReadingError::ParsingError(ChunkParsingError::ChunkNotGenerated)) => {}
                    Err(ChunkReadingError::ChunkNotExist) => {}
                    Err(e) => panic!("{:?}", e),
                }

                println!("=========== OK ===========");
            }
        }

        assert!(actually_tested);
    }
    */
}
