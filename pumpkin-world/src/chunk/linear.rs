use std::fs::{File, OpenOptions};
use std::io::{IoSlice, Read, Seek, SeekFrom, Write};
use std::mem::transmute;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use pumpkin_config::ADVANCED_CONFIG;
use zstd::zstd_safe::OutBuffer;

use crate::{chunk::ChunkWritingError, level::LevelFolder};

use super::anvil::AnvilChunkFormat;
use super::{
    ChunkData, ChunkReader, ChunkReadingError, ChunkSerializingError, ChunkWriter, CompressionError,
};

const REGION_SIZE: usize = 32; // 32x32 chunks
const CHUNK_COUNT: usize = REGION_SIZE * REGION_SIZE;

const SIGNATURE: [u8; 8] = (0xc3ff13183cca9d9a_u64).to_be_bytes(); // as described in https://gist.github.com/Aaron2550/5701519671253d4c6190bde6706f9f98
const CHUNK_HEADER_BYTES_SIZE: usize = CHUNK_COUNT * size_of::<LinearChunkHeader>();

#[derive(Default, PartialEq, Eq, Clone, Copy)]
pub enum LinearVersion {
    #[default]
    None = 0x00, // used for defaults and invalid values
    V1 = 0x01, //used by linear.py in xymb-endcrystalme/LinearRegionFileFormatTools
    V2 = 0x02, //Requires investigation about this value/version
}

struct LinearFileHeader {
    version: LinearVersion, // ( 0.. 1 Byte)
    newest_timestamp: u32,  // ( 1.. 5 Byte) newest chunk timestamp
    compression_level: u8,  // ( 5.. 6 Byte) compression level used with zlib
    chunks_count: u8,       // ( 6.. 7 Byte) number of non 0 size chunks
    chunks_bytes: u32, // ( 7..11 Byte) size of the Compressed Chunk Heades Bytes (fixed size) + Chunk Data Bytes (dynamic size)
    region_hash: u64,  // (11..19 Byte) hash of the region file (apparently not used)
}

#[derive(Default, Clone, Copy)]
#[repr(C)]
struct LinearChunkHeader {
    size: u32,
    timestamp: u32,
}

struct LinearFile {
    header: LinearFileHeader,
    chunks_headers: [LinearChunkHeader; CHUNK_COUNT],
    chunks_data: Vec<u8>,
}

#[derive(Clone, Default)]
pub struct LinearChunkFormat;

impl From<u8> for LinearVersion {
    fn from(value: u8) -> Self {
        match value {
            0x01 => LinearVersion::V1,
            0x02 => LinearVersion::V2,
            _ => LinearVersion::None,
        }
    }
}

impl LinearFileHeader {
    const FILE_HEADER_SIZE: usize = 19;

    fn check_version(&self) -> Result<(), ChunkReadingError> {
        match self.version {
            LinearVersion::None => Err(ChunkReadingError::InvalidHeader),
            _ => Ok(()),
        }
    }
    fn from_bytes(bytes: &[u8; Self::FILE_HEADER_SIZE]) -> Self {
        LinearFileHeader {
            version: bytes[0].into(),
            newest_timestamp: u32::from_be_bytes(bytes[1..5].try_into().unwrap()),
            compression_level: bytes[5],
            chunks_count: bytes[6],
            chunks_bytes: u32::from_be_bytes(bytes[7..11].try_into().unwrap()),
            region_hash: u64::from_be_bytes(bytes[11..19].try_into().unwrap()),
        }
    }

    fn to_bytes(&self) -> [u8; Self::FILE_HEADER_SIZE] {
        let mut bytes = [0u8; LinearFileHeader::FILE_HEADER_SIZE];
        bytes[0] = self.version as u8;
        bytes[1..5].copy_from_slice(&self.newest_timestamp.to_be_bytes());
        bytes[5] = self.compression_level;
        bytes[6] = self.chunks_count;
        bytes[7..11].copy_from_slice(&self.chunks_bytes.to_be_bytes());
        bytes[11..19].copy_from_slice(&self.region_hash.to_be_bytes());

        bytes
    }
}

impl LinearFile {
    fn new() -> Self {
        LinearFile {
            header: LinearFileHeader {
                version: LinearVersion::V1,
                newest_timestamp: 0,
                compression_level: ADVANCED_CONFIG.chunk.compression.compression_level as u8,
                chunks_count: 0,
                chunks_bytes: 0,
                region_hash: 0,
            },
            chunks_headers: [LinearChunkHeader::default(); CHUNK_COUNT],
            chunks_data: vec![],
        }
    }
    fn check_signature(file: &mut File) -> Result<(), ChunkReadingError> {
        let mut signature = [0; 8];

        file.seek(SeekFrom::Start(0))
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?; //seek to the start of the file
        file.read_exact(&mut signature)
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;
        if signature != SIGNATURE {
            return Err(ChunkReadingError::InvalidHeader);
        }

        file.seek(SeekFrom::End(-8))
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?; //seek to the end of the file
        file.read_exact(&mut signature)
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;
        if signature != SIGNATURE {
            return Err(ChunkReadingError::InvalidHeader);
        }

        file.rewind()
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?; //rewind the file

        Ok(())
    }
    fn load(path: &Path) -> Result<Self, ChunkReadingError> {
        let mut file =
            OpenOptions::new()
                .read(true)
                .open(path)
                .map_err(|err| match err.kind() {
                    kind => ChunkReadingError::IoError(kind),
                })?;

        Self::check_signature(&mut file)?;

        //Skip the signature
        file.seek(SeekFrom::Start(8))
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;

        // We read the header
        let mut header_bytes = [0; LinearFileHeader::FILE_HEADER_SIZE];
        file.read_exact(&mut header_bytes)
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;

        // Parse the header
        let file_header = LinearFileHeader::from_bytes(&header_bytes);
        file_header.check_version()?;

        // Uncompress the data (header + chunks)
        let mut decoded_data = zstd::decode_all(file.take(file_header.chunks_bytes as u64))
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;

        // Parse the chunk headers
        let headers_buffer: [u8; CHUNK_HEADER_BYTES_SIZE] = decoded_data
            .drain(..CHUNK_HEADER_BYTES_SIZE)
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap();

        let headers: [LinearChunkHeader; CHUNK_COUNT] =
            unsafe { std::mem::transmute(headers_buffer) };

        Ok(LinearFile {
            header: file_header,
            chunks_headers: headers,
            chunks_data: decoded_data,
        })
    }

    fn save(&mut self, path: &Path) -> Result<(), ChunkWritingError> {
        // Parse the headers to a buffer
        let headers_buffer: [u8; CHUNK_HEADER_BYTES_SIZE] =
            unsafe { transmute(self.chunks_headers) };

        // Compose the data buffer (chunk_headers + chunks_data)
        let mut data_buffer = vec![0; CHUNK_HEADER_BYTES_SIZE + self.chunks_data.len()];
        data_buffer[..CHUNK_HEADER_BYTES_SIZE].copy_from_slice(&headers_buffer);
        data_buffer[CHUNK_HEADER_BYTES_SIZE..].copy_from_slice(&self.chunks_data);

        // Compress the data buffer
        let compressed_buffer = zstd::encode_all(
            data_buffer.as_slice(),
            ADVANCED_CONFIG.chunk.compression.compression_level as i32,
        )
        .map_err(|err| ChunkWritingError::Compression(CompressionError::ZstdError(err)))?;

        // Update the header
        self.header.chunks_bytes = compressed_buffer.len() as u32;
        self.header.compression_level = ADVANCED_CONFIG.chunk.compression.compression_level as u8;

        // Write/OverWrite the data to the file
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        file.write_vectored(&[
            IoSlice::new(&SIGNATURE),
            IoSlice::new(&self.header.to_bytes()),
            IoSlice::new(&compressed_buffer),
            IoSlice::new(&SIGNATURE),
        ])
        .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        file.flush()
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        Ok(())
    }

    fn get_chunk(
        &self,
        at: &pumpkin_util::math::vector2::Vector2<i32>,
    ) -> Result<ChunkData, ChunkReadingError> {
        // We check if the chunk exists
        let chunk_index: usize = LinearChunkFormat::get_chunk_index(at);
        let last_chunk_size = self.chunks_headers[chunk_index].size as usize;
        if last_chunk_size == 0 {
            return Err(ChunkReadingError::ChunkNotExist);
        }

        // We iterate over the headers to sum the size of the chunks until the desired one
        let mut bytes: usize = 0;
        for i in 0..chunk_index {
            bytes += self.chunks_headers[i].size as usize;
        }

        ChunkData::from_bytes(&self.chunks_data[bytes..bytes + last_chunk_size], *at)
            .map_err(ChunkReadingError::ParsingError)
    }

    fn put_chunk(
        &mut self,
        chunk: &ChunkData,
        at: &pumpkin_util::math::vector2::Vector2<i32>,
    ) -> Result<(), ChunkSerializingError> {
        let chunk_index: usize = LinearChunkFormat::get_chunk_index(at);
        let chunk_raw = AnvilChunkFormat {} //We use Anvil format to serialize the chunk
            .to_bytes(chunk)?;

        let new_chunk_size = chunk_raw.len();
        let old_chunk_size = self.chunks_headers[chunk_index].size as usize;

        // We update headers
        match (old_chunk_size, new_chunk_size) {
            (0, 0) => (),
            (0, _) => self.header.chunks_count += 1,
            (_, 0) => self.header.chunks_count -= 1,
            _ => (),
        }

        self.header.newest_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        self.chunks_headers[chunk_index] = LinearChunkHeader {
            size: new_chunk_size as u32,
            timestamp: self.header.newest_timestamp,
        };

        // We calculate the start point of the chunk in the data buffer
        let mut bytes: usize = 0;
        for i in 0..chunk_index {
            bytes += self.chunks_headers[i].size as usize;
        }

        let old_total_size = self.chunks_data.len();
        let new_total_size = (old_total_size + new_chunk_size) - old_chunk_size;

        // We update the data buffer (avoiding reallocations)
        if new_chunk_size > old_chunk_size {
            self.chunks_data.resize(new_total_size, 0);
        }
        self.chunks_data.copy_within(
            bytes + old_chunk_size..old_total_size,
            bytes + new_chunk_size,
        );
        self.chunks_data[bytes..bytes + new_chunk_size].copy_from_slice(&chunk_raw);

        if new_chunk_size < old_chunk_size {
            self.chunks_data.truncate(new_total_size);
        }

        Ok(())
    }
}

impl LinearChunkFormat {
    const fn get_region_coords(at: &pumpkin_util::math::vector2::Vector2<i32>) -> (i32, i32) {
        (at.x >> 5, at.z >> 5) // Divide by 32 for the region coordinates
    }

    const fn get_chunk_index(at: &pumpkin_util::math::vector2::Vector2<i32>) -> usize {
        let (region_x, region_z) = Self::get_region_coords(at);

        // we need only the 5 last bits of the x and z coordinates
        let decode_x = (at.x ^ (region_x << 5)) as usize;
        let decode_z = (at.z ^ (region_z << 5)) as usize;

        // we calculate the index of the chunk in the region file
        decode_x + (decode_z * REGION_SIZE)
    }
}

impl ChunkReader for LinearChunkFormat {
    fn read_chunk(
        &self,
        save_file: &LevelFolder,
        at: &pumpkin_util::math::vector2::Vector2<i32>,
    ) -> Result<ChunkData, ChunkReadingError> {
        let (region_x, region_z) = LinearChunkFormat::get_region_coords(at);

        let path = save_file
            .region_folder
            .join(format!("r.{}.{}.linear", region_x, region_z));

        if !path.is_file() {
            return Err(ChunkReadingError::ChunkNotExist);
        }
        let file_data = LinearFile::load(&path)?;

        file_data.get_chunk(at)
    }
}

impl ChunkWriter for LinearChunkFormat {
    fn write_chunk(
        &self,
        chunk: &ChunkData,
        level_folder: &LevelFolder,
        at: &pumpkin_util::math::vector2::Vector2<i32>,
    ) -> Result<(), ChunkWritingError> {
        let (region_x, region_z) = LinearChunkFormat::get_region_coords(at);

        let path = level_folder
            .region_folder
            .join(format!("r.{}.{}.linear", region_x, region_z));

        let mut file_data = if !path.is_file() {
            LinearFile::new()
        } else {
            LinearFile::load(&path).map_err(|err| match err {
                ChunkReadingError::IoError(err) => ChunkWritingError::IoError(err),
                ChunkReadingError::InvalidHeader => {
                    ChunkWritingError::IoError(std::io::ErrorKind::InvalidData)
                }
                _ => ChunkWritingError::IoError(std::io::ErrorKind::Other),
            })?
        };

        file_data
            .put_chunk(chunk, at)
            .map_err(|err| ChunkWritingError::ChunkSerializingError(err.to_string()))?;

        file_data.save(&path)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pumpkin_util::math::vector2::Vector2;
    use std::fs;
    use std::path::PathBuf;

    use crate::chunk::linear::LinearChunkFormat;
    use crate::chunk::ChunkWriter;
    use crate::generation::{get_world_gen, Seed};
    use crate::{
        chunk::{ChunkReader, ChunkReadingError},
        level::LevelFolder,
    };

    #[test]
    fn not_existing() {
        let region_path = PathBuf::from("not_existing");
        let result = LinearChunkFormat.read_chunk(
            &LevelFolder {
                root_folder: PathBuf::from(""),
                region_folder: region_path,
            },
            &Vector2::new(0, 0),
        );
        assert!(matches!(result, Err(ChunkReadingError::ChunkNotExist)));
    }

    #[test]
    fn test_writing() {
        let generator = get_world_gen(Seed(0));
        let level_folder = LevelFolder {
            root_folder: PathBuf::from("./tmp"),
            region_folder: PathBuf::from("./tmp/region"),
        };
        if fs::exists(&level_folder.root_folder).unwrap() {
            fs::remove_dir_all(&level_folder.root_folder).expect("Could not delete directory");
        }

        fs::create_dir_all(&level_folder.region_folder).expect("Could not create directory");

        // Generate chunks
        let mut chunks = vec![];
        for x in -5..5 {
            for y in -5..5 {
                let position = Vector2::new(x, y);
                chunks.push((position, generator.generate_chunk(position)));
            }
        }

        for i in 0..5 {
            println!("Iteration {}", i + 1);
            for (at, chunk) in &chunks {
                LinearChunkFormat
                    .write_chunk(chunk, &level_folder, at)
                    .expect("Failed to write chunk");
            }

            let mut read_chunks = vec![];
            for (at, _chunk) in &chunks {
                read_chunks.push(
                    LinearChunkFormat
                        .read_chunk(&level_folder, at)
                        .expect("Could not read chunk"),
                );
            }

            for (at, chunk) in &chunks {
                let read_chunk = read_chunks
                    .iter()
                    .find(|chunk| chunk.position == *at)
                    .expect("Missing chunk");
                assert_eq!(chunk.subchunks, read_chunk.subchunks, "Chunks don't match");
            }
        }

        fs::remove_dir_all(&level_folder.root_folder).expect("Could not delete directory");

        println!("Checked chunks successfully");
    }
}
