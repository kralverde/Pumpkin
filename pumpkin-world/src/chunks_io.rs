use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{ErrorKind, Read, Write},
    path::Path,
    sync::Arc,
};

use log::error;
use pumpkin_util::math::vector2::Vector2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    chunk::{ChunkReadingError, ChunkSerializingError, ChunkWritingError, FileLocksManager},
    level::LevelFolder,
};

pub enum LoadedData<R>
where
    R: Send,
{
    LoadedData(R),
    MissingData(Vector2<i32>),
}

pub trait ChunkIO<R>
where
    Self: Send + Sync,
    R: Send + Sized,
{
    fn load_chunks(
        &self,
        folder: &LevelFolder,
        chunk_coords: &[Vector2<i32>],
    ) -> Result<Vec<LoadedData<R>>, ChunkReadingError>;

    fn save_chunks(
        &self,
        folder: &LevelFolder,
        chunks_data: &[(Vector2<i32>, &R)],
    ) -> Result<(), ChunkWritingError>;
}

pub trait ChunkSerializer: Send + Sync + Sized + Default {
    type Data: Send;

    fn get_chunk_key(chunk: Vector2<i32>) -> String;

    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self, ChunkReadingError>;

    fn add_chunk_data(&mut self, chunk_data: &Self::Data) -> Result<(), ChunkSerializingError>;
    fn get_chunk_data(
        &self,
        chunk: Vector2<i32>,
    ) -> Result<LoadedData<Self::Data>, ChunkReadingError>;
}

pub struct ChunkFileManager<S: ChunkSerializer> {
    file_locks: Arc<FileLocksManager>,
    _serializer: std::marker::PhantomData<S>,
}

impl<S: ChunkSerializer> Default for ChunkFileManager<S> {
    fn default() -> Self {
        Self {
            file_locks: Arc::default(),
            _serializer: std::marker::PhantomData,
        }
    }
}

impl<S: ChunkSerializer> ChunkFileManager<S> {
    pub fn read_file(&self, path: &Path) -> Result<Vec<u8>, ChunkReadingError> {
        let file = OpenOptions::new()
            .write(false)
            .read(true)
            .create(false)
            .truncate(false)
            .open(path)
            .map_err(|err| match err.kind() {
                ErrorKind::NotFound => ChunkReadingError::ChunkNotExist,
                kind => ChunkReadingError::IoError(kind),
            })?;

        let file_bytes = file
            .bytes()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| ChunkReadingError::IoError(err.kind()))?;

        Ok(file_bytes)
    }

    pub fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<(), ChunkWritingError> {
        let mut file = OpenOptions::new()
            .write(true)
            .read(false)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        file.write_all(bytes)
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        file.flush()
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        Ok(())
    }
}

impl<R, S> ChunkIO<R> for ChunkFileManager<S>
where
    R: Send + Sized,
    for<'a> &'a R: Send,
    S: ChunkSerializer<Data = R>,
{
    fn load_chunks(
        &self,
        folder: &LevelFolder,
        chunk_coords: &[Vector2<i32>],
    ) -> Result<Vec<LoadedData<R>>, ChunkReadingError> {
        let mut regions_chunks: BTreeMap<String, Vec<Vector2<i32>>> = BTreeMap::new();

        for &at in chunk_coords {
            let key = S::get_chunk_key(at);

            regions_chunks
                .entry(key)
                .and_modify(|chunks| chunks.push(at))
                .or_insert(vec![at]);
        }

        let chunks_by_region: Vec<Result<Vec<LoadedData<R>>, _>> = regions_chunks
            .into_par_iter()
            .map(|(file_name, chunks)| {
                tokio::task::block_in_place(|| {
                    let path = folder.region_folder.join(file_name);
                    let _reader_guard = self.file_locks.get_read_guard(&path);

                    let chunk_serializer = match self.read_file(&path) {
                        Ok(file) => Ok(S::from_bytes(&file)?),
                        Err(ChunkReadingError::ChunkNotExist) => Ok(S::default()),
                        Err(err) => Err(err),
                    }?;

                    let chunks_data = chunks
                        .iter()
                        .map(|&at| chunk_serializer.get_chunk_data(at).unwrap())
                        .collect();

                    Ok(chunks_data)
                })
            })
            .collect();

        let mut final_chunks: Vec<_> = Vec::with_capacity(chunk_coords.len());
        for chunks in chunks_by_region {
            final_chunks.extend(chunks?)
        }

        Ok(final_chunks)
    }

    fn save_chunks(
        &self,
        folder: &LevelFolder,
        chunks_data: &[(Vector2<i32>, &R)],
    ) -> Result<(), ChunkWritingError> {
        let mut regions_chunks: BTreeMap<String, Vec<&R>> = BTreeMap::new();

        for &(at, chunk) in chunks_data {
            let key = S::get_chunk_key(at);

            regions_chunks
                .entry(key)
                .and_modify(|chunks| chunks.push(chunk))
                .or_insert(vec![chunk]);
        }

        regions_chunks
            .into_par_iter()
            .try_for_each(|(file_name, chunks)| {
                tokio::task::block_in_place(|| {
                    let path = folder.region_folder.join(file_name);
                    let _write_guard = self.file_locks.get_write_guard(&path);

                    let mut chunk_serializer = match self.read_file(&path) {
                        Ok(file) => Ok(S::from_bytes(&file).map_err(|err| {
                            error!("Error parsign the data before write: {:?}", err);
                            ChunkWritingError::IoError(std::io::ErrorKind::Other)
                        })?),
                        Err(ChunkReadingError::ChunkNotExist) => Ok(S::default()),
                        Err(ChunkReadingError::IoError(err)) => {
                            error!("Error reading the data before write: {}", err);
                            return Err(ChunkWritingError::IoError(err));
                        }
                        Err(err) => {
                            error!("Error reading the data before write: {:?}", err);
                            return Err(ChunkWritingError::IoError(std::io::ErrorKind::Other));
                        }
                    }?;

                    chunks
                        .iter()
                        .for_each(|&chunk| chunk_serializer.add_chunk_data(chunk).unwrap());

                    self.write_file(&path, chunk_serializer.to_bytes().as_slice())?;

                    Ok(())
                })?;

                Ok(())
            })?;

        Ok(())
    }
}
