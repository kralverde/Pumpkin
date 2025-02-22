use std::{
    collections::BTreeMap,
    error,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use log::{error, info, trace};
use pumpkin_util::math::vector2::Vector2;
use tokio::{fs::OpenOptions, io::AsyncReadExt};
use tokio::{
    io::AsyncWriteExt,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::{
    chunk::{ChunkReadingError, ChunkWritingError},
    level::LevelFolder,
};

pub enum LoadedData<D, Err: error::Error>
where
    D: Send,
{
    /// The chunk data was loaded successfully
    Loaded(D),
    /// The chunk data was not found
    Missing(Vector2<i32>),

    /// An error occurred while loading the chunk data
    Error((Vector2<i32>, Err)),
}

/// Trait to handle the IO of chunks
/// for loading and saving chunks data
/// can be implemented for different types of IO
/// or with different optimizations
///
/// The `R` type is the type of the data that will be loaded/saved
/// like ChunkData or EntityData
#[async_trait]
pub trait ChunkIO<D>
where
    Self: Send + Sync,
    D: Send + Sized,
{
    /// Load the chunks data
    async fn load_chunks(
        &self,
        folder: &LevelFolder,
        chunk_coords: &[Vector2<i32>],
    ) -> Vec<LoadedData<D, ChunkReadingError>>;

    /// Persist the chunks data
    async fn save_chunks(
        &self,
        folder: &LevelFolder,
        chunks_data: Vec<(Vector2<i32>, &D)>,
    ) -> Result<(), ChunkWritingError>;

    fn cache_count(&self) -> usize;

    fn wait_for_lock_releases(&self);
}

/// Trait to serialize and deserialize the chunk data to and from bytes.
///
/// The `Data` type is the type of the data that will be updated or serialized/deserialized
/// like ChunkData or EntityData
pub trait ChunkSerializer: Send + Sync + Sized + Default {
    type Data: Send;

    fn get_chunk_key(chunk: Vector2<i32>) -> String;

    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self, ChunkReadingError>;

    fn add_chunks_data(&mut self, chunk_data: &[&Self::Data]) -> Result<(), ChunkWritingError>;

    fn get_chunks_data(
        &self,
        chunk: &[Vector2<i32>],
    ) -> Vec<LoadedData<Self::Data, ChunkReadingError>>;
}

/// A simple implementation of the ChunkSerializer trait
/// that load and save the data from a file in the disk
/// using parallelism and a cache for the files with ongoing IO operations.
///
/// It also avoid IO operations that could produce dataraces thanks to the
/// DashMap that manages the locks for the files.
pub struct ChunkFileManager<S: ChunkSerializer> {
    // TODO: Make file reading/writing async

    // Dashmap has rw-locks on shards, but we want per-serializer
    file_locks: DashMap<PathBuf, Arc<RwLock<S>>>,
    _serializer: std::marker::PhantomData<S>,
}

impl<S: ChunkSerializer> Default for ChunkFileManager<S> {
    fn default() -> Self {
        Self {
            file_locks: DashMap::default(),
            _serializer: std::marker::PhantomData,
        }
    }
}

pub struct ChunkFileManagerLockGuard<'a, S: ChunkSerializer> {
    lock: Arc<tokio::sync::RwLock<S>>,
    parent: &'a ChunkFileManager<S>,
    key: PathBuf,
}

impl<'a, S: ChunkSerializer> ChunkFileManagerLockGuard<'a, S> {
    fn new(
        key: PathBuf,
        parent: &'a ChunkFileManager<S>,
        lock: Arc<tokio::sync::RwLock<S>>,
    ) -> Self {
        Self { key, parent, lock }
    }

    async fn write(&self) -> RwLockWriteGuard<'_, S> {
        self.lock.write().await
    }

    async fn read(&self) -> RwLockReadGuard<'_, S> {
        self.lock.read().await
    }
}

impl<S: ChunkSerializer> Drop for ChunkFileManagerLockGuard<'_, S> {
    fn drop(&mut self) {
        // If we can aquire a write lock, that means nothing else is locking this -> drop it
        let file = self
            .parent
            .file_locks
            .remove_if(&self.key, |_, value| Arc::strong_count(value) <= 1);

        if let Some((path, _)) = file {
            //TODO: ITS A TRACE
            info!("Removed lock for file: {:?}", path);
        }
    }
}

impl<S: ChunkSerializer> ChunkFileManager<S> {
    pub async fn read_file(
        &self,
        path: &Path,
    ) -> Result<ChunkFileManagerLockGuard<'_, S>, ChunkReadingError> {
        // We get the entry from the DashMap and try to insert a new lock if it doesn't exist
        // using dead-lock safe methods like `or_try_insert_with`

        async fn read_from_disk<S: ChunkSerializer>(
            path: &Path,
        ) -> Result<Arc<RwLock<S>>, ChunkReadingError> {
            let file = OpenOptions::new()
                .read(true)
                .write(false)
                .create(false)
                .truncate(false)
                .open(path)
                .await
                .map_err(|err| match err.kind() {
                    ErrorKind::NotFound => ChunkReadingError::ChunkNotExist,
                    kind => ChunkReadingError::IoError(kind),
                });

            let value = match file {
                Ok(mut file) => {
                    let mut file_bytes = Vec::new();
                    file.read_to_end(&mut file_bytes)
                        .await
                        .map_err(|err| ChunkReadingError::IoError(err.kind()))?;

                    tokio::task::block_in_place(|| S::from_bytes(&file_bytes))?
                }
                Err(ChunkReadingError::ChunkNotExist) => S::default(),
                Err(err) => return Err(err),
            };

            Ok(Arc::new(RwLock::new(value)))
        }

        let lock = if let Some(serializer) = self.file_locks.get(path) {
            trace!("Using cached chunk serializer: {:?}", path);
            serializer.value().clone()
        } else {
            match self.file_locks.entry(path.to_path_buf()) {
                Entry::Occupied(entry) => {
                    trace!("Using cached chunk serializer: {:?}", path);
                    entry.into_ref().downgrade()
                }
                Entry::Vacant(entry) => {
                    trace!("Reading file from Disk: {:?}", path);
                    entry.insert(read_from_disk(path).await?).downgrade()
                }
            }
            .value()
            .clone()
        };

        Ok(ChunkFileManagerLockGuard::new(
            path.to_path_buf(),
            self,
            lock,
        ))
    }

    pub async fn write_file(&self, path: &Path, serializer: &S) -> Result<(), ChunkWritingError> {
        trace!("Writing file to Disk: {:?}", path);

        // We use tmp files to avoid corruption of the data if the process is abruptly interrupted.
        let tmp_path = &path.with_extension("tmp");

        let mut file = OpenOptions::new()
            .read(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(tmp_path)
            .await
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        file.write_all(serializer.to_bytes().as_slice())
            .await
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        file.flush()
            .await
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        // The rename of the file works like an atomic operation ensuring
        // that the data is not corrupted before the rename is completed
        tokio::fs::rename(tmp_path, path)
            .await
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        Ok(())
    }
}

#[async_trait]
impl<D, S> ChunkIO<D> for ChunkFileManager<S>
where
    D: Send + Sized + Sync,
    S: ChunkSerializer<Data = D>,
{
    async fn load_chunks(
        &self,
        folder: &LevelFolder,
        chunk_coords: &[Vector2<i32>],
    ) -> Vec<LoadedData<D, ChunkReadingError>> {
        let mut regions_chunks: BTreeMap<String, Vec<Vector2<i32>>> = BTreeMap::new();

        for &at in chunk_coords {
            let key = S::get_chunk_key(at);

            regions_chunks
                .entry(key)
                .and_modify(|chunks| chunks.push(at))
                .or_insert(vec![at]);
        }

        let chunks_tasks = regions_chunks
            .into_iter()
            .map(async |(file_name, chunks)| {
                let path = folder.region_folder.join(file_name);

                let chunk_serializer = match self.read_file(&path).await {
                    Ok(chunk_serializer) => chunk_serializer,
                    Err(ChunkReadingError::ChunkNotExist) => {
                        unreachable!("Default Serializer must be created")
                    }
                    Err(err) => return vec![LoadedData::Error((chunks[0], err))],
                };

                // We need to block the read to avoid other threads to write/modify the data
                let chunk_guard = &chunk_serializer.read().await;
                let fetched_chunks =
                    tokio::task::block_in_place(|| chunk_guard.get_chunks_data(chunks.as_slice()));

                fetched_chunks
            })
            .collect::<Vec<_>>();

        let mut chunks_by_region = Vec::with_capacity(chunk_coords.len());
        for task in chunks_tasks {
            chunks_by_region.extend(task.await);
        }

        chunks_by_region
    }

    async fn save_chunks(
        &self,
        folder: &LevelFolder,
        chunks_data: Vec<(Vector2<i32>, &D)>,
    ) -> Result<(), ChunkWritingError> {
        let mut regions_chunks: BTreeMap<String, Vec<&D>> = BTreeMap::new();

        for (at, chunk) in &chunks_data {
            let key = S::get_chunk_key(*at);

            regions_chunks
                .entry(key)
                .and_modify(|chunks| chunks.push(chunk))
                .or_insert(vec![chunk]);
        }

        let tasks = regions_chunks
            .into_iter()
            .map(async |(file_name, chunks)| {
                let path = folder.region_folder.join(file_name);

                //TODO: Do we need to read the chunk from file to write it every time? Cant we just write to
                //offsets in the file? Especially for the anvil format.
                let chunk_serializer = match self.read_file(&path).await {
                    Ok(file) => Ok(file),
                    Err(ChunkReadingError::ChunkNotExist) => {
                        unreachable!("Must be managed by the cache")
                    }
                    Err(ChunkReadingError::IoError(err)) => {
                        error!("Error reading the data before write: {}", err);
                        Err(ChunkWritingError::IoError(err))
                    }
                    Err(err) => {
                        error!("Error reading the data before write: {:?}", err);
                        Err(ChunkWritingError::IoError(std::io::ErrorKind::Other))
                    }
                }?;

                // We need to block the read to avoid other threads to write/modify/read the data
                let mut chunk_guard = chunk_serializer.write().await;
                tokio::task::block_in_place(|| chunk_guard.add_chunks_data(chunks.as_slice()))?;

                // With the modification done, we can drop the write lock but keep the read lock
                // to avoid other threads to write/modify the data, but allow other threads to read it
                let chunk_guard = &chunk_guard.downgrade();
                self.write_file(&path, chunk_guard).await?;

                Ok(())
            })
            .collect::<Vec<_>>();

        for task in tasks {
            task.await?;
        }

        Ok(())
    }

    fn cache_count(&self) -> usize {
        self.file_locks.len()
    }

    fn wait_for_lock_releases(&self) {
        let locks: Vec<_> = self
            .file_locks
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        // Aquire a write lock on all entries to verify they are complete
        for lock in locks {
            let _lock = lock.write();
        }
    }
}
