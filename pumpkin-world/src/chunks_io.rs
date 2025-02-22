use std::{
    collections::BTreeMap,
    error,
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, LazyLock},
};

use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use log::{error, trace};
use pumpkin_util::math::vector2::Vector2;
use tokio::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    chunk::{ChunkReadingError, ChunkWritingError},
    level::LevelFolder,
};

pub struct LockedWriteFile<'a> {
    file: File,
    path: PathBuf,
    parent: &'a FileLockManager,
    _lock: Option<RwLockWriteGuard<'a, ()>>,
}

impl Drop for LockedWriteFile<'_> {
    fn drop(&mut self) {
        let lock_guard = self
            ._lock
            .take()
            .expect("The lock guard is only taken in the drop method");
        drop(lock_guard);

        self.parent.locks.remove_if(&self.path, |_, value| {
            // If we can aquire the lock, no one else is using it
            value.try_write().is_ok()
        });
    }
}

impl Write for LockedWriteFile<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }

    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
        self.file.write_vectored(bufs)
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.file.write_all(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

pub struct LockedReadFile<'a> {
    file: File,
    path: PathBuf,
    parent: &'a FileLockManager,
    _lock: Option<RwLockReadGuard<'a, ()>>,
}

impl Drop for LockedReadFile<'_> {
    fn drop(&mut self) {
        let lock_guard = self
            ._lock
            .take()
            .expect("The lock guard is only taken in the drop method");
        drop(lock_guard);

        self.parent.locks.remove_if(&self.path, |_, value| {
            // If we can aquire the lock, no one else is using it
            value.try_write().is_ok()
        });
    }
}

impl Read for LockedReadFile<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        self.file.read_vectored(bufs)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> std::io::Result<usize> {
        self.file.read_to_end(buf)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.file.read_exact(buf)
    }
}

pub static FILE_LOCK_MANAGER: LazyLock<FileLockManager> = LazyLock::new(|| FileLockManager {
    locks: DashMap::default(),
});

pub struct FileLockManager {
    locks: DashMap<PathBuf, RwLock<()>>,
}

impl FileLockManager {
    pub fn print_log(&self) {
        log::debug!("{} File locks remain in map", self.locks.len());
    }

    /// Saftey: `lock_read` or `lock_write` cannot be called after this method
    pub async fn await_tasks(&self) {
        for entry in self.locks.iter() {
            let write_guard = entry.write().await;
            drop(write_guard);
        }
    }

    pub async fn lock_read<'me>(
        &'me self,
        path: &Path,
    ) -> Result<LockedReadFile<'me>, std::io::Error> {
        let entry = self
            .locks
            .entry(path.to_path_buf())
            .or_insert_with(|| RwLock::new(()))
            .downgrade();

        // Unsafety: the lock is owned by self, so it has the same lifetime as self. The dashmap
        // implementation doesn't include the map lifetime when coersing raw pointers, so it is
        // elided which makes the lifetime too short to leave this function
        let lock = unsafe { std::mem::transmute::<&'_ RwLock<()>, &'me RwLock<()>>(entry.value()) };
        let lock = lock.read().await;

        let file = OpenOptions::new()
            .write(false)
            .read(true)
            .create(false)
            .truncate(false)
            .open(path)?;

        Ok(LockedReadFile {
            file,
            path: path.to_path_buf(),
            parent: self,
            _lock: Some(lock),
        })
    }

    pub async fn lock_write<'me>(
        &'me self,
        path: &Path,
        truncate: bool,
    ) -> Result<LockedWriteFile<'me>, std::io::Error> {
        let entry = self
            .locks
            .entry(path.to_path_buf())
            .or_insert_with(|| RwLock::new(()))
            .downgrade();

        // Unsafety: the lock is owned by self, so it has the same lifetime as self. The dashmap
        // implementation doesn't include the map lifetime when coersing raw pointers, so it is
        // elided which makes the lifetime too short to leave this function
        let lock = unsafe { std::mem::transmute::<&'_ RwLock<()>, &'me RwLock<()>>(entry.value()) };
        let lock = lock.write().await;

        let file = OpenOptions::new()
            .write(true)
            .read(false)
            .create(true)
            .truncate(truncate)
            .open(path)?;

        Ok(LockedWriteFile {
            file,
            path: path.to_path_buf(),
            parent: self,
            _lock: Some(lock),
        })
    }
}

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
        // TODO: We want to stream the chunks as soon as they're ready instead of collecting them first
    ) -> Box<[LoadedData<D, ChunkReadingError>]>;

    /// Persist the chunks data
    async fn save_chunks(
        &self,
        folder: &LevelFolder,
        chunks_data: Vec<(Vector2<i32>, &D)>,
    ) -> Result<(), ChunkWritingError>;

    fn print_log(&self);
}

/// Trait to serialize and deserialize the chunk data to and from bytes.
///
/// The `Data` type is the type of the data that will be updated or serialized/deserialized
/// like ChunkData or EntityData
pub trait ChunkSerializer: Send + Sync + Sized + Default {
    type Data: Send;

    // TODO: Make this async
    fn read_from_file(file: LockedReadFile) -> Result<Self, ChunkReadingError>;

    // TODO: Make this async
    fn write_to_file(&self, file: LockedWriteFile) -> Result<(), ChunkWritingError>;

    fn get_chunk_path(chunk: Vector2<i32>) -> String;

    fn update_chunks(&mut self, chunk_data: &[&Self::Data]) -> Result<(), ChunkWritingError>;

    fn read_chunks(
        &self,
        chunk: &[Vector2<i32>],
    ) -> Box<[LoadedData<Self::Data, ChunkReadingError>]>;
}

/// A simple implementation of the ChunkSerializer trait
/// that load and save the data from a file in the disk
/// using parallelism and a cache for the files with ongoing IO operations.
///
/// It also avoid IO operations that could produce dataraces thanks to the
/// DashMap that manages the locks for the files.
pub struct ChunkFileManager<S: ChunkSerializer> {
    // Serializers that have been read from file and have all known information about the chunk
    // TODO: Should this be a LRU cache? How would we determine the size?
    populated_chunk_serializers: DashMap<PathBuf, Arc<RwLock<S>>>,
    // Try to clean on the next load chunks call
    paths_to_maybe_clean: Mutex<Vec<PathBuf>>,
}

impl<S: ChunkSerializer> ChunkFileManager<S> {
    async fn get_populated_serializer(
        &self,
        key: PathBuf,
    ) -> Result<Arc<RwLock<S>>, ChunkReadingError> {
        let entry = self.populated_chunk_serializers.entry(key.clone());
        let reference = match entry {
            Entry::Vacant(entry) => {
                let file = FILE_LOCK_MANAGER.lock_read(&key).await;
                let serializer = match file {
                    Ok(file) => S::read_from_file(file),
                    Err(err) => match err.kind() {
                        // If the file does not exist, then we have no information about the chunks
                        // in the region
                        ErrorKind::NotFound => Ok(S::default()),
                        kind => Err(ChunkReadingError::IoError(kind)),
                    },
                }?;

                entry.insert(Arc::new(RwLock::new(serializer))).downgrade()
            }
            Entry::Occupied(entry) => entry.into_ref().downgrade(),
        };
        Ok(reference.value().clone())
    }

    async fn write_file(&self, path: &Path, serializer: &S) -> Result<(), ChunkWritingError> {
        trace!("Writing file to Disk: {:?}", path);

        // We use tmp files to avoid corruption of the data if the process is abruptly interrupted.
        let tmp_path = path.with_extension("tmp");

        let file = FILE_LOCK_MANAGER
            .lock_write(&tmp_path, true)
            .await
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        serializer.write_to_file(file)?;

        // The rename of the file works like an atomic operation ensuring
        // that the data is not corrupted before the rename is completed
        let _file = FILE_LOCK_MANAGER
            .lock_write(path, false)
            .await
            .map_err(|err| ChunkWritingError::IoError(err.kind()))?;

        fs::rename(tmp_path, path).map_err(|err| ChunkWritingError::IoError(err.kind()))?;
        drop(_file);

        Ok(())
    }

    fn clean_cache(&self, paths: &[PathBuf]) {
        // Remove the locks that are not being used,
        // there will be always at least one strong reference, the one in the DashMap
        paths.iter().for_each(|path| {
            let removed = self
                .populated_chunk_serializers
                .remove_if(path, |_, lock| Arc::strong_count(lock) <= 1);

            if let Some((path, _)) = removed {
                trace!("Removed lock for file: {:?}", path);
            }
        });
    }
}

impl<S: ChunkSerializer> Default for ChunkFileManager<S> {
    fn default() -> Self {
        Self {
            populated_chunk_serializers: DashMap::default(),
            paths_to_maybe_clean: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl<D, S> ChunkIO<D> for ChunkFileManager<S>
where
    D: Send + Sized,
    for<'a> &'a D: Send,
    S: ChunkSerializer<Data = D>,
{
    async fn load_chunks(
        &self,
        folder: &LevelFolder,
        chunk_coords: &[Vector2<i32>],
    ) -> Box<[LoadedData<D, ChunkReadingError>]> {
        let mut regions_chunks: BTreeMap<String, Vec<Vector2<i32>>> = BTreeMap::new();

        for &at in chunk_coords {
            let key = S::get_chunk_path(at);

            regions_chunks
                .entry(key)
                .and_modify(|chunks| chunks.push(at))
                .or_insert(vec![at]);
        }

        let paths = regions_chunks
            .keys()
            .map(|key| folder.region_folder.join(key))
            .collect::<Vec<_>>();

        let chunks_tasks = regions_chunks
            .into_iter()
            .map(async |(file_name, chunks)| {
                let path = folder.region_folder.join(file_name);

                let chunk_serializer = match self.get_populated_serializer(path).await {
                    Ok(chunk_serializer) => chunk_serializer,
                    Err(ChunkReadingError::ChunkNotExist) => {
                        unreachable!("Default Serializer must be created")
                    }
                    Err(err) => {
                        return vec![LoadedData::Error((chunks[0], err))].into_boxed_slice();
                    }
                };

                // We need to block the read to avoid other threads to write/modify the data
                let chunk_guard = chunk_serializer.read().await;
                chunk_guard.read_chunks(&chunks)
            })
            .collect::<Vec<_>>();

        let mut chunks_by_region = Vec::with_capacity(chunk_coords.len());

        // TODO: This gets chunk data synchronously, but we want concurrency!
        for task in chunks_tasks {
            chunks_by_region.extend(task.await);
        }

        // Call this after creating new strong references with arcs
        let mut paths_to_clean = self.paths_to_maybe_clean.lock().await;
        self.clean_cache(&paths_to_clean);
        paths_to_clean.clear();
        paths_to_clean.extend(paths);

        chunks_by_region.into_boxed_slice()
    }

    async fn save_chunks(
        &self,
        folder: &LevelFolder,
        chunks_data: Vec<(Vector2<i32>, &D)>,
    ) -> Result<(), ChunkWritingError> {
        let mut regions_chunks: BTreeMap<String, Vec<&D>> = BTreeMap::new();

        for (at, chunk) in &chunks_data {
            let key = S::get_chunk_path(*at);

            regions_chunks
                .entry(key)
                .and_modify(|chunks| chunks.push(chunk))
                .or_insert(vec![chunk]);
        }

        let paths = regions_chunks
            .keys()
            .map(|key| folder.region_folder.join(key))
            .collect::<Vec<_>>();

        {
            let mut cache = self.paths_to_maybe_clean.lock().await;
            cache.extend(paths);
        }

        let tasks = regions_chunks
            .into_iter()
            .map(async |(file_name, chunks)| {
                let path = folder.region_folder.join(file_name);

                //TODO: Do we need to read the chunk from file to write it every time? Cant we just write to
                //offsets in the file? Especially for the anvil format.
                let chunk_serializer = match self.get_populated_serializer(path.clone()).await {
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
                chunk_guard.update_chunks(chunks.as_slice())?;

                // With the modification done, we can drop the write lock but keep the read lock
                // to avoid other threads to write/modify the data, but allow other threads to read it
                let chunk_guard = RwLockWriteGuard::downgrade(chunk_guard);
                self.write_file(&path, &chunk_guard).await?;
                Ok(())
            })
            .collect::<Vec<_>>();

        // TODO: This writes chunk data synchronously, but we want concurrency!
        for task in tasks {
            task.await?;
        }

        Ok(())
    }

    fn print_log(&self) {
        log::debug!(
            "{} Chunk Serializers remain cached",
            self.populated_chunk_serializers.len()
        );
    }
}
