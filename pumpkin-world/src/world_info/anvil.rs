use std::{
    fs::OpenOptions,
    io::{Read, Write},
    time::{SystemTime, UNIX_EPOCH},
};

use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Serialize};

use crate::level::LevelFolder;

use super::{LevelData, WorldInfoError, WorldInfoReader, WorldInfoWriter};

pub const LEVEL_DAT_FILE_NAME: &str = "level.dat";
pub const LEVEL_DAT_BACKUP_FILE_NAME: &str = "level.dat_old";

pub struct AnvilLevelInfo;

impl WorldInfoReader for AnvilLevelInfo {
    fn read_world_info(&self, level_folder: &LevelFolder) -> Result<LevelData, WorldInfoError> {
        let path = level_folder.root_folder.join(LEVEL_DAT_FILE_NAME);

        let world_info_file = OpenOptions::new().read(true).open(path)?;
        // try to decompress using GZip
        let mut decoder = GzDecoder::new(world_info_file);
        let mut decompressed_data = Vec::new();
        decoder.read_to_end(&mut decompressed_data)?;

        let info = fastnbt::from_bytes::<LevelDat>(&decompressed_data)
            .map_err(|e| WorldInfoError::DeserializationError(e.to_string()))?;

        // todo check version

        Ok(info.data)
    }
}

impl WorldInfoWriter for AnvilLevelInfo {
    fn write_world_info(
        &self,
        data: LevelData,
        level_folder: &LevelFolder,
    ) -> Result<(), WorldInfoError> {
        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let mut data = data.clone();
        data.last_played = since_the_epoch.as_millis() as i64;
        let level_dat = LevelDat { data };

        // convert it into nbt
        // TODO: Doesn't seem like pumpkin_nbt is working
        // TODO: fastnbt doesnt serialize bools
        let nbt = fastnbt::to_bytes(&level_dat).unwrap();

        // open file
        let path = level_folder.root_folder.join(LEVEL_DAT_FILE_NAME);
        let world_info_file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(path)?;
        // now compress using GZip, TODO: im not sure about the to_vec, but writer is not implemented for BytesMut, see https://github.com/tokio-rs/bytes/pull/478
        let mut encoder = GzEncoder::new(world_info_file, Compression::best());

        // write compressed data into file
        encoder.write_all(&nbt)?;

        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct LevelDat {
    // This tag contains all the level data.
    #[serde(rename = "Data")]
    pub data: LevelData,
}

#[cfg(test)]
mod test {
    use std::{fs, path::PathBuf};

    use crate::{level::LevelFolder, world_info::LevelData};

    use super::{AnvilLevelInfo, WorldInfoReader, WorldInfoWriter};

    #[test]
    fn test_perserve_level_dat_seed() {
        let seed = 1337;

        let mut data = LevelData::default();
        data.world_gen_settings.seed = seed;

        let level_folder = LevelFolder {
            root_folder: PathBuf::from("./tmp_Info"),
            region_folder: PathBuf::from("./tmp_Info/region"),
        };
        if fs::exists(&level_folder.root_folder).unwrap() {
            fs::remove_dir_all(&level_folder.root_folder).expect("Could not delete directory");
        }
        fs::create_dir_all(&level_folder.region_folder).expect("Could not create directory");

        AnvilLevelInfo
            .write_world_info(data, &level_folder)
            .unwrap();
        let data = AnvilLevelInfo.read_world_info(&level_folder).unwrap();

        fs::remove_dir_all(&level_folder.root_folder).expect("Could not delete directory");

        assert_eq!(data.world_gen_settings.seed, seed);
    }
}
