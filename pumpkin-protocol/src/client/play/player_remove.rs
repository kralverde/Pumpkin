use pumpkin_data::packet::clientbound::PLAY_PLAYER_INFO_REMOVE;
use pumpkin_macros::packet;
use serde::{Serialize, ser::SerializeSeq};

use crate::VarInt;

#[derive(Serialize)]
#[packet(PLAY_PLAYER_INFO_REMOVE)]
pub struct CRemovePlayerInfo {
    players_count: VarInt,
    #[serde(serialize_with = "serialize_slice_uuids")]
    players: Box<[uuid::Uuid]>,
}

impl CRemovePlayerInfo {
    pub fn new(players_count: VarInt, players: Box<[uuid::Uuid]>) -> Self {
        Self {
            players_count,
            players,
        }
    }
}

fn serialize_slice_uuids<S: serde::Serializer>(
    uuids: &[uuid::Uuid],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_seq(Some(uuids.len()))?;
    for uuid in uuids {
        seq.serialize_element(uuid.as_bytes())?;
    }
    seq.end()
}
