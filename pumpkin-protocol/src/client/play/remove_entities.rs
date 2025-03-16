use pumpkin_data::packet::clientbound::PLAY_REMOVE_ENTITIES;
use pumpkin_macros::packet;
use serde::Serialize;

use crate::VarInt;

#[derive(Serialize)]
#[packet(PLAY_REMOVE_ENTITIES)]
pub struct CRemoveEntities {
    entity_count: VarInt,
    entity_ids: Box<[VarInt]>,
}

impl CRemoveEntities {
    pub fn new(entity_ids: Box<[VarInt]>) -> Self {
        Self {
            entity_count: entity_ids.len().into(),
            entity_ids,
        }
    }
}
