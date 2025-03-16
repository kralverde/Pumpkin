use crate::{Property, VarInt};

pub enum PlayerAction {
    AddPlayer {
        name: String,
        properties: Box<[Property]>,
    },
    InitializeChat(u8),
    /// Gamemode ?
    UpdateGameMode(VarInt),
    /// Listed ?
    UpdateListed(bool),
    UpdateLatency(u8),
    UpdateDisplayName(u8),
    UpdateListOrder,
}
