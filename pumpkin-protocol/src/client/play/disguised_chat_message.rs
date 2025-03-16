use pumpkin_data::packet::clientbound::PLAY_DISGUISED_CHAT;
use pumpkin_util::text::TextComponent;

use pumpkin_macros::packet;
use serde::Serialize;

use crate::VarInt;

#[derive(Serialize)]
#[packet(PLAY_DISGUISED_CHAT)]
pub struct CDisguisedChatMessage {
    message: TextComponent,
    chat_type: VarInt,
    sender_name: TextComponent,
    target_name: Option<TextComponent>,
}

impl CDisguisedChatMessage {
    pub fn new(
        message: TextComponent,
        chat_type: VarInt,
        sender_name: TextComponent,
        target_name: Option<TextComponent>,
    ) -> Self {
        Self {
            message,
            chat_type,
            sender_name,
            target_name,
        }
    }
}
