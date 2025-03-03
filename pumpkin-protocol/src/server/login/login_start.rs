use crate::ser::{ByteBufMut, NetworkRead};
use pumpkin_data::packet::serverbound::LOGIN_HELLO;
use pumpkin_macros::packet;

use crate::{ClientPacket, ServerPacket, ser::ReadingError};

#[packet(LOGIN_HELLO)]
pub struct SLoginStart {
    pub name: String, // 16
    pub uuid: uuid::Uuid,
}

impl ClientPacket for SLoginStart {
    fn write(&self, bytebuf: &mut impl bytes::BufMut) {
        bytebuf.put_string_len(&self.name, 16);
        bytebuf.put_uuid(&self.uuid);
    }
}

impl ServerPacket for SLoginStart {
    fn read(read: impl NetworkRead) -> Result<Self, ReadingError> {
        let mut read = read;

        Ok(Self {
            name: read.get_string_bounded(16)?,
            uuid: read.get_uuid()?,
        })
    }
}
