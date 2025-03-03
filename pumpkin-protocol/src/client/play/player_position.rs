use bytes::BufMut;
use pumpkin_data::packet::clientbound::PLAY_PLAYER_POSITION;
use pumpkin_macros::packet;
use pumpkin_util::math::vector3::Vector3;

use crate::{
    ClientPacket, PositionFlag, ServerPacket, VarInt,
    ser::{ByteBufMut, NetworkRead},
};

#[packet(PLAY_PLAYER_POSITION)]
pub struct CPlayerPosition<'a> {
    pub teleport_id: VarInt,
    pub position: Vector3<f64>,
    pub delta: Vector3<f64>,
    pub yaw: f32,
    pub pitch: f32,
    pub releatives: &'a [PositionFlag],
}

impl<'a> CPlayerPosition<'a> {
    pub fn new(
        teleport_id: VarInt,
        position: Vector3<f64>,
        delta: Vector3<f64>,
        yaw: f32,
        pitch: f32,
        releatives: &'a [PositionFlag],
    ) -> Self {
        Self {
            teleport_id,
            position,
            delta,
            yaw,
            pitch,
            releatives,
        }
    }
}

impl ServerPacket for CPlayerPosition<'_> {
    fn read(read: impl NetworkRead) -> Result<Self, crate::ser::ReadingError> {
        let mut read = read;

        fn get_vec(
            read_helper: &mut impl NetworkRead,
        ) -> Result<Vector3<f64>, crate::ser::ReadingError> {
            Ok(Vector3::new(
                read_helper.get_f64_be()?,
                read_helper.get_f64_be()?,
                read_helper.get_f64_be()?,
            ))
        }

        Ok(Self {
            teleport_id: read.get_var_int()?,
            position: get_vec(&mut read)?,
            delta: get_vec(&mut read)?,
            yaw: read.get_f32_be()?,
            pitch: read.get_f32_be()?,
            releatives: &[], // TODO
        })
    }
}

impl ClientPacket for CPlayerPosition<'_> {
    fn write(&self, bytebuf: &mut impl BufMut) {
        bytebuf.put_var_int(&self.teleport_id);
        bytebuf.put_f64(self.position.x);
        bytebuf.put_f64(self.position.y);
        bytebuf.put_f64(self.position.z);
        bytebuf.put_f64(self.delta.x);
        bytebuf.put_f64(self.delta.y);
        bytebuf.put_f64(self.delta.z);
        bytebuf.put_f32(self.yaw);
        bytebuf.put_f32(self.pitch);
        // not sure about that
        bytebuf.put_i32(PositionFlag::get_bitfield(self.releatives));
    }
}
