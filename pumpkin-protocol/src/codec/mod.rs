use std::num::NonZeroUsize;

use bytes::BufMut;

use crate::ser::{NetworkRead, ReadingError};

pub mod bit_set;
pub mod identifier;
pub mod slot;
pub mod var_int;
pub mod var_long;

pub trait Codec<T> {
    const MAX_SIZE: NonZeroUsize;

    fn written_size(&self) -> usize;

    fn encode(&self, write: &mut impl BufMut);

    fn decode(read: &mut impl NetworkRead) -> Result<T, ReadingError>;
}
