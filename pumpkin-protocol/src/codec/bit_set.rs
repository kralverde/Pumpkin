use std::num::NonZeroUsize;

use bytes::BufMut;
use serde::{Serialize, Serializer};

use crate::ser::ByteBufMut;
use crate::ser::NetworkRead;
use crate::ser::ReadingError;

use super::{Codec, var_int::VarInt};

pub struct BitSet(pub Box<[i64]>);

impl Codec<BitSet> for BitSet {
    /// The maximum size of the BitSet is `remaining / 8`.
    const MAX_SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(usize::MAX) };

    fn written_size(&self) -> usize {
        todo!()
    }

    fn encode(&self, write: &mut impl BufMut) {
        write.put_var_int(&VarInt::from(self.0.len()));
        for b in &self.0 {
            write.put_i64(*b);
        }
    }

    fn decode(read: &mut impl NetworkRead) -> Result<Self, ReadingError> {
        // read length
        let length = read.get_var_int()?;
        let mut array: Vec<i64> = Vec::with_capacity(length.0 as usize);
        for _ in 0..length.0 {
            let long = read.get_i64_be()?;
            array.push(long);
        }
        Ok(BitSet(array.into_boxed_slice()))
    }
}

impl Serialize for BitSet {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        todo!()
    }
}
