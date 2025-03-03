use core::str;
use std::io::Read;

use crate::{
    FixedBitSet,
    codec::{Codec, bit_set::BitSet, identifier::Identifier, var_int::VarInt, var_long::VarLong},
};
use bytes::BufMut;

mod deserializer;
use thiserror::Error;
pub mod packet;
pub mod serializer;

#[derive(Debug, Error)]
pub enum ReadingError {
    /// End-of-File
    #[error("EOF, Tried to read {0} but No bytes left to consume")]
    EOF(String),
    #[error("{0} is Incomplete")]
    Incomplete(String),
    #[error("{0} is too Large")]
    TooLarge(String),
    #[error("{0}")]
    Message(String),
}

pub trait NetworkRead {
    fn get_i8_be(&mut self) -> Result<i8, ReadingError>;
    fn get_u8_be(&mut self) -> Result<u8, ReadingError>;
    fn get_i16_be(&mut self) -> Result<i16, ReadingError>;
    fn get_u16_be(&mut self) -> Result<u16, ReadingError>;
    fn get_i32_be(&mut self) -> Result<i32, ReadingError>;
    fn get_u32_be(&mut self) -> Result<u32, ReadingError>;
    fn get_i64_be(&mut self) -> Result<i64, ReadingError>;
    fn get_u64_be(&mut self) -> Result<u64, ReadingError>;
    fn get_f32_be(&mut self) -> Result<f32, ReadingError>;
    fn get_f64_be(&mut self) -> Result<f64, ReadingError>;
    fn read_boxed_slice(&mut self, count: usize) -> Result<Box<[u8]>, ReadingError>;

    fn read_remaining_to_boxed_slice(&mut self, bound: usize) -> Result<Box<[u8]>, ReadingError>;

    fn get_bool(&mut self) -> Result<bool, ReadingError>;
    fn get_var_int(&mut self) -> Result<VarInt, ReadingError>;
    fn get_var_long(&mut self) -> Result<VarLong, ReadingError>;
    fn get_string_bounded(&mut self, bound: usize) -> Result<String, ReadingError>;
    fn get_string(&mut self) -> Result<String, ReadingError>;
    fn get_identifier(&mut self) -> Result<Identifier, ReadingError>;
    fn get_uuid(&mut self) -> Result<uuid::Uuid, ReadingError>;
    fn get_fixed_bitset(&mut self, bits: usize) -> Result<FixedBitSet, ReadingError>;

    fn get_option<G>(
        &mut self,
        parse: impl FnOnce(&mut Self) -> Result<G, ReadingError>,
    ) -> Result<Option<G>, ReadingError>;

    fn get_list<G>(
        &mut self,
        parse: impl Fn(&mut Self) -> Result<G, ReadingError>,
    ) -> Result<Vec<G>, ReadingError>;
}

/*
impl<R: NetworkRead> NetworkRead for &mut R {
    fn get_i8_be(&mut self) -> Result<i8, ReadingError> {
        (*self).get_i8_be()
    }

    fn get_u8_be(&mut self) -> Result<u8, ReadingError> {
        (*self).get_u8_be()
    }

    fn get_i16_be(&mut self) -> Result<i16, ReadingError> {
        (*self).get_i16_be()
    }

    fn get_u16_be(&mut self) -> Result<u16, ReadingError> {
        (*self).get_u16_be()
    }

    fn get_i32_be(&mut self) -> Result<i32, ReadingError> {
        (*self).get_i32_be()
    }

    fn get_u32_be(&mut self) -> Result<u32, ReadingError> {
        (*self).get_u32_be()
    }

    fn get_i64_be(&mut self) -> Result<i64, ReadingError> {
        (*self).get_i64_be()
    }

    fn get_u64_be(&mut self) -> Result<u64, ReadingError> {
        (*self).get_u64_be()
    }

    fn get_f32_be(&mut self) -> Result<f32, ReadingError> {
        (*self).get_f32_be()
    }

    fn get_f64_be(&mut self) -> Result<f64, ReadingError> {
        (*self).get_f64_be()
    }

    fn read_boxed_slice(&mut self, count: usize) -> Result<Box<[u8]>, ReadingError> {
        (*self).read_boxed_slice(count)
    }

    fn read_remaining_to_boxed_slice(&mut self, bound: usize) -> Result<Box<[u8]>, ReadingError> {
        (*self).read_remaining_to_boxed_slice(bound)
    }

    fn get_bool(&mut self) -> Result<bool, ReadingError> {
        (*self).get_bool()
    }

    fn get_var_int(&mut self) -> Result<VarInt, ReadingError> {
        (*self).get_var_int()
    }

    fn get_var_long(&mut self) -> Result<VarLong, ReadingError> {
        (*self).get_var_long()
    }

    fn get_string_bounded(&mut self, bound: usize) -> Result<String, ReadingError> {
        (*self).get_string_bounded(bound)
    }

    fn get_string(&mut self) -> Result<String, ReadingError> {
        (*self).get_string()
    }

    fn get_identifier(&mut self) -> Result<Identifier, ReadingError> {
        (*self).get_identifier()
    }

    fn get_uuid(&mut self) -> Result<uuid::Uuid, ReadingError> {
        (*self).get_uuid()
    }

    fn get_fixed_bitset(&mut self, bits: usize) -> Result<FixedBitSet, ReadingError> {
        (*self).get_fixed_bitset(bits)
    }

    fn get_option<G>(
        &mut self,
        parse: impl FnOnce(&mut Self) -> Result<G, ReadingError>,
    ) -> Result<Option<G>, ReadingError> {
        (*self).get_option(parse)
    }

    fn get_list<G>(
        &mut self,
        parse: impl Fn(&mut Self) -> Result<G, ReadingError>,
    ) -> Result<Vec<G>, ReadingError> {
        (*self).get_list(parse)
    }
}
*/

impl<R: Read> NetworkRead for R {
    //TODO: Macroize this
    fn get_i8_be(&mut self) -> Result<i8, ReadingError> {
        let mut buf = [0u8];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(i8::from_be_bytes(buf))
    }

    fn get_u8_be(&mut self) -> Result<u8, ReadingError> {
        let mut buf = [0u8];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(u8::from_be_bytes(buf))
    }

    fn get_i16_be(&mut self) -> Result<i16, ReadingError> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(i16::from_be_bytes(buf))
    }

    fn get_u16_be(&mut self) -> Result<u16, ReadingError> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(u16::from_be_bytes(buf))
    }

    fn get_i32_be(&mut self) -> Result<i32, ReadingError> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(i32::from_be_bytes(buf))
    }

    fn get_u32_be(&mut self) -> Result<u32, ReadingError> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(u32::from_be_bytes(buf))
    }

    fn get_i64_be(&mut self) -> Result<i64, ReadingError> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(i64::from_be_bytes(buf))
    }

    fn get_u64_be(&mut self) -> Result<u64, ReadingError> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(u64::from_be_bytes(buf))
    }
    fn get_f32_be(&mut self) -> Result<f32, ReadingError> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(f32::from_be_bytes(buf))
    }

    fn get_f64_be(&mut self) -> Result<f64, ReadingError> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(f64::from_be_bytes(buf))
    }

    fn read_boxed_slice(&mut self, count: usize) -> Result<Box<[u8]>, ReadingError> {
        let mut buf = vec![0u8; count];
        self.read_exact(&mut buf)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

        Ok(buf.into())
    }

    fn read_remaining_to_boxed_slice(&mut self, bound: usize) -> Result<Box<[u8]>, ReadingError> {
        let mut return_buf = Vec::new();

        // TODO: We can probably remove the temp buffer somehow
        let mut temp_buf = [0; 1024];
        loop {
            let bytes_read = self
                .read(&mut temp_buf)
                .map_err(|err| ReadingError::Incomplete(err.to_string()))?;

            if bytes_read == 0 {
                break;
            }

            if return_buf.len() + bytes_read > bound {
                return Err(ReadingError::TooLarge(
                    "Read remaining too long".to_string(),
                ));
            }

            return_buf.extend(&temp_buf[..bytes_read]);
        }
        Ok(return_buf.into_boxed_slice())
    }

    fn get_bool(&mut self) -> Result<bool, ReadingError> {
        let byte = self.get_u8_be()?;
        Ok(byte != 0)
    }

    fn get_var_int(&mut self) -> Result<VarInt, ReadingError> {
        VarInt::decode(self)
    }

    fn get_var_long(&mut self) -> Result<VarLong, ReadingError> {
        VarLong::decode(self)
    }

    fn get_string_bounded(&mut self, bound: usize) -> Result<String, ReadingError> {
        let size = self.get_var_int()?.0 as usize;
        if size > bound {
            return Err(ReadingError::TooLarge("string".to_string()));
        }

        let data = self.read_boxed_slice(size)?;
        String::from_utf8(data.into()).map_err(|e| ReadingError::Message(e.to_string()))
    }

    fn get_string(&mut self) -> Result<String, ReadingError> {
        self.get_string_bounded(i16::MAX as usize)
    }

    fn get_identifier(&mut self) -> Result<Identifier, ReadingError> {
        Identifier::decode(self)
    }

    fn get_uuid(&mut self) -> Result<uuid::Uuid, ReadingError> {
        let mut bytes = [0u8; 16];
        self.read_exact(&mut bytes)
            .map_err(|err| ReadingError::Incomplete(err.to_string()))?;
        Ok(uuid::Uuid::from_slice(&bytes).expect("Failed to parse UUID"))
    }

    fn get_fixed_bitset(&mut self, bits: usize) -> Result<FixedBitSet, ReadingError> {
        let bytes = self.read_boxed_slice(bits.div_ceil(8))?;
        Ok(bytes)
    }

    fn get_option<G>(
        &mut self,
        parse: impl FnOnce(&mut Self) -> Result<G, ReadingError>,
    ) -> Result<Option<G>, ReadingError> {
        if self.get_bool()? {
            Ok(Some(parse(self)?))
        } else {
            Ok(None)
        }
    }

    fn get_list<G>(
        &mut self,
        parse: impl Fn(&mut Self) -> Result<G, ReadingError>,
    ) -> Result<Vec<G>, ReadingError> {
        let len = self.get_var_int()?.0 as usize;
        let mut list = Vec::with_capacity(len);
        for _ in 0..len {
            list.push(parse(self)?);
        }
        Ok(list)
    }
}

pub trait ByteBufMut {
    fn put_bool(&mut self, v: bool);

    fn put_uuid(&mut self, v: &uuid::Uuid);

    fn put_string(&mut self, val: &str);

    fn put_string_len(&mut self, val: &str, max_size: usize);

    fn put_string_array(&mut self, array: &[&str]);

    fn put_bit_set(&mut self, set: &BitSet);

    /// Writes `true` if the option is Some, or `false` if None. If the option is
    /// some, then it also calls the `write` closure.
    fn put_option<G>(&mut self, val: &Option<G>, write: impl FnOnce(&mut Self, &G));

    fn put_list<G>(&mut self, list: &[G], write: impl Fn(&mut Self, &G));

    fn put_identifier(&mut self, val: &Identifier);

    fn put_var_int(&mut self, value: &VarInt);

    fn put_varint_arr(&mut self, v: &[i32]);
}

impl<T: BufMut> ByteBufMut for T {
    fn put_bool(&mut self, v: bool) {
        if v {
            self.put_u8(1);
        } else {
            self.put_u8(0);
        }
    }

    fn put_uuid(&mut self, v: &uuid::Uuid) {
        // thats the vanilla way
        let pair = v.as_u64_pair();
        self.put_u64(pair.0);
        self.put_u64(pair.1);
    }

    fn put_string(&mut self, val: &str) {
        self.put_string_len(val, i16::MAX as usize);
    }

    fn put_string_len(&mut self, val: &str, max_size: usize) {
        if val.len() > max_size {
            // Should be panic?, I mean its our fault
            panic!("String is too big");
        }
        self.put_var_int(&val.len().into());
        self.put(val.as_bytes());
    }

    fn put_string_array(&mut self, array: &[&str]) {
        for string in array {
            self.put_string(string)
        }
    }

    fn put_var_int(&mut self, var_int: &VarInt) {
        var_int.encode(self);
    }

    fn put_bit_set(&mut self, bit_set: &BitSet) {
        bit_set.encode(self);
    }

    fn put_option<G>(&mut self, val: &Option<G>, write: impl FnOnce(&mut Self, &G)) {
        self.put_bool(val.is_some());
        if let Some(v) = val {
            write(self, v)
        }
    }

    fn put_list<G>(&mut self, list: &[G], write: impl Fn(&mut Self, &G)) {
        self.put_var_int(&list.len().into());
        for v in list {
            write(self, v);
        }
    }

    fn put_varint_arr(&mut self, v: &[i32]) {
        self.put_list(v, |p, &v| p.put_var_int(&v.into()))
    }

    fn put_identifier(&mut self, val: &Identifier) {
        val.encode(self);
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use bytes::BytesMut;
    use serde::{Deserialize, Serialize};

    use crate::{
        VarInt,
        ser::{deserializer, serializer},
    };

    #[test]
    fn test_i32_reserialize() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Debug)]
        struct Foo {
            bar: i32,
        }
        let foo = Foo { bar: 69 };
        let mut bytes = BytesMut::new();
        let mut serializer = serializer::Serializer::new(&mut bytes);
        foo.serialize(&mut serializer).unwrap();

        let cursor = Cursor::new(bytes);
        let deserialized: Foo =
            Foo::deserialize(&mut deserializer::Deserializer::new(cursor)).unwrap();

        assert_eq!(foo, deserialized);
    }

    #[test]
    fn test_varint_reserialize() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Debug)]
        struct Foo {
            bar: VarInt,
        }
        let foo = Foo { bar: 69.into() };
        let mut bytes = BytesMut::new();
        let mut serializer = serializer::Serializer::new(&mut bytes);
        foo.serialize(&mut serializer).unwrap();

        let cursor = Cursor::new(bytes);
        let deserialized: Foo =
            Foo::deserialize(&mut deserializer::Deserializer::new(cursor)).unwrap();

        assert_eq!(foo, deserialized);
    }
}
