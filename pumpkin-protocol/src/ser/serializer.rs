use std::fmt::Display;

use serde::{
    Serialize,
    ser::{self},
};

use super::{NetworkWriteExt, Write, WritingError};

pub struct Serializer<W: Write> {
    pub write: W,
}

impl<W: Write> Serializer<W> {
    pub fn new(w: W) -> Self {
        Self { write: w }
    }
}

impl ser::Error for WritingError {
    fn custom<T: Display>(msg: T) -> Self {
        Self::Message(msg.to_string())
    }
}

// General notes on the serializer:
//
// Primitives are written as-is
// Strings automatically prepend a VarInt
// Enums are written as a VarInt of the index
// Structs are ignored
// Iterables' values are written in order, but NO information (e.g. size) about the
// iterable itself is written (list sizes should be a separate field)
impl<W: Write> ser::Serializer for &mut Serializer<W> {
    type Ok = ();
    type Error = WritingError;

    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_bool(self, v: bool) -> Result<Self::Ok, Self::Error> {
        self.write.write_bool(v)
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
        self.write.write_slice(v)
    }
    fn serialize_char(self, _v: char) -> Result<Self::Ok, Self::Error> {
        unimplemented!()
    }
    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        self.write.write_f32_be(v)
    }
    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        self.write.write_f64_be(v)
    }
    fn serialize_i128(self, _v: i128) -> Result<Self::Ok, Self::Error> {
        unimplemented!()
    }
    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        self.write.write_i16_be(v)
    }
    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        self.write.write_i32_be(v)
    }
    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        self.write.write_i64_be(v)
    }
    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        self.write.write_i8_be(v)
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        unimplemented!()
    }
    fn serialize_newtype_struct<T>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        // TODO: This is super sketchy... is there a way to do it better? Can we choose what
        // serializer to use on a struct somehow from within the struct?
        if name == "TextComponent" {
            let mut buf = Vec::new();
            let mut nbt_serializer = pumpkin_nbt::serializer::Serializer::new(&mut buf, None);
            value
                .serialize(&mut nbt_serializer)
                .expect("Failed to serialize NBT for TextComponent within the network serializer");

            self.serialize_bytes(&buf)
        } else {
            value.serialize(self)
        }
    }
    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.write.write_var_int(&variant_index.into())?;
        value.serialize(self)
    }
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.write.write_bool(false)
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        // here is where all arrays/list getting written, usually we prefix the length of every length with an var int. The problem is
        // that byte arrays also getting thrown in here, and we don't want to prefix them
        Ok(self)
    }
    fn serialize_some<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Serialize,
    {
        self.write.write_bool(true)?;
        value.serialize(self)
    }
    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        self.write.write_string(v)
    }
    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        unimplemented!()
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Ok(self)
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        unimplemented!()
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        // Serialize ENUM index as varint
        self.write.write_var_int(&variant_index.into())?;
        Ok(self)
    }
    fn serialize_u128(self, _v: u128) -> Result<Self::Ok, Self::Error> {
        unimplemented!()
    }
    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        self.write.write_u16_be(v)
    }
    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        self.write.write_u32_be(v)
    }
    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        self.write.write_u64_be(v)
    }
    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        self.write.write_u8_be(v)
    }
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        unimplemented!()
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        // For ENUMs, only write enum index as varint
        self.write.write_var_int(&variant_index.into())
    }
    fn is_human_readable(&self) -> bool {
        false
    }
}

impl<W: Write> ser::SerializeSeq for &mut Serializer<W> {
    // Must match the `Ok` type of the serializer.
    type Ok = ();
    // Must match the `Error` type of the serializer.
    type Error = WritingError;

    // Serialize a single element of the sequence.
    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(&mut **self)
    }

    // Close the sequence.
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<W: Write> ser::SerializeTuple for &mut Serializer<W> {
    type Ok = ();
    type Error = WritingError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

// Same thing but for tuple structs.
impl<W: Write> ser::SerializeTupleStruct for &mut Serializer<W> {
    type Ok = ();
    type Error = WritingError;

    fn serialize_field<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        todo!()
    }

    fn end(self) -> Result<(), Self::Error> {
        todo!()
    }
}

// Tuple variants are a little different. Refer back to the
// `serialize_tuple_variant` method above:
//
//    self.write += "{";
//    variant.serialize(&mut *self)?;
//    self.write += ":[";
//
// So the `end` method in this impl is responsible for closing both the `]` and
// the `}`.
impl<W: Write> ser::SerializeTupleVariant for &mut Serializer<W> {
    type Ok = ();
    type Error = WritingError;

    fn serialize_field<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        todo!()
    }

    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

// Some `Serialize` types are not able to hold a key and value in memory at the
// same time, so `SerializeMap` implementations are required to support
// `serialize_key` and `serialize_value` individually.
//
// There is a third optional method on the `SerializeMap` trait. The
// `serialize_entry` method allows serializers to optimize for the case where
// key and value are both available simultaneously. In JSON it doesn't make a
// difference, so the default behavior for `serialize_entry` is fine.
impl<W: Write> ser::SerializeMap for &mut Serializer<W> {
    type Ok = ();
    type Error = WritingError;

    // The Serde data model allows map keys to be any serializable type. JSON
    // only allows string keys, so the implementation below will produce invalid
    // JSON if the key serializes as something other than a string.
    //
    // A real JSON serializer would need to validate that map keys are strings.
    // This can be done by using a different `Serializer` to serialize the key
    // (instead of `&mut **self`) and having that other `Serializer` only
    // implement `serialize_str` and return an error on any other data type.
    fn serialize_key<T>(&mut self, _key: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        todo!()
    }

    // It doesn't make a difference whether the colon is printed at the end of
    // `serialize_key` or at the beginning of `serialize_value`. In this case,
    // the code is a bit simpler having it here.
    fn serialize_value<T>(&mut self, _value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        todo!()
    }

    fn end(self) -> Result<(), Self::Error> {
        todo!()
    }
}

// Structs are like maps in which the keys are constrained to be compile-time
// constant strings.
impl<W: Write> ser::SerializeStruct for &mut Serializer<W> {
    type Ok = ();
    type Error = WritingError;

    fn serialize_field<T>(&mut self, _key: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(&mut **self)
    }

    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn skip_field(&mut self, key: &'static str) -> Result<(), Self::Error> {
        let _ = key;
        Ok(())
    }
}

// Similar to `SerializeTupleVariant`, here the `end` method is responsible for
// closing both of the curly braces opened by `serialize_struct_variant`.
impl<W: Write> ser::SerializeStructVariant for &mut Serializer<W> {
    type Ok = ();
    type Error = WritingError;

    fn serialize_field<T>(&mut self, _key: &'static str, _value: &T) -> Result<(), Self::Error>
    where
        T: ?Sized + Serialize,
    {
        todo!()
    }

    fn end(self) -> Result<(), Self::Error> {
        todo!()
    }
}
