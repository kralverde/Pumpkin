use std::io::Cursor;

use aes::cipher::{BlockDecryptMut, BlockSizeUser, KeyIvInit, generic_array::GenericArray};
use bytes::{Buf, Bytes, BytesMut};
use flate2::read::ZlibDecoder;
use thiserror::Error;

use crate::{
    MAX_PACKET_SIZE, RawPacket, RawPacketPayload, VarInt,
    codec::Codec,
    ser::{NetworkRead, ReadingError},
};

type Cipher = cfb8::Decryptor<aes::Aes128>;

/// Decoder: Client -> Server
/// Supports ZLib decoding/decompression
/// Supports Aes128 Encryption
#[derive(Default)]
pub struct PacketDecoder {
    // This will grow infinitely if we don't make a new one
    buf: BytesMut,
    cipher: Option<Cipher>,
    expect_compression: bool,
}

impl PacketDecoder {
    pub fn decode(&mut self) -> Result<Option<RawPacket<Cursor<Bytes>>>, PacketDecodeError> {
        let packet_len = match VarInt::decode(&mut &self.buf[..]) {
            Ok(len) => len,
            Err(ReadingError::Incomplete(_)) => return Ok(None),
            _ => Err(PacketDecodeError::MalformedLength)?,
        };

        let packet_len_byte_len = packet_len.written_size();
        let packet_len = packet_len.0 as usize;

        if !(0..=MAX_PACKET_SIZE).contains(&packet_len) {
            Err(PacketDecodeError::OutOfBounds)?
        }

        if self.buf.len() - packet_len_byte_len < packet_len {
            // Not enough data arrived yet.
            return Ok(None);
        }

        let total_packet_len = packet_len_byte_len + packet_len;
        let mut data = self.buf.split_to(total_packet_len).freeze();

        // TODO: How to handle this?
        // We don't want to own `data`, so create a new `BytesMut`
        /*
        let mut new_buf = BytesMut::with_capacity(self.buf.len());
        new_buf.extend(&self.buf);
        self.buf = new_buf;
        */

        // Go past the packet length VarInt
        data.advance(packet_len_byte_len);
        let mut reader = if self.expect_compression {
            let data_len =
                VarInt::decode(&mut &data[..]).map_err(|_| PacketDecodeError::TooLong)?;
            let data_len_byte_len = data_len.written_size();
            let data_len = data_len.0 as usize;

            if !(0..=MAX_PACKET_SIZE).contains(&data_len) {
                Err(PacketDecodeError::OutOfBounds)?
            }

            // Is this packet compressed?
            data.advance(data_len_byte_len);
            if data_len > 0 {
                RawPacketPayload::Decompress(ZlibDecoder::new(Cursor::new(data)))
            } else {
                debug_assert_eq!(data_len, 0);
                RawPacketPayload::Standard(Cursor::new(data))
            }
        } else {
            // no compression
            RawPacketPayload::Standard(Cursor::new(data))
        };

        let packet_id = reader
            .get_var_int()
            .map_err(|_| PacketDecodeError::DecodeID)?
            .0;

        Ok(Some(RawPacket {
            id: packet_id,
            payload: reader,
        }))
    }

    pub fn set_encryption(&mut self, key: Option<&[u8; 16]>) {
        if let Some(key) = key {
            assert!(self.cipher.is_none(), "encryption is already enabled");
            let mut cipher = Cipher::new_from_slices(key, key).expect("invalid key");
            // Don't forget to decrypt the data we already have.
            Self::decrypt_bytes(&mut cipher, &mut self.buf);
            self.cipher = Some(cipher);
        } else {
            assert!(self.cipher.is_some(), "encryption is already disabled");
            self.cipher = None;
        }
    }

    /// Sets ZLib Decompression
    pub fn set_compression(&mut self, compression: bool) {
        self.expect_compression = compression;
    }

    fn decrypt_bytes(cipher: &mut Cipher, bytes: &mut [u8]) {
        for chunk in bytes.chunks_mut(Cipher::block_size()) {
            let gen_arr = GenericArray::from_mut_slice(chunk);
            cipher.decrypt_block_mut(gen_arr);
        }
    }

    pub fn queue_bytes(&mut self, mut bytes: BytesMut) {
        if let Some(cipher) = &mut self.cipher {
            Self::decrypt_bytes(cipher, &mut bytes);
        }

        self.buf.unsplit(bytes);
    }

    pub fn queue_slice(&mut self, bytes: &[u8]) {
        let len = self.buf.len();

        self.buf.extend_from_slice(bytes);

        if let Some(cipher) = &mut self.cipher {
            let slice = &mut self.buf[len..];
            Self::decrypt_bytes(cipher, slice);
        }
    }

    pub fn take_capacity(&mut self) -> BytesMut {
        self.buf.split_off(self.buf.len())
    }

    pub fn reserve(&mut self, additional: usize) {
        self.buf.reserve(additional);
    }
}

#[derive(Error, Debug)]
pub enum PacketDecodeError {
    #[error("failed to decode packet ID")]
    DecodeID,
    #[error("packet exceeds maximum length")]
    TooLong,
    #[error("packet length is out of bounds")]
    OutOfBounds,
    #[error("malformed packet length VarInt")]
    MalformedLength,
    #[error("failed to decompress packet: {0}")]
    FailedDecompression(String), // Updated to include error details
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use crate::ser::ByteBufMut;

    use super::*;
    use aes::Aes128;
    use bytes::BufMut;
    use cfb8::Encryptor as Cfb8Encryptor;
    use cfb8::cipher::AsyncStreamCipher;
    use flate2::{Compression, write::ZlibEncoder};

    /// Helper function to compress data using libdeflater's Zlib compressor
    fn compress_zlib(data: &[u8]) -> Vec<u8> {
        let mut compressed = Vec::new();
        ZlibEncoder::new(&mut compressed, Compression::default())
            .write_all(data)
            .unwrap();
        compressed
    }

    /// Helper function to encrypt data using AES-128 CFB-8 mode
    fn encrypt_aes128(data: &mut [u8], key: &[u8; 16], iv: &[u8; 16]) {
        let encryptor = Cfb8Encryptor::<Aes128>::new_from_slices(key, iv).expect("Invalid key/iv");
        encryptor.encrypt(data);
    }

    /// Helper function to build a packet with optional compression and encryption
    fn build_packet(
        packet_id: i32,
        payload: &[u8],
        compress: bool,
        key: Option<&[u8; 16]>,
        iv: Option<&[u8; 16]>,
    ) -> Vec<u8> {
        let mut buffer = BytesMut::new();

        if compress {
            // Create a buffer that includes packet_id_varint and payload
            let mut data_to_compress = BytesMut::new();
            let packet_id_varint = VarInt(packet_id);
            data_to_compress.put_var_int(&packet_id_varint);
            data_to_compress.put_slice(payload);

            // Compress the combined data
            let compressed_payload = compress_zlib(&data_to_compress);
            let data_len = data_to_compress.len() as i32; // 1 + payload.len()
            let data_len_varint = VarInt(data_len);
            buffer.put_var_int(&data_len_varint);
            buffer.put_slice(&compressed_payload);
        } else {
            // No compression; data_len is payload length
            let packet_id_varint = VarInt(packet_id);
            buffer.put_var_int(&packet_id_varint);
            buffer.put_slice(payload);
        }

        // Calculate packet length: length of buffer
        let packet_len = buffer.len() as i32;
        let packet_len_varint = VarInt(packet_len);
        let mut packet_length_encoded = Vec::new();
        {
            packet_len_varint.encode(&mut packet_length_encoded);
        }

        // Create a new buffer for the entire packet
        let mut packet = Vec::new();
        packet.extend_from_slice(&packet_length_encoded);
        packet.extend_from_slice(&buffer);

        // Encrypt if key and iv are provided
        if let (Some(k), Some(v)) = (key, iv) {
            encrypt_aes128(&mut packet, k, v);
            packet
        } else {
            packet
        }
    }

    /// Test decoding without compression and encryption
    #[test]
    fn test_decode_without_compression_and_encryption() {
        // Sample packet data: packet_id = 1, payload = "Hello"
        let packet_id = 1;
        let payload = b"Hello";

        // Build the packet without compression and encryption
        let packet = build_packet(packet_id, payload, false, None, None);

        // Initialize the decoder without compression and encryption
        let mut decoder = PacketDecoder::default();
        decoder.set_compression(false);

        // Feed the packet to the decoder
        decoder.queue_slice(&packet);

        // Attempt to decode
        let result = decoder.decode().expect("Decoding failed");
        assert!(result.is_some());

        let mut raw_packet = result.unwrap();
        let mut decoded_payload = Vec::new();
        raw_packet
            .payload
            .read_to_end(&mut decoded_payload)
            .unwrap();

        assert_eq!(raw_packet.id, packet_id);
        assert_eq!(decoded_payload, payload);
    }

    /// Test decoding with compression
    #[test]
    fn test_decode_with_compression() {
        // Sample packet data: packet_id = 2, payload = "Hello, compressed world!"
        let packet_id = 2;
        let payload = b"Hello, compressed world!";

        // Build the packet with compression enabled
        let packet = build_packet(packet_id, payload, true, None, None);

        // Initialize the decoder with compression enabled
        let mut decoder = PacketDecoder::default();
        decoder.set_compression(true);

        // Feed the packet to the decoder
        decoder.queue_slice(&packet);

        // Attempt to decode
        let result = decoder.decode().expect("Decoding failed");
        assert!(result.is_some());

        let mut raw_packet = result.unwrap();
        let mut decoded_payload = Vec::new();
        raw_packet
            .payload
            .read_to_end(&mut decoded_payload)
            .unwrap();

        assert_eq!(raw_packet.id, packet_id);
        assert_eq!(decoded_payload, payload);
    }

    /// Test decoding with encryption
    #[test]
    fn test_decode_with_encryption() {
        // Sample packet data: packet_id = 3, payload = "Hello, encrypted world!"
        let packet_id = 3;
        let payload = b"Hello, encrypted world!";

        // Define encryption key and IV
        let key = [0x00u8; 16]; // Example key
        let iv = [0x00u8; 16]; // Example IV

        // Build the packet with encryption enabled (no compression)
        let packet = build_packet(packet_id, payload, false, Some(&key), Some(&iv));

        // Initialize the decoder with encryption enabled
        let mut decoder = PacketDecoder::default();
        decoder.set_compression(false);
        decoder.set_encryption(Some(&key));

        // Feed the encrypted packet to the decoder
        decoder.queue_slice(&packet);

        // Attempt to decode
        let result = decoder.decode().expect("Decoding failed");
        assert!(result.is_some());

        let mut raw_packet = result.unwrap();
        let mut decoded_payload = Vec::new();
        raw_packet
            .payload
            .read_to_end(&mut decoded_payload)
            .unwrap();

        assert_eq!(raw_packet.id, packet_id);
        assert_eq!(decoded_payload, payload);
    }

    /// Test decoding with both compression and encryption
    #[test]
    fn test_decode_with_compression_and_encryption() {
        // Sample packet data: packet_id = 4, payload = "Hello, compressed and encrypted world!"
        let packet_id = 4;
        let payload = b"Hello, compressed and encrypted world!";

        // Define encryption key and IV
        let key = [0x01u8; 16]; // Example key
        let iv = [0x01u8; 16]; // Example IV

        // Build the packet with both compression and encryption enabled
        let packet = build_packet(packet_id, payload, true, Some(&key), Some(&iv));

        // Initialize the decoder with both compression and encryption enabled
        let mut decoder = PacketDecoder::default();
        decoder.set_compression(true);
        decoder.set_encryption(Some(&key));

        // Feed the encrypted and compressed packet to the decoder
        decoder.queue_slice(&packet);

        // Attempt to decode
        let result = decoder.decode().expect("Decoding failed");
        assert!(result.is_some());

        let mut raw_packet = result.unwrap();
        let mut decoded_payload = Vec::new();
        raw_packet
            .payload
            .read_to_end(&mut decoded_payload)
            .unwrap();

        assert_eq!(raw_packet.id, packet_id);
        assert_eq!(decoded_payload, payload);
    }

    /// Test decoding with invalid compressed data
    #[test]
    fn test_decode_with_invalid_compressed_data() {
        // Sample packet data: packet_id = 5, payload_len = 10, but compressed data is invalid
        let data_len = 10; // Expected decompressed size
        let invalid_compressed_data = vec![0xFF, 0xFF, 0xFF]; // Invalid Zlib data

        // Build the packet with compression enabled but invalid compressed data
        let mut buffer = BytesMut::new();
        let data_len_varint = VarInt(data_len);
        buffer.put_var_int(&data_len_varint);
        buffer.put_slice(&invalid_compressed_data);

        // Calculate packet length: VarInt(data_len) + invalid compressed data
        let packet_len = buffer.len() as i32;
        let packet_len_varint = VarInt(packet_len);

        // Create a new buffer for the entire packet
        let mut packet_buffer = BytesMut::new();
        packet_buffer.put_var_int(&packet_len_varint);
        packet_buffer.put_slice(&buffer);

        let packet_bytes = packet_buffer;

        // Initialize the decoder with compression enabled
        let mut decoder = PacketDecoder::default();
        decoder.set_compression(true);

        // Feed the invalid compressed packet to the decoder
        decoder.queue_slice(&packet_bytes);

        // Attempt to decode and expect a decompression error
        let result = decoder.decode();

        if result.is_ok() {
            panic!("This should have errored!");
        }
    }

    /// Test decoding with a zero-length packet
    #[test]
    fn test_decode_with_zero_length_packet() {
        // Sample packet data: packet_id = 7, payload = "" (empty)
        let packet_id = 7;
        let payload = b"";

        // Build the packet without compression and encryption
        let packet = build_packet(packet_id, payload, false, None, None);

        // Initialize the decoder without compression and encryption
        let mut decoder = PacketDecoder::default();
        decoder.set_compression(false);

        // Feed the packet to the decoder
        decoder.queue_slice(&packet);

        // Attempt to decode
        let result = decoder.decode().expect("Decoding failed");
        assert!(result.is_some());

        let mut raw_packet = result.unwrap();
        let mut decoded_payload = Vec::new();
        raw_packet
            .payload
            .read_to_end(&mut decoded_payload)
            .unwrap();

        assert_eq!(raw_packet.id, packet_id);
        assert_eq!(decoded_payload, payload);
    }

    /// Test decoding with maximum length packet
    #[test]
    fn test_decode_with_maximum_length_packet() {
        // Sample packet data: packet_id = 8, payload = "A" repeated MAX_PACKET_SIZE times
        // Sample packet data: packet_id = 8, payload = "A" repeated (MAX_PACKET_SIZE - 1) times
        let packet_id = 8;
        let payload = vec![0x41u8; MAX_PACKET_SIZE - 1]; // "A" repeated

        // Build the packet with compression enabled
        let packet = build_packet(packet_id, &payload, true, None, None);
        println!(
            "Built packet (with compression, maximum length): {:?}",
            packet
        );

        // Initialize the decoder with compression enabled
        let mut decoder = PacketDecoder::default();
        decoder.set_compression(true);

        // Feed the packet to the decoder
        decoder.queue_slice(&packet);

        // Attempt to decode
        let result = decoder.decode().expect("Decoding failed");
        assert!(
            result.is_some(),
            "Decoder returned None when it should have decoded a packet"
        );

        let mut raw_packet = result.unwrap();
        let mut decoded_payload = Vec::new();
        raw_packet
            .payload
            .read_to_end(&mut decoded_payload)
            .unwrap();

        assert_eq!(raw_packet.id, packet_id);
        assert_eq!(decoded_payload, payload);
    }
}
