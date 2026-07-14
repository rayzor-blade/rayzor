//! Minimal GGUF v3 parser shared by native and WASM runtimes.
//!
//! The parser copies metadata keys/values and tensor index records into owned
//! structures. Tensor payloads stay in the caller's `ByteSource`.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

pub const GGUF_MAGIC: u32 = 0x4655_4747;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

pub trait ByteSource {
    fn read_at(&self, offset: u64, len: usize) -> Result<&[u8], GgufError>;
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufError {
    OutOfBounds,
    BadMagic,
    UnsupportedVersion,
    InvalidUtf8,
    UnknownMetaType,
    InvalidTensorIndex,
    SizeOverflow,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(u32, Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn type_tag(&self) -> u32 {
        match self {
            MetaValue::U8(_) => 0,
            MetaValue::I8(_) => 1,
            MetaValue::U16(_) => 2,
            MetaValue::I16(_) => 3,
            MetaValue::U32(_) => 4,
            MetaValue::I32(_) => 5,
            MetaValue::F32(_) => 6,
            MetaValue::Bool(_) => 7,
            MetaValue::Str(_) => 8,
            MetaValue::Arr(_, _) => 9,
            MetaValue::U64(_) => 10,
            MetaValue::I64(_) => 11,
            MetaValue::F64(_) => 12,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataKv {
    pub key: String,
    pub value: MetaValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<MetadataKv>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_start: u64,
}

impl GgufFile {
    pub fn parse<S: ByteSource>(source: &S) -> Result<Self, GgufError> {
        let mut p = Parser { source, pos: 0 };
        let magic = p.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = p.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion);
        }
        let n_tensors = p.read_u64()? as usize;
        let n_meta = p.read_u64()? as usize;

        let mut metadata = Vec::with_capacity(n_meta);
        for _ in 0..n_meta {
            let key = p.read_string()?;
            let tag = p.read_u32()?;
            let value = p.read_value(tag)?;
            metadata.push(MetadataKv { key, value });
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = p.read_string()?;
            let ndim = p.read_u32()? as usize;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(p.read_u64()?);
            }
            let dtype = p.read_u32()?;
            let offset = p.read_u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        let alignment = metadata
            .iter()
            .find(|kv| kv.key == "general.alignment")
            .and_then(|kv| match kv.value {
                MetaValue::U32(v) => Some(v as u64),
                MetaValue::I32(v) if v > 0 => Some(v as u64),
                MetaValue::U64(v) => Some(v),
                MetaValue::I64(v) if v > 0 => Some(v as u64),
                _ => None,
            })
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let rem = p.pos % alignment;
        let data_start = if rem == 0 {
            p.pos
        } else {
            p.pos + (alignment - rem)
        };
        if data_start > source.len() {
            return Err(GgufError::OutOfBounds);
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            alignment,
            data_start,
        })
    }

    pub fn tensor_byte_size(info: &TensorInfo) -> Result<usize, GgufError> {
        let mut n = 1u64;
        for &d in &info.dims {
            n = n.checked_mul(d).ok_or(GgufError::SizeOverflow)?;
        }
        let bytes = match info.dtype {
            0 => n.checked_mul(4).ok_or(GgufError::SizeOverflow)?,
            1 => n.checked_mul(2).ok_or(GgufError::SizeOverflow)?,
            8 => n
                .checked_add((n / 32).checked_mul(2).ok_or(GgufError::SizeOverflow)?)
                .ok_or(GgufError::SizeOverflow)?,
            12 => div_ceil(n, 256)
                .checked_mul(144)
                .ok_or(GgufError::SizeOverflow)?,
            14 => div_ceil(n, 256)
                .checked_mul(210)
                .ok_or(GgufError::SizeOverflow)?,
            _ => return Err(GgufError::UnknownMetaType),
        };
        usize::try_from(bytes).map_err(|_| GgufError::SizeOverflow)
    }

    pub fn tensor_bytes<'a, S: ByteSource>(
        &self,
        source: &'a S,
        idx: usize,
    ) -> Result<&'a [u8], GgufError> {
        let info = self.tensors.get(idx).ok_or(GgufError::InvalidTensorIndex)?;
        let len = Self::tensor_byte_size(info)?;
        source.read_at(
            self.data_start
                .checked_add(info.offset)
                .ok_or(GgufError::SizeOverflow)?,
            len,
        )
    }
}

fn div_ceil(n: u64, d: u64) -> u64 {
    if n == 0 {
        0
    } else {
        ((n - 1) / d) + 1
    }
}

struct Parser<'a, S: ByteSource> {
    source: &'a S,
    pos: u64,
}

impl<S: ByteSource> Parser<'_, S> {
    fn read_exact(&mut self, len: usize) -> Result<&[u8], GgufError> {
        let out = self.source.read_at(self.pos, len)?;
        self.pos = self
            .pos
            .checked_add(len as u64)
            .ok_or(GgufError::SizeOverflow)?;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let b = self.read_exact(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::SizeOverflow)?;
        let b = self.read_exact(len)?;
        core::str::from_utf8(b)
            .map(String::from)
            .map_err(|_| GgufError::InvalidUtf8)
    }

    fn read_value(&mut self, tag: u32) -> Result<MetaValue, GgufError> {
        Ok(match tag {
            0 => MetaValue::U8(self.read_u8()?),
            1 => MetaValue::I8(self.read_u8()? as i8),
            2 => MetaValue::U16(self.read_u16()?),
            3 => MetaValue::I16(self.read_u16()? as i16),
            4 => MetaValue::U32(self.read_u32()?),
            5 => MetaValue::I32(self.read_u32()? as i32),
            6 => MetaValue::F32(f32::from_bits(self.read_u32()?)),
            7 => MetaValue::Bool(self.read_u8()? != 0),
            8 => MetaValue::Str(self.read_string()?),
            9 => {
                let elem = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.read_value(elem)?);
                }
                MetaValue::Arr(elem, values)
            }
            10 => MetaValue::U64(self.read_u64()?),
            11 => MetaValue::I64(self.read_u64()? as i64),
            12 => MetaValue::F64(f64::from_bits(self.read_u64()?)),
            _ => return Err(GgufError::UnknownMetaType),
        })
    }
}

impl ByteSource for &[u8] {
    fn read_at(&self, offset: u64, len: usize) -> Result<&[u8], GgufError> {
        let start = usize::try_from(offset).map_err(|_| GgufError::OutOfBounds)?;
        let end = start.checked_add(len).ok_or(GgufError::OutOfBounds)?;
        self.get(start..end).ok_or(GgufError::OutOfBounds)
    }

    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn parses_minimal_gguf() {
        let bytes = crate::gguf::tests_support::minimal_gguf();
        let src = bytes.as_slice();
        let parsed = GgufFile::parse(&src).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.metadata.len(), 1);
        assert_eq!(parsed.metadata[0].key, "answer");
        assert_eq!(parsed.metadata[0].value, MetaValue::U32(42));
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].name, "w");
        assert_eq!(parsed.tensors[0].dims, vec![2]);
        assert_eq!(parsed.tensors[0].dtype, 0);
        assert_eq!(parsed.tensor_bytes(&src, 0).unwrap().len(), 8);
    }
}

#[cfg(test)]
pub mod tests_support {
    use alloc::vec::Vec;

    pub fn minimal_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        push_u32(&mut b, super::GGUF_MAGIC);
        push_u32(&mut b, 3);
        push_u64(&mut b, 1);
        push_u64(&mut b, 1);

        push_str(&mut b, "answer");
        push_u32(&mut b, 4);
        push_u32(&mut b, 42);

        push_str(&mut b, "w");
        push_u32(&mut b, 1);
        push_u64(&mut b, 2);
        push_u32(&mut b, 0);
        push_u64(&mut b, 0);

        let rem = b.len() % super::GGUF_DEFAULT_ALIGNMENT as usize;
        if rem != 0 {
            b.resize(b.len() + (super::GGUF_DEFAULT_ALIGNMENT as usize - rem), 0);
        }
        push_f32(&mut b, 1.25);
        push_f32(&mut b, -2.5);
        b
    }

    fn push_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_f32(b: &mut Vec<u8>, v: f32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_str(b: &mut Vec<u8>, s: &str) {
        push_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }
}
