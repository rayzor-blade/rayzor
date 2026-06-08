//! WASM-owned GGUF byte-source and FFI accessors.

use rayzor_runtime_core::gguf::{ByteSource, GgufError, GgufFile, MetaValue};
use std::boxed::Box;
use std::vec::Vec;

struct WasmBytes {
    bytes: Vec<u8>,
}

impl ByteSource for WasmBytes {
    fn read_at(&self, offset: u64, len: usize) -> Result<&[u8], GgufError> {
        let start = usize::try_from(offset).map_err(|_| GgufError::OutOfBounds)?;
        let end = start.checked_add(len).ok_or(GgufError::OutOfBounds)?;
        self.bytes.get(start..end).ok_or(GgufError::OutOfBounds)
    }

    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
}

struct WasmGguf {
    source: WasmBytes,
    parsed: GgufFile,
}

#[repr(C)]
pub struct GgufMetadataKvView {
    pub key_ptr: i32,
    pub key_len: i32,
    pub type_tag: i32,
    pub value_i64: i64,
    pub value_f64: f64,
    pub value_ptr: i32,
    pub value_len: i32,
}

#[repr(C)]
pub struct GgufTensorInfoView {
    pub name_ptr: i32,
    pub name_len: i32,
    pub ndim: i32,
    pub dims_ptr: i32,
    pub dtype: i32,
    pub offset: u64,
    pub nbytes: i32,
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gguf_open_from_bytes(ptr: i32, len: i32) -> i32 {
    if ptr == 0 || len <= 0 {
        return 0;
    }
    let bytes = core::slice::from_raw_parts(ptr as *const u8, len as usize).to_vec();
    let source = WasmBytes { bytes };
    let parsed = match GgufFile::parse(&source) {
        Ok(parsed) => parsed,
        Err(_) => return 0,
    };
    Box::into_raw(Box::new(WasmGguf { source, parsed })) as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gguf_metadata_count(handle: i32) -> i32 {
    if handle == 0 {
        return 0;
    }
    (*(handle as *const WasmGguf)).parsed.metadata.len() as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gguf_metadata_kv(handle: i32, idx: i32) -> GgufMetadataKvView {
    if handle == 0 || idx < 0 {
        return empty_kv();
    }
    let gguf = &*(handle as *const WasmGguf);
    let Some(kv) = gguf.parsed.metadata.get(idx as usize) else {
        return empty_kv();
    };
    let (value_i64, value_f64, value_ptr, value_len) = match &kv.value {
        MetaValue::U8(v) => (*v as i64, *v as f64, 0, 0),
        MetaValue::I8(v) => (*v as i64, *v as f64, 0, 0),
        MetaValue::U16(v) => (*v as i64, *v as f64, 0, 0),
        MetaValue::I16(v) => (*v as i64, *v as f64, 0, 0),
        MetaValue::U32(v) => (*v as i64, *v as f64, 0, 0),
        MetaValue::I32(v) => (*v as i64, *v as f64, 0, 0),
        MetaValue::F32(v) => (0, *v as f64, 0, 0),
        MetaValue::Bool(v) => (*v as i64, *v as u8 as f64, 0, 0),
        MetaValue::Str(v) => (0, 0.0, v.as_ptr() as i32, v.len() as i32),
        MetaValue::Arr(_, values) => (values.len() as i64, 0.0, 0, values.len() as i32),
        MetaValue::U64(v) => (*v as i64, *v as f64, 0, 0),
        MetaValue::I64(v) => (*v, *v as f64, 0, 0),
        MetaValue::F64(v) => (0, *v, 0, 0),
    };
    GgufMetadataKvView {
        key_ptr: kv.key.as_ptr() as i32,
        key_len: kv.key.len() as i32,
        type_tag: kv.value.type_tag() as i32,
        value_i64,
        value_f64,
        value_ptr,
        value_len,
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gguf_tensor_count(handle: i32) -> i32 {
    if handle == 0 {
        return 0;
    }
    (*(handle as *const WasmGguf)).parsed.tensors.len() as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gguf_tensor_info(handle: i32, idx: i32) -> GgufTensorInfoView {
    if handle == 0 || idx < 0 {
        return empty_tensor();
    }
    let gguf = &*(handle as *const WasmGguf);
    let Some(info) = gguf.parsed.tensors.get(idx as usize) else {
        return empty_tensor();
    };
    let nbytes = GgufFile::tensor_byte_size(info).unwrap_or(0) as i32;
    GgufTensorInfoView {
        name_ptr: info.name.as_ptr() as i32,
        name_len: info.name.len() as i32,
        ndim: info.dims.len() as i32,
        dims_ptr: info.dims.as_ptr() as i32,
        dtype: info.dtype as i32,
        offset: info.offset,
        nbytes,
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gguf_tensor_bytes(
    handle: i32,
    idx: i32,
    dst: i32,
    len: i32,
) -> i32 {
    if handle == 0 || idx < 0 || dst == 0 || len < 0 {
        return -1;
    }
    let gguf = &*(handle as *const WasmGguf);
    let bytes = match gguf.parsed.tensor_bytes(&gguf.source, idx as usize) {
        Ok(bytes) => bytes,
        Err(_) => return -1,
    };
    if len as usize > bytes.len() {
        return -1;
    }
    core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, len as usize);
    len
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gguf_close(handle: i32) {
    if handle != 0 {
        drop(Box::from_raw(handle as *mut WasmGguf));
    }
}

fn empty_kv() -> GgufMetadataKvView {
    GgufMetadataKvView {
        key_ptr: 0,
        key_len: 0,
        type_tag: -1,
        value_i64: 0,
        value_f64: 0.0,
        value_ptr: 0,
        value_len: 0,
    }
}

fn empty_tensor() -> GgufTensorInfoView {
    GgufTensorInfoView {
        name_ptr: 0,
        name_len: 0,
        ndim: 0,
        dims_ptr: 0,
        dtype: -1,
        offset: 0,
        nbytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_gguf_round_trips_through_ffi() {
        let bytes = minimal_gguf();
        unsafe {
            let h = rayzor_gguf_open_from_bytes(bytes.as_ptr() as i32, bytes.len() as i32);
            assert!(h != 0);
            assert_eq!(rayzor_gguf_metadata_count(h), 1);
            let kv = rayzor_gguf_metadata_kv(h, 0);
            let key = core::slice::from_raw_parts(kv.key_ptr as *const u8, kv.key_len as usize);
            assert_eq!(core::str::from_utf8(key).unwrap(), "answer");
            assert_eq!(kv.type_tag, 4);
            assert_eq!(kv.value_i64, 42);

            assert_eq!(rayzor_gguf_tensor_count(h), 1);
            let info = rayzor_gguf_tensor_info(h, 0);
            let name =
                core::slice::from_raw_parts(info.name_ptr as *const u8, info.name_len as usize);
            assert_eq!(core::str::from_utf8(name).unwrap(), "w");
            assert_eq!(info.ndim, 1);
            assert_eq!(*(info.dims_ptr as *const u64), 2);
            assert_eq!(info.dtype, 0);
            assert_eq!(info.nbytes, 8);

            let mut out = [0u8; 8];
            assert_eq!(
                rayzor_gguf_tensor_bytes(h, 0, out.as_mut_ptr() as i32, 8),
                8
            );
            let a = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
            let b = f32::from_le_bytes([out[4], out[5], out[6], out[7]]);
            assert_eq!(a, 1.25);
            assert_eq!(b, -2.5);
            rayzor_gguf_close(h);
        }
    }

    fn minimal_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        push_u32(&mut b, rayzor_runtime_core::gguf::GGUF_MAGIC);
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

        let align = rayzor_runtime_core::gguf::GGUF_DEFAULT_ALIGNMENT as usize;
        let rem = b.len() % align;
        if rem != 0 {
            b.resize(b.len() + (align - rem), 0);
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
