//! Rayzor carriers consumed by xgpu's generated model and shared backend.

use rayzor_plugin::host_abi;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub enum ErrorKind {
    Type,
    Runtime,
}

pub mod host {
    use super::{ErrorKind, host_abi};

    pub fn raise(_kind: ErrorKind, message: &str) {
        unsafe { host_abi::rayzor_plugin_raise(message.as_ptr(), message.len()) }
    }
}

pub trait NativeEnum: Copy + Default {
    fn native(self) -> i32;
    fn from_native(value: i32) -> Option<Self>;
}

#[repr(transparent)]
pub struct Enum<T: NativeEnum>(i64, PhantomData<fn() -> T>);

impl<T: NativeEnum> Copy for Enum<T> {}
impl<T: NativeEnum> Clone for Enum<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: NativeEnum> Enum<T> {
    pub fn get(self) -> T {
        T::from_native(self.0 as i32).unwrap_or_else(|| {
            host::raise(ErrorKind::Type, "native enum value is not declared");
            T::default()
        })
    }
}
impl<T: NativeEnum> From<T> for Enum<T> {
    fn from(value: T) -> Self {
        Self(value.native() as i64, PhantomData)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Text(*mut u8);

impl Text {
    pub const NULL: Self = Self(std::ptr::null_mut());

    pub fn new(value: &str) -> Self {
        Self(unsafe { host_abi::rayzor_plugin_string_new(value.as_ptr(), value.len()) })
    }

    pub fn as_str(&self) -> &str {
        if self.0.is_null() {
            return "";
        }
        let data = unsafe { host_abi::rayzor_plugin_string_data(self.0) };
        let len = unsafe { host_abi::rayzor_plugin_string_len(self.0) };
        if data.is_null() || len == 0 {
            return "";
        }
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(data, len)) }
    }

    pub fn value(self) -> Value {
        Value(self.0)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Buffer(*mut u8);

impl Buffer {
    pub const NULL: Self = Self(std::ptr::null_mut());

    pub fn new(value: &[u8]) -> Self {
        Self(unsafe { host_abi::rayzor_plugin_bytes_new(value.as_ptr(), value.len()) })
    }

    pub fn len(&self) -> usize {
        unsafe { host_abi::rayzor_plugin_bytes_len(self.0) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        unsafe { host_abi::rayzor_plugin_bytes_data(self.0) }
    }

    pub fn as_mut_ptr(&self) -> Option<*mut u8> {
        let pointer = unsafe { host_abi::rayzor_plugin_bytes_data_mut(self.0) };
        (!pointer.is_null()).then_some(pointer)
    }

    pub fn is_read_only(&self) -> bool {
        false
    }

    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.is_empty() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BufferMut(Buffer);

impl BufferMut {
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.0.as_mut_ptr().unwrap_or(std::ptr::null_mut())
    }

    pub fn buffer(&self) -> Buffer {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Value(*mut u8);

impl Value {
    pub const fn null() -> Self {
        Self(std::ptr::null_mut())
    }
}

#[repr(transparent)]
pub struct Future<T = Value>(*mut u8, PhantomData<fn() -> T>);

impl<T> Copy for Future<T> {}
impl<T> Clone for Future<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> Send for Future<T> {}
unsafe impl<T> Sync for Future<T> {}

impl<T> Future<T> {
    pub const NULL: Self = Self(std::ptr::null_mut(), PhantomData);

    pub fn new() -> Self {
        Self(
            unsafe { host_abi::rayzor_plugin_future_pending(1) },
            PhantomData,
        )
    }

    pub fn resolve(self, value: Value) -> bool {
        unsafe { host_abi::rayzor_plugin_future_resolve(self.0, value.0 as i64) != 0 }
    }

    pub fn reject(self, error: Value) -> bool {
        let text = Text(error.0);
        unsafe {
            host_abi::rayzor_plugin_future_reject(
                self.0,
                text.as_str().as_ptr(),
                text.as_str().len(),
            ) != 0
        }
    }

    pub fn resolve_boxed(self, value: Box<T>) -> bool {
        let value = Box::into_raw(value).cast::<u8>();
        unsafe { host_abi::rayzor_plugin_future_resolve(self.0, value as i64) != 0 }
    }
}

pub trait Rootable: Copy {
    type Root: Clone;
    fn root(self) -> Self::Root;
    fn get(root: &Self::Root) -> Self;
}

pub struct Rooted<T: Rootable> {
    root: T::Root,
}

impl<T: Rootable> Rooted<T> {
    pub fn new(value: T) -> Self {
        Self { root: value.root() }
    }

    pub fn get(&self) -> T {
        T::get(&self.root)
    }
}
impl<T: Rootable> Clone for Rooted<T> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
        }
    }
}

#[repr(C)]
struct TextHeader {
    data: *mut u8,
    len: usize,
    cap: usize,
}

#[doc(hidden)]
pub struct TextStorage {
    _bytes: Box<[u8]>,
    header: TextHeader,
}

unsafe impl Send for TextStorage {}
unsafe impl Sync for TextStorage {}

impl Rootable for Text {
    type Root = Arc<TextStorage>;

    fn root(self) -> Self::Root {
        let mut bytes = self.as_str().as_bytes().to_vec().into_boxed_slice();
        let data = bytes.as_mut_ptr();
        let len = bytes.len();
        Arc::new(TextStorage {
            _bytes: bytes,
            header: TextHeader { data, len, cap: 0 },
        })
    }

    fn get(root: &Self::Root) -> Self {
        Self((&root.header as *const TextHeader).cast_mut().cast())
    }
}

#[doc(hidden)]
pub struct BufferRoot(*mut u8);
unsafe impl Send for BufferRoot {}
unsafe impl Sync for BufferRoot {}

impl Clone for BufferRoot {
    fn clone(&self) -> Self {
        Self(unsafe { host_abi::rayzor_plugin_bytes_retain(self.0) })
    }
}
impl Drop for BufferRoot {
    fn drop(&mut self) {
        unsafe { host_abi::rayzor_plugin_bytes_release(self.0) }
    }
}
impl Rootable for Buffer {
    type Root = BufferRoot;

    fn root(self) -> Self::Root {
        BufferRoot(unsafe { host_abi::rayzor_plugin_bytes_retain(self.0) })
    }

    fn get(root: &Self::Root) -> Self {
        Self(root.0)
    }
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct FutureRoot(*mut u8);
unsafe impl Send for FutureRoot {}
unsafe impl Sync for FutureRoot {}

impl<T> Rootable for Future<T> {
    type Root = FutureRoot;

    fn root(self) -> Self::Root {
        FutureRoot(self.0)
    }

    fn get(root: &Self::Root) -> Self {
        Self(root.0, PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link_test_host() {
        std::hint::black_box([
            rayzor_runtime::plugin_carriers::rayzor_plugin_bytes_data as *const (),
            rayzor_runtime::plugin_carriers::rayzor_plugin_bytes_len as *const (),
            rayzor_runtime::plugin_carriers::rayzor_plugin_bytes_new as *const (),
            rayzor_runtime::plugin_carriers::rayzor_plugin_bytes_retain as *const (),
            rayzor_runtime::plugin_carriers::rayzor_plugin_bytes_release as *const (),
            rayzor_runtime::plugin_carriers::rayzor_plugin_raise as *const (),
        ]);
    }

    #[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
    enum TestEnum {
        #[default]
        Zero,
        Seven,
    }

    impl NativeEnum for TestEnum {
        fn native(self) -> i32 {
            match self {
                Self::Zero => 0,
                Self::Seven => 7,
            }
        }

        fn from_native(value: i32) -> Option<Self> {
            match value {
                0 => Some(Self::Zero),
                7 => Some(Self::Seven),
                _ => None,
            }
        }
    }

    #[test]
    fn enum_codes_and_rooted_bytes_keep_the_runtime_identity() {
        link_test_host();
        assert_eq!(Enum::from(TestEnum::Seven).get(), TestEnum::Seven);
        let bytes = Buffer::new(b"xgpu");
        let rooted = Rooted::new(bytes);
        unsafe { host_abi::rayzor_plugin_bytes_release(bytes.0) };
        let kept = rooted.get();
        assert_eq!(unsafe { kept.as_slice() }, b"xgpu");
    }
}
