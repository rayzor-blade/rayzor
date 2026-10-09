//! Bound String methods use packed Dynamic arguments for optional parameters.

use crate::closure_entries::{haxe_dynamic_to_slot, haxe_reflect_make_var_args};
use crate::haxe_array::{HaxeArray, haxe_array_get_erased};
use crate::haxe_string::HaxeString;
use crate::haxe_sys;
use crate::type_system::{
    TYPE_NULL, TYPE_STRING, dynamic_box_at, haxe_box_array_ptr, haxe_box_haxestring_ptr,
    haxe_box_int_ptr, haxe_unbox_if_tag,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    CharAt,
    CharCodeAt,
    IndexOf,
    LastIndexOf,
    Split,
    Substr,
    Substring,
    ToLowerCase,
    ToUpperCase,
    ToString,
}

impl Method {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "charAt" => Self::CharAt,
            "charCodeAt" => Self::CharCodeAt,
            "indexOf" => Self::IndexOf,
            "lastIndexOf" => Self::LastIndexOf,
            "split" => Self::Split,
            "substr" => Self::Substr,
            "substring" => Self::Substring,
            "toLowerCase" => Self::ToLowerCase,
            "toUpperCase" => Self::ToUpperCase,
            "toString" => Self::ToString,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
struct Binding {
    receiver: *const HaxeString,
    method: Method,
}

pub(super) fn field(receiver: *const HaxeString, name: &str) -> *mut u8 {
    let Some(method) = Method::from_name(name).filter(|_| !receiver.is_null()) else {
        return std::ptr::null_mut();
    };
    let binding = Box::into_raw(Box::new(Binding { receiver, method }));
    let callback =
        Box::into_raw(Box::new([invoke as *const () as usize, binding as usize])) as *mut u8;
    haxe_reflect_make_var_args(callback)
}

fn integer(argument: *mut u8, default: i32) -> i32 {
    if argument.is_null() || dynamic_box_at(argument).is_some_and(|d| d.type_id == TYPE_NULL) {
        default
    } else {
        haxe_dynamic_to_slot(argument) as i32
    }
}

extern "C" fn invoke(env: usize, args: *mut u8) -> *mut u8 {
    let Binding { receiver, method } = unsafe { *(env as *const Binding) };
    let first = haxe_array_get_erased(args, 0, 0) as *mut u8;
    let second = haxe_array_get_erased(args, 1, 0) as *mut u8;
    let length = crate::haxe_string::haxe_string_length(receiver) as i32;
    let string = match method {
        Method::CharAt => haxe_sys::haxe_string_char_at_ptr(receiver, integer(first, 0) as i64),
        Method::CharCodeAt => {
            let code = haxe_sys::haxe_string_char_code_at_ptr(receiver, integer(first, 0) as i64);
            return if code < 0 {
                std::ptr::null_mut()
            } else {
                haxe_box_int_ptr(code)
            };
        }
        Method::IndexOf | Method::LastIndexOf => {
            let needle = haxe_unbox_if_tag(first, TYPE_STRING.0) as *const HaxeString;
            let index = if method == Method::IndexOf {
                haxe_sys::haxe_string_index_of_ptr(receiver, needle, integer(second, 0))
            } else {
                haxe_sys::haxe_string_last_index_of_ptr(receiver, needle, integer(second, length))
            };
            return haxe_box_int_ptr(index as i64);
        }
        Method::Split => {
            let delimiter = haxe_unbox_if_tag(first, TYPE_STRING.0) as *const HaxeString;
            let mut count = 0;
            let parts = haxe_sys::haxe_string_split_ptr(receiver, delimiter, &mut count);
            let array = Box::into_raw(Box::new(HaxeArray {
                ptr: parts as *mut u8,
                len: count as usize,
                cap: count as usize,
                elem_size: 8,
                flags: 0,
            }));
            // String slots retain their layout when the array crosses Dynamic.
            return haxe_box_array_ptr(array as *mut u8, 5);
        }
        Method::Substr => {
            haxe_sys::haxe_string_substr_ptr(receiver, integer(first, 0), integer(second, length))
        }
        Method::Substring => haxe_sys::haxe_string_substring_ptr(
            receiver,
            integer(first, 0),
            integer(second, length),
        ),
        Method::ToLowerCase => haxe_sys::haxe_string_lower(receiver),
        Method::ToUpperCase => haxe_sys::haxe_string_upper(receiver),
        Method::ToString => receiver as *mut HaxeString,
    };
    haxe_box_haxestring_ptr(string as *mut u8)
}

fn binding(closure: *mut u8) -> Option<Binding> {
    if !crate::closure_entries::haxe_closure_is_varargs(closure) {
        return None;
    }
    unsafe {
        let callback = *(closure as *const *const usize).add(1);
        if callback.is_null() || *callback != invoke as *const () as usize {
            return None;
        }
        Some(*(*callback.add(1) as *const Binding))
    }
}

pub(super) fn compare_methods(left: *mut u8, right: *mut u8) -> Option<bool> {
    let (left, right) = (binding(left)?, binding(right)?);
    Some(
        left.method == right.method
            && crate::haxe_string::haxe_string_compare(left.receiver, right.receiver) == 0,
    )
}
