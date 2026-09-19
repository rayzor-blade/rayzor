//! Iteration over a value whose protocol the compiler could not name: an
//! anonymous structure `{ iterator: f }` or `{ hasNext: f, next: f }`, whose
//! entry points are closures read from its fields at run time, and the
//! iterator such a closure hands back, which may itself be an iteration
//! handle built at some boundary.
//!
//! The handle layout and tag are the compiler's (`ir/mir/helpers/iter_handle.rs`).

/// First word of an iteration handle.
const ITER_HANDLE_TAG: i64 = 0x52_5A_49_54_45_52_00_01;

/// A closure record is `[code, env]`; its code takes the env first.
unsafe fn call_closure(record: u64, args: &[i64]) -> i64 {
    if record < 0x1000 || record & 7 != 0 {
        return 0;
    }
    let code = unsafe { *(record as *const usize) };
    let env = unsafe { *(record as *const usize).add(1) } as i64;
    if code == 0 {
        return 0;
    }
    unsafe {
        match args {
            [] => {
                let f: extern "C" fn(i64) -> i64 = std::mem::transmute(code);
                f(env)
            }
            [a] => {
                let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(code);
                f(env, *a)
            }
            _ => 0,
        }
    }
}

fn is_handle(value: i64) -> bool {
    value >= 0x1000 && value & 7 == 0 && unsafe { *(value as *const i64) } == ITER_HANDLE_TAG
}

/// The iterator a handle iterates: its object, made an iterator once through
/// slot 16 when that slot names an `iterator()`.
unsafe fn handle_iterator(handle: i64) -> i64 {
    let obj = unsafe { *((handle + 8) as *const i64) };
    let make = unsafe { *((handle + 16) as *const u64) };
    if make == 0 {
        return obj;
    }
    let it = unsafe { call_closure(make, &[obj]) };
    unsafe {
        *((handle + 8) as *mut i64) = it;
        *((handle + 16) as *mut u64) = 0;
    }
    it
}

/// `{ iterator: f }`: the iterator `f()` answers.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_anon_iterable_iterator(obj: *mut u8) -> i64 {
    match crate::anon_object::anon_raw_field(obj, "iterator") {
        Some(f) => unsafe { call_closure(f, &[]) },
        None => 0,
    }
}

/// `hasNext()` of an iterator value of unknown shape: a handle's own entry
/// point, or a structure's `hasNext` closure. Anything else has nothing.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_iter_value_has_next(it: i64) -> i32 {
    if it < 0x1000 {
        return 0;
    }
    if is_handle(it) {
        let obj = unsafe { handle_iterator(it) };
        let hn = unsafe { *((it + 24) as *const u64) };
        return (unsafe { call_closure(hn, &[obj]) } & 0xff != 0) as i32;
    }
    if is_class_instance(it) {
        return 0;
    }
    match crate::anon_object::anon_raw_field(it as *mut u8, "hasNext") {
        Some(f) => (unsafe { call_closure(f, &[]) } & 0xff != 0) as i32,
        None => 0,
    }
}

/// `next()` of the same iterator value.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_iter_value_next(it: i64) -> i64 {
    if it < 0x1000 {
        return 0;
    }
    if is_handle(it) {
        let obj = unsafe { handle_iterator(it) };
        let nx = unsafe { *((it + 32) as *const u64) };
        return unsafe { call_closure(nx, &[obj]) };
    }
    if is_class_instance(it) {
        return 0;
    }
    match crate::anon_object::anon_raw_field(it as *mut u8, "next") {
        Some(f) => unsafe { call_closure(f, &[]) },
        None => 0,
    }
}

/// A class instance carries its type id in the first word; a structure's
/// handle points at its Arc, which is never that small.
fn is_class_instance(value: i64) -> bool {
    let header = unsafe { *(value as *const i64) };
    (0..0x1000_0000).contains(&header)
}
