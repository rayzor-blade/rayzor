//! Stackful fibers backing Haxe coroutines (`rayzor.coro.Fiber`).
//!
//! A fiber runs a Haxe closure on its own krio stack. The exception handler
//! stack and the shadow call stack are thread-local, so each fiber keeps its
//! own and `rayzor_fiber_resume` swaps them in for the duration of a resume:
//! a fiber's try/catch frames never interleave with its resumer's.

use std::cell::Cell;

use krio_fiber::{Fiber, FiberStep};

/// Stack size of a coroutine fiber; pages are committed on first touch.
const FIBER_STACK_SIZE: usize = 1024 * 1024;

struct HaxeFiber {
    fiber: Fiber,
    handlers: crate::exception::HandlerStack,
    shadow: crate::native_stack_trace::ShadowStack,
    /// The Haxe object that continues this fiber.
    data: usize,
}

thread_local! {
    static CURRENT: Cell<*mut HaxeFiber> = const { Cell::new(std::ptr::null_mut()) };
}

/// A fiber that will run the closure `{fn, env}` when first resumed.
///
/// # Safety
/// `closure_obj` must point to a closure object `{ fn_ptr, env_ptr }`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rayzor_fiber_new(closure_obj: *const u8) -> *mut u8 {
    if closure_obj.is_null() {
        return std::ptr::null_mut();
    }
    let (func_addr, env_addr) = unsafe {
        let words = closure_obj as *const usize;
        (*words, *words.add(1))
    };
    let fiber = Fiber::with_stack_size(FIBER_STACK_SIZE, move || {
        type ClosureFn = extern "C" fn(*const u8) -> i64;
        let func: ClosureFn = unsafe { std::mem::transmute(func_addr) };
        func(env_addr as *const u8);
    });
    Box::into_raw(Box::new(HaxeFiber {
        fiber,
        handlers: Default::default(),
        shadow: Default::default(),
        data: 0,
    })) as *mut u8
}

/// Run `handle` until it yields (0) or its closure returns (1). A closure
/// that panicked reports 2.
///
/// # Safety
/// `handle` must come from `rayzor_fiber_new` and not be finished or freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rayzor_fiber_resume(handle: *mut u8) -> i32 {
    if handle.is_null() {
        return 1;
    }
    let fiber = handle as *mut HaxeFiber;
    let outer_handlers =
        crate::exception::replace_handlers(std::mem::take(unsafe { &mut (*fiber).handlers }));
    let outer_shadow = crate::native_stack_trace::replace_shadow_stack(std::mem::take(unsafe {
        &mut (*fiber).shadow
    }));
    let previous = CURRENT.with(|c| c.replace(fiber));
    let step = unsafe { (*fiber).fiber.resume() };
    CURRENT.with(|c| c.set(previous));
    unsafe {
        (*fiber).handlers = crate::exception::replace_handlers(outer_handlers);
        (*fiber).shadow = crate::native_stack_trace::replace_shadow_stack(outer_shadow);
    }
    match step {
        FiberStep::Yielded => 0,
        FiberStep::Done => 1,
        FiberStep::Errored => 2,
    }
}

/// Suspend the running fiber back to whoever resumed it. A no-op outside
/// any fiber.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_fiber_yield() {
    if CURRENT.with(|c| c.get()).is_null() {
        return;
    }
    krio_fiber::yield_now();
}

/// Release a fiber that is finished or will never be resumed again.
///
/// # Safety
/// `handle` must come from `rayzor_fiber_new`, and must not be running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rayzor_fiber_free(handle: *mut u8) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle as *mut HaxeFiber) });
    }
}

/// Attach the Haxe object that continues `handle`.
///
/// # Safety
/// `handle` must come from `rayzor_fiber_new` and not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rayzor_fiber_set_data(handle: *mut u8, data: *mut u8) {
    if !handle.is_null() {
        unsafe { (*(handle as *mut HaxeFiber)).data = data as usize };
    }
}

/// The object attached to the running fiber, or null outside any fiber.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_fiber_current_data() -> *mut u8 {
    let fiber = CURRENT.with(|c| c.get());
    if fiber.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*fiber).data as *mut u8 }
    }
}
