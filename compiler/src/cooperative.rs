//! Optional checkpoints for a host that schedules compiler work cooperatively.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

thread_local! {
    static HOOK: RefCell<Option<Rc<dyn Fn()>>> = const { RefCell::new(None) };
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TICKS: Cell<u32> = const { Cell::new(0) };
}

pub struct Guard(Option<Rc<dyn Fn()>>);

impl Drop for Guard {
    fn drop(&mut self) {
        ACTIVE.set(self.0.is_some());
        HOOK.with_borrow_mut(|hook| *hook = self.0.take());
    }
}

/// Installs a thread-local checkpoint hook until the guard is dropped.
pub fn install(hook: Rc<dyn Fn()>) -> Guard {
    let old = HOOK.with_borrow_mut(|current| current.replace(hook));
    ACTIVE.set(true);
    Guard(old)
}

/// Calls the host outside the hook's RefCell borrow, allowing stack suspension.
pub fn checkpoint() {
    if !ACTIVE.get() {
        return;
    }
    let tick = TICKS.get().wrapping_add(1);
    TICKS.set(tick);
    if tick.is_multiple_of(128) {
        poll();
    }
}

/// Checks the host at coarse module boundaries without hot-loop throttling.
pub fn poll() {
    if !ACTIVE.get() {
        return;
    }
    let hook = HOOK.with_borrow(Clone::clone);
    if let Some(hook) = hook {
        hook();
    }
}
