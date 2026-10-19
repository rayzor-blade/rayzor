//! A time budget for macro evaluation on the current thread, so a macro that
//! never finishes fails with an error instead of stalling its compile.

use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    static TICKS: Cell<u32> = const { Cell::new(0) };
}

/// Restores the previous budget when dropped.
pub struct Budget(Option<Instant>);

impl Drop for Budget {
    fn drop(&mut self) {
        DEADLINE.with(|d| d.set(self.0));
    }
}

/// Macros evaluated on this thread until the returned guard drops may run for
/// `budget` in all; `None` lifts the limit.
pub fn limit(budget: Option<Duration>) -> Budget {
    let previous = DEADLINE.with(|d| d.replace(budget.map(|b| Instant::now() + b)));
    Budget(previous)
}

/// Whether the budget has run out. Cheap enough for every loop iteration and
/// call: the clock is read once per 1024 checks.
pub fn exhausted() -> bool {
    let Some(deadline) = DEADLINE.with(Cell::get) else {
        return false;
    };
    let tick = TICKS.with(|t| {
        let n = t.get().wrapping_add(1);
        t.set(n);
        n
    });
    tick % 1024 == 0 && Instant::now() >= deadline
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spent_budget_is_reported_and_lifted_by_the_guard() {
        assert!(!(0..5000).any(|_| exhausted()));
        {
            let _b = limit(Some(Duration::ZERO));
            assert!((0..5000).any(|_| exhausted()));
        }
        assert!(!(0..5000).any(|_| exhausted()));
    }
}
