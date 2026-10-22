//! A time budget for macro evaluation on the current thread, so a macro that
//! never finishes fails with an error instead of stalling its compile.

use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    static TICKS: Cell<u32> = const { Cell::new(0) };
    static ALLOWANCE: Cell<Option<Duration>> = const { Cell::new(None) };
    static IN_MACRO: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous budget when dropped.
pub struct Budget(Option<Instant>, Option<Duration>);

impl Drop for Budget {
    fn drop(&mut self) {
        DEADLINE.with(|d| d.set(self.0));
        ALLOWANCE.with(|allowance| allowance.set(self.1));
    }
}

/// Configure the evaluation allowance on this thread; `None` lifts the limit.
pub fn limit(budget: Option<Duration>) -> Budget {
    let previous = DEADLINE.with(|d| d.replace(budget.map(|b| Instant::now() + b)));
    let allowance = ALLOWANCE.with(|allowance| allowance.replace(budget));
    Budget(previous, allowance)
}

pub struct Invocation(Option<Option<Instant>>);

impl Drop for Invocation {
    fn drop(&mut self) {
        if let Some(previous) = self.0 {
            DEADLINE.with(|deadline| deadline.set(previous));
            IN_MACRO.with(|active| active.set(false));
        }
    }
}

/// Each top-level macro gets the configured allowance; nested work shares it.
pub fn invocation() -> Invocation {
    if IN_MACRO.with(|active| active.replace(true)) {
        return Invocation(None);
    }
    let deadline = ALLOWANCE.with(|allowance| allowance.get().map(|duration| Instant::now() + duration));
    Invocation(Some(DEADLINE.with(|current| current.replace(deadline))))
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

    #[test]
    fn invocation_excludes_prior_indexing_and_keeps_the_nested_deadline() {
        let _budget = limit(Some(Duration::from_secs(30)));
        let spent = Instant::now() - Duration::from_secs(1);
        DEADLINE.with(|deadline| deadline.set(Some(spent)));
        {
            let _call = invocation();
            assert!(!(0..5000).any(|_| exhausted()));
            DEADLINE.with(|deadline| deadline.set(Some(spent)));
            let _nested = invocation();
            assert!((0..5000).any(|_| exhausted()));
        }
        assert!((0..5000).any(|_| exhausted()));
    }
}
