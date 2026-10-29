//! A time budget for macro evaluation on the current thread, so a macro that
//! never finishes fails with an error instead of stalling its compile.

use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    static TICKS: Cell<u32> = const { Cell::new(0) };
    static ALLOWANCE: Cell<Option<Duration>> = const { Cell::new(None) };
    static IN_MACRO: Cell<bool> = const { Cell::new(false) };
    static PREVIEW: Cell<bool> = const { Cell::new(false) };
}

/// A suspended compiler task's macro allowance; paused time is not execution.
#[derive(Default)]
pub struct State {
    remaining: Option<Duration>,
    allowance: Option<Duration>,
    active: bool,
    ticks: u32,
    preview: bool,
}

/// Exchanges the current thread's budget with a task's saved state.
pub fn replace_state(state: State) -> State {
    let now = Instant::now();
    State {
        remaining: DEADLINE
            .replace(state.remaining.map(|remaining| now + remaining))
            .map(|deadline| deadline.saturating_duration_since(now)),
        allowance: ALLOWANCE.replace(state.allowance),
        active: IN_MACRO.replace(state.active),
        ticks: TICKS.replace(state.ticks),
        preview: PREVIEW.replace(state.preview),
    }
}

pub struct Preview {
    _budget: Budget,
    previous: bool,
}

impl Drop for Preview {
    fn drop(&mut self) {
        PREVIEW.set(self.previous);
    }
}

/// Provisional inference defers long macros and external effects to full typing.
pub fn preview() -> Preview {
    Preview {
        _budget: limit(Some(Duration::from_millis(100))),
        previous: PREVIEW.replace(true),
    }
}

pub fn previewing() -> bool {
    PREVIEW.get()
}

pub fn require_effects(location: crate::tast::SourceLocation) -> Result<(), super::MacroError> {
    if previewing() {
        Err(super::MacroError::PreviewPending { location })
    } else {
        Ok(())
    }
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
    let deadline =
        ALLOWANCE.with(|allowance| allowance.get().map(|duration| Instant::now() + duration));
    Invocation(Some(DEADLINE.with(|current| current.replace(deadline))))
}

/// Whether the budget has run out. Cheap enough for every loop iteration and
/// call: the clock is read once per 1024 checks.
pub fn exhausted() -> bool {
    crate::cooperative::checkpoint();
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
    fn preview_policy_is_scoped_and_follows_its_suspended_task() {
        assert!(!previewing());
        {
            let _preview = preview();
            assert!(previewing());
            let suspended = replace_state(State::default());
            assert!(!previewing());
            let _ = replace_state(suspended);
            assert!(previewing());
        }
        assert!(!previewing());
        assert!(ALLOWANCE.get().is_none());
    }

    #[test]
    fn suspended_tasks_keep_separate_allowances_and_invocation_state() {
        let _budget = limit(Some(Duration::from_secs(30)));
        let _call = invocation();
        TICKS.set(17);
        let suspended = replace_state(State::default());
        assert!(!IN_MACRO.get());
        assert_eq!(ALLOWANCE.get(), None);
        {
            let _other = limit(Some(Duration::from_secs(2)));
            let _other_call = invocation();
            assert_eq!(ALLOWANCE.get(), Some(Duration::from_secs(2)));
        }
        let empty = replace_state(suspended);
        assert!(!empty.active);
        assert!(IN_MACRO.get());
        assert_eq!(ALLOWANCE.get(), Some(Duration::from_secs(30)));
        assert_eq!(TICKS.get(), 17);
        assert!(DEADLINE.get().unwrap() > Instant::now() + Duration::from_secs(29));
    }

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
