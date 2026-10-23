//! `RAYZOR_STARTUP_TIMES=1`: when each step of `rayzor run` finished, from
//! process start, printed to stderr once the program returns. Off by default;
//! safe to leave on, it only reads the clock at each step.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static START: OnceLock<Instant> = OnceLock::new();
static STEPS: Mutex<Vec<(&'static str, Instant)>> = Mutex::new(Vec::new());

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RAYZOR_STARTUP_TIMES").is_some())
}

/// Call first in `main`: the origin every step is measured from.
pub fn process_started() {
    START.get_or_init(Instant::now);
}

/// Record that `step` has just finished.
pub fn finished(step: &'static str) {
    if enabled() {
        STEPS.lock().unwrap().push((step, Instant::now()));
    }
}

/// Print each step with its end time and duration. `native` is the backend's
/// own native compilation, split into what ran before `main` and during it.
pub fn report(native_before_main: Duration, native_total: Duration, passes: u32, bailouts: u32) {
    if !enabled() {
        return;
    }
    let Some(&start) = START.get() else {
        return;
    };
    let steps = STEPS.lock().unwrap();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let mut previous = start;
    eprintln!("[startup] {:>9} {:>9}  step", "at ms", "took ms");
    for &(step, at) in steps.iter() {
        eprintln!(
            "[startup] {:9.1} {:9.1}  {}",
            ms(at - start),
            ms(at - previous),
            step
        );
        previous = at;
    }
    eprintln!(
        "[startup] native compile: {:.1} ms before main, {:.1} ms while main ran ({} passes, {} interpreter bailouts)",
        ms(native_before_main),
        ms(native_total.saturating_sub(native_before_main)),
        passes,
        bailouts
    );
}
