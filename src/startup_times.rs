//! `RAYZOR_STARTUP_TIMES=1`: when each step of `rayzor run` finished, from
//! process start, printed to stderr once the program returns. Off by default;
//! safe to leave on, it only reads the clock at each step. A `--features
//! profile` build also prints each step's live heap and its peak.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static START: OnceLock<Instant> = OnceLock::new();
/// Each step's end, with the live heap and the step's peak when tracked.
/// A step's name, when it finished, and the heap then (profile builds only).
type Step = (&'static str, Instant, Option<(u64, u64)>);

static STEPS: Mutex<Vec<Step>> = Mutex::new(Vec::new());

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
        STEPS
            .lock()
            .unwrap()
            .push((step, Instant::now(), heap_at_step()));
    }
}

#[cfg(feature = "profile")]
fn heap_at_step() -> Option<(u64, u64)> {
    Some((
        rayzor_runtime::live_bytes(),
        rayzor_runtime::take_window_peak(),
    ))
}

#[cfg(not(feature = "profile"))]
fn heap_at_step() -> Option<(u64, u64)> {
    None
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
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[startup] {:>9} {:>9} {:>9} {:>9}  step",
        "at ms", "took ms", "live MB", "peak MB"
    );
    for &(step, at, heap) in steps.iter() {
        let (live, peak) = heap.map_or((String::from("-"), String::from("-")), |(l, p)| {
            (format!("{:.1}", mb(l)), format!("{:.1}", mb(p)))
        });
        eprintln!(
            "[startup] {:9.1} {:9.1} {:>9} {:>9}  {}",
            ms(at - start),
            ms(at - previous),
            live,
            peak,
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
