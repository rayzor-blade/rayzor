//! Compilation progress display for `rayzor run` and `rayzor build`.
//!
//! Uses ratatui for animated progress during compilation phases,
//! with a spinner, phase pipeline, and cache/shake summary.

use crossterm::style::{Color, Stylize};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::style;

// ── Spinner ──────────────────────────────────────────────────────

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// An animated spinner that runs in a background thread.
pub struct Spinner {
    done: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Spinner {
    /// Start a spinner with a message. Returns a handle to stop it.
    pub fn start(message: String) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let is_tty = style::is_tty();

        let handle = std::thread::spawn(move || {
            if !is_tty {
                eprintln!("  {}...", message);
                return;
            }
            let mut i = 0;
            while !done_clone.load(Ordering::Relaxed) {
                let frame = SPINNER_FRAMES[i % SPINNER_FRAMES.len()];
                eprint!(
                    "\r  {} {} ",
                    frame.with(Color::Cyan),
                    message.as_str().with(Color::White),
                );
                let _ = std::io::stderr().flush();
                std::thread::sleep(Duration::from_millis(80));
                i += 1;
            }
            // Clear the spinner line
            eprint!("\r{}\r", " ".repeat(60));
            let _ = std::io::stderr().flush();
        });

        Self {
            done,
            handle: Some(handle),
        }
    }

    /// Stop the spinner and optionally print a completion message.
    pub fn stop(mut self, result_line: Option<&str>) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if let Some(line) = result_line {
            eprintln!("{}", line);
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ── Phase Pipeline ───────────────────────────────────────────────

/// Tracks compilation phases with timing for a pipeline display.
pub struct PhasePipeline {
    start: Instant,
    phases: Vec<(String, f64)>,
    current_spinner: Option<Spinner>,
}

impl PhasePipeline {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            phases: Vec::new(),
            current_spinner: None,
        }
    }

    /// Begin a named phase (starts spinner).
    pub fn begin(&mut self, name: &str) {
        // Stop previous spinner
        if let Some(spinner) = self.current_spinner.take() {
            spinner.stop(None);
        }
        self.current_spinner = Some(Spinner::start(name.to_string()));
    }

    /// End the current phase, recording its duration.
    pub fn end(&mut self, name: &str, ms: f64) {
        if let Some(spinner) = self.current_spinner.take() {
            spinner.stop(None);
        }
        self.phases.push((name.to_string(), ms));
    }

    /// Print the phase pipeline summary.
    pub fn print_summary(&self) {
        if self.phases.is_empty() {
            return;
        }
        let tty = style::is_tty();
        let total_ms: f64 = self.phases.iter().map(|(_, ms)| ms).sum();

        if tty {
            // Phase pipeline with colored bars
            let bar_width = 40;
            eprint!("  ");
            for (i, (name, ms)) in self.phases.iter().enumerate() {
                let frac = ms / total_ms.max(0.01);
                let seg_width = (frac * bar_width as f64).round() as usize;
                let seg_width = seg_width.max(1);
                let color = phase_color(name);
                let bar = "\u{2588}".repeat(seg_width);
                eprint!("{}", bar.with(color));
                if i < self.phases.len() - 1 {
                    eprint!("{}", "\u{2502}".with(Color::DarkGrey));
                }
            }
            eprintln!(
                " {}",
                format!("{:.0}ms", total_ms).with(Color::White).bold()
            );

            // Legend: phase labels below the bar
            eprint!("  ");
            for (name, ms) in &self.phases {
                let color = phase_color(name);
                eprint!(
                    "{} {} ",
                    "\u{25CF}".with(color),
                    format!("{} {:.0}ms", name, ms).with(Color::DarkGrey),
                );
            }
            eprintln!();
        } else {
            let parts: Vec<String> = self
                .phases
                .iter()
                .map(|(name, ms)| format!("{} {:.0}ms", name, ms))
                .collect();
            eprintln!("  {}", parts.join(" → "));
        }
    }
}

fn phase_color(name: &str) -> Color {
    match name {
        "parse" | "frontend" => Color::Blue,
        "shake" | "tree-shake" => Color::Yellow,
        "optimize" | "opt" => Color::Magenta,
        "jit" | "codegen" => Color::Cyan,
        _ => Color::White,
    }
}

// ── Run banner ───────────────────────────────────────────────────

/// Print a run banner with styled output.
pub fn print_run_banner(file: &str, profile: &str, preset: &str) {
    if style::is_tty() {
        eprintln!(
            "\n {} {} {} {}",
            "\u{25B6}".with(Color::Cyan),
            file.with(Color::White).bold(),
            format!("[{}]", profile).with(Color::DarkGrey),
            format!("[{}]", preset).with(Color::DarkGrey),
        );
    } else {
        println!("Running {} [{}] [preset: {}]...", file, profile, preset);
    }
}

/// Print cache summary line.
pub fn print_cache_summary(warm: usize, cold: usize) {
    if warm == 0 && cold == 0 {
        return;
    }
    if style::is_tty() {
        let warm_s = format!("{} cached", warm).with(Color::Green).to_string();
        let cold_s = format!("{} compiled", cold).with(Color::Yellow).to_string();
        eprintln!(
            "  {} {} | {}",
            "\u{25CF}".with(Color::DarkGrey),
            warm_s,
            cold_s,
        );
    } else {
        eprintln!("  cache: {} cached | {} compiled", warm, cold);
    }
}

/// Print tree-shake summary line.
pub fn print_shake_summary(before: usize, after: usize) {
    if before <= after {
        return;
    }
    let removed = before - after;
    let pct = (removed as f64 / before as f64 * 100.0) as usize;
    if style::is_tty() {
        eprintln!(
            "  {} {} → {} functions {}",
            "\u{25CF}".with(Color::DarkGrey),
            before.to_string().with(Color::DarkGrey),
            after.to_string().with(Color::Green),
            format!("({}% stripped)", pct).with(Color::DarkGrey),
        );
    } else {
        eprintln!(
            "  shake: {} → {} functions ({}% removed)",
            before, after, pct
        );
    }
}
