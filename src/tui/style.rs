//! Color palette and styling utilities for rayzor CLI output.

use crossterm::style::{Color, Stylize};

/// Check if stdout is a terminal (for color/TUI decisions).
pub fn is_tty() -> bool {
    std::io::IsTerminal::is_terminal(&std::io::stdout())
}

// ── Rayzor color palette ──────────────────────────────────────────

pub const RAYZOR_CYAN: Color = Color::Cyan;
pub const RAYZOR_GREEN: Color = Color::Green;
pub const RAYZOR_YELLOW: Color = Color::Yellow;
pub const RAYZOR_RED: Color = Color::Red;
pub const RAYZOR_MAGENTA: Color = Color::Magenta;
pub const RAYZOR_DIM: Color = Color::DarkGrey;
pub const RAYZOR_BLUE: Color = Color::Blue;

// ── Styled print helpers ──────────────────────────────────────────

/// Print a phase label + timing: `  parse    12ms`
pub fn print_phase(label: &str, ms: f64) {
    if is_tty() {
        let timing = format!("{:.0}ms", ms);
        println!(
            "  {:10} {}",
            label.with(RAYZOR_DIM),
            timing.with(RAYZOR_CYAN)
        );
    } else {
        println!("  {:10} {:.0}ms", label, ms);
    }
}

/// Print a success line: `  ✓ message`
pub fn print_ok(msg: &str) {
    if is_tty() {
        println!("  {} {}", "✓".with(RAYZOR_GREEN), msg);
    } else {
        println!("  OK {}", msg);
    }
}

/// Print a warning line: `  ⚠ message`
pub fn print_warn(msg: &str) {
    if is_tty() {
        eprintln!("  {} {}", "⚠".with(RAYZOR_YELLOW), msg.with(RAYZOR_YELLOW));
    } else {
        eprintln!("  WARN {}", msg);
    }
}

/// Print an error line: `  ✗ message`
pub fn print_err(msg: &str) {
    if is_tty() {
        eprintln!("  {} {}", "✗".with(RAYZOR_RED), msg.with(RAYZOR_RED));
    } else {
        eprintln!("  ERR {}", msg);
    }
}

/// Print a labeled value: `  label    value`
pub fn print_kv(label: &str, value: &str) {
    if is_tty() {
        println!("  {:10} {}", label.with(RAYZOR_DIM), value);
    } else {
        println!("  {:10} {}", label, value);
    }
}

/// Print a section header
pub fn print_header(title: &str) {
    if is_tty() {
        println!("{}", title.with(Color::White).bold());
    } else {
        println!("{}", title);
    }
}

/// Format a duration as a colored string based on speed.
pub fn format_ms(ms: f64, good_threshold: f64, bad_threshold: f64) -> String {
    if !is_tty() {
        return format!("{:.0}ms", ms);
    }
    let s = format!("{:.0}ms", ms);
    if ms <= good_threshold {
        s.with(RAYZOR_GREEN).to_string()
    } else if ms >= bad_threshold {
        s.with(RAYZOR_RED).to_string()
    } else {
        s.with(RAYZOR_YELLOW).to_string()
    }
}

/// Render a compact horizontal bar.
pub fn render_bar(filled: usize, total: usize, color: Color) -> String {
    if !is_tty() || total == 0 {
        return "#".repeat(filled.min(total));
    }
    let fill = "\u{2588}".repeat(filled);
    let empty = "\u{2591}".repeat(total.saturating_sub(filled));
    format!("{}{}", fill.with(color), empty.with(RAYZOR_DIM))
}
