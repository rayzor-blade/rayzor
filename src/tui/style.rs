//! Color palette and styling utilities for rayzor CLI output.

/// Check if stdout is a terminal (for color/TUI decisions).
pub fn is_tty() -> bool {
    std::io::IsTerminal::is_terminal(&std::io::stdout())
}
