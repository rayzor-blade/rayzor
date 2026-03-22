//! Terminal UI rendering for rayzor CLI output.
//!
//! Provides colored, structured output for compilation progress,
//! cache stats, tree-shake results, and diagnostic summaries.
//! Falls back to plain text when stdout is not a TTY.

pub mod style;
pub mod progress;

pub use style::is_tty;
