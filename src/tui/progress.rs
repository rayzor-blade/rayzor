//! Compilation progress TUI using ratatui.
//!
//! Renders a live terminal UI during `rayzor run` showing:
//! - Current phase with animated spinner
//! - Phase pipeline with timing bars
//! - Cache hit/miss summary
//! - Tree-shake stats

use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Bar, BarChart, BarGroup, Block, Borders, Gauge, Paragraph, Row, Table},
    Frame, Terminal,
};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::style::is_tty;

// ── Compilation state shared with TUI render thread ──────────────

#[derive(Clone, Debug)]
pub struct PhaseEntry {
    pub name: String,
    pub duration_ms: f64,
    pub color: Color,
}

#[derive(Clone, Debug, Default)]
pub struct CompilationState {
    pub file: String,
    pub profile: String,
    pub preset: String,
    pub current_phase: String,
    pub phases: Vec<PhaseEntry>,
    pub cache_warm: usize,
    pub cache_cold: usize,
    pub shake_before: usize,
    pub shake_after: usize,
    pub total_functions: usize,
    pub done: bool,
    pub error: Option<String>,
    pub output_lines: Vec<String>,
}

fn phase_color(name: &str) -> Color {
    match name {
        "parse" | "frontend" => Color::Blue,
        "stdlib" => Color::Cyan,
        "shake" | "tree-shake" => Color::Yellow,
        "optimize" | "opt" => Color::Magenta,
        "jit" | "codegen" => Color::Green,
        "link" => Color::Red,
        _ => Color::White,
    }
}

// ── Live TUI ─────────────────────────────────────────────────────

/// A live compilation progress TUI that renders in the terminal.
pub struct ProgressTui {
    state: Arc<Mutex<CompilationState>>,
    start: Instant,
}

impl ProgressTui {
    pub fn new(file: &str, profile: &str, preset: &str) -> Self {
        let state = CompilationState {
            file: file.to_string(),
            profile: profile.to_string(),
            preset: preset.to_string(),
            ..Default::default()
        };
        Self {
            state: Arc::new(Mutex::new(state)),
            start: Instant::now(),
        }
    }

    /// Get a handle to update state from the compilation thread.
    pub fn handle(&self) -> ProgressHandle {
        ProgressHandle {
            state: self.state.clone(),
        }
    }

    /// Run the TUI render loop. Call this from a background thread.
    /// Returns when state.done is set to true.
    pub fn run(&self) -> io::Result<()> {
        if !is_tty() {
            // Non-TTY: just wait for done
            loop {
                std::thread::sleep(Duration::from_millis(100));
                let state = self.state.lock().unwrap();
                if state.done {
                    // Print plain text summary
                    print_plain_summary(&state, self.start.elapsed());
                    return Ok(());
                }
            }
        }

        // Set up terminal
        terminal::enable_raw_mode()?;
        let mut stdout = io::stderr();
        stdout.execute(EnterAlternateScreen)?;
        stdout.execute(cursor::Hide)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let tick_rate = Duration::from_millis(80);
        let mut spinner_idx: usize = 0;

        loop {
            let state = self.state.lock().unwrap().clone();
            let elapsed = self.start.elapsed();

            terminal.draw(|frame| {
                render_progress(frame, &state, elapsed, spinner_idx);
            })?;

            if state.done {
                // Brief pause to show final state
                std::thread::sleep(Duration::from_millis(300));
                break;
            }

            // Handle input (q to quit) or tick
            if event::poll(tick_rate)? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') {
                        break;
                    }
                }
            }

            spinner_idx += 1;
        }

        // Restore terminal
        terminal::disable_raw_mode()?;
        let mut stdout = io::stderr();
        stdout.execute(LeaveAlternateScreen)?;
        stdout.execute(cursor::Show)?;

        // Print final summary to normal stderr after leaving alternate screen
        let state = self.state.lock().unwrap();
        print_styled_summary(&state, self.start.elapsed());

        Ok(())
    }
}

/// Handle for updating compilation state from the main thread.
#[derive(Clone)]
pub struct ProgressHandle {
    state: Arc<Mutex<CompilationState>>,
}

impl ProgressHandle {
    pub fn begin_phase(&self, name: &str) {
        let mut state = self.state.lock().unwrap();
        state.current_phase = name.to_string();
    }

    pub fn end_phase(&self, name: &str, duration_ms: f64) {
        let mut state = self.state.lock().unwrap();
        state.phases.push(PhaseEntry {
            name: name.to_string(),
            duration_ms,
            color: phase_color(name),
        });
        state.current_phase.clear();
    }

    pub fn set_cache_stats(&self, warm: usize, cold: usize) {
        let mut state = self.state.lock().unwrap();
        state.cache_warm = warm;
        state.cache_cold = cold;
    }

    pub fn set_shake_stats(&self, before: usize, after: usize) {
        let mut state = self.state.lock().unwrap();
        state.shake_before = before;
        state.shake_after = after;
    }

    pub fn set_functions(&self, count: usize) {
        let mut state = self.state.lock().unwrap();
        state.total_functions = count;
    }

    pub fn add_output(&self, line: &str) {
        let mut state = self.state.lock().unwrap();
        state.output_lines.push(line.to_string());
    }

    pub fn set_error(&self, err: &str) {
        let mut state = self.state.lock().unwrap();
        state.error = Some(err.to_string());
    }

    pub fn finish(&self) {
        let mut state = self.state.lock().unwrap();
        state.done = true;
    }
}

// ── Render functions ─────────────────────────────────────────────

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn render_progress(frame: &mut Frame, state: &CompilationState, elapsed: Duration, tick: usize) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Length(3), // Spinner / current phase
            Constraint::Min(5),   // Phase timeline + stats
            Constraint::Length(3), // Footer
        ])
        .split(area);

    // ── Header ───────────────────────────────────────────────────
    let elapsed_s = elapsed.as_secs_f64();
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" ▶ ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(&state.file, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(
            format!("[{}]", state.profile),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("  "),
        Span::styled(
            format!("[{}]", state.preset),
            Style::default().fg(Color::DarkGray),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " rayzor ",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )),
    );
    frame.render_widget(header, chunks[0]);

    // ── Spinner + current phase ──────────────────────────────────
    if !state.done {
        let spinner_char = SPINNER[tick % SPINNER.len()];
        let phase_text = if state.current_phase.is_empty() {
            "preparing...".to_string()
        } else {
            state.current_phase.clone()
        };

        let spinner_line = Paragraph::new(Line::from(vec![
            Span::styled(format!("  {} ", spinner_char), Style::default().fg(Color::Cyan)),
            Span::styled(&phase_text, Style::default().fg(Color::White)),
            Span::styled(
                format!("  {:.1}s", elapsed_s),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        frame.render_widget(spinner_line, chunks[1]);
    } else {
        let done_line = Paragraph::new(Line::from(vec![
            Span::styled("  ✓ ", Style::default().fg(Color::Green)),
            Span::styled("done", Style::default().fg(Color::Green)),
            Span::styled(
                format!("  {:.1}s", elapsed_s),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        frame.render_widget(done_line, chunks[1]);
    }

    // ── Phase timeline + stats ───────────────────────────────────
    let inner_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // Phase bars
            Constraint::Min(1),   // Stats
        ])
        .split(chunks[2]);

    // Phase bar chart
    if !state.phases.is_empty() {
        let bars: Vec<Bar> = state
            .phases
            .iter()
            .map(|p| {
                Bar::default()
                    .value(p.duration_ms as u64)
                    .label(Line::from(format!("{} {:.0}ms", p.name, p.duration_ms)))
                    .style(Style::default().fg(p.color))
            })
            .collect();

        let chart = BarChart::default()
            .block(Block::default().title(" phases ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)))
            .data(BarGroup::default().bars(&bars))
            .bar_width(
                (inner_chunks[0].width as usize / state.phases.len().max(1)).max(3) as u16,
            )
            .bar_gap(1)
            .bar_style(Style::default().fg(Color::Cyan))
            .value_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));

        frame.render_widget(chart, inner_chunks[0]);
    }

    // Stats table
    let mut rows = Vec::new();

    if state.cache_warm > 0 || state.cache_cold > 0 {
        rows.push(Row::new(vec![
            Span::styled("cache", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} cached", state.cache_warm),
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!("{} compiled", state.cache_cold),
                Style::default().fg(Color::Yellow),
            ),
        ]));
    }

    if state.shake_before > 0 && state.shake_after < state.shake_before {
        let pct = ((state.shake_before - state.shake_after) as f64
            / state.shake_before as f64
            * 100.0) as usize;
        rows.push(Row::new(vec![
            Span::styled("shake", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{} → {}", state.shake_before, state.shake_after),
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!("{}% stripped", pct),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    if state.total_functions > 0 {
        rows.push(Row::new(vec![
            Span::styled("functions", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", state.total_functions),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("compiled", Style::default().fg(Color::DarkGray)),
        ]));
    }

    if !rows.is_empty() {
        let table = Table::new(
            rows,
            [
                Constraint::Length(12),
                Constraint::Length(20),
                Constraint::Min(10),
            ],
        )
        .block(
            Block::default()
                .title(" stats ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(table, inner_chunks[1]);
    }

    // ── Footer ───────────────────────────────────────────────────
    if let Some(ref err) = state.error {
        let footer = Paragraph::new(Line::from(vec![
            Span::styled(" ✗ ", Style::default().fg(Color::Red)),
            Span::styled(err.as_str(), Style::default().fg(Color::Red)),
        ]));
        frame.render_widget(footer, chunks[3]);
    } else {
        let total_ms: f64 = state.phases.iter().map(|p| p.duration_ms).sum();
        let footer = Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" total: {:.0}ms", total_ms),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw("  "),
            Span::styled("q quit", Style::default().fg(Color::DarkGray)),
        ]))
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(footer, chunks[3]);
    }
}

// ── Fallback output ──────────────────────────────────────────────

fn print_plain_summary(state: &CompilationState, elapsed: Duration) {
    if !state.phases.is_empty() {
        let parts: Vec<String> = state
            .phases
            .iter()
            .map(|p| format!("{} {:.0}ms", p.name, p.duration_ms))
            .collect();
        eprintln!("  {}", parts.join(" → "));
    }
    if state.shake_before > state.shake_after && state.shake_before > 0 {
        let pct = ((state.shake_before - state.shake_after) as f64
            / state.shake_before as f64
            * 100.0) as usize;
        eprintln!(
            "  shake: {} → {} ({}% removed)",
            state.shake_before, state.shake_after, pct
        );
    }
}

fn print_styled_summary(state: &CompilationState, elapsed: Duration) {
    use crossterm::style::Stylize;

    // One-line phase pipeline
    if !state.phases.is_empty() {
        let total_ms: f64 = state.phases.iter().map(|p| p.duration_ms).sum();
        let parts: Vec<String> = state
            .phases
            .iter()
            .map(|p| {
                let c = match p.color {
                    Color::Blue => crossterm::style::Color::Blue,
                    Color::Cyan => crossterm::style::Color::Cyan,
                    Color::Yellow => crossterm::style::Color::Yellow,
                    Color::Magenta => crossterm::style::Color::Magenta,
                    Color::Green => crossterm::style::Color::Green,
                    Color::Red => crossterm::style::Color::Red,
                    _ => crossterm::style::Color::White,
                };
                format!(
                    "{} {}",
                    p.name.as_str().with(crossterm::style::Color::DarkGrey),
                    format!("{:.0}ms", p.duration_ms).with(c)
                )
            })
            .collect();
        eprintln!(
            "  {} total {}",
            parts.join(" → "),
            format!("{:.0}ms", total_ms)
                .with(crossterm::style::Color::White)
                .bold()
        );
    }

    // Shake stats
    if state.shake_before > state.shake_after && state.shake_before > 0 {
        let pct = ((state.shake_before - state.shake_after) as f64
            / state.shake_before as f64
            * 100.0) as usize;
        eprintln!(
            "  {} {} → {} functions ({}% stripped)",
            "shake".with(crossterm::style::Color::DarkGrey),
            state
                .shake_before
                .to_string()
                .with(crossterm::style::Color::DarkGrey),
            state
                .shake_after
                .to_string()
                .with(crossterm::style::Color::Green),
            pct,
        );
    }
}

// ── Simple non-TUI banner (used when verbose is off) ─────────────

/// Print a run banner with styled output (non-TUI, just a single line).
pub fn print_run_banner(file: &str, profile: &str, preset: &str) {
    if is_tty() {
        use crossterm::style::Stylize;
        eprintln!(
            " {} {} {} {}",
            "\u{25B6}".with(crossterm::style::Color::Cyan),
            file.with(crossterm::style::Color::White).bold(),
            format!("[{}]", profile).with(crossterm::style::Color::DarkGrey),
            format!("[{}]", preset).with(crossterm::style::Color::DarkGrey),
        );
    } else {
        eprintln!("Running {} [{}] [preset: {}]...", file, profile, preset);
    }
}

/// Print tree-shake summary (non-TUI fallback).
pub fn print_shake_summary(before: usize, after: usize) {
    if before <= after {
        return;
    }
    let pct = ((before - after) as f64 / before as f64 * 100.0) as usize;
    if is_tty() {
        use crossterm::style::Stylize;
        eprintln!(
            "  {} {} → {} functions ({}% stripped)",
            "shake".with(crossterm::style::Color::DarkGrey),
            before
                .to_string()
                .with(crossterm::style::Color::DarkGrey),
            after.to_string().with(crossterm::style::Color::Green),
            pct,
        );
    } else {
        eprintln!(
            "  shake: {} → {} functions ({}% removed)",
            before, after, pct
        );
    }
}
