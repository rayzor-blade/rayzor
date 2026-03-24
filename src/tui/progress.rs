//! Compilation progress TUI using ratatui.
//!
//! Renders inline progress during compilation, then a final summary
//! with program output in a bordered panel.

use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table, Wrap},
    Terminal,
};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::style::is_tty;

/// Format milliseconds with appropriate precision.
/// >= 10ms: "123ms", >= 1ms: "1.2ms", >= 0.01ms: "0.12ms", < 0.01ms: "5µs"
fn fmt_ms(ms: f64) -> String {
    if ms >= 10.0 {
        format!("{:.0}ms", ms)
    } else if ms >= 1.0 {
        format!("{:.1}ms", ms)
    } else if ms >= 0.01 {
        format!("{:.2}ms", ms)
    } else {
        format!("{:.0}µs", ms * 1000.0)
    }
}

// ── Shared state ─────────────────────────────────────────────────

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
    #[allow(dead_code)]
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
        "compile" | "parse" => Color::Blue,
        "stdlib" => Color::Cyan,
        "shake" | "tree-shake" => Color::Yellow,
        "optimize" | "opt" => Color::Magenta,
        "jit" | "codegen" => Color::Green,
        _ => Color::White,
    }
}

// ── Spinner constants ────────────────────────────────────────────

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ── ProgressTui ──────────────────────────────────────────────────

pub struct ProgressTui {
    state: Arc<Mutex<CompilationState>>,
    start: Instant,
}

impl ProgressTui {
    pub fn new(file: &str, profile: &str, preset: &str) -> Self {
        Self {
            state: Arc::new(Mutex::new(CompilationState {
                file: file.to_string(),
                profile: profile.to_string(),
                preset: preset.to_string(),
                ..Default::default()
            })),
            start: Instant::now(),
        }
    }

    pub fn handle(&self) -> ProgressHandle {
        ProgressHandle {
            state: self.state.clone(),
        }
    }

    /// Run the spinner loop during compilation. Returns when done.
    pub fn run(&self) -> io::Result<()> {
        if !is_tty() {
            loop {
                std::thread::sleep(Duration::from_millis(100));
                if self.state.lock().unwrap().done {
                    let state = self.state.lock().unwrap();
                    print_plain_summary(&state, self.start.elapsed());
                    return Ok(());
                }
            }
        }

        let mut stderr = io::stderr();
        stderr.execute(cursor::Hide)?;

        let mut tick: usize = 0;
        loop {
            let state = self.state.lock().unwrap().clone();
            let elapsed = self.start.elapsed();

            // Overwrite spinner line
            let frame_char = SPINNER[tick % SPINNER.len()];
            let phase = if state.current_phase.is_empty() {
                "preparing"
            } else {
                &state.current_phase
            };
            eprint!(
                "\r  {} {}  {:.1}s  ",
                frame_char,
                phase,
                elapsed.as_secs_f64()
            );
            let _ = stderr.flush();

            if state.done {
                // Clear the spinner line
                eprint!("\r{}\r", " ".repeat(60));
                let _ = stderr.flush();
                break;
            }

            std::thread::sleep(Duration::from_millis(80));
            tick += 1;
        }

        stderr.execute(cursor::Show)?;
        Ok(())
    }

    /// Render the final summary using ratatui inline viewport.
    /// Call this AFTER execution is complete and output has been captured.
    pub fn render_final(&self) -> io::Result<()> {
        if !is_tty() {
            return Ok(());
        }

        let state = self.state.lock().unwrap().clone();
        let elapsed = self.start.elapsed();

        // Calculate height dynamically based on actual content
        let phase_rows = state.phases.len().max(1) as u16;
        let has_stats = state.cache_warm > 0 || state.cache_cold > 0;
        let stats_rows = if has_stats { 1 } else { 0 };
        let output_lines = state.output_lines.len() as u16;
        let total_height = (2 // header + status
            + phase_rows
            + stats_rows
            + 2  // output panel borders
            + output_lines.max(1))
        .min(30); // cap height

        // Use ratatui inline viewport
        terminal::enable_raw_mode()?;
        let backend = CrosstermBackend::new(io::stderr());
        let mut terminal = Terminal::with_options(
            backend,
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Inline(total_height),
            },
        )?;

        terminal.draw(|frame| {
            render_final_frame(frame, &state, elapsed);
        })?;

        terminal::disable_raw_mode()?;
        eprintln!(); // newline after viewport

        Ok(())
    }

    /// Run a full interactive TUI session (alternate screen, scrollable, searchable).
    /// Stays alive until user presses `q` or Esc.
    pub fn run_interactive(&self) -> io::Result<()> {
        if !is_tty() {
            // Fallback: just print output
            let state = self.state.lock().unwrap();
            for line in &state.output_lines {
                println!("{}", line);
            }
            return Ok(());
        }

        let state = self.state.lock().unwrap().clone();
        let elapsed = self.start.elapsed();

        // Enter alternate screen
        terminal::enable_raw_mode()?;
        let mut stderr = io::stderr();
        stderr.execute(EnterAlternateScreen)?;
        stderr.execute(cursor::Hide)?;

        let backend = CrosstermBackend::new(stderr);
        let mut terminal = Terminal::new(backend)?;

        let mut app = InteractiveApp {
            state,
            elapsed,
            scroll_offset: 0,
            search_query: String::new(),
            search_mode: false,
            show_stats: true,
        };

        loop {
            terminal.draw(|frame| app.render(frame))?;

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if app.search_mode {
                        match key.code {
                            KeyCode::Esc => {
                                app.search_mode = false;
                                app.search_query.clear();
                            }
                            KeyCode::Enter => {
                                app.search_mode = false;
                                // Jump to first match
                                app.jump_to_match();
                            }
                            KeyCode::Backspace => {
                                app.search_query.pop();
                            }
                            KeyCode::Char(c) => {
                                app.search_query.push(c);
                            }
                            _ => {}
                        }
                    } else {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => break,
                            KeyCode::Char('/') => {
                                app.search_mode = true;
                                app.search_query.clear();
                            }
                            KeyCode::Char('s') => app.show_stats = !app.show_stats,
                            KeyCode::Up | KeyCode::Char('k') => {
                                app.scroll_offset = app.scroll_offset.saturating_sub(1);
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                app.scroll_offset = (app.scroll_offset + 1)
                                    .min(app.state.output_lines.len().saturating_sub(1));
                            }
                            KeyCode::PageUp => {
                                app.scroll_offset = app.scroll_offset.saturating_sub(20);
                            }
                            KeyCode::PageDown => {
                                app.scroll_offset = (app.scroll_offset + 20)
                                    .min(app.state.output_lines.len().saturating_sub(1));
                            }
                            KeyCode::Home | KeyCode::Char('g') => {
                                app.scroll_offset = 0;
                            }
                            KeyCode::End | KeyCode::Char('G') => {
                                app.scroll_offset = app.state.output_lines.len().saturating_sub(1);
                            }
                            KeyCode::Char('n') => {
                                // Next search match
                                if !app.search_query.is_empty() {
                                    app.jump_to_next_match();
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Restore terminal
        terminal::disable_raw_mode()?;
        let mut stderr = io::stderr();
        stderr.execute(LeaveAlternateScreen)?;
        stderr.execute(cursor::Show)?;

        Ok(())
    }
}

// ── Interactive App ──────────────────────────────────────────────

struct InteractiveApp {
    state: CompilationState,
    elapsed: Duration,
    scroll_offset: usize,
    search_query: String,
    search_mode: bool,
    show_stats: bool,
}

impl InteractiveApp {
    fn render(&self, frame: &mut ratatui::Frame) {
        let area = frame.area();

        let stats_height = if self.show_stats {
            let phase_rows = self.state.phases.len() as u16;
            3 + phase_rows // header + status + phases + border
        } else {
            0
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(stats_height),
                Constraint::Min(5),    // Output
                Constraint::Length(1), // Status bar
            ])
            .split(area);

        // ── Stats panel (toggleable with 's') ────────────────
        if self.show_stats {
            let stats_area = chunks[0];
            let stats_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1), // Header
                    Constraint::Length(1), // Status
                    Constraint::Min(1),    // Phase bars
                ])
                .split(stats_area);

            // Header
            let header = Line::from(vec![
                Span::styled(
                    " ▶ rayzor ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    self.state.file.as_str(),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("[{}]", self.state.profile),
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            frame.render_widget(Paragraph::new(header), stats_chunks[0]);

            // Status
            let total_ms = self.elapsed.as_secs_f64() * 1000.0;
            let mut status_spans = vec![
                Span::styled(
                    " ✓ ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{:.0}ms", total_ms),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {} functions", self.state.total_functions),
                    Style::default().fg(Color::DarkGray),
                ),
            ];
            if self.state.shake_before > self.state.shake_after && self.state.shake_before > 0 {
                let pct = ((self.state.shake_before - self.state.shake_after) as f64
                    / self.state.shake_before as f64
                    * 100.0) as usize;
                status_spans.push(Span::styled(
                    format!(
                        "  ({} → {}, {}% stripped)",
                        self.state.shake_before, self.state.shake_after, pct
                    ),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            frame.render_widget(Paragraph::new(Line::from(status_spans)), stats_chunks[1]);

            // Phase bars
            if !self.state.phases.is_empty() {
                let max_ms = self
                    .state
                    .phases
                    .iter()
                    .map(|p| p.duration_ms)
                    .fold(0.0_f64, f64::max);
                let bar_max = (stats_chunks[2].width as usize).saturating_sub(22).min(40);
                let rows: Vec<Row> = self
                    .state
                    .phases
                    .iter()
                    .map(|p| {
                        let frac = if max_ms > 0.0 {
                            p.duration_ms / max_ms
                        } else {
                            0.0
                        };
                        let bar_len = if p.duration_ms < 0.001 {
                            0
                        } else {
                            (frac * bar_max as f64).round().max(1.0) as usize
                        };
                        Row::new(vec![
                            Line::from(Span::styled(
                                format!(" {:>9} ", p.name),
                                Style::default().fg(Color::White),
                            )),
                            Line::from(vec![
                                Span::styled(
                                    "\u{2588}".repeat(bar_len),
                                    Style::default().fg(p.color),
                                ),
                                Span::styled(
                                    format!(" {}", fmt_ms(p.duration_ms)),
                                    Style::default().fg(if p.duration_ms == max_ms {
                                        Color::White
                                    } else {
                                        Color::DarkGray
                                    }),
                                ),
                            ]),
                        ])
                    })
                    .collect();
                let table = Table::new(rows, [Constraint::Length(11), Constraint::Min(10)]);
                frame.render_widget(table, stats_chunks[2]);
            }
        }

        // ── Output panel (scrollable) ────────────────────────
        let output_area = chunks[1];
        let visible_height = output_area.height.saturating_sub(2) as usize; // -2 for borders

        let output_lines: Vec<Line> = if self.state.output_lines.is_empty() {
            vec![Line::from(Span::styled(
                "(no output)",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            self.state
                .output_lines
                .iter()
                .skip(self.scroll_offset)
                .take(visible_height)
                .map(|l| {
                    if !self.search_query.is_empty() && l.contains(&self.search_query) {
                        // Highlight matching lines
                        highlight_search_line(l, &self.search_query)
                    } else {
                        Line::from(Span::raw(l.as_str()))
                    }
                })
                .collect()
        };

        let total_lines = self.state.output_lines.len();
        let scroll_info = if total_lines > visible_height {
            format!(" {}/{} ", self.scroll_offset + 1, total_lines)
        } else {
            String::new()
        };

        let output_block = Block::default()
            .title(Span::styled(
                " output ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ))
            .title_bottom(
                Line::from(Span::styled(
                    scroll_info,
                    Style::default().fg(Color::DarkGray),
                ))
                .right_aligned(),
            )
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        frame.render_widget(
            Paragraph::new(output_lines).block(output_block),
            output_area,
        );

        // ── Status bar ───────────────────────────────────────
        let status_bar = if self.search_mode {
            Line::from(vec![
                Span::styled(
                    " /",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    self.search_query.as_str(),
                    Style::default().fg(Color::White),
                ),
                Span::styled("▏", Style::default().fg(Color::Yellow)),
            ])
        } else {
            Line::from(vec![
                Span::styled(
                    " q",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" quit  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "/",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" search  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "s",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" stats  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "↑↓",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" scroll  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "n",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" next match", Style::default().fg(Color::DarkGray)),
            ])
        };
        frame.render_widget(Paragraph::new(status_bar), chunks[2]);
    }

    fn jump_to_match(&mut self) {
        if self.search_query.is_empty() {
            return;
        }
        for (i, line) in self.state.output_lines.iter().enumerate() {
            if line.contains(&self.search_query) {
                self.scroll_offset = i;
                return;
            }
        }
    }

    fn jump_to_next_match(&mut self) {
        if self.search_query.is_empty() {
            return;
        }
        let start = self.scroll_offset + 1;
        for i in start..self.state.output_lines.len() {
            if self.state.output_lines[i].contains(&self.search_query) {
                self.scroll_offset = i;
                return;
            }
        }
        // Wrap around
        for i in 0..start.min(self.state.output_lines.len()) {
            if self.state.output_lines[i].contains(&self.search_query) {
                self.scroll_offset = i;
                return;
            }
        }
    }
}

/// Highlight search matches within a line.
fn highlight_search_line<'a>(line: &'a str, query: &str) -> Line<'a> {
    let mut spans = Vec::new();
    let mut remaining = line;
    while let Some(pos) = remaining.find(query) {
        if pos > 0 {
            spans.push(Span::raw(&remaining[..pos]));
        }
        spans.push(Span::styled(
            &remaining[pos..pos + query.len()],
            Style::default().fg(Color::Black).bg(Color::Yellow),
        ));
        remaining = &remaining[pos + query.len()..];
    }
    if !remaining.is_empty() {
        spans.push(Span::raw(remaining));
    }
    Line::from(spans)
}

// ── Handle ───────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ProgressHandle {
    state: Arc<Mutex<CompilationState>>,
}

impl ProgressHandle {
    pub fn begin_phase(&self, name: &str) {
        self.state.lock().unwrap().current_phase = name.to_string();
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

    #[allow(dead_code)]
    pub fn set_cache_stats(&self, warm: usize, cold: usize) {
        let mut s = self.state.lock().unwrap();
        s.cache_warm = warm;
        s.cache_cold = cold;
    }

    pub fn set_shake_stats(&self, before: usize, after: usize) {
        let mut s = self.state.lock().unwrap();
        s.shake_before = before;
        s.shake_after = after;
    }

    pub fn set_functions(&self, count: usize) {
        self.state.lock().unwrap().total_functions = count;
    }

    pub fn add_output_line(&self, line: String) {
        self.state.lock().unwrap().output_lines.push(line);
    }

    #[allow(dead_code)]
    pub fn set_error(&self, err: &str) {
        self.state.lock().unwrap().error = Some(err.to_string());
    }

    pub fn finish(&self) {
        self.state.lock().unwrap().done = true;
    }
}

// ── Final frame render ───────────────────────────────────────────

fn render_final_frame(frame: &mut ratatui::Frame, state: &CompilationState, elapsed: Duration) {
    let area = frame.area();

    // Dynamic heights based on content
    let phase_rows = state.phases.len().max(1) as u16;
    let has_stats = state.cache_warm > 0 || state.cache_cold > 0;
    let stats_height = if has_stats { 1 } else { 0 };
    let output_height = state.output_lines.len().max(1) as u16 + 2; // +2 for border
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),                  // Header
            Constraint::Length(1),                  // Status
            Constraint::Length(phase_rows),         // Phase bars (1 row per phase)
            Constraint::Length(stats_height),       // Stats (0 if no cache info)
            Constraint::Min(output_height.min(15)), // Output panel
        ])
        .split(area);

    // ── Header ───────────────────────────────────────────────
    let header = Line::from(vec![
        Span::styled(
            " ▶ rayzor ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            state.file.as_str(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("[{}]", state.profile),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(Paragraph::new(header), chunks[0]);

    // ── Status line ──────────────────────────────────────────
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let status = if let Some(ref err) = state.error {
        Line::from(vec![
            Span::styled(
                " ✗ ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(err.as_str(), Style::default().fg(Color::Red)),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                " ✓ ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:.0}ms", total_ms),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {} functions", state.total_functions),
                Style::default().fg(Color::DarkGray),
            ),
            if state.shake_before > state.shake_after && state.shake_before > 0 {
                let pct = ((state.shake_before - state.shake_after) as f64
                    / state.shake_before as f64
                    * 100.0) as usize;
                Span::styled(
                    format!(
                        "  ({} → {}, {}% stripped)",
                        state.shake_before, state.shake_after, pct
                    ),
                    Style::default().fg(Color::DarkGray),
                )
            } else {
                Span::raw("")
            },
        ])
    };
    frame.render_widget(Paragraph::new(status), chunks[1]);

    // ── Phase bars: label | bar | time ─────────────────────────
    if !state.phases.is_empty() {
        let max_ms = state
            .phases
            .iter()
            .map(|p| p.duration_ms)
            .fold(0.0_f64, f64::max);
        let bar_max_width = (chunks[2].width as usize).saturating_sub(22).min(40); // cap bar length

        let rows: Vec<Row> = state
            .phases
            .iter()
            .map(|p| {
                let frac = if max_ms > 0.0 {
                    p.duration_ms / max_ms
                } else {
                    0.0
                };
                let bar_len = if p.duration_ms < 0.5 {
                    0
                } else {
                    (frac * bar_max_width as f64).round().max(1.0) as usize
                };
                let bar_str = "\u{2588}".repeat(bar_len);
                let time_color = if p.duration_ms == max_ms {
                    Color::White
                } else {
                    Color::DarkGray
                };
                let time_mod = if p.duration_ms == max_ms {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                };

                Row::new(vec![
                    Line::from(Span::styled(
                        format!(" {:>9} ", p.name),
                        Style::default().fg(Color::White),
                    )),
                    Line::from(vec![
                        Span::styled(bar_str, Style::default().fg(p.color)),
                        Span::styled(
                            format!(" {}", fmt_ms(p.duration_ms)),
                            Style::default().fg(time_color).add_modifier(time_mod),
                        ),
                    ]),
                ])
            })
            .collect();

        let table = Table::new(rows, [Constraint::Length(11), Constraint::Min(10)]);
        frame.render_widget(table, chunks[2]);
    }

    // ── Stats row ────────────────────────────────────────────
    let mut stat_spans: Vec<Span> = Vec::new();
    if state.cache_warm > 0 || state.cache_cold > 0 {
        stat_spans.push(Span::styled(
            " cache ",
            Style::default().fg(Color::DarkGray),
        ));
        stat_spans.push(Span::styled(
            format!("{} hit", state.cache_warm),
            Style::default().fg(Color::Green),
        ));
        stat_spans.push(Span::styled("  ", Style::default()));
        stat_spans.push(Span::styled(
            format!("{} miss", state.cache_cold),
            Style::default().fg(Color::Yellow),
        ));
    }
    if !stat_spans.is_empty() {
        frame.render_widget(Paragraph::new(Line::from(stat_spans)), chunks[3]);
    }

    // ── Output panel ─────────────────────────────────────────
    let output_text: Vec<Line> = if state.output_lines.is_empty() {
        vec![Line::from(Span::styled(
            "(no output)",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        state
            .output_lines
            .iter()
            .map(|l| Line::from(Span::raw(l.as_str())))
            .collect()
    };

    let output_panel = Paragraph::new(output_text)
        .block(
            Block::default()
                .title(Span::styled(
                    " output ",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(output_panel, chunks[4]);
}

// ── Fallback ─────────────────────────────────────────────────────

fn print_plain_summary(state: &CompilationState, _elapsed: Duration) {
    if !state.phases.is_empty() {
        let parts: Vec<String> = state
            .phases
            .iter()
            .map(|p| format!("{} {}", p.name, fmt_ms(p.duration_ms)))
            .collect();
        eprintln!("  {}", parts.join(" → "));
    }
    if state.shake_before > state.shake_after && state.shake_before > 0 {
        let pct = ((state.shake_before - state.shake_after) as f64 / state.shake_before as f64
            * 100.0) as usize;
        eprintln!(
            "  shake: {} → {} ({}% removed)",
            state.shake_before, state.shake_after, pct
        );
    }
    // Print output
    for line in &state.output_lines {
        println!("{}", line);
    }
}

/// Print a simple run banner (non-TUI fallback).
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
#[allow(dead_code)]
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
            before.to_string().with(crossterm::style::Color::DarkGrey),
            after.to_string().with(crossterm::style::Color::Green),
            pct,
        );
    } else {
        eprintln!("  shake: {} → {} ({}% removed)", before, after, pct);
    }
}
