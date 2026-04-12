//! Interactive MIR dump viewer using ratatui.
//!
//! Full-screen TUI with:
//! - Left panel: function list (scrollable, filterable)
//! - Right panel: MIR code (syntax highlighted, scrollable)
//! - / search with highlighting
//! - Tab to switch panels
//! - q to quit

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
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Terminal,
};
use std::io;
use std::time::Duration;

/// A parsed MIR function for the viewer.
struct MirFunction {
    name: String,
    #[allow(dead_code)]
    id: String,
    code: String,
    line_count: usize,
}

/// The interactive MIR viewer app.
pub fn run_mir_viewer(mir_text: &str, module_name: &str, total_functions: usize) -> io::Result<()> {
    // Parse functions from MIR text
    let functions = parse_mir_functions(mir_text);

    if functions.is_empty() {
        eprintln!("No functions to display.");
        return Ok(());
    }

    // Enter TUI
    terminal::enable_raw_mode()?;
    let mut stderr = io::stderr();
    stderr.execute(EnterAlternateScreen)?;
    stderr.execute(cursor::Hide)?;

    let backend = CrosstermBackend::new(stderr);
    let mut terminal = Terminal::new(backend)?;

    let mut app = MirViewerApp {
        functions,
        module_name: module_name.to_string(),
        total_functions,
        selected_func: 0,
        func_list_state: ListState::default(),
        code_scroll: 0,
        search_query: String::new(),
        search_mode: false,
        active_panel: Panel::FuncList,
    };
    app.func_list_state.select(Some(0));

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
                        KeyCode::Tab => {
                            app.active_panel = match app.active_panel {
                                Panel::FuncList => Panel::Code,
                                Panel::Code => Panel::FuncList,
                            };
                        }
                        KeyCode::Char('/') => {
                            app.search_mode = true;
                            app.search_query.clear();
                        }
                        KeyCode::Up | KeyCode::Char('k') => match app.active_panel {
                            Panel::FuncList => app.select_prev(),
                            Panel::Code => app.code_scroll = app.code_scroll.saturating_sub(1),
                        },
                        KeyCode::Down | KeyCode::Char('j') => match app.active_panel {
                            Panel::FuncList => app.select_next(),
                            Panel::Code => {
                                let max = app.current_line_count().saturating_sub(1);
                                app.code_scroll = (app.code_scroll + 1).min(max);
                            }
                        },
                        KeyCode::PageUp => {
                            app.code_scroll = app.code_scroll.saturating_sub(20);
                        }
                        KeyCode::PageDown => {
                            let max = app.current_line_count().saturating_sub(1);
                            app.code_scroll = (app.code_scroll + 20).min(max);
                        }
                        KeyCode::Home => app.code_scroll = 0,
                        KeyCode::End => {
                            app.code_scroll = app.current_line_count().saturating_sub(1);
                        }
                        KeyCode::Enter if app.active_panel == Panel::FuncList => {
                            app.code_scroll = 0;
                            app.active_panel = Panel::Code;
                        }
                        KeyCode::Char('n') => app.jump_to_next_match(),
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

#[derive(PartialEq)]
enum Panel {
    FuncList,
    Code,
}

struct MirViewerApp {
    functions: Vec<MirFunction>,
    module_name: String,
    total_functions: usize,
    selected_func: usize,
    func_list_state: ListState,
    code_scroll: usize,
    search_query: String,
    search_mode: bool,
    active_panel: Panel,
}

impl MirViewerApp {
    fn select_next(&mut self) {
        if self.selected_func < self.functions.len() - 1 {
            self.selected_func += 1;
            self.func_list_state.select(Some(self.selected_func));
            self.code_scroll = 0;
        }
    }

    fn select_prev(&mut self) {
        if self.selected_func > 0 {
            self.selected_func -= 1;
            self.func_list_state.select(Some(self.selected_func));
            self.code_scroll = 0;
        }
    }

    fn current_line_count(&self) -> usize {
        self.functions
            .get(self.selected_func)
            .map(|f| f.line_count)
            .unwrap_or(0)
    }

    fn jump_to_next_match(&mut self) {
        if self.search_query.is_empty() {
            return;
        }
        if let Some(func) = self.functions.get(self.selected_func) {
            let lines: Vec<&str> = func.code.lines().collect();
            let start = self.code_scroll + 1;
            // Search forward from current position
            for (i, line) in lines.iter().enumerate().skip(start) {
                if line.contains(self.search_query.as_str()) {
                    self.code_scroll = i;
                    return;
                }
            }
            // Wrap around
            for (i, line) in lines.iter().enumerate().take(start.min(lines.len())) {
                if line.contains(self.search_query.as_str()) {
                    self.code_scroll = i;
                    return;
                }
            }
        }
    }

    fn render(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();

        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header
                Constraint::Min(5),    // Content
                Constraint::Length(1), // Status bar
            ])
            .split(area);

        // ── Header ───────────────────────────────────────────
        let header = Line::from(vec![
            Span::styled(
                " MIR ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {}  ", self.module_name),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!("{} functions", self.total_functions),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
        frame.render_widget(Paragraph::new(header), main_chunks[0]);

        // ── Content: function list | code ────────────────────
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(30), // Function list
                Constraint::Min(40),    // Code
            ])
            .split(main_chunks[1]);

        // Function list
        let func_border_style = if self.active_panel == Panel::FuncList {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let func_items: Vec<ListItem> = self
            .functions
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let style = if i == self.selected_func {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else if !self.search_query.is_empty() && f.name.contains(&self.search_query) {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::White)
                };
                ListItem::new(Span::styled(format!(" {}", f.name), style))
            })
            .collect();

        let func_list = List::new(func_items)
            .block(
                Block::default()
                    .title(Span::styled(
                        " functions ",
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL)
                    .border_style(func_border_style),
            )
            .highlight_style(Style::default().bg(Color::DarkGray));

        frame.render_stateful_widget(func_list, content_chunks[0], &mut self.func_list_state);

        // Code panel
        let code_border_style = if self.active_panel == Panel::Code {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let visible_height = content_chunks[1].height.saturating_sub(2) as usize;

        let code_lines: Vec<Line> = if let Some(func) = self.functions.get(self.selected_func) {
            func.code
                .lines()
                .enumerate()
                .skip(self.code_scroll)
                .take(visible_height)
                .map(|(line_num, line)| {
                    let line_no = Span::styled(
                        format!("{:>4} ", line_num + 1),
                        Style::default().fg(Color::DarkGray),
                    );

                    if !self.search_query.is_empty() && line.contains(&self.search_query) {
                        let highlighted = highlight_mir_line_with_search(line, &self.search_query);
                        let mut spans = vec![line_no];
                        spans.extend(highlighted);
                        Line::from(spans)
                    } else {
                        let highlighted = highlight_mir_line(line);
                        let mut spans = vec![line_no];
                        spans.extend(highlighted);
                        Line::from(spans)
                    }
                })
                .collect()
        } else {
            vec![Line::from(Span::styled(
                "(no function selected)",
                Style::default().fg(Color::DarkGray),
            ))]
        };

        let total_lines = self.current_line_count();
        let scroll_info = if total_lines > visible_height {
            format!(" {}/{} ", self.code_scroll + 1, total_lines)
        } else {
            String::new()
        };

        let func_name = self
            .functions
            .get(self.selected_func)
            .map(|f| f.name.as_str())
            .unwrap_or("");

        let code_block = Block::default()
            .title(Span::styled(
                format!(" {} ", func_name),
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
            .border_style(code_border_style);

        frame.render_widget(
            Paragraph::new(code_lines).block(code_block),
            content_chunks[1],
        );

        // ── Status bar ───────────────────────────────────────
        let status = if self.search_mode {
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
                    "Tab",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" switch  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "/",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" search  ", Style::default().fg(Color::DarkGray)),
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
                Span::styled(" next  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Enter",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" view", Style::default().fg(Color::DarkGray)),
            ])
        };
        frame.render_widget(Paragraph::new(status), main_chunks[2]);
    }
}

// ── MIR parsing ──────────────────────────────────────────────────

fn parse_mir_functions(mir_text: &str) -> Vec<MirFunction> {
    let mut functions = Vec::new();
    let mut current_id = String::new();
    let mut current_name = String::new();
    let mut current_lines = Vec::new();

    for line in mir_text.lines() {
        // Function header: "; fn2 = @main"
        if line.starts_with("; fn") && line.contains(" = @") {
            // Save previous function
            if !current_name.is_empty() {
                let code = current_lines.join("\n");
                let line_count = current_lines.len();
                functions.push(MirFunction {
                    name: current_name.clone(),
                    id: current_id.clone(),
                    code,
                    line_count,
                });
                current_lines.clear();
            }

            // Parse new function
            if let Some(at_pos) = line.find('@') {
                current_name = line[at_pos + 1..].to_string();
            }
            if let Some(eq_pos) = line.find(" = ") {
                current_id = line[2..eq_pos].to_string();
            }
        } else if line.starts_with("fn @")
            || line.starts_with("; Module")
            || line.starts_with("; Functions")
        {
            // Function body start or module header — add to current lines
            current_lines.push(line.to_string());
        } else {
            current_lines.push(line.to_string());
        }
    }

    // Save last function
    if !current_name.is_empty() {
        let code = current_lines.join("\n");
        let line_count = current_lines.len();
        functions.push(MirFunction {
            name: current_name,
            id: current_id,
            code,
            line_count,
        });
    }

    functions
}

// ── Syntax highlighting ──────────────────────────────────────────

fn highlight_mir_line(line: &str) -> Vec<Span<'_>> {
    let trimmed = line.trim();

    // Comment lines
    if trimmed.starts_with(';') {
        return vec![Span::styled(line, Style::default().fg(Color::DarkGray))];
    }

    // Block labels: "  bb0:" or "  bb1:"
    if trimmed.starts_with("bb") && trimmed.ends_with(':') {
        return vec![Span::styled(
            line,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )];
    }

    // Function signatures: "fn @name(...) -> type {"
    if trimmed.starts_with("fn @") {
        return vec![Span::styled(
            line,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
    }

    // Keywords and opcodes
    let mut spans = Vec::new();
    let mut remaining = line;

    // Simple token-level highlighting
    if let Some(eq_pos) = remaining.find(" = ") {
        // Register assignment: "$0 = ..."
        let (dest, rest) = remaining.split_at(eq_pos);
        spans.push(Span::styled(dest, Style::default().fg(Color::Green)));
        spans.push(Span::styled(" = ", Style::default().fg(Color::DarkGray)));
        remaining = &rest[3..];

        // Highlight the opcode
        let opcode_end = remaining.find(' ').unwrap_or(remaining.len());
        let (opcode, rest) = remaining.split_at(opcode_end);
        let opcode_color = match opcode {
            "const" => Color::Magenta,
            "call" => Color::Cyan,
            "load" | "store" => Color::Yellow,
            "gep" => Color::Yellow,
            "cast" | "bitcast" => Color::Blue,
            "add" | "sub" | "mul" | "div" | "mod" | "cmp" => Color::Red,
            "ret" | "br" | "condbr" => Color::Yellow,
            "phi" => Color::Magenta,
            _ => Color::White,
        };
        spans.push(Span::styled(opcode, Style::default().fg(opcode_color)));
        spans.push(Span::styled(rest, Style::default().fg(Color::White)));
    } else if trimmed.starts_with("ret ") || trimmed == "ret" {
        spans.push(Span::styled(
            line,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    } else if trimmed.starts_with("call ") {
        spans.push(Span::styled(line, Style::default().fg(Color::Cyan)));
    } else if trimmed.starts_with("store ") {
        spans.push(Span::styled(line, Style::default().fg(Color::Yellow)));
    } else {
        spans.push(Span::styled(line, Style::default().fg(Color::White)));
    }

    spans
}

fn highlight_mir_line_with_search<'a>(line: &'a str, query: &str) -> Vec<Span<'a>> {
    let mut spans = Vec::new();
    let mut remaining = line;

    while let Some(pos) = remaining.find(query) {
        if pos > 0 {
            spans.push(Span::styled(
                &remaining[..pos],
                Style::default().fg(Color::White),
            ));
        }
        spans.push(Span::styled(
            &remaining[pos..pos + query.len()],
            Style::default().fg(Color::Black).bg(Color::Yellow),
        ));
        remaining = &remaining[pos + query.len()..];
    }
    if !remaining.is_empty() {
        spans.push(Span::styled(remaining, Style::default().fg(Color::White)));
    }

    spans
}
