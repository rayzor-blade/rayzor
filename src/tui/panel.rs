//! Reusable inline TUI panel for CLI command output.
//!
//! Renders a bordered ratatui panel inline (no alternate screen).
//! Falls back to plain text when not a TTY.

use ratatui::{
    backend::CrosstermBackend,
    layout::Constraint,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Terminal,
};
use std::io;

use super::style::is_tty;

/// A row in an info panel: label + value, optionally colored.
pub struct InfoRow {
    pub label: String,
    pub value: String,
    pub value_color: Color,
}

impl InfoRow {
    pub fn new(label: &str, value: &str) -> Self {
        Self {
            label: label.to_string(),
            value: value.to_string(),
            value_color: Color::White,
        }
    }

    pub fn colored(label: &str, value: &str, color: Color) -> Self {
        Self {
            label: label.to_string(),
            value: value.to_string(),
            value_color: color,
        }
    }
}

/// Render an info panel with a title, rows, and optional footer.
pub fn render_info_panel(
    title: &str,
    rows: &[InfoRow],
    footer: Option<&str>,
) -> io::Result<()> {
    if !is_tty() {
        // Plain text fallback
        println!("{}", title);
        for row in rows {
            println!("  {:16} {}", row.label, row.value);
        }
        if let Some(f) = footer {
            println!("  {}", f);
        }
        return Ok(());
    }

    let row_count = rows.len() as u16;
    let height = (row_count + 3).min(25); // +3 for borders + footer

    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(io::stderr());
    let mut terminal = Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(height),
        },
    )?;

    terminal.draw(|frame| {
        let tui_rows: Vec<Row> = rows
            .iter()
            .map(|r| {
                Row::new(vec![
                    Span::styled(
                        format!(" {}", r.label),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(r.value.as_str(), Style::default().fg(r.value_color)),
                ])
            })
            .collect();

        let mut block = Block::default()
            .title(Span::styled(
                format!(" {} ", title),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        if let Some(f) = footer {
            block = block.title_bottom(
                Line::from(Span::styled(
                    format!(" {} ", f),
                    Style::default().fg(Color::DarkGray),
                ))
                .right_aligned(),
            );
        }

        let table = Table::new(
            tui_rows,
            [Constraint::Length(18), Constraint::Min(20)],
        )
        .block(block);

        frame.render_widget(table, frame.area());
    })?;

    crossterm::terminal::disable_raw_mode()?;
    eprintln!();

    Ok(())
}

/// Render a simple message panel (title + paragraph text).
pub fn render_message_panel(title: &str, lines: &[&str]) -> io::Result<()> {
    if !is_tty() {
        println!("{}", title);
        for line in lines {
            println!("  {}", line);
        }
        return Ok(());
    }

    let height = (lines.len() as u16 + 3).min(20);

    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(io::stderr());
    let mut terminal = Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(height),
        },
    )?;

    terminal.draw(|frame| {
        let text: Vec<Line> = lines
            .iter()
            .map(|l| Line::from(Span::styled(*l, Style::default().fg(Color::White))))
            .collect();

        let panel = Paragraph::new(text).block(
            Block::default()
                .title(Span::styled(
                    format!(" {} ", title),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

        frame.render_widget(panel, frame.area());
    })?;

    crossterm::terminal::disable_raw_mode()?;
    eprintln!();

    Ok(())
}
