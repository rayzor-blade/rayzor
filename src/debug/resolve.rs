//! `rayzor debug resolve` — turn hex PCs into Haxe function names + lines.
//!
//! Reads PCs from positional args or stdin (one per line, with or without
//! `0x` prefix). Looks up each PC in the JIT symbol CSV (defaults to the
//! most recent `/tmp/rayzor_jit_symbols.*.csv`). Joins against the file
//! table CSV (`/tmp/rayzor_file_table.csv`) to expand the numeric
//! `file_id` column into a real source path.

use super::DebugCommands;
use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

pub fn execute(cmd: DebugCommands) -> Result<()> {
    let DebugCommands::Resolve {
        pcs,
        csv,
        file_table,
    } = cmd
    else {
        return Err(anyhow!(
            "debug::resolve::execute called with non-Resolve variant"
        ));
    };

    let csv_path = csv.map(Ok).unwrap_or_else(latest_jit_csv)?;
    let file_table_path = file_table.unwrap_or_else(|| PathBuf::from("/tmp/rayzor_file_table.csv"));

    println!("[resolve] jit-map:    {}", csv_path.display());
    println!(
        "[resolve] file-table: {} ({})",
        file_table_path.display(),
        if file_table_path.exists() {
            "present"
        } else {
            "MISSING - line numbers won't expand to paths"
        }
    );

    let mut symbols = load_symbols(&csv_path)?;
    symbols.sort_by_key(|s| s.start);
    let file_map = load_file_table(&file_table_path).unwrap_or_default();

    let pc_strings: Vec<String> = if !pcs.is_empty() {
        pcs
    } else {
        read_stdin_lines()?
    };

    println!();
    for pc_str in pc_strings {
        let pc_str = pc_str.trim();
        if pc_str.is_empty() {
            continue;
        }
        let pc = match parse_hex(pc_str) {
            Some(v) => v,
            None => {
                println!("{pc_str:20}  PARSE ERROR");
                continue;
            }
        };
        match resolve_pc(pc, &symbols, &file_map) {
            Some(r) => println!("0x{pc:016x}  {}", r.format()),
            None => println!("0x{pc:016x}  unresolved (no JIT range covers this PC)"),
        }
    }
    Ok(())
}

struct Symbol {
    start: u64,
    end: u64,
    file_id: u32,
    line: u32,
    qname: String,
}

struct Resolved<'a> {
    sym: &'a Symbol,
    offset: u64,
    file_path: Option<&'a str>,
}

impl<'a> Resolved<'a> {
    fn format(&self) -> String {
        let mut out = format!("{}  +0x{:x}", self.sym.qname, self.offset);
        if let Some(p) = self.file_path {
            out.push_str(&format!("  ({p}:{})", self.sym.line));
        } else if self.sym.line != 0 {
            out.push_str(&format!(
                "  (file_id={}:{})",
                self.sym.file_id, self.sym.line
            ));
        }
        out
    }
}

fn resolve_pc<'a>(
    pc: u64,
    symbols: &'a [Symbol],
    file_map: &'a HashMap<u32, String>,
) -> Option<Resolved<'a>> {
    // Binary search for the largest start <= pc, then verify end > pc.
    let idx = symbols.partition_point(|s| s.start <= pc).saturating_sub(1);
    let sym = symbols.get(idx)?;
    if pc < sym.start || pc >= sym.end {
        return None;
    }
    Some(Resolved {
        sym,
        offset: pc - sym.start,
        file_path: file_map.get(&sym.file_id).map(|s| s.as_str()),
    })
}

fn load_symbols(path: &PathBuf) -> Result<Vec<Symbol>> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    let mut first = true;
    for line in BufReader::new(f).lines() {
        let line = line?;
        if first {
            first = false;
            if line.starts_with("backend_id,") {
                continue;
            }
        }
        // CSV: backend_id,start_hex,end_hex,size_bytes,func_id,file_id,line,column,qname
        let mut fields = line.splitn(9, ',');
        let _backend = fields.next();
        let start = fields.next().and_then(parse_hex).unwrap_or(0);
        let end = fields.next().and_then(parse_hex).unwrap_or(0);
        let _size = fields.next();
        let _func_id = fields.next();
        let file_id: u32 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let line_no: u32 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let _column = fields.next();
        let qname = fields
            .next()
            .map(|s| s.trim_matches('"').to_string())
            .unwrap_or_default();
        if start == 0 && end == 0 {
            continue;
        }
        out.push(Symbol {
            start,
            end,
            file_id,
            line: line_no,
            qname,
        });
    }
    Ok(out)
}

fn load_file_table(path: &PathBuf) -> Result<HashMap<u32, String>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = HashMap::new();
    let mut first = true;
    for line in BufReader::new(f).lines() {
        let line = line?;
        if first {
            first = false;
            if line.starts_with("file_id,") {
                continue;
            }
        }
        let mut fields = line.splitn(2, ',');
        let id: u32 = match fields.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let path = fields.next().unwrap_or("").trim_matches('"').to_string();
        out.insert(id, path);
    }
    Ok(out)
}

fn latest_jit_csv() -> Result<PathBuf> {
    // Prefer per-PID files (richer attribution) over the shared one
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir("/tmp")? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with("rayzor_jit_symbols.") && name.ends_with(".csv")) {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            if let Ok(mt) = meta.modified() {
                let p = entry.path();
                if newest.as_ref().map(|(t, _)| mt > *t).unwrap_or(true) {
                    newest = Some((mt, p));
                }
            }
        }
    }
    newest
        .map(|(_, p)| p)
        .ok_or_else(|| {
            anyhow!("no /tmp/rayzor_jit_symbols.*.csv found; run with RAYZOR_DUMP_JIT_MAP=1 first")
        })
        .or_else(|_| {
            let shared = PathBuf::from("/tmp/rayzor_jit_symbols.csv");
            if shared.exists() {
                Ok(shared)
            } else {
                bail!("no JIT symbol CSV available; pass --csv or run with --jit-map enabled")
            }
        })
}

fn parse_hex(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

fn read_stdin_lines() -> Result<Vec<String>> {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(s.lines().map(|l| l.to_string()).collect())
}
