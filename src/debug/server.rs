//! `rayzor debug server` — minimal HTTP server + embedded dashboard.
//!
//! Single-threaded TCP listener, manual HTTP/1.1 handler. No external
//! deps. Reads metric state from the files rayzor's runtime + compiler
//! drop into `/tmp` (alloc-stats line in stderr is captured via an
//! optional sidecar file; JIT map + crash backtraces are read directly
//! from the most recent `/tmp/rayzor_*.csv` / `/tmp/rayzor-crash-*.txt`).
//!
//! The dashboard polls `/api/metrics` every second. Because we re-read
//! the on-disk state on each request, multiple rayzor processes can hand
//! data to the same dashboard without coordination.

use super::DebugCommands;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const DASHBOARD_HTML: &str = include_str!("dashboard.html");

pub fn execute(cmd: DebugCommands) -> Result<()> {
    let DebugCommands::Server { port, host, open } = cmd else {
        return Err(anyhow!(
            "debug::server::execute called with non-Server variant"
        ));
    };
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("=== rayzor debug server ===");
    println!("listening on http://{addr}");
    println!("dashboard: http://{addr}/");
    println!("endpoints: /api/metrics  /api/jit-map  /api/crashes  /api/file-table");
    println!();

    if open {
        let url = format!("http://{addr}/");
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(&url).spawn();
        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
    }

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        if let Err(e) = handle_request(&mut stream) {
            // Best-effort: only log; never crash the server.
            eprintln!("[server] {e}");
        }
    }
    Ok(())
}

fn handle_request(stream: &mut std::net::TcpStream) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();

    // Drain remaining headers
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }

    let (status, content_type, body) = route(&path);
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.write_all(&body)?;
    Ok(())
}

fn route(path: &str) -> (&'static str, &'static str, Vec<u8>) {
    let path_no_query = path.split('?').next().unwrap_or(path);
    let query = path.split('?').nth(1).unwrap_or("");
    match path_no_query {
        "/" | "/index.html" => ("200 OK", "text/html; charset=utf-8", DASHBOARD_HTML.into()),
        "/api/metrics" => ("200 OK", "application/json", metrics_json().into_bytes()),
        "/api/crashes" => ("200 OK", "application/json", crashes_json().into_bytes()),
        "/api/jit-map" => (
            "200 OK",
            "application/json",
            jit_map_json(parse_limit(query)).into_bytes(),
        ),
        "/api/file-table" => ("200 OK", "application/json", file_table_json().into_bytes()),
        _ => ("404 Not Found", "text/plain", b"not found\n".to_vec()),
    }
}

fn parse_limit(query: &str) -> usize {
    for kv in query.split('&') {
        if let Some(v) = kv.strip_prefix("limit=") {
            if let Ok(n) = v.parse::<usize>() {
                return n;
            }
        }
    }
    100
}

// ---- /api/metrics --------------------------------------------------------

/// Read the most recent alloc-stats line from `/tmp/rayzor-metrics.txt`
/// (written by an opt-in sidecar) or, if not present, return whatever
/// snapshot we can recover from process state. The "tensor" and "alloc"
/// halves come from the two stat dump functions in `runtime/src/`.
fn metrics_json() -> String {
    let alloc = read_kv("/tmp/rayzor-metrics-alloc.kv");
    let tensor = read_kv("/tmp/rayzor-metrics-tensor.kv");
    let alloc_block = stat_block(&alloc);
    let tensor_block = stat_block_tensor(&tensor);
    format!("{{\"alloc\":{alloc_block},\"tensor\":{tensor_block}}}")
}

fn stat_block(m: &HashMap<String, String>) -> String {
    let g = |k: &str| m.get(k).and_then(|v| v.parse::<u64>().ok());
    let allocs = g("allocs");
    let frees = g("frees");
    let alloc_bytes = g("alloc_bytes");
    let free_bytes = g("free_bytes");
    let live = alloc_bytes
        .zip(free_bytes)
        .map(|(a, f)| a.saturating_sub(f));
    let peak = g("peak");
    json_obj(&[
        ("allocs", json_opt_num(allocs)),
        ("frees", json_opt_num(frees)),
        ("live", json_opt_num(live)),
        ("peak", json_opt_num(peak)),
    ])
}

fn stat_block_tensor(m: &HashMap<String, String>) -> String {
    let g = |k: &str| m.get(k).and_then(|v| v.parse::<u64>().ok());
    let alloc_bytes = g("alloc_bytes");
    let free_bytes = g("free_bytes");
    let live = alloc_bytes
        .zip(free_bytes)
        .map(|(a, f)| a.saturating_sub(f));
    let pool_hits = g("pool_hits").unwrap_or(0);
    let pool_misses = g("pool_misses").unwrap_or(0);
    let hit_rate = if pool_hits + pool_misses == 0 {
        0.0
    } else {
        100.0 * pool_hits as f64 / (pool_hits + pool_misses) as f64
    };
    json_obj(&[
        ("allocs", json_opt_num(g("allocs"))),
        ("frees", json_opt_num(g("frees"))),
        ("alloc_bytes", json_opt_num(alloc_bytes)),
        ("free_bytes", json_opt_num(free_bytes)),
        ("live", json_opt_num(live)),
        ("peak", json_opt_num(g("peak"))),
        ("pool_hits", format!("{pool_hits}")),
        ("pool_misses", format!("{pool_misses}")),
        ("pool_hit_rate", format!("{hit_rate:.2}")),
    ])
}

fn read_kv(path: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Ok(s) = std::fs::read_to_string(path) {
        for line in s.lines() {
            if let Some((k, v)) = line.split_once('=') {
                m.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    m
}

// ---- /api/crashes --------------------------------------------------------

fn crashes_json() -> String {
    let mut entries = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/tmp") {
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if !(name.starts_with("rayzor-crash-") && name.ends_with(".txt")) {
                continue;
            }
            let mtime = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .unwrap_or(SystemTime::UNIX_EPOCH);
            entries.push((mtime, p));
        }
    }
    entries.sort_by(|a, b| b.0.cmp(&a.0));
    entries.truncate(20);
    let items: Vec<String> = entries
        .into_iter()
        .map(|(_, p)| {
            let body = std::fs::read_to_string(&p).unwrap_or_default();
            format!(
                "{{\"path\":{},\"body\":{}}}",
                json_str(&p.to_string_lossy()),
                json_str(&body)
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

// ---- /api/jit-map --------------------------------------------------------

fn jit_map_json(limit: usize) -> String {
    let csv = match latest_jit_csv() {
        Some(p) => p,
        None => return "[]".into(),
    };
    let mut entries = Vec::new();
    if let Ok(f) = std::fs::File::open(&csv) {
        let mut first = true;
        for line in BufReader::new(f).lines().map_while(|l| l.ok()) {
            if first {
                first = false;
                if line.starts_with("backend_id,") {
                    continue;
                }
            }
            let mut fields = line.splitn(9, ',');
            let _backend = fields.next();
            let start = fields.next().unwrap_or("0");
            let end = fields.next().unwrap_or("0");
            let size: u64 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let _func_id = fields.next();
            let file_id: u32 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let line_no: u32 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let _col = fields.next();
            let qname = fields.next().unwrap_or("").trim_matches('"').to_string();
            entries.push((
                qname,
                size,
                start.to_string(),
                end.to_string(),
                file_id,
                line_no,
            ));
        }
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.truncate(limit);
    let file_map = load_file_table_inner();
    let items: Vec<String> = entries
        .into_iter()
        .map(|(qname, size, _start, _end, file_id, line)| {
            let file = file_map.get(&file_id).cloned().unwrap_or_default();
            format!(
                "{{\"qname\":{},\"size_bytes\":{size},\"file\":{},\"line\":{line}}}",
                json_str(&qname),
                json_str(&file),
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

fn latest_jit_csv() -> Option<PathBuf> {
    let mut newest: Option<(SystemTime, PathBuf)> = None;
    for e in std::fs::read_dir("/tmp").ok()?.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with("rayzor_jit_symbols.") && name.ends_with(".csv")) {
            continue;
        }
        if let Ok(mt) = e.metadata().and_then(|m| m.modified()) {
            let p = e.path();
            if newest.as_ref().map(|(t, _)| mt > *t).unwrap_or(true) {
                newest = Some((mt, p));
            }
        }
    }
    let shared = PathBuf::from("/tmp/rayzor_jit_symbols.csv");
    newest
        .map(|(_, p)| p)
        .or_else(|| if shared.exists() { Some(shared) } else { None })
}

fn load_file_table_inner() -> HashMap<u32, String> {
    let p = Path::new("/tmp/rayzor_file_table.csv");
    if !p.exists() {
        return HashMap::new();
    }
    let mut m = HashMap::new();
    if let Ok(f) = std::fs::File::open(p) {
        let mut first = true;
        for line in BufReader::new(f).lines().map_while(|l| l.ok()) {
            if first {
                first = false;
                if line.starts_with("file_id,") {
                    continue;
                }
            }
            let mut it = line.splitn(2, ',');
            if let (Some(k), Some(v)) = (it.next(), it.next()) {
                if let Ok(id) = k.parse::<u32>() {
                    m.insert(id, v.trim_matches('"').to_string());
                }
            }
        }
    }
    m
}

// ---- /api/file-table -----------------------------------------------------

fn file_table_json() -> String {
    let m = load_file_table_inner();
    let items: Vec<String> = m
        .into_iter()
        .map(|(id, p)| format!("{{\"id\":{id},\"path\":{}}}", json_str(&p)))
        .collect();
    format!("[{}]", items.join(","))
}

// ---- json helpers --------------------------------------------------------

fn json_obj(kvs: &[(&str, String)]) -> String {
    let inner: Vec<String> = kvs
        .iter()
        .map(|(k, v)| format!("{}:{}", json_str(k), v))
        .collect();
    format!("{{{}}}", inner.join(","))
}

fn json_opt_num(v: Option<u64>) -> String {
    v.map(|n| n.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_str(s: &str) -> String {
    let escaped: String = s
        .chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32).chars().collect::<Vec<_>>(),
            c => vec![c],
        })
        .collect();
    format!("\"{escaped}\"")
}
