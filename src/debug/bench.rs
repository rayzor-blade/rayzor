//! `rayzor debug bench` — run a file N times, aggregate per-run metrics.
//!
//! Replaces `tools/llama-diff/peak_bench.sh` with a typed, in-process
//! harness. Each run is a fresh subprocess; we capture stdout/stderr,
//! parse the `[tensor-data]` and `[done]` lines, classify failures by
//! signal, and report min/median/mean/max + success rate.

use super::DebugCommands;
use anyhow::{anyhow, Result};
use clap::ValueEnum;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Tensor-data live peak in MB (deterministic; requires `--features profile`)
    PeakMem,
    /// Wall-clock decode tok/s from `[done]` line (llama-chat-style apps)
    TokPerS,
    /// Total wall time of the run, seconds
    WallTime,
}

impl Metric {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Metric::PeakMem => "peak (MB)",
            Metric::TokPerS => "tok/s",
            Metric::WallTime => "wall (s)",
        }
    }

    fn extract(&self, stdout: &str) -> Option<f64> {
        match self {
            Metric::PeakMem => extract_peak_mb(stdout),
            Metric::TokPerS => extract_tok_per_s(stdout),
            Metric::WallTime => None, // populated by the harness, not stdout
        }
    }
}

pub fn execute(cmd: DebugCommands) -> Result<()> {
    let DebugCommands::Bench {
        file,
        runs,
        metric,
        timeout,
        no_cache_scrub,
        program_args,
    } = cmd
    else {
        return Err(anyhow!(
            "debug::bench::execute called with non-Bench variant"
        ));
    };

    run_bench(&file, runs, metric, timeout, !no_cache_scrub, &program_args)?;
    Ok(())
}

pub(crate) struct BenchSummary {
    pub successful: usize,
    pub failed: usize,
    pub samples: Vec<f64>,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub mean: f64,
}

pub(crate) fn run_bench(
    file: &Path,
    runs: usize,
    metric: Metric,
    timeout_s: u64,
    scrub_cache: bool,
    program_args: &[String],
) -> Result<BenchSummary> {
    let self_exe = std::env::current_exe()?;

    println!("=== rayzor debug bench ===");
    println!("file:    {}", file.display());
    println!("runs:    {runs}");
    println!("metric:  {}", metric.label());
    println!();

    let mut samples = Vec::new();
    let mut failed = 0;

    for i in 1..=runs {
        if scrub_cache {
            scrub_rayzor_caches()?;
        }

        let mut child = Command::new(&self_exe);
        child
            .arg("run")
            .arg(file)
            .arg("--preset")
            .arg("application")
            .env("RAYZOR_DUMP_ALLOC_AT_EXIT", "1")
            .env("RAYZOR_DUMP_JIT_MAP", "1");

        if !program_args.is_empty() {
            child.arg("--");
            for a in program_args {
                child.arg(a);
            }
        }

        let start = Instant::now();
        let output = match run_with_timeout(child, Duration::from_secs(timeout_s)) {
            Ok(out) => out,
            Err(e) => {
                println!("  run {i:>3}  FAILED  ({e})");
                failed += 1;
                continue;
            }
        };
        let wall = start.elapsed().as_secs_f64();

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}{stderr}");

        // Detect crash signature
        let exit_code = output.status.code().unwrap_or(-1);
        let died_to_signal = exit_code >= 128
            || combined.contains("trace trap")
            || combined.contains("sig5-back")
            || combined.contains("sig11-back");

        let sample = match metric {
            Metric::WallTime => Some(wall),
            _ => metric.extract(&combined),
        };

        match sample {
            Some(v) if !died_to_signal => {
                samples.push(v);
                let label = match metric {
                    Metric::PeakMem => format!("peak={v:.1} MB"),
                    Metric::TokPerS => format!("tok/s={v:.2}"),
                    Metric::WallTime => format!("wall={v:.2}s"),
                };
                println!("  run {i:>3}  ok       {label}");
            }
            _ => {
                let why = if died_to_signal {
                    let sig = combined
                        .lines()
                        .find_map(|l| {
                            l.find("sig")
                                .map(|i| l[i..].split_whitespace().next().unwrap_or("sig?"))
                        })
                        .unwrap_or("trap");
                    format!("crashed ({sig}, exit {exit_code})")
                } else {
                    format!("no metric in stdout (exit {exit_code})")
                };
                println!("  run {i:>3}  FAIL     {why}");
                failed += 1;
            }
        }
    }

    let summary = summarize(samples, failed);

    println!();
    println!("=== summary ===");
    println!("successful: {}/{}", summary.successful, runs);
    if summary.successful > 0 {
        println!(
            "{:11}: min={:.2}  median={:.2}  mean={:.2}  max={:.2}",
            metric.label(),
            summary.min,
            summary.median,
            summary.mean,
            summary.max
        );
    }
    Ok(summary)
}

fn summarize(mut samples: Vec<f64>, failed: usize) -> BenchSummary {
    let successful = samples.len();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = samples.first().copied().unwrap_or(f64::NAN);
    let max = samples.last().copied().unwrap_or(f64::NAN);
    let median = if samples.is_empty() {
        f64::NAN
    } else if samples.len() % 2 == 1 {
        samples[samples.len() / 2]
    } else {
        let m = samples.len() / 2;
        (samples[m - 1] + samples[m]) / 2.0
    };
    let mean = if samples.is_empty() {
        f64::NAN
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    };
    BenchSummary {
        successful,
        failed,
        samples,
        min,
        max,
        median,
        mean,
    }
}

fn extract_peak_mb(out: &str) -> Option<f64> {
    // [tensor-data] ... peak=BYTES (MB MB) ...
    out.lines()
        .find(|l| l.starts_with("[tensor-data]"))
        .and_then(|l| {
            let i = l.find("peak=")? + "peak=".len();
            let mb_open = l[i..].find('(')? + i + 1;
            let mb_close = l[mb_open..].find(' ')? + mb_open;
            l[mb_open..mb_close].parse::<f64>().ok()
        })
}

fn extract_tok_per_s(out: &str) -> Option<f64> {
    // [done] N tokens in Ts (X tok/s)
    out.lines()
        .find(|l| l.starts_with("[done] "))
        .and_then(|l| {
            let paren = l.find('(')? + 1;
            let close = l[paren..].find(' ')? + paren;
            l[paren..close].parse::<f64>().ok()
        })
}

fn scrub_rayzor_caches() -> Result<()> {
    let repo_root = repo_root().ok_or_else(|| anyhow!("not inside a git repo"))?;
    // `find <repo> -name .rayzor -type d -prune -exec rm -rf {} +` style scrub.
    // Walk only top-level repo dirs, skip target/.
    for entry in walkdir::WalkDir::new(&repo_root)
        .max_depth(4)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let p = entry.path();
        if p.is_dir() && p.file_name().and_then(|s| s.to_str()) == Some(".rayzor") {
            let _ = std::fs::remove_dir_all(p);
        }
    }
    Ok(())
}

fn repo_root() -> Option<std::path::PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    Some(std::path::PathBuf::from(s.trim()))
}

/// Run a command with a wall-clock timeout. Kills the process tree if
/// it overruns.
///
/// Drains stdout/stderr in dedicated reader threads so child pipes never
/// block — without this, a chatty rayzor run filled the OS pipe buffer
/// (~16 KB) and the child stalled waiting for us to read, but we were
/// busy looping on try_wait. Resulted in spurious "timeout" failures
/// even on runs that would have completed in seconds.
fn run_with_timeout(mut cmd: Command, dur: Duration) -> Result<std::process::Output> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn()?;
    let pid = child.id();

    let so = Arc::new(Mutex::new(Vec::new()));
    let se = Arc::new(Mutex::new(Vec::new()));

    let so_thread = {
        let so = so.clone();
        let pipe = child.stdout.take();
        thread::spawn(move || {
            if let Some(mut p) = pipe {
                let mut buf = Vec::new();
                let _ = std::io::Read::read_to_end(&mut p, &mut buf);
                so.lock().unwrap().extend(buf);
            }
        })
    };
    let se_thread = {
        let se = se.clone();
        let pipe = child.stderr.take();
        thread::spawn(move || {
            if let Some(mut p) = pipe {
                let mut buf = Vec::new();
                let _ = std::io::Read::read_to_end(&mut p, &mut buf);
                se.lock().unwrap().extend(buf);
            }
        })
    };

    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() > dur {
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
            let s = child.wait()?;
            // Let the reader threads finish draining the (now closed) pipes.
            let _ = so_thread.join();
            let _ = se_thread.join();
            let _ = s;
            return Err(anyhow!("timeout after {}s", dur.as_secs()));
        }
        thread::sleep(Duration::from_millis(50));
    };

    let _ = so_thread.join();
    let _ = se_thread.join();
    let stdout = std::mem::take(&mut *so.lock().unwrap());
    let stderr = std::mem::take(&mut *se.lock().unwrap());
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}
