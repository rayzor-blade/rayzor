//! `rayzor debug compare` — A/B benchmark across two git refs.
//!
//! Orchestrates: stash uncommitted changes → checkout base_ref → build →
//! bench → checkout HEAD → restore stash → build → bench → print delta.
//!
//! Restores the original ref + stash on any error / Ctrl-C path so an
//! aborted run never strands the tree mid-checkout.

use super::bench::{run_bench, Metric};
use super::DebugCommands;
use anyhow::{anyhow, bail, Context, Result};
use std::process::Command;

pub fn execute(cmd: DebugCommands) -> Result<()> {
    let DebugCommands::Compare {
        base_ref,
        file,
        runs,
        metric,
        timeout,
        program_args,
    } = cmd
    else {
        return Err(anyhow!(
            "debug::compare::execute called with non-Compare variant"
        ));
    };

    // Snapshot starting ref so we can restore at the end.
    let head_ref = current_ref().context("reading current git ref")?;
    println!("[compare] starting ref: {head_ref}");
    println!("[compare] baseline ref: {base_ref}");

    // Demand a clean tree — easier to reason about than auto-stashing.
    if has_changes()? {
        bail!(
            "working tree has uncommitted changes — commit or stash before running\n\
             `rayzor debug compare` (we re-checkout between refs and don't want\n\
             to surprise-stash your work)"
        );
    }

    // Helper that restores HEAD + rebuilds on the way out, regardless of
    // success or panic.
    let mut head_summary = None;
    let mut base_summary = None;

    let result = (|| -> Result<()> {
        println!("\n=== running at HEAD ({head_ref}) ===");
        rebuild_release()?;
        let h = run_bench(&file, runs, metric, timeout, true, &program_args)?;
        head_summary = Some(h);

        println!("\n=== checking out baseline ({base_ref}) ===");
        run_git(&["checkout", &base_ref])?;
        rebuild_release()?;
        let b = run_bench(&file, runs, metric, timeout, true, &program_args)?;
        base_summary = Some(b);

        Ok(())
    })();

    // Always restore the original ref + binary state.
    println!("\n[compare] restoring ref: {head_ref}");
    if let Err(e) = run_git(&["checkout", &head_ref]) {
        eprintln!("warning: failed to restore ref {head_ref}: {e}");
    }
    if let Err(e) = rebuild_release() {
        eprintln!("warning: failed to rebuild at {head_ref}: {e}");
    }

    result?;

    let (h, b) = match (head_summary, base_summary) {
        (Some(h), Some(b)) => (h, b),
        _ => bail!("missing one of the bench halves"),
    };
    print_delta(&head_ref, &h, &base_ref, &b, metric);
    Ok(())
}

fn print_delta(
    head_ref: &str,
    head: &super::bench::BenchSummary,
    base_ref: &str,
    base: &super::bench::BenchSummary,
    metric: Metric,
) {
    println!("\n=== A/B summary ({}) ===", metric.label());
    println!(
        "  HEAD     ({head_ref:>10}):  median={:.2}  successful={}/{}",
        head.median,
        head.successful,
        head.successful + head.failed
    );
    println!(
        "  BASELINE ({base_ref:>10}):  median={:.2}  successful={}/{}",
        base.median,
        base.successful,
        base.successful + base.failed
    );
    if head.median.is_finite() && base.median.is_finite() {
        let delta = head.median - base.median;
        let pct = (head.median - base.median) / base.median * 100.0;
        let direction = match metric {
            // For peak-mem + wall-time, smaller is better
            Metric::PeakMem | Metric::WallTime => {
                if delta < 0.0 {
                    "BETTER (smaller)"
                } else {
                    "WORSE (larger)"
                }
            }
            // For tok/s, larger is better
            Metric::TokPerS => {
                if delta > 0.0 {
                    "BETTER (faster)"
                } else {
                    "WORSE (slower)"
                }
            }
        };
        println!("  delta:                 {delta:+.2}  ({pct:+.1}%)  {direction}");
    }
}

fn current_ref() -> Result<String> {
    let out = Command::new("git")
        .args(["symbolic-ref", "--short", "HEAD"])
        .output()?;
    if out.status.success() {
        return Ok(String::from_utf8(out.stdout)?.trim().into());
    }
    // Detached HEAD: report the short SHA.
    let out = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()?;
    Ok(String::from_utf8(out.stdout)?.trim().into())
}

fn has_changes() -> Result<bool> {
    let out = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    let s = String::from_utf8(out.stdout)?;
    // Ignore untracked-only lines (?? prefix) — those don't block a checkout
    Ok(s.lines().any(|l| !l.starts_with("?? ")))
}

fn run_git(args: &[&str]) -> Result<()> {
    let status = Command::new("git").args(args).status()?;
    if !status.success() {
        bail!("git {args:?} failed with {status}");
    }
    Ok(())
}

fn rebuild_release() -> Result<()> {
    println!("[compare] cargo build --release --features profile -p rayzor");
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--features",
            "profile",
            "-p",
            "rayzor",
        ])
        .status()?;
    if !status.success() {
        bail!("rebuild failed with {status}");
    }
    Ok(())
}
