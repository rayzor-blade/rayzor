//! `rayzor debug bench` — run a file N times, aggregate per-run metrics.
//!
//! Replaces `tools/llama-diff/peak_bench.sh` with a typed, in-process
//! harness. Each run is a fresh subprocess; we capture stdout/stderr,
//! parse the `[tensor-data]` and `[done]` lines, classify failures by
//! signal, and report min/median/mean/max + success rate.

use super::DebugCommands;
use anyhow::{anyhow, Result};
use clap::ValueEnum;
use std::path::{Path, PathBuf};
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
        release,
        llvm,
        native_libs,
        debug_dumps,
        preset,
        tier_thresholds,
        preset_override_toml,
        tier_sample_rate,
        tier_start_interpreted,
        tier_promotion,
        decode_profile,
        cooldown_ms,
        program_args,
    } = cmd
    else {
        return Err(anyhow!(
            "debug::bench::execute called with non-Bench variant"
        ));
    };

    let options = BenchOptions {
        preset,
        tier_thresholds,
        preset_override_toml,
        tier_sample_rate,
        tier_start_interpreted,
        tier_promotion,
        release,
        llvm,
        native_libs,
        debug_dumps,
        decode_profile,
        cooldown_ms,
    };
    run_bench_with_options(
        &file,
        runs,
        metric,
        timeout,
        !no_cache_scrub,
        &program_args,
        &options,
    )?;
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct BenchOptions {
    pub preset: String,
    pub tier_thresholds: Vec<String>,
    pub preset_override_toml: bool,
    pub tier_sample_rate: Option<u64>,
    pub tier_start_interpreted: Option<bool>,
    pub tier_promotion: Option<bool>,
    pub release: bool,
    pub llvm: bool,
    pub native_libs: Vec<PathBuf>,
    pub debug_dumps: bool,
    pub decode_profile: bool,
    pub cooldown_ms: u64,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            preset: "application".to_string(),
            tier_thresholds: Vec::new(),
            preset_override_toml: false,
            tier_sample_rate: None,
            tier_start_interpreted: None,
            tier_promotion: None,
            release: false,
            llvm: false,
            native_libs: Vec::new(),
            debug_dumps: false,
            decode_profile: false,
            cooldown_ms: 0,
        }
    }
}

pub(crate) struct BenchSummary {
    pub successful: usize,
    pub failed: usize,
    #[allow(dead_code)] // retained on the summary for future per-sample reporting
    pub samples: Vec<f64>,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub mean: f64,
    pub stddev: f64,
}

pub(crate) fn run_bench(
    file: &Path,
    runs: usize,
    metric: Metric,
    timeout_s: u64,
    scrub_cache: bool,
    program_args: &[String],
) -> Result<BenchSummary> {
    run_bench_with_options(
        file,
        runs,
        metric,
        timeout_s,
        scrub_cache,
        program_args,
        &BenchOptions::default(),
    )
}

pub(crate) fn run_bench_with_options(
    file: &Path,
    runs: usize,
    metric: Metric,
    timeout_s: u64,
    scrub_cache: bool,
    program_args: &[String],
    options: &BenchOptions,
) -> Result<BenchSummary> {
    let self_exe = std::env::current_exe()?;
    let variants = bench_variants(options)?;
    let total_runs = runs * variants.len();

    println!("=== rayzor debug bench ===");
    println!("file:    {}", file.display());
    println!("runs:    {runs} per variant ({total_runs} subprocesses)");
    println!("metric:  {}", metric.label());
    println!("preset:  {}", options.preset);
    if variants.len() > 1 {
        let labels = variants
            .iter()
            .map(|v| v.label.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        println!("variants: {labels}");
        println!("order:   round-robin");
    }
    if options.preset_override_toml {
        println!("tier:    CLI --preset overrides rayzor.toml [tier]");
    }
    if let Some(sample_rate) = options.tier_sample_rate {
        println!("sample:  {sample_rate}");
    }
    if let Some(start_interpreted) = options.tier_start_interpreted {
        println!("start:   interpreted={start_interpreted}");
    }
    if let Some(tier_promotion) = options.tier_promotion {
        println!("promote: {tier_promotion}");
    }
    if options.release {
        println!("mode:    release");
    }
    if options.llvm {
        println!("llvm:    enabled");
    }
    if !options.native_libs.is_empty() {
        let libs = options
            .native_libs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!("native:  {libs}");
    }
    if options.debug_dumps {
        println!("debug:   dumps enabled");
    }
    if options.decode_profile {
        println!("decode:  tail-latency profile enabled");
    }
    if options.cooldown_ms > 0 {
        println!("cooldown: {} ms between subprocesses", options.cooldown_ms);
    }
    println!();

    let mut accumulators = variants
        .into_iter()
        .map(VariantAccumulator::new)
        .collect::<Vec<_>>();
    let mut ordinal = 0usize;

    for i in 1..=runs {
        for idx in 0..accumulators.len() {
            ordinal += 1;
            if scrub_cache {
                scrub_rayzor_caches()?;
            }

            let variant = accumulators[idx].variant.clone();
            let outcome = run_one(
                &self_exe,
                file,
                metric,
                timeout_s,
                program_args,
                options,
                &variant,
            );

            match outcome {
                Ok(SingleRunOutcome::Ok {
                    value,
                    label,
                    profile,
                    pool_profile,
                }) => {
                    accumulators[idx].samples.push(value);
                    if let Some(profile) = profile {
                        accumulators[idx].profiles.push(profile);
                    }
                    if let Some(pool_profile) = pool_profile {
                        accumulators[idx].pool_profiles.push(pool_profile);
                    }
                    println!(
                        "  run {i:>3}  {:<14} ok       {label}",
                        accumulators[idx].variant.label
                    );
                }
                Ok(SingleRunOutcome::Fail { why }) => {
                    accumulators[idx].failed += 1;
                    println!(
                        "  run {i:>3}  {:<14} FAIL     {why}",
                        accumulators[idx].variant.label
                    );
                }
                Err(e) => {
                    accumulators[idx].failed += 1;
                    println!(
                        "  run {i:>3}  {:<14} FAILED  ({e})",
                        accumulators[idx].variant.label
                    );
                }
            }

            if options.cooldown_ms > 0 && ordinal < total_runs {
                std::thread::sleep(Duration::from_millis(options.cooldown_ms));
            }
        }
    }

    let mut combined_samples = Vec::new();
    let mut combined_failed = 0usize;

    println!();
    println!("=== summary ===");
    for acc in &accumulators {
        combined_samples.extend(acc.samples.iter().copied());
        combined_failed += acc.failed;
        let summary = summarize(acc.samples.clone(), acc.failed);
        println!(
            "{:<14} successful: {}/{}",
            acc.variant.label, summary.successful, runs
        );
        if summary.successful > 0 {
            println!(
                "{:14} {}: min={:.2}  median={:.2}  mean={:.2}  max={:.2}  stddev={:.2}  drift={:.2}",
                "",
                metric.label(),
                summary.min,
                summary.median,
                summary.mean,
                summary.max,
                summary.stddev,
                drift(&acc.samples)
            );
            if options.decode_profile && !acc.profiles.is_empty() {
                print_profile_summary(&acc.profiles);
            }
            if !acc.pool_profiles.is_empty() {
                print_pool_summary(&acc.pool_profiles);
            }
        }
    }

    Ok(summarize(combined_samples, combined_failed))
}

#[derive(Clone, Debug)]
struct BenchVariant {
    label: String,
    tier_thresholds: Option<String>,
}

impl BenchVariant {
    fn manifest() -> Self {
        Self {
            label: "toml".to_string(),
            tier_thresholds: None,
        }
    }

    fn thresholds(spec: String) -> Self {
        Self {
            label: spec.clone(),
            tier_thresholds: Some(spec),
        }
    }
}

#[derive(Clone, Debug)]
struct VariantAccumulator {
    variant: BenchVariant,
    samples: Vec<f64>,
    profiles: Vec<DecodeProfile>,
    pool_profiles: Vec<PoolProfile>,
    failed: usize,
}

impl VariantAccumulator {
    fn new(variant: BenchVariant) -> Self {
        Self {
            variant,
            samples: Vec::new(),
            profiles: Vec::new(),
            pool_profiles: Vec::new(),
            failed: 0,
        }
    }
}

enum SingleRunOutcome {
    Ok {
        value: f64,
        label: String,
        profile: Option<DecodeProfile>,
        pool_profile: Option<PoolProfile>,
    },
    Fail {
        why: String,
    },
}

fn bench_variants(options: &BenchOptions) -> Result<Vec<BenchVariant>> {
    if options.tier_thresholds.is_empty() {
        return Ok(vec![BenchVariant::manifest()]);
    }

    options
        .tier_thresholds
        .iter()
        .map(|spec| {
            validate_threshold_spec(spec)?;
            Ok(BenchVariant::thresholds(spec.clone()))
        })
        .collect()
}

fn validate_threshold_spec(spec: &str) -> Result<()> {
    let parts = spec
        .split(|c| matches!(c, '/' | ',' | ':'))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>();
    if !(parts.len() == 3 || parts.len() == 4) {
        return Err(anyhow!(
            "invalid --tier-thresholds `{spec}`; expected I/W/H or I/W/H/B"
        ));
    }
    for part in parts {
        match part.to_ascii_lowercase().as_str() {
            "max" | "never" => {}
            _ => {
                part.parse::<u64>().map_err(|_| {
                    anyhow!("invalid threshold component `{part}` in --tier-thresholds `{spec}`")
                })?;
            }
        }
    }
    Ok(())
}

fn run_one(
    self_exe: &Path,
    file: &Path,
    metric: Metric,
    timeout_s: u64,
    program_args: &[String],
    options: &BenchOptions,
    variant: &BenchVariant,
) -> Result<SingleRunOutcome> {
    let mut child = Command::new(self_exe);
    child
        .arg("run")
        .arg(file)
        .arg("--preset")
        .arg(&options.preset)
        .arg("--stats");
    if options.debug_dumps {
        child
            .env("RAYZOR_DUMP_ALLOC_AT_EXIT", "1")
            .env("RAYZOR_DUMP_JIT_MAP", "1");
    }
    if options.release {
        child.arg("--release");
    }
    if options.llvm {
        child.arg("--llvm");
    }
    for native_lib in &options.native_libs {
        child.arg("--native-lib").arg(native_lib);
    }
    if options.decode_profile {
        child.env("RAYZOR_PROFILE_DECODE", "1");
    }

    if options.preset_override_toml {
        child.arg("--preset-override-toml");
    }
    if let Some(spec) = &variant.tier_thresholds {
        child.arg("--tier-thresholds").arg(spec);
    }
    if let Some(sample_rate) = options.tier_sample_rate {
        child.arg("--tier-sample-rate").arg(sample_rate.to_string());
    }
    if let Some(start_interpreted) = options.tier_start_interpreted {
        child
            .arg("--tier-start-interpreted")
            .arg(start_interpreted.to_string());
    }
    if let Some(tier_promotion) = options.tier_promotion {
        child
            .arg("--tier-promotion")
            .arg(tier_promotion.to_string());
    }
    if !program_args.is_empty() {
        child.arg("--");
        for a in program_args {
            child.arg(a);
        }
    }

    let start = Instant::now();
    let output = match run_with_timeout(child, Duration::from_secs(timeout_s)) {
        Ok(out) => out,
        Err(e) => return Ok(SingleRunOutcome::Fail { why: e.to_string() }),
    };
    let wall = start.elapsed().as_secs_f64();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

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
        Some(value) if !died_to_signal => {
            let mut label = metric_value_label(metric, value);
            let profile = if options.decode_profile {
                extract_decode_profile(&combined)
            } else {
                None
            };
            let pool_profile = extract_pool_profile(&combined);
            if options.decode_profile {
                if let Some(profile) = profile {
                    label.push_str(&format!(
                        "  p95={:.2}ms p99={:.2}ms max={:.2}ms@{}",
                        profile.step_p95_ms,
                        profile.step_p99_ms,
                        profile.step_max_ms,
                        profile.step_max_i
                    ));
                    if profile.tier_events.total > 0 {
                        label.push_str(&format!(
                            "  tier_events={} near={} routes={} installs={} compiles={}",
                            profile.tier_events.total,
                            profile.tier_events.near_max,
                            profile.tier_events.routes,
                            profile.tier_events.installs,
                            profile.tier_events.compiles
                        ));
                    }
                    if profile.tier_events_all.total > profile.tier_events.total {
                        label.push_str(&format!(
                            "  tier_process_events={}",
                            profile.tier_events_all.total
                        ));
                    }
                } else {
                    label.push_str("  profile=missing");
                }
            }
            if let Some(pool) = pool_profile {
                label.push_str(&format!(
                    "  pool_band={:.1}ms pool_quant={:.1}ms pool_dispatches={}",
                    pool.band_ms, pool.quant_ms, pool.dispatches
                ));
                if let Some(workers) = pool.workers {
                    label.push_str(&format!(" pool_workers={workers}"));
                }
            }
            Ok(SingleRunOutcome::Ok {
                value,
                label,
                profile,
                pool_profile,
            })
        }
        _ => Ok(SingleRunOutcome::Fail {
            why: failure_reason(&combined, exit_code, died_to_signal),
        }),
    }
}

fn metric_value_label(metric: Metric, value: f64) -> String {
    match metric {
        Metric::PeakMem => format!("peak={value:.1} MB"),
        Metric::TokPerS => format!("tok/s={value:.2}"),
        Metric::WallTime => format!("wall={value:.2}s"),
    }
}

fn print_profile_summary(profiles: &[DecodeProfile]) {
    let p95 = summarize(
        profiles.iter().map(|p| p.step_p95_ms).collect::<Vec<_>>(),
        0,
    );
    let p99 = summarize(
        profiles.iter().map(|p| p.step_p99_ms).collect::<Vec<_>>(),
        0,
    );
    let worst = profiles
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.step_max_ms
                .partial_cmp(&b.step_max_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, p)| (i + 1, *p));

    if let Some((run, worst)) = worst {
        println!(
            "{:14} profile: p95_med={:.2}ms p95_max={:.2}ms  p99_med={:.2}ms p99_max={:.2}ms  worst={:.2}ms@{} run={}",
            "", p95.median, p95.max, p99.median, p99.max, worst.step_max_ms, worst.step_max_i, run
        );
        let events = profiles
            .iter()
            .fold(TierEventCounts::default(), |mut acc, p| {
                acc.total += p.tier_events.total;
                acc.routes += p.tier_events.routes;
                acc.registers += p.tier_events.registers;
                acc.installs += p.tier_events.installs;
                acc.compiles += p.tier_events.compiles;
                acc.errors += p.tier_events.errors;
                acc.near_max += p.tier_events.near_max;
                acc
            });
        if events.total > 0 {
            println!(
                "{:14} tier: decode_events={} near_max={} routes={} registers={} installs={} compiles={} errors={}",
                "",
                events.total,
                events.near_max,
                events.routes,
                events.registers,
                events.installs,
                events.compiles,
                events.errors
            );
        }
        let all_events = profiles
            .iter()
            .fold(TierEventCounts::default(), |mut acc, p| {
                acc.total += p.tier_events_all.total;
                acc.routes += p.tier_events_all.routes;
                acc.registers += p.tier_events_all.registers;
                acc.installs += p.tier_events_all.installs;
                acc.compiles += p.tier_events_all.compiles;
                acc.errors += p.tier_events_all.errors;
                acc.near_max += p.tier_events_all.near_max;
                acc
            });
        if all_events.total > events.total {
            println!(
                "{:14} tier-process: events={} routes={} registers={} installs={} compiles={} errors={}",
                "",
                all_events.total,
                all_events.routes,
                all_events.registers,
                all_events.installs,
                all_events.compiles,
                all_events.errors
            );
        }
    }
}

fn print_pool_summary(profiles: &[PoolProfile]) {
    let band = summarize(profiles.iter().map(|p| p.band_ms).collect::<Vec<_>>(), 0);
    let quant = summarize(profiles.iter().map(|p| p.quant_ms).collect::<Vec<_>>(), 0);
    let dispatches = summarize(
        profiles
            .iter()
            .map(|p| p.dispatches as f64)
            .collect::<Vec<_>>(),
        0,
    );
    let workers = profiles
        .iter()
        .filter_map(|p| p.workers)
        .map(|v| v.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("/");
    let worker_label = if workers.is_empty() {
        String::new()
    } else {
        format!("  workers={workers}")
    };
    println!(
        "{:14} pool: band_med={:.1}ms band_max={:.1}ms  quant_med={:.1}ms quant_max={:.1}ms  dispatches_med={:.0}{}",
        "",
        band.median,
        band.max,
        quant.median,
        quant.max,
        dispatches.median,
        worker_label
    );
}

fn failure_reason(combined: &str, exit_code: i32, died_to_signal: bool) -> String {
    if died_to_signal {
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
    }
}

#[derive(Clone, Copy, Debug)]
struct DecodeProfile {
    decode_start_s: f64,
    decode_end_s: f64,
    step_p95_ms: f64,
    step_p99_ms: f64,
    step_max_ms: f64,
    step_max_i: i64,
    step_max_s: f64,
    tier_events: TierEventCounts,
    tier_events_all: TierEventCounts,
}

#[derive(Clone, Copy, Debug)]
struct PoolProfile {
    band_ms: f64,
    quant_ms: f64,
    dispatches: u64,
    workers: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TierEventCounts {
    total: u64,
    routes: u64,
    registers: u64,
    installs: u64,
    compiles: u64,
    errors: u64,
    near_max: u64,
}

#[derive(Clone, Debug)]
struct TierEvent {
    t: f64,
    kind: String,
}

fn extract_pool_profile(out: &str) -> Option<PoolProfile> {
    let workers = out.lines().rev().find_map(|line| {
        let (_, payload) = line.split_once("[pool] ")?;
        parse_key(payload, "workers")?.parse::<u64>().ok()
    });
    let line = out
        .lines()
        .rev()
        .find(|line| line.contains("[profile-pool] "))?;
    let (_, payload) = line.split_once("[profile-pool] ")?;
    Some(PoolProfile {
        band_ms: parse_key(payload, "band_ms")?.parse::<f64>().ok()?,
        quant_ms: parse_key(payload, "quant_ms")?.parse::<f64>().ok()?,
        dispatches: parse_key(payload, "dispatches")?.parse::<u64>().ok()?,
        workers,
    })
}

fn parse_key<'a>(payload: &'a str, wanted: &str) -> Option<&'a str> {
    payload.split_whitespace().find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (key == wanted).then_some(value)
    })
}

fn extract_decode_profile(out: &str) -> Option<DecodeProfile> {
    let line = out.lines().find(|l| l.contains("[profile-decode] "))?;
    let (_, payload) = line.split_once("[profile-decode] ")?;
    let mut decode_start_s = None;
    let mut decode_end_s = None;
    let mut p95 = None;
    let mut p99 = None;
    let mut max = None;
    let mut max_i = None;
    let mut max_s = None;
    for part in payload.split_whitespace() {
        let (key, value) = part.split_once('=')?;
        match key {
            "decode_start_s" => decode_start_s = value.parse::<f64>().ok(),
            "decode_end_s" => decode_end_s = value.parse::<f64>().ok(),
            "step_p95_ms" => p95 = value.parse::<f64>().ok(),
            "step_p99_ms" => p99 = value.parse::<f64>().ok(),
            "step_max_ms" => max = value.parse::<f64>().ok(),
            "step_max_i" => max_i = value.parse::<i64>().ok(),
            "step_max_s" => max_s = value.parse::<f64>().ok(),
            _ => {}
        }
    }
    let mut profile = DecodeProfile {
        decode_start_s: decode_start_s.unwrap_or(0.0),
        decode_end_s: decode_end_s.unwrap_or(0.0),
        step_p95_ms: p95?,
        step_p99_ms: p99?,
        step_max_ms: max?,
        step_max_i: max_i?,
        step_max_s: max_s.unwrap_or(0.0),
        tier_events: TierEventCounts::default(),
        tier_events_all: TierEventCounts::default(),
    };
    let tier_events = extract_tier_events(out);
    profile.tier_events = correlate_tier_events(&profile, &tier_events);
    profile.tier_events_all = count_tier_events(&tier_events);
    Some(profile)
}

fn extract_tier_events(out: &str) -> Vec<TierEvent> {
    out.lines()
        .filter_map(|line| {
            let (_, payload) = line.split_once("[tier-event] ")?;
            let mut t = None;
            let mut kind = None;
            for part in payload.split_whitespace() {
                let Some((key, value)) = part.split_once('=') else {
                    continue;
                };
                match key {
                    "t" => t = value.parse::<f64>().ok(),
                    "kind" => kind = Some(value.to_string()),
                    _ => {}
                }
            }
            Some(TierEvent { t: t?, kind: kind? })
        })
        .collect()
}

fn correlate_tier_events(profile: &DecodeProfile, events: &[TierEvent]) -> TierEventCounts {
    let mut counts = TierEventCounts::default();
    let has_window = profile.decode_start_s > 0.0 && profile.decode_end_s > profile.decode_start_s;
    let near_max_window_s = 2.0;
    for event in events {
        if has_window && (event.t < profile.decode_start_s || event.t > profile.decode_end_s) {
            continue;
        }
        add_tier_event_count(&mut counts, event);
        if profile.step_max_s > 0.0 && (event.t - profile.step_max_s).abs() <= near_max_window_s {
            counts.near_max += 1;
        }
    }
    counts
}

fn count_tier_events(events: &[TierEvent]) -> TierEventCounts {
    let mut counts = TierEventCounts::default();
    for event in events {
        add_tier_event_count(&mut counts, event);
    }
    counts
}

fn add_tier_event_count(counts: &mut TierEventCounts, event: &TierEvent) {
    counts.total += 1;
    if event.kind.contains("route") || event.kind == "promote_target" {
        counts.routes += 1;
    }
    if event.kind.contains("register") {
        counts.registers += 1;
    }
    if event.kind.contains("install") {
        counts.installs += 1;
    }
    if event.kind.contains("compile") || event.kind.contains("llvm") {
        counts.compiles += 1;
    }
    if event.kind.contains("error") || event.kind.contains("timeout") {
        counts.errors += 1;
    }
}

fn drift(samples: &[f64]) -> f64 {
    match (samples.first(), samples.last()) {
        (Some(first), Some(last)) if samples.len() > 1 => last - first,
        _ => f64::NAN,
    }
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
    let stddev = if samples.is_empty() {
        f64::NAN
    } else {
        let variance =
            samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / samples.len() as f64;
        variance.sqrt()
    };
    BenchSummary {
        successful,
        failed,
        samples,
        min,
        max,
        median,
        mean,
        stddev,
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
