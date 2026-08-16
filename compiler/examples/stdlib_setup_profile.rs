//! What the compiler spends before it looks at a line of user code.
//!
//! Preparing the compiler — constructing the unit and registering the standard
//! library's declarations — is the same work for every program, so it belongs
//! to a setup phase done once. This reports what it actually costs per process,
//! against the compile it precedes, so the two are never read as one number.

use compiler::compilation::{CompilationConfig, CompilationUnit};
use std::time::Instant;

const SOURCE: &str = r#"
class Main {
    static function main() {
        var total = 0;
        for (i in 0...10) total += i;
        Sys.println(Std.string(total));
    }
}
"#;

fn main() {
    let runs: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(5);

    // `loop` mode does setup and nothing else, long enough for a sampling
    // profiler to say where the time inside it goes.
    if std::env::args().any(|a| a == "loop") {
        let iterations: usize = std::env::args()
            .nth(1)
            .and_then(|a| a.parse().ok())
            .unwrap_or(200);
        let start = Instant::now();
        for _ in 0..iterations {
            let config = CompilationConfig {
                extra_defines: vec!["jit".to_string()],
                ..Default::default()
            };
            let mut unit = CompilationUnit::new(config);
            unit.load_stdlib().expect("stdlib");
            std::hint::black_box(&unit);
        }
        let total = start.elapsed();
        println!(
            "{iterations} registrations in {:.2}s — {:.2}ms each",
            total.as_secs_f64(),
            total.as_secs_f64() * 1000.0 / iterations as f64
        );
        return;
    }

    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>7}",
        "run", "new()", "load_stdlib", "compile", "setup %"
    );

    for run in 1..=runs {
        let config = CompilationConfig {
            extra_defines: vec!["jit".to_string()],
            ..Default::default()
        };

        let t0 = Instant::now();
        let mut unit = CompilationUnit::new(config);
        let construct = t0.elapsed();

        let t1 = Instant::now();
        unit.load_stdlib().expect("stdlib");
        let register = t1.elapsed();

        let t2 = Instant::now();
        unit.add_file(SOURCE, "profile.hx").expect("parse");
        unit.lower_to_tast().expect("tast");
        let _ = unit.get_mir_modules();
        let compile = t2.elapsed();

        let setup = construct + register;
        let share = setup.as_secs_f64() / (setup + compile).as_secs_f64() * 100.0;
        println!(
            "{:>8}  {:>9.2}ms  {:>9.2}ms  {:>9.2}ms  {:>6.1}%",
            run,
            construct.as_secs_f64() * 1000.0,
            register.as_secs_f64() * 1000.0,
            compile.as_secs_f64() * 1000.0,
            share
        );
    }
}
