//! Lower the standard library and write the archive the CLI embeds.
//!
//! Compiling a program means lowering whatever part of the standard library it
//! reaches — for anything touching `Sys`, a closure of about forty modules.
//! That work is the same for every program, so doing it here means no program
//! pays for it, including the first one run in a fresh checkout.
//!
//! This is a normal workspace binary rather than a build script: the compiler
//! depends on the runtime, and pulling that into a build script's host graph
//! collides with the same crate in the target graph.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Reach the parts of the library a real program reaches. An empty `main`
/// lowers almost nothing, leaving exactly the closure that costs a cold
/// compile unprepared.
const WARMUP: &str = r#"
class Main {
    static function main() {
        var m = new Map<String, Int>();
        m.set("k", 1);
        var b = new StringBuf();
        b.add(StringTools.trim(" v "));
        var xs = [1, 2, 3];
        var total = 0;
        for (x in xs) total += x + m.get("k") + Std.int(Math.sqrt(x));
        Sys.println(b.toString() + Std.string(total));
    }
}
"#;

/// An artifact is only found again under the same cache discriminator, and the
/// discriminator follows the preprocessor defines. A run selects one of these
/// by the tier it compiles for, so the archive carries all three.
const TIERS: [&str; 3] = ["jit", "llvm", "interp"];

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/stdlib_snapshot.bin"));

    let staging = std::env::temp_dir().join("rayzor-snapshot-staging");
    let _ = std::fs::remove_dir_all(&staging);

    for tier in TIERS {
        lower_for(tier, &staging);
    }

    let mut entries = BTreeMap::new();
    collect(&staging, &staging, &mut entries);
    if entries.is_empty() {
        eprintln!("no standard library artifacts were produced");
        std::process::exit(1);
    }

    let archive = compiler::ir::snapshot::encode(&entries);
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out, &archive).expect("failed to write the snapshot archive");
    // The compiler that produced these artifacts. A cache entry is only valid
    // for a compiler with the same id, so the build records it beside the
    // archive and warns when the two have drifted apart.
    std::fs::write(out.with_extension("build-id"), compiler::BUILD_ID)
        .expect("failed to write the snapshot build id");
    let _ = std::fs::remove_dir_all(&staging);

    println!(
        "{} modules, {} KB -> {}",
        entries.len(),
        archive.len() / 1024,
        out.display()
    );
}

fn lower_for(tier: &str, staging: &Path) {
    use compiler::compilation::{CompilationConfig, CompilationUnit};

    // Lower from source, never from the library this binary already carries:
    // a restored module writes no artifact, so a build whose carried snapshot
    // is still valid would otherwise produce an empty archive.
    std::env::set_var("RAYZOR_IGNORE_EMBEDDED_SNAPSHOT", "1");

    let config = CompilationConfig {
        extra_defines: vec![tier.to_string()],
        cache_dir: Some(staging.to_path_buf()),
        ..Default::default()
    };

    let mut unit = CompilationUnit::new(config);
    if unit.load_stdlib().is_err() || unit.add_file(WARMUP, "snapshot_warmup.hx").is_err() {
        return;
    }
    let _ = unit.lower_to_tast();
}

/// Key every artifact by its path relative to the staging root, which is
/// `<discriminator>/<module>.blade` — the key the loader asks for.
fn collect(root: &Path, dir: &Path, entries: &mut BTreeMap<String, Vec<u8>>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(root, &path, entries);
        } else if path.extension().is_some_and(|e| e == "blade") {
            if let (Ok(rel), Ok(bytes)) = (path.strip_prefix(root), std::fs::read(&path)) {
                if let Some(key) = rel.to_str() {
                    entries.insert(key.replace('\\', "/"), bytes);
                }
            }
        }
    }
}
