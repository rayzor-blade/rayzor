//! Embed the lowered standard library the binary carries, if one was produced.
//!
//! The archive is built by the `snapshot-gen` binary rather than here: the
//! compiler depends on the runtime, and reaching the compiler from a build
//! script would put that crate in the host graph alongside the target graph's
//! copy, which collide over the same output filename.
//!
//! Without an archive the binary still works — the loader finds nothing and
//! compiles the library from source, exactly as before — so a plain
//! `cargo build` needs no extra step.

use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let dest = out_dir.join("stdlib_snapshot.bin");

    // Being told where the archive is and finding one there are different
    // contracts. A plain `cargo build` may find nothing and fall back to
    // source, as documented above; a build given a path has no second reading,
    // and quietly shipping a binary that carries nothing is the failure this
    // whole mechanism exists to avoid.
    let requested = std::env::var_os("RAYZOR_SNAPSHOT").map(PathBuf::from);
    let source = requested
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/stdlib_snapshot.bin"));

    println!("cargo:rerun-if-env-changed=RAYZOR_SNAPSHOT");
    println!("cargo:rerun-if-changed={}", source.display());

    match std::fs::read(&source) {
        Ok(bytes) => {
            // A stale or truncated archive still parses; it just indexes to
            // nothing, which is the same empty binary by another route.
            if requested.is_some() && module_count(&bytes) == 0 {
                panic!(
                    "RAYZOR_SNAPSHOT={} carries no modules; regenerate it with \
                     `cargo run --release -p snapshot-gen -- {}`",
                    source.display(),
                    source.display()
                );
            }
            println!(
                "cargo:warning=embedding standard library snapshot ({} KB)",
                bytes.len() / 1024
            );
            std::fs::write(&dest, bytes).expect("failed to stage the snapshot");
        }
        Err(err) => {
            if requested.is_some() {
                panic!("RAYZOR_SNAPSHOT={} cannot be read: {err}", source.display());
            }
            // An empty archive keeps the include site valid; the loader reads
            // it as "carries nothing" and compilation falls back to source.
            std::fs::write(&dest, empty_archive()).expect("failed to stage an empty snapshot");
        }
    }
}

/// Entry count of an archive, or zero if it is not one. This reads the same
/// header `compiler::ir::snapshot::index` does, spelled out again because a
/// build script cannot depend on the crate it builds.
fn module_count(bytes: &[u8]) -> u32 {
    if bytes.len() < 8 || &bytes[..4] != b"RZSN" {
        return 0;
    }
    u32::from_le_bytes(bytes[4..8].try_into().expect("8 bytes checked above"))
}

/// `RZSN` followed by a zero entry count — see `compiler::ir::snapshot`.
fn empty_archive() -> Vec<u8> {
    let mut bytes = b"RZSN".to_vec();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes
}
