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

    let source = std::env::var_os("RAYZOR_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/stdlib_snapshot.bin"));

    println!("cargo:rerun-if-env-changed=RAYZOR_SNAPSHOT");
    println!("cargo:rerun-if-changed={}", source.display());

    match std::fs::read(&source) {
        Ok(bytes) => {
            println!(
                "cargo:warning=embedding standard library snapshot ({} KB)",
                bytes.len() / 1024
            );
            std::fs::write(&dest, bytes).expect("failed to stage the snapshot");
        }
        Err(_) => {
            // An empty archive keeps the include site valid; the loader reads
            // it as "carries nothing" and compilation falls back to source.
            std::fs::write(&dest, empty_archive()).expect("failed to stage an empty snapshot");
        }
    }
}

/// `RZSN` followed by a zero entry count — see `compiler::ir::snapshot`.
fn empty_archive() -> Vec<u8> {
    let mut bytes = b"RZSN".to_vec();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes
}
