#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unreachable_patterns,
    unused_mut,
    unused_assignments,
    unused_parens
)]
#![allow(
    clippy::single_component_path_imports,
    clippy::for_kv_map,
    clippy::explicit_auto_deref
)]
#![allow(
    clippy::println_empty_string,
    clippy::len_zero,
    clippy::useless_vec,
    clippy::field_reassign_with_default
)]
#![allow(
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::bool_assert_comparison
)]
#![allow(
    clippy::empty_line_after_doc_comments,
    clippy::useless_format,
    clippy::clone_on_copy
)]
//! Check preprocessor output for Bytes.hx

use parser::preprocessor::{PreprocessorConfig, preprocess};

fn main() {
    let source = std::fs::read_to_string("compiler/haxe-std/haxe/io/Bytes.hx")
        .expect("Failed to read Bytes.hx");

    // Test with rayzor define
    let mut config = PreprocessorConfig::default();
    config.defines.insert("rayzor".to_string());
    let result = preprocess(&source, &config);

    println!("=== Preprocessed output (first 500 chars) ===");
    println!("{}", &result[..result.len().min(500)]);

    // Show the non-comment lines
    println!("\n=== Non-comment lines ===");
    for (i, line) in result.lines().enumerate().take(40) {
        let trimmed = line.trim();
        if !trimmed.is_empty()
            && !trimmed.starts_with("/*")
            && !trimmed.starts_with("*")
            && !trimmed.starts_with("//")
        {
            println!("{:3}: {}", i + 1, line);
        }
    }

    // Check if it contains typedef or class
    if result.contains("typedef Bytes") {
        println!("\n✅ Found 'typedef Bytes' - preprocessor correctly chose rayzor branch");
    } else if result.contains("class Bytes") {
        println!("\n❌ Found 'class Bytes' - preprocessor incorrectly chose non-rayzor branch");
    } else {
        println!("\n⚠️ Neither typedef nor class found");
    }
}
