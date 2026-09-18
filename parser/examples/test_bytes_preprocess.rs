use parser::parse_haxe_file_with_diagnostics;
use parser::preprocessor::{PreprocessorConfig, preprocess};

fn main() {
    let source = std::fs::read_to_string("compiler/haxe-std/haxe/io/Bytes.hx")
        .expect("Failed to read Bytes.hx");

    let config = PreprocessorConfig::default();
    let result = preprocess(&source, &config);

    // Print lines of preprocessed output
    let lines: Vec<&str> = result.lines().collect();
    println!("Total preprocessed lines: {}", lines.len());

    // Print first 30 lines
    println!("\n--- First 30 lines ---");
    for (i, line) in lines.iter().enumerate().take(30) {
        println!("{:4}: {}", i + 1, line);
    }

    // Check for untyped
    println!("\n--- Lines containing 'untyped' ---");
    for (i, line) in lines.iter().enumerate() {
        if line.contains("untyped") {
            println!("{:4}: {}", i + 1, line);
        }
    }

    // Now try to parse the preprocessed source
    println!("\n--- Attempting to parse preprocessed Bytes.hx ---");
    match parse_haxe_file_with_diagnostics("Bytes.hx", &result) {
        Ok(file) => {
            println!(
                "SUCCESS: Parsed {} declarations",
                file.file.declarations.len()
            );
        }
        Err(e) => {
            println!("FAILED to parse: {}", e);
        }
    }
}
