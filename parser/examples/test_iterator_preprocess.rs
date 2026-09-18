use parser::preprocessor::{PreprocessorConfig, preprocess};

fn main() {
    let source =
        std::fs::read_to_string("compiler/haxe-std/haxe/iterators/StringIteratorUnicode.hx")
            .expect("Failed to read StringIteratorUnicode.hx");

    let config = PreprocessorConfig::default();
    let result = preprocess(&source, &config);

    println!("=== Preprocessed StringIteratorUnicode.hx ===");
    for (i, line) in result.lines().enumerate() {
        println!("{:3}: {}", i + 1, line);
    }
    println!("=== End ===");
}
