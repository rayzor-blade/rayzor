use parser::preprocessor::{PreprocessorConfig, preprocess};

#[test]
fn test_preprocess_with_leading_comment() {
    let source = r#"/*
 * Copyright notice
 */
#if jvm
code for jvm
#end
package;
"#;

    let config = PreprocessorConfig::default();
    let result = preprocess(source, &config);

    println!("=== ORIGINAL ===\n{}", source);
    println!("\n=== PREPROCESSED ===\n{}", result);
    println!("\n=== Result bytes: {} ===", result.len());

    // The preprocessed result should still have the comment and package
    assert!(result.contains("Copyright notice"));
    assert!(result.contains("package"));

    // jvm block should be removed
    assert!(!result.contains("code for jvm"));
}
