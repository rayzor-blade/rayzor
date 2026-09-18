#[cfg(test)]
mod tests {
    use crate::tast::{AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeTable};
    use parser::{ErrorFormatter, parse_haxe_file_with_diagnostics};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_parser_error_formatting() {
        // This Haxe code has intentional syntax errors
        let haxe_code = r#"
class TestErrors {
    private var field:Int

    public function new() {
        this.field = 42;
    }

    public function test() {
        var x = this.field
        // Missing semicolon above

        if (x > 10 {  // Missing closing parenthesis
            trace("Large");
        }
    }
}"#;

        // Parse to get diagnostics
        match parse_haxe_file_with_diagnostics("test_errors.hx", haxe_code) {
            Ok(parse_result) => {
                // Even if parsing succeeded, check for diagnostics
                if parse_result.diagnostics.is_empty() {
                    panic!("Expected parse errors but no diagnostics were collected");
                }

                let formatter = ErrorFormatter::with_colors();
                let error_output = formatter
                    .format_diagnostics(&parse_result.diagnostics, &parse_result.source_map);
                println!("Parser diagnostics:\n{}", error_output);

                // Verify the error output contains expected formatting
                assert!(
                    error_output.contains("error") || error_output.contains("warning"),
                    "Error output should contain severity level"
                );
                assert!(
                    error_output.contains("test_errors.hx"),
                    "Error output should contain filename"
                );
                assert!(
                    error_output.contains("-->"),
                    "Error output should contain source location arrow"
                );

                // The error formatter should show the problematic lines with context
                assert!(
                    error_output.contains("var x = this.field"),
                    "Error should show the line missing semicolon"
                );
                assert!(
                    error_output.contains("if (x > 10 {"),
                    "Error should show the line with missing parenthesis"
                );
            }
            Err(error_output) => {
                println!("Parser error output:\n{}", error_output);

                // Also verify error formatting in case of complete failure
                assert!(
                    error_output.contains("error"),
                    "Error output should contain 'error' severity"
                );
                assert!(
                    error_output.contains("test_errors.hx"),
                    "Error output should contain filename"
                );
            }
        }
    }

    #[test]
    fn test_parser_error_with_suggestions() {
        let haxe_code = r#"
class Test {
    function method() {
        var x = 10
        var y = 20;
    }
}"#;

        match parse_haxe_file_with_diagnostics("test_semicolon.hx", haxe_code) {
            Ok(parse_result) => {
                if parse_result.diagnostics.is_empty() {
                    panic!("Expected diagnostics for missing semicolon");
                }

                let formatter = ErrorFormatter::with_colors();
                let error_output = formatter
                    .format_diagnostics(&parse_result.diagnostics, &parse_result.source_map);
                println!("Semicolon diagnostics:\n{}", error_output);

                // Check for proper error formatting
                assert!(
                    error_output.contains("error") || error_output.contains("warning"),
                    "Should show severity level"
                );
                assert!(
                    error_output.contains("test_semicolon.hx"),
                    "Should show filename"
                );
                assert!(
                    error_output.contains("var x = 10"),
                    "Should show the problematic line"
                );

                // The diagnostics system should provide a helpful suggestion
                // Note: The exact message depends on the parser implementation
            }
            Err(error_output) => {
                println!("Parser error output:\n{}", error_output);

                // Also verify error formatting in case of complete failure
                assert!(error_output.contains("error"), "Should show error severity");
                assert!(
                    error_output.contains("test_semicolon.hx"),
                    "Should show filename"
                );
            }
        }
    }
}
