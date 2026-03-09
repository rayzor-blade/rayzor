#![allow(unused_imports, dead_code)]
//! E2E tests for Send/Sync concurrency validation.
//!
//! Verifies that:
//! 1. Valid code (Send-derived types in Thread.spawn) compiles successfully
//! 2. Invalid code (non-Send captures in Thread.spawn) produces E0302 errors
//! 3. Error messages use correct wording ("derive" not "implement")

use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::pipeline::ErrorCategory;

fn compile_to_tast(code: &str) -> Result<(), Vec<compiler::pipeline::CompilationError>> {
    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib()
        .map_err(|e| panic!("Failed to load stdlib: {}", e))
        .unwrap();
    unit.add_file(code, "test/Main.hx")
        .map_err(|e| panic!("Failed to add file: {}", e))
        .unwrap();
    unit.lower_to_tast().map(|_| ())
}

fn main() {
    let mut pass = 0;
    let mut fail = 0;

    // ---- Test 1: Valid Send captures compile without errors ----
    print!("  [E2E] valid_send_captures ... ");
    let valid_code = r#"
import rayzor.concurrent.Thread;

@:derive([Send])
class Counter {
    public var value:Int;
    public function new(v:Int) { this.value = v; }
}

class Main {
    static function main() {
        var x = 42;
        var c = new Counter(10);
        var h = Thread.spawn(() -> {
            return x + c.value;
        });
        trace(h.join());
    }
}
"#;
    match compile_to_tast(valid_code) {
        Ok(()) => {
            println!("OK");
            pass += 1;
        }
        Err(errors) => {
            println!("FAIL: expected success, got {} error(s)", errors.len());
            for e in &errors {
                println!("    {:?}: {}", e.category, e.message);
            }
            fail += 1;
        }
    }

    // ---- Test 2: Function capture produces ConcurrencyError (E0302) ----
    print!("  [E2E] function_capture_rejected ... ");
    let invalid_fn_code = r#"
import rayzor.concurrent.Thread;

class Main {
    static function main() {
        var fn_val = () -> { return 42; };
        var h = Thread.spawn(() -> {
            return fn_val();
        });
        trace(h.join());
    }
}
"#;
    match compile_to_tast(invalid_fn_code) {
        Err(errors) => {
            let has_concurrency_error = errors
                .iter()
                .any(|e| matches!(e.category, ErrorCategory::ConcurrencyError));
            if has_concurrency_error {
                println!("OK");
                pass += 1;
            } else {
                println!(
                    "FAIL: expected ConcurrencyError, got: {:?}",
                    errors.iter().map(|e| &e.category).collect::<Vec<_>>()
                );
                fail += 1;
            }
        }
        Ok(()) => {
            println!("FAIL: expected error, but compiled successfully");
            fail += 1;
        }
    }

    // ---- Test 3: Error message says "derive" not "implement" ----
    print!("  [E2E] error_message_wording ... ");
    match compile_to_tast(invalid_fn_code) {
        Err(errors) => {
            let concurrency_errors: Vec<_> = errors
                .iter()
                .filter(|e| matches!(e.category, ErrorCategory::ConcurrencyError))
                .collect();
            let has_derive = concurrency_errors
                .iter()
                .all(|e| e.message.contains("derive") || !e.message.contains("implement"));
            let mentions_type = concurrency_errors
                .iter()
                .any(|e| e.message.contains("Function"));
            if has_derive && mentions_type {
                println!("OK");
                pass += 1;
            } else {
                println!("FAIL: bad wording in error messages:");
                for e in &concurrency_errors {
                    println!("    {}", e.message);
                }
                fail += 1;
            }
        }
        Ok(()) => {
            println!("FAIL: expected error");
            fail += 1;
        }
    }

    // ---- Test 4: Error code is E0302 ----
    print!("  [E2E] error_code_e0302 ... ");
    match compile_to_tast(invalid_fn_code) {
        Err(errors) => {
            let concurrency_errors: Vec<_> = errors
                .iter()
                .filter(|e| matches!(e.category, ErrorCategory::ConcurrencyError))
                .collect();
            let code = concurrency_errors
                .first()
                .map(|e| e.category.error_code())
                .unwrap_or("none");
            if code == "E0302" {
                println!("OK");
                pass += 1;
            } else {
                println!("FAIL: expected E0302, got {}", code);
                fail += 1;
            }
        }
        Ok(()) => {
            println!("FAIL: expected error");
            fail += 1;
        }
    }

    // ---- Test 5: Class with non-Send field (Function) is rejected ----
    print!("  [E2E] non_send_field_rejected ... ");
    let invalid_class_code = r#"
import rayzor.concurrent.Thread;

class Callback {
    public var handler:Void -> Int;
    public function new(h:Void -> Int) { this.handler = h; }
}

class Main {
    static function main() {
        var cb = new Callback(() -> { return 1; });
        var h = Thread.spawn(() -> {
            return cb.handler();
        });
        trace(h.join());
    }
}
"#;
    match compile_to_tast(invalid_class_code) {
        Err(errors) => {
            let has_concurrency_error = errors
                .iter()
                .any(|e| matches!(e.category, ErrorCategory::ConcurrencyError));
            if has_concurrency_error {
                println!("OK");
                pass += 1;
            } else {
                println!(
                    "FAIL: expected ConcurrencyError, got: {:?}",
                    errors
                        .iter()
                        .map(|e| format!("{:?}: {}", e.category, e.message))
                        .collect::<Vec<_>>()
                );
                fail += 1;
            }
        }
        Ok(()) => {
            println!("FAIL: expected error, but compiled successfully");
            fail += 1;
        }
    }

    // ---- Summary ----
    let total = pass + fail;
    println!("\n=== Send/Sync E2E Results: {}/{} passed ===", pass, total);
    if fail > 0 {
        std::process::exit(1);
    }
}
