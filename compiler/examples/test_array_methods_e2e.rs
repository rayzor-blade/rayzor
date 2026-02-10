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
    clippy::clone_on_copy,
    clippy::vec_init_then_push
)]
//! Array methods end-to-end test suite
//!
//! Tests the complete pipeline for Array methods added in this session:
//! - indexOf, lastIndexOf, contains
//! - concat, splice, shift, unshift
//! - resize, toString

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};

/// Test result
#[derive(Debug)]
enum TestResult {
    Success,
    Failed { error: String },
}

impl TestResult {
    fn is_success(&self) -> bool {
        matches!(self, TestResult::Success)
    }
}

/// A single end-to-end test case
struct E2ETestCase {
    name: String,
    haxe_source: String,
}

impl E2ETestCase {
    fn new(name: &str, haxe_source: &str) -> Self {
        Self {
            name: name.to_string(),
            haxe_source: haxe_source.to_string(),
        }
    }

    fn run(&self) -> TestResult {
        println!("\n{}", "=".repeat(70));
        println!("TEST: {}", self.name);
        println!("{}", "=".repeat(70));

        let mut unit = CompilationUnit::new(CompilationConfig::fast());

        if let Err(e) = unit.load_stdlib() {
            return TestResult::Failed {
                error: format!("Failed to load stdlib: {}", e),
            };
        }

        let filename = format!("{}.hx", self.name);
        if let Err(e) = unit.add_file(&self.haxe_source, &filename) {
            return TestResult::Failed {
                error: format!("Failed to add file: {}", e),
            };
        }

        println!("  Compiling to TAST...");
        let typed_files = match unit.lower_to_tast() {
            Ok(files) => {
                println!("  TAST ({} files)", files.len());
                files
            }
            Err(errors) => {
                return TestResult::Failed {
                    error: format!("TAST failed: {:?}", errors),
                };
            }
        };

        println!("  Lowering to MIR...");
        let mir_modules = unit.get_mir_modules();
        if mir_modules.is_empty() {
            return TestResult::Failed {
                error: "No MIR modules generated".to_string(),
            };
        }
        println!("  MIR ({} modules)", mir_modules.len());

        println!("  Compiling to native...");
        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        let mut backend = match CraneliftBackend::with_symbols(&symbols_ref) {
            Ok(b) => b,
            Err(e) => {
                return TestResult::Failed {
                    error: format!("Backend init failed: {}", e),
                };
            }
        };

        for module in &mir_modules {
            if let Err(e) = backend.compile_module(module) {
                return TestResult::Failed {
                    error: format!("Codegen failed: {}", e),
                };
            }
        }
        println!("  Codegen succeeded");

        println!("  Executing...");
        for module in mir_modules.iter().rev() {
            if let Ok(()) = backend.call_main(module) {
                println!("  Execution succeeded");
                return TestResult::Success;
            }
        }

        TestResult::Failed {
            error: "Failed to execute main".to_string(),
        }
    }
}

fn main() {
    let mut tests = Vec::new();

    // ========================================================================
    // TEST 1: indexOf
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_indexOf",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30, 40, 50];
        var idx = arr.indexOf(30);
        trace(idx);  // 2
        var not_found = arr.indexOf(99);
        trace(not_found);  // -1
    }
}
"#,
    ));

    // ========================================================================
    // TEST 2: lastIndexOf
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_lastIndexOf",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30, 20, 10];
        var idx = arr.lastIndexOf(20);
        trace(idx);  // 3
    }
}
"#,
    ));

    // ========================================================================
    // TEST 3: contains
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_contains",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30];
        trace(arr.contains(20));  // true
        trace(arr.contains(99));  // false
    }
}
"#,
    ));

    // ========================================================================
    // TEST 4: concat
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_concat",
        r#"
class Main {
    static function main() {
        var a = [1, 2, 3];
        var b = [4, 5, 6];
        var c = a.concat(b);
        trace(c.length);  // 6
        trace(a.length);  // 3 (unchanged)
    }
}
"#,
    ));

    // ========================================================================
    // TEST 5: splice
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_splice",
        r#"
class Main {
    static function main() {
        var arr = [1, 2, 3, 4, 5];
        var removed = arr.splice(1, 2);
        trace(arr.length);      // 3 (1, 4, 5)
        trace(removed.length);  // 2 (2, 3)
    }
}
"#,
    ));

    // ========================================================================
    // TEST 6: shift
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_shift",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30];
        var first = arr.shift();
        trace(first);       // 10
        trace(arr.length);  // 2
    }
}
"#,
    ));

    // ========================================================================
    // TEST 7: unshift
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_unshift",
        r#"
class Main {
    static function main() {
        var arr = [2, 3, 4];
        arr.unshift(1);
        trace(arr.length);  // 4
    }
}
"#,
    ));

    // ========================================================================
    // TEST 8: resize
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_resize",
        r#"
class Main {
    static function main() {
        var arr = [1, 2, 3, 4, 5];
        arr.resize(3);
        trace(arr.length);  // 3
        arr.resize(6);
        trace(arr.length);  // 6
    }
}
"#,
    ));

    // ========================================================================
    // TEST 9: toString
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_toString",
        r#"
class Main {
    static function main() {
        var arr = [1, 2, 3];
        var s = arr.toString();
        trace(s);  // [1, 2, 3]
    }
}
"#,
    ));

    // ========================================================================
    // TEST 10: Combined operations
    // ========================================================================
    tests.push(E2ETestCase::new(
        "array_combined",
        r#"
class Main {
    static function main() {
        var arr = [1, 2, 3];
        arr.push(4);
        arr.unshift(0);
        // arr = [0, 1, 2, 3, 4]
        trace(arr.length);       // 5
        trace(arr.contains(3));  // true
        trace(arr.indexOf(2));   // 2
        var first = arr.shift();
        trace(first);            // 0
        trace(arr.length);       // 4
    }
}
"#,
    ));

    // Run all tests
    println!("========================================================================");
    println!("  Array Methods — E2E Test Suite");
    println!("========================================================================");

    let results: Vec<(String, TestResult)> =
        tests.iter().map(|t| (t.name.clone(), t.run())).collect();

    println!("\n\n{}", "=".repeat(70));
    println!("TEST SUMMARY");
    println!("{}", "=".repeat(70));

    let total = results.len();
    let passed = results.iter().filter(|(_, r)| r.is_success()).count();
    let failed = total - passed;

    println!("\nOverall:");
    println!("   Total:  {}", total);
    println!("   Passed: {} ({}%)", passed, passed * 100 / total);
    println!("   Failed: {}", failed);

    println!("\nResults:");
    for (name, result) in &results {
        match result {
            TestResult::Success => {
                println!("   PASS {} ", name);
            }
            TestResult::Failed { error } => {
                println!("   FAIL {} -- {}", name, error);
            }
        }
    }

    if failed == 0 {
        println!("\nAll tests passed!");
    } else {
        println!("\n{} test(s) failed", failed);
        std::process::exit(1);
    }
}
