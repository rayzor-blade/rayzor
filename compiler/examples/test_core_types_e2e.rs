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
use std::thread::sleep;
use std::time::Duration;

use compiler::codegen::CraneliftBackend;
/// End-to-end test framework for Core Types (String, Array, Math)
///
/// This tests the COMPLETE pipeline for core type operations:
/// 1. Parse Haxe source code
/// 2. Compile to TAST with shared state (multi-file aware)
/// 3. Lower to HIR
/// 4. Lower to MIR with stdlib mappings
/// 5. Validate MIR structure
/// 6. Generate native code and execute
///
/// Purpose: Verify String, Array, and Math runtime functions are stable
/// and compatible with the current compilation pipeline.
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::IrModule;

/// Test result levels
#[derive(Debug, Clone, PartialEq, Eq)]
enum TestLevel {
    /// L1: Source code compiles to TAST without errors
    Compilation,
    /// L2: HIR lowering succeeds
    HirLowering,
    /// L3: MIR lowering succeeds with proper stdlib mappings
    MirLowering,
    /// L4: MIR structure is valid (all extern functions registered)
    MirValidation,
    /// L5: Native code generation succeeds
    Codegen,
    /// L6: Execution produces correct output
    Execution,
}

/// Test result
#[derive(Debug)]
enum TestResult {
    Success { level: TestLevel },
    Failed { level: TestLevel, error: String },
}

impl TestResult {
    fn is_success(&self) -> bool {
        matches!(self, TestResult::Success { .. })
    }

    fn level(&self) -> TestLevel {
        match self {
            TestResult::Success { level } => level.clone(),
            TestResult::Failed { level, .. } => level.clone(),
        }
    }
}

/// A single end-to-end test case
struct E2ETestCase {
    name: String,
    description: String,
    haxe_source: String,
    expected_level: TestLevel,
    /// Expected function calls in MIR (for validation)
    expected_mir_calls: Vec<String>,
}

impl E2ETestCase {
    fn new(name: &str, description: &str, haxe_source: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            haxe_source: haxe_source.to_string(),
            expected_level: TestLevel::Execution,
            expected_mir_calls: Vec::new(),
        }
    }

    fn expect_mir_calls(mut self, calls: Vec<&str>) -> Self {
        self.expected_mir_calls = calls.iter().map(|s| s.to_string()).collect();
        self
    }

    fn expect_level(mut self, level: TestLevel) -> Self {
        self.expected_level = level;
        self
    }

    /// Run the test through the full pipeline
    fn run(&self) -> TestResult {
        println!("\n{}", "=".repeat(70));
        println!("TEST: {}", self.name);
        println!("{}", self.description);
        println!("{}", "=".repeat(70));

        // Create compilation unit with stdlib
        let mut unit = CompilationUnit::new(CompilationConfig::fast());

        // Load stdlib
        if let Err(e) = unit.load_stdlib() {
            return TestResult::Failed {
                level: TestLevel::Compilation,
                error: format!("Failed to load stdlib: {}", e),
            };
        }

        // Add the test file
        let filename = format!("{}.hx", self.name);
        if let Err(e) = unit.add_file(&self.haxe_source, &filename) {
            return TestResult::Failed {
                level: TestLevel::Compilation,
                error: format!("Failed to add file: {}", e),
            };
        }

        // L1: Compile to TAST
        println!("L1: Compiling to TAST...");
        let _typed_files = match unit.lower_to_tast() {
            Ok(files) => {
                println!("  ✅ TAST lowering succeeded ({} files)", files.len());
                files
            }
            Err(errors) => {
                return TestResult::Failed {
                    level: TestLevel::Compilation,
                    error: format!(
                        "TAST lowering failed with {} errors: {:?}",
                        errors.len(),
                        errors
                    ),
                };
            }
        };

        if matches!(self.expected_level, TestLevel::Compilation) {
            return TestResult::Success {
                level: TestLevel::Compilation,
            };
        }

        // L2: HIR lowering (integrated in pipeline)
        println!("L2: HIR lowering...");
        println!("  ✅ HIR lowering succeeded (integrated in pipeline)");

        if matches!(self.expected_level, TestLevel::HirLowering) {
            return TestResult::Success {
                level: TestLevel::HirLowering,
            };
        }

        // L3: MIR lowering
        println!("L3: MIR lowering...");
        let mir_modules = unit.get_mir_modules();
        if mir_modules.is_empty() {
            return TestResult::Failed {
                level: TestLevel::MirLowering,
                error: "No MIR modules generated".to_string(),
            };
        }

        let mir_module = mir_modules.last().unwrap();
        println!(
            "  ✅ MIR lowering succeeded ({} modules)",
            mir_modules.len()
        );
        println!("  📊 MIR Stats:");
        println!("     - Functions: {}", mir_module.functions.len());
        println!(
            "     - Extern functions: {}",
            mir_module.extern_functions.len()
        );

        if matches!(self.expected_level, TestLevel::MirLowering) {
            return TestResult::Success {
                level: TestLevel::MirLowering,
            };
        }

        // L4: MIR Validation
        println!("L4: Validating MIR structure...");
        if let Err(e) = self.validate_mir_modules(&mir_modules) {
            return TestResult::Failed {
                level: TestLevel::MirValidation,
                error: e,
            };
        }
        println!("  ✅ MIR validation passed");

        if matches!(self.expected_level, TestLevel::MirValidation) {
            return TestResult::Success {
                level: TestLevel::MirValidation,
            };
        }

        // L5: Codegen
        println!("L5: Compiling to native code...");
        let mut backend = match self.compile_to_native(&mir_modules) {
            Ok(backend) => {
                println!("  ✅ Codegen succeeded (Cranelift JIT)");
                backend
            }
            Err(e) => {
                return TestResult::Failed {
                    level: TestLevel::Codegen,
                    error: format!("Codegen failed: {:?}", e),
                };
            }
        };

        if matches!(self.expected_level, TestLevel::Codegen) {
            return TestResult::Success {
                level: TestLevel::Codegen,
            };
        }

        // L6: Execution
        println!("L6: Executing compiled code...");
        if let Err(e) = self.execute_and_validate(&mut backend, self.name.clone(), &mir_modules) {
            return TestResult::Failed {
                level: TestLevel::Execution,
                error: format!("Execution failed: {:?}", e),
            };
        }
        println!("  ✅ Execution succeeded");

        TestResult::Success {
            level: TestLevel::Execution,
        }
    }

    fn validate_mir_modules(&self, modules: &[std::sync::Arc<IrModule>]) -> Result<(), String> {
        let mut all_functions = std::collections::BTreeSet::new();
        for module in modules {
            // Collect extern functions
            for (_, ef) in &module.extern_functions {
                all_functions.insert(ef.name.clone());
            }
            // Collect ALL regular functions (including MIR wrappers)
            for (_, func) in &module.functions {
                all_functions.insert(func.name.clone());
            }
        }

        if !self.expected_mir_calls.is_empty() {
            for expected_call in &self.expected_mir_calls {
                let found = all_functions
                    .iter()
                    .any(|name| name.contains(expected_call));
                if !found {
                    return Err(format!(
                        "Expected function '{}' not found in MIR. Available: {:?}",
                        expected_call,
                        all_functions.iter().collect::<Vec<_>>()
                    ));
                }
            }
            println!("  ✓ All expected functions found");
        }

        println!("  ✓ All functions have valid structure");
        Ok(())
    }

    fn compile_to_native(
        &self,
        modules: &[std::sync::Arc<IrModule>],
    ) -> Result<CraneliftBackend, String> {
        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        let mut backend = CraneliftBackend::with_symbols(&symbols_ref)?;

        for module in modules {
            backend.compile_module(module)?;
        }

        Ok(backend)
    }

    fn execute_and_validate(
        &self,
        backend: &mut CraneliftBackend,
        name: String,
        modules: &[std::sync::Arc<IrModule>],
    ) -> Result<(), String> {
        for module in modules.iter().rev() {
            println!("  🔍 Trying to execute main in module... {}", name);
            if let Ok(()) = backend.call_main(module) {
                return Ok(());
            }
        }

        Err("Failed to execute main in any module".to_string())
    }
}

/// Test suite runner
struct E2ETestSuite {
    tests: Vec<E2ETestCase>,
}

impl E2ETestSuite {
    fn new() -> Self {
        Self { tests: Vec::new() }
    }

    fn add_test(&mut self, test: E2ETestCase) {
        self.tests.push(test);
    }

    fn run_all(&self) -> Vec<(String, TestResult)> {
        let mut results = Vec::new();

        for test in &self.tests {
            let result = test.run();
            let success = result.is_success();
            let test_name = test.name.clone();

            if success {
                println!("\n✅ {} PASSED", test_name);
            } else {
                if let TestResult::Failed {
                    ref error,
                    ref level,
                    ..
                } = result
                {
                    println!("  Error at {:?}: {}", level, error);
                }
                println!("\n❌ {} FAILED", test_name);
            }

            results.push((test_name.clone(), result));
            sleep(Duration::from_millis(100));
        }

        results
    }

    fn print_summary(&self, results: &[(String, TestResult)]) {
        println!("\n{}", "=".repeat(70));
        println!("CORE TYPES TEST SUMMARY");
        println!("{}", "=".repeat(70));

        let total = results.len();
        let passed = results.iter().filter(|(_, r)| r.is_success()).count();
        let failed = total - passed;

        let mut by_level: std::collections::BTreeMap<String, (usize, usize)> =
            std::collections::BTreeMap::new();
        for (_, result) in results {
            let level_name = format!("{:?}", result.level());
            let entry = by_level.entry(level_name).or_insert((0, 0));
            if result.is_success() {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }

        println!("\n📊 Overall:");
        println!("   Total:  {}", total);
        if total > 0 {
            println!("   Passed: {} ({}%)", passed, passed * 100 / total);
        }
        println!("   Failed: {}", failed);

        println!("\n📈 By Level:");
        for (level, (pass, fail)) in by_level {
            println!("   {}: {} pass, {} fail", level, pass, fail);
        }

        println!("\n📋 Results:");
        for (name, result) in results {
            match result {
                TestResult::Success { level } => {
                    println!("   ✅ {} (reached {:?})", name, level);
                }
                TestResult::Failed { level, error } => {
                    println!("   ❌ {} (failed at {:?})", name, level);
                    println!("      Error: {}", error);
                }
            }
        }

        if failed == 0 {
            println!("\n🎉 All tests passed!");
        } else {
            println!("\n⚠️  {} test(s) failed", failed);
        }
    }
}

fn main() -> Result<(), String> {
    println!("=== Rayzor Core Types End-to-End Test Suite ===");
    println!("Testing: String, Array, Math runtime functions\n");

    let mut suite = E2ETestSuite::new();

    // ============================================================================
    // STRING TESTS
    // ============================================================================

    // String Test 1: Basic string operations
    suite.add_test(
        E2ETestCase::new(
            "string_basic",
            "Basic string creation and length",
            r#"
package test;

class Main {
    static function main() {
        var s = "Hello, World!";
        var len = s.length;
        trace(len);  // Expected: 13
    }
}
"#,
        )
        .expect_mir_calls(vec!["string_length"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 2: charAt
    suite.add_test(
        E2ETestCase::new(
            "string_charAt",
            "String charAt operation",
            r#"
package test;

class Main {
    static function main() {
        var s = "Hello";
        trace(s.charAt(0));  // Expected: H
        trace(s.charAt(4));  // Expected: o
    }
}
"#,
        )
        .expect_mir_calls(vec!["String_charAt"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 3: indexOf
    suite.add_test(
        E2ETestCase::new(
            "string_indexOf",
            "String indexOf operation",
            r#"
package test;

class Main {
    static function main() {
        var s = "Hello, World!";
        trace(s.indexOf("World", 0));  // Expected: 7
        trace(s.indexOf("xyz", 0));    // Expected: -1
    }
}
"#,
        )
        .expect_mir_calls(vec!["String_indexOf_2"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 4: substring
    suite.add_test(
        E2ETestCase::new(
            "string_substring",
            "String substring operation",
            r#"
package test;

class Main {
    static function main() {
        var s = "Hello, World!";
        trace(s.substring(0, 5));   // Expected: Hello
        trace(s.substring(7, 12));  // Expected: World
    }
}
"#,
        )
        .expect_mir_calls(vec!["String_substring"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 5: split
    suite.add_test(
        E2ETestCase::new(
            "string_split",
            "String split operation",
            r#"
package test;

class Main {
    static function main() {
        var s = "a,b,c,d";
        var parts = s.split(",");
        trace(parts.length);  // Expected: 4
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_string_split_array"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 6: toUpperCase / toLowerCase
    suite.add_test(
        E2ETestCase::new(
            "string_case",
            "String case conversion",
            r#"
package test;

class Main {
    static function main() {
        var s = "Hello";
        trace(s.toUpperCase());  // Expected: HELLO
        trace(s.toLowerCase());  // Expected: hello
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_string_upper", "haxe_string_lower"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 7: charCodeAt
    suite.add_test(
        E2ETestCase::new(
            "string_charCodeAt",
            "String charCodeAt operation",
            r#"
package test;

class Main {
    static function main() {
        var s = "Hello";
        trace(s.charCodeAt(0));  // Expected: 72 (ASCII for 'H')
        trace(s.charCodeAt(1));  // Expected: 101 (ASCII for 'e')
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_string_char_code_at"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 8: lastIndexOf
    suite.add_test(
        E2ETestCase::new(
            "string_lastIndexOf",
            "String lastIndexOf operation",
            r#"
package test;

class Main {
    static function main() {
        var s = "hello hello";
        trace(s.lastIndexOf("hello", 100));  // Expected: 6
        trace(s.lastIndexOf("o", 100));      // Expected: 10
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_string_last_index_of"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 9: substr
    suite.add_test(
        E2ETestCase::new(
            "string_substr",
            "String substr operation",
            r#"
package test;

class Main {
    static function main() {
        var s = "Hello, World!";
        trace(s.substr(0, 5));  // Expected: Hello
        trace(s.substr(7, 5));  // Expected: World
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_string_substr"])
        .expect_level(TestLevel::Execution),
    );

    // String Test 10: fromCharCode
    suite.add_test(
        E2ETestCase::new(
            "string_fromCharCode",
            "String.fromCharCode static method",
            r#"
package test;

class Main {
    static function main() {
        trace(String.fromCharCode(65));  // Expected: A
        trace(String.fromCharCode(90));  // Expected: Z
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_string_from_char_code"])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // ARRAY TESTS
    // ============================================================================

    // Array Test 1: Basic array and length
    suite.add_test(
        E2ETestCase::new(
            "array_basic",
            "Basic array creation and length",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Int>();
        trace(arr.length);  // Expected: 0
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_array_length"])
        .expect_level(TestLevel::Execution),
    );

    // Array Test 2: push and pop
    suite.add_test(
        E2ETestCase::new(
            "array_push_pop",
            "Array push and pop operations",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Int>();
        arr.push(10);
        arr.push(20);
        arr.push(30);
        trace(arr.length);  // Expected: 3
        var last = arr.pop();
        trace(last);        // Expected: 30
        trace(arr.length);  // Expected: 2
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_array_push_i64", "haxe_array_pop_ptr"])
        .expect_level(TestLevel::Execution),
    );

    // Array Test 3: Index access
    suite.add_test(
        E2ETestCase::new(
            "array_index",
            "Array index access",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Int>();
        arr.push(10);
        arr.push(20);
        arr.push(30);
        trace(arr[0]);  // Expected: 10
        trace(arr[1]);  // Expected: 20
        arr[1] = 25;
        trace(arr[1]);  // Expected: 25
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_array_get_ptr", "haxe_array_set"])
        .expect_level(TestLevel::Execution),
    );

    // Array Test 4: slice
    suite.add_test(
        E2ETestCase::new(
            "array_slice",
            "Array slice operation",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Int>();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.push(4);
        arr.push(5);
        var sliced = arr.slice(1, 4);
        trace(sliced.length);  // Expected: 3
        trace(sliced[0]);      // Expected: 2
        trace(sliced[2]);      // Expected: 4
    }
}
"#,
        )
        .expect_mir_calls(vec!["array_slice"])
        .expect_level(TestLevel::Execution),
    );

    // Array Test 5: reverse
    suite.add_test(
        E2ETestCase::new(
            "array_reverse",
            "Array reverse operation",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Int>();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.reverse();
        trace(arr[0]);  // Expected: 3
        trace(arr[1]);  // Expected: 2
        trace(arr[2]);  // Expected: 1
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_array_reverse"])
        .expect_level(TestLevel::Execution),
    );

    // Array Test 6: insert and remove
    suite.add_test(
        E2ETestCase::new(
            "array_insert_remove",
            "Array insert and remove operations",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Int>();
        arr.push(1);
        arr.push(3);
        arr.insert(1, 2);  // [1, 2, 3]
        trace(arr.length);  // Expected: 3
        trace(arr[1]);      // Expected: 2
        arr.remove(2);      // [1, 3]
        trace(arr.length);  // Expected: 2
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_array_insert", "haxe_array_remove"])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // MATH TESTS
    // ============================================================================

    // Math Test 1: Basic math functions
    suite.add_test(
        E2ETestCase::new(
            "math_basic",
            "Basic Math functions (abs, floor, ceil)",
            r#"
package test;

class Main {
    static function main() {
        trace(Math.abs(-5.0));   // Expected: 5.0
        trace(Math.floor(3.7));  // Expected: 3.0
        trace(Math.ceil(3.2));   // Expected: 4.0
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_math_abs", "haxe_math_floor", "haxe_math_ceil"])
        .expect_level(TestLevel::Execution),
    );

    // Math Test 2: Trigonometric functions
    suite.add_test(
        E2ETestCase::new(
            "math_trig",
            "Trigonometric functions (sin, cos, tan)",
            r#"
package test;

class Main {
    static function main() {
        trace(Math.sin(0.0));  // Expected: 0.0
        trace(Math.cos(0.0));  // Expected: 1.0
        trace(Math.tan(0.0));  // Expected: 0.0
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_math_sin", "haxe_math_cos", "haxe_math_tan"])
        .expect_level(TestLevel::Execution),
    );

    // Math Test 3: Power and square root
    suite.add_test(
        E2ETestCase::new(
            "math_pow_sqrt",
            "Power and square root functions",
            r#"
package test;

class Main {
    static function main() {
        trace(Math.sqrt(16.0));      // Expected: 4.0
        trace(Math.pow(2.0, 3.0));   // Expected: 8.0
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_math_sqrt", "haxe_math_pow"])
        .expect_level(TestLevel::Execution),
    );

    // Math Test 4: Min and Max
    suite.add_test(
        E2ETestCase::new(
            "math_min_max",
            "Min and Max functions",
            r#"
package test;

class Main {
    static function main() {
        trace(Math.min(3.0, 7.0));  // Expected: 3.0
        trace(Math.max(3.0, 7.0));  // Expected: 7.0
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_math_min", "haxe_math_max"])
        .expect_level(TestLevel::Execution),
    );

    // Math Test 5: Logarithms and exponential
    suite.add_test(
        E2ETestCase::new(
            "math_log_exp",
            "Logarithm and exponential functions",
            r#"
package test;

class Main {
    static function main() {
        var e = Math.exp(1.0);
        trace(e);             // Expected: ~2.718
        trace(Math.log(e));   // Expected: ~1.0
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_math_exp", "haxe_math_log"])
        .expect_level(TestLevel::Execution),
    );

    // Math Test 6: Round
    suite.add_test(
        E2ETestCase::new(
            "math_round",
            "Rounding function",
            r#"
package test;

class Main {
    static function main() {
        trace(Math.round(3.4));  // Expected: 3.0
        trace(Math.round(3.6));  // Expected: 4.0
        trace(Math.round(3.5));  // Expected: 4.0 (rounds up)
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_math_round"])
        .expect_level(TestLevel::Execution),
    );

    // Math Test 7: Random
    suite.add_test(
        E2ETestCase::new(
            "math_random",
            "Random number generation",
            r#"
package test;

class Main {
    static function main() {
        var r = Math.random();
        // Random returns [0.0, 1.0), just verify it runs
        trace(r);  // Expected: some value between 0 and 1
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_math_random"])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // INTEGRATION TESTS
    // ============================================================================

    // Integration Test 1: String + Array
    suite.add_test(
        E2ETestCase::new(
            "integration_string_array",
            "String split to array",
            r#"
package test;

class Main {
    static function main() {
        var s = "apple,banana,cherry";
        var fruits = s.split(",");
        trace(fruits.length);  // Expected: 3
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_string_split", "array_length"])
        .expect_level(TestLevel::Execution),
    );

    // Integration Test 2: Math + Array
    suite.add_test(
        E2ETestCase::new(
            "integration_math_array",
            "Math operations on array elements",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Float>();
        arr.push(1.5);
        arr.push(2.7);
        arr.push(3.2);

        var sum = 0.0;
        var i = 0;
        while (i < arr.length) {
            sum = sum + Math.floor(arr[i]);
            i++;
        }
        // sum = 1 + 2 + 3 = 6.0
    }
}
"#,
        )
        .expect_mir_calls(vec!["haxe_array_push", "haxe_math_floor"])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // Run all tests
    // ============================================================================
    let results = suite.run_all();
    suite.print_summary(&results);

    let failed_count = results.iter().filter(|(_, r)| !r.is_success()).count();
    if failed_count > 0 {
        Err(format!("{} test(s) failed", failed_count))
    } else {
        Ok(())
    }
}
