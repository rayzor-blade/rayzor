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
//! SIMD4f end-to-end test suite
//!
//! Tests the complete pipeline for rayzor.SIMD4f:
//! - SIMD4f.splat(): broadcast scalar to all 4 lanes
//! - SIMD4f.make(): construct from 4 individual values
//! - Arithmetic operators: +, -, *, /
//! - Lane access: extract, insert
//! - Reductions: sum, dot

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
                println!("  ✅ TAST ({} files)", files.len());
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
        println!("  ✅ MIR ({} modules)", mir_modules.len());

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
        println!("  ✅ Codegen succeeded");

        println!("  Executing...");
        for module in mir_modules.iter().rev() {
            if let Ok(()) = backend.call_main(module) {
                println!("  ✅ Execution succeeded");
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

    // ============================================================================
    // TEST 1: SIMD4f.splat — broadcast scalar to all 4 lanes
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_splat",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.splat(3.0);
        trace(true);  // splat created successfully
    }
}
"#,
    ));

    // ============================================================================
    // TEST 2: SIMD4f.make — construct from 4 individual values
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_make",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(1.0, 2.0, 3.0, 4.0);
        trace(true);  // make created successfully
    }
}
"#,
    ));

    // ============================================================================
    // TEST 3: SIMD4f arithmetic — just add two vectors
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_arithmetic",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.splat(1.0);
        var b = SIMD4f.splat(2.0);
        var c = a + b;
        trace(true);
    }
}
"#,
    ));

    // ============================================================================
    // TEST 4: SIMD4f.sum — horizontal reduction
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_sum",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(1.0, 2.0, 3.0, 4.0);
        var s:Float = a.sum();
        trace(s);  // 10.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 5: SIMD4f.dot — dot product
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_dot",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(1.0, 2.0, 3.0, 4.0);
        var b = SIMD4f.splat(2.0);
        var d = a.dot(b);
        trace(d);  // 20.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 6: Tuple literal construction — var a:SIMD4f = (1.0, 2.0, 3.0, 4.0)
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_tuple_literal",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a:SIMD4f = (1.0, 2.0, 3.0, 4.0);
        var s = a.sum();
        trace(s);  // 10.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 7: @:from Array literal — var a:SIMD4f = [1.0, 2.0, 3.0, 4.0]
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_from_array",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a:SIMD4f = [1.0, 2.0, 3.0, 4.0];
        var s = a.sum();
        trace(s);  // 10.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 8: SIMD4f.sqrt — element-wise square root
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_sqrt",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(4.0, 9.0, 16.0, 25.0);
        var b = a.sqrt();
        trace(b.sum());  // 2+3+4+5 = 14.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 9: SIMD4f.abs — element-wise absolute value
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_abs",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(-1.0, 2.0, -3.0, 4.0);
        var b = a.abs();
        trace(b.sum());  // 1+2+3+4 = 10.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 10: SIMD4f.min / max
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_min_max",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(1.0, 5.0, 3.0, 7.0);
        var b = SIMD4f.make(4.0, 2.0, 6.0, 1.0);
        var lo = a.min(b);
        var hi = a.max(b);
        trace(lo.sum());  // 1+2+3+1 = 7.0
        trace(hi.sum());  // 4+5+6+7 = 22.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 11: SIMD4f.ceil / floor / round
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_rounding",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(1.3, 2.7, -1.3, -2.7);
        var c = a.ceil();
        var f = a.floor();
        trace(c.sum());   // 2+3+(-1)+(-2) = 2.0
        trace(f.sum());   // 1+2+(-2)+(-3) = -2.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 12: SIMD4f.normalize — unit vector
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_normalize",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(3.0, 0.0, 0.0, 0.0);
        var n = a.normalize();
        trace(true);  // normalize completed
    }
}
"#,
    ));

    // ============================================================================
    // TEST 13: SIMD4f.magnitude — vector magnitude
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_len",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(3.0, 4.0, 0.0, 0.0);
        var l = a.len();
        trace(l);  // 5.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 14: SIMD4f.lerp — linear interpolation
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_lerp",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.splat(0.0);
        var b = SIMD4f.splat(10.0);
        var mid = a.lerp(b, 0.5);
        trace(mid.sum());  // 5+5+5+5 = 20.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 15: SIMD4f.cross3 — 3D cross product
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_cross3",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var x = SIMD4f.make(1.0, 0.0, 0.0, 0.0);
        var y = SIMD4f.make(0.0, 1.0, 0.0, 0.0);
        var z = x.cross3(y);
        trace(z.sum());  // 0+0+1+0 = 1.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 16: SIMD4f.distance
    // ============================================================================
    tests.push(E2ETestCase::new(
        "simd4f_distance",
        r#"
package test;

import rayzor.SIMD4f;

class Main {
    static function main() {
        var a = SIMD4f.make(0.0, 0.0, 0.0, 0.0);
        var b = SIMD4f.make(3.0, 4.0, 0.0, 0.0);
        var d = a.distance(b);
        trace(d);  // 5.0
    }
}
"#,
    ));

    // Run all tests
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║            SIMD4f — E2E Test Suite                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let results: Vec<(String, TestResult)> =
        tests.iter().map(|t| (t.name.clone(), t.run())).collect();

    println!("\n\n{}", "=".repeat(70));
    println!("TEST SUMMARY");
    println!("{}", "=".repeat(70));

    let total = results.len();
    let passed = results.iter().filter(|(_, r)| r.is_success()).count();
    let failed = total - passed;

    println!("\n📊 Overall:");
    println!("   Total:  {}", total);
    println!("   Passed: {} ({}%)", passed, passed * 100 / total);
    println!("   Failed: {}", failed);

    println!("\n📋 Results:");
    for (name, result) in &results {
        match result {
            TestResult::Success => {
                println!("   ✅ {} (reached Execution)", name);
            }
            TestResult::Failed { error } => {
                println!("   ❌ {} — {}", name, error);
            }
        }
    }

    if failed == 0 {
        println!("\n🎉 All tests passed!");
    } else {
        println!("\n⚠️  {} test(s) failed", failed);
        std::process::exit(1);
    }
}
