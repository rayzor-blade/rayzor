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
//! Tensor end-to-end test suite
//!
//! Tests the complete pipeline for rayzor.ds.Tensor:
//! - Construction: zeros, ones, full, fromArray
//! - Properties: ndim, numel
//! - Arithmetic: +, -, *, /
//! - Reductions: sum, mean, dot
//! - Linear algebra: matmul
//! - Math: sqrt, exp, relu

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

    // ============================================================================
    // TEST 1: Tensor.zeros — create zero tensor
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_zeros",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.zeros([2, 3], DType.F32);
        var s = t.sum();
        trace(s);  // 0.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 2: Tensor.ones — create ones tensor and sum
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_ones",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.ones([2, 3], DType.F32);
        var s = t.sum();
        trace(s);  // 6.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 3: Tensor.full — fill with constant
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_full",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.full([2, 2], 5.0, DType.F32);
        var s = t.sum();
        trace(s);  // 20.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 4: Tensor add — elementwise addition
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_add",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var a = Tensor.ones([3], DType.F32);
        var b = Tensor.full([3], 2.0, DType.F32);
        var c = a.add(b);
        trace(c.mean());  // 3.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 5: Tensor mul — elementwise multiplication
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_mul",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var a = Tensor.full([4], 3.0, DType.F32);
        var b = Tensor.full([4], 2.0, DType.F32);
        var c = a.mul(b);
        trace(c.mean());  // 6.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 6: Tensor.mean — average
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_mean",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.full([4], 3.0, DType.F32);
        var m = t.mean();
        trace(m);  // 3.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 7: Tensor.dot — dot product
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_dot",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var a = Tensor.full([3], 2.0, DType.F32);
        var b = Tensor.full([3], 3.0, DType.F32);
        var d = a.dot(b);
        trace(d);  // 18.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 8: Tensor.sqrt — elementwise square root
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_sqrt",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.full([4], 9.0, DType.F32);
        var s = t.sqrt();
        trace(s.mean());  // 3.0 (sqrt(9) = 3)
    }
}
"#,
    ));

    // ============================================================================
    // TEST 9: Tensor.relu — ReLU activation
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_relu",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.full([2], -3.0, DType.F32);
        var r = t.relu();
        trace(r.mean());  // 0.0 (all negative -> 0)
    }
}
"#,
    ));

    // ============================================================================
    // TEST 10: Tensor properties — ndim, numel
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_properties",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.zeros([2, 3, 4], DType.F32);
        trace(t.ndim());   // 3
        trace(t.numel());  // 24
    }
}
"#,
    ));

    // ============================================================================
    // TEST 11: Tensor.matmul — matrix multiplication
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_matmul",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var a = Tensor.ones([2, 3], DType.F32);
        var b = Tensor.ones([3, 2], DType.F32);
        var c = a.matmul(b);
        trace(c.mean());  // each element = 3, 2x2 matrix, mean = 3.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 12: Tensor.free — explicit deallocation
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_free",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.ones([100], DType.F32);
        t.free();
        trace(true);  // free succeeded
    }
}
"#,
    ));

    // ============================================================================
    // TEST 13: Tensor.max / Tensor.min reductions
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_max_min",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.fromArray([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], DType.F32);
        trace(t.max());  // 9.0
        trace(t.min());  // 1.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 14: Tensor.gelu / Tensor.silu — activations sanity
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_gelu_silu",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        // GELU(0) ~= 0, SiLU(0) = 0
        var z = Tensor.zeros([4], DType.F32);
        trace(z.gelu().sum());  // ~ 0
        trace(z.silu().sum());  // 0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 15: Tensor.softmax — sums to 1.0 along last dim
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_softmax",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], DType.F32);
        var s = t.softmax();
        trace(s.sum());  // ~ 1.0
    }
}
"#,
    ));

    // ============================================================================
    // TEST 16: Tensor.layerNorm — normalized vector has mean ~0
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_layer_norm",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], DType.F32);
        var n = t.layerNorm(0.00001);
        // After layer norm over last dim, sum of normalized values ~= 0
        trace(n.sum());
    }
}
"#,
    ));

    // ============================================================================
    // TEST 17: Tensor.rmsNorm — magnitude property
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_rms_norm",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], DType.F32);
        var r = t.rmsNorm(0.00001);
        // RMS-norm preserves direction; finite sum
        trace(r.sum());
    }
}
"#,
    ));

    // ============================================================================
    // TEST 18: Tensor.permute — reorder axes (no-copy view)
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_permute",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.ones([2, 3, 4], DType.F32);
        var p = t.permute([2, 0, 1]);
        var s = p.shape();
        trace(s[0]);  // 4
        trace(s[1]);  // 2
        trace(s[2]);  // 3
        trace(p.numel());  // 24
    }
}
"#,
    ));

    // ============================================================================
    // TEST 19a: Tensor operator overloading (@:op)
    //
    // Exercises @:op(A + B), (A - B), (A * B), (A / B) declared on Tensor.hx.
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_operator_overload",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var a = Tensor.full([4], 2.0, DType.F32);
        var b = Tensor.full([4], 3.0, DType.F32);

        var added = a + b;   // @:op(A + B) -> [5,5,5,5]
        var subbed = a - b;  // @:op(A - B) -> [-1,-1,-1,-1]
        var muled = a * b;   // @:op(A * B) -> [6,6,6,6]
        var divved = a / b;  // @:op(A / B) -> [2/3,...]

        trace(added.sum());   // 20
        trace(subbed.sum());  // -4
        trace(muled.sum());   // 24
        trace(divved.sum());  // 2.6666...
    }
}
"#,
    ));

    // ============================================================================
    // TEST 19: Tensor.slice — slice along an axis
    // ============================================================================
    tests.push(E2ETestCase::new(
        "tensor_slice",
        r#"
package test;

import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.ones([6], DType.F32);
        var s = t.slice(0, 1, 4);  // 3 elements
        trace(s.numel());  // 3
        trace(s.sum());    // 3.0
    }
}
"#,
    ));

    // Run all tests
    println!("+---------------------------------------------------------------------------+");
    println!("|            Tensor -- E2E Test Suite                                        |");
    println!("+---------------------------------------------------------------------------+");

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
                println!("   PASS {} (reached Execution)", name);
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
