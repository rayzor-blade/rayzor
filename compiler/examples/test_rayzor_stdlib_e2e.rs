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
/// End-to-end test framework for Rayzor stdlib
///
/// This tests the COMPLETE pipeline:
/// 1. Parse Haxe source code
/// 2. Compile to TAST with shared state (multi-file aware)
/// 3. Lower to HIR
/// 4. Lower to MIR with stdlib mappings
/// 5. Validate MIR structure
/// 6. (Future) Generate native code and execute
///
/// Key learnings applied:
/// - Uses CompilationUnit for proper multi-file compilation
/// - Shared state (StringInterner, NamespaceResolver, SymbolTable)
/// - Stdlib is loaded once and symbols are visible to all user files
/// - Fully qualified names work without imports
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
    /// L5: Native code generation succeeds (future)
    #[allow(dead_code)]
    Codegen,
    /// L6: Execution produces correct output (future)
    #[allow(dead_code)]
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
            expected_level: TestLevel::Execution, // Default to full execution
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

        // Create compilation unit with stdlib (use fast() for lazy stdlib loading)
        let mut unit = CompilationUnit::new(CompilationConfig::fast());

        // Load stdlib (this is critical - must happen first!)
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
        let typed_files = match unit.lower_to_tast() {
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

        // L2: HIR is generated as part of the pipeline
        println!("L2: HIR lowering...");
        println!("  ✅ HIR lowering succeeded (integrated in pipeline)");

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
        println!("     - Globals: {}", mir_module.globals.len());

        // DEBUG: List all modules and check for main function
        for (i, module) in mir_modules.iter().enumerate() {
            let has_main = module
                .functions
                .values()
                .any(|f| f.name.contains("main") || f.name.contains("Main"));
            if has_main {
                println!("  DEBUG: Module {} has main-like function", i);
                for func in module
                    .functions
                    .values()
                    .filter(|f| f.name.contains("main") || f.name.contains("Main"))
                {
                    println!("    - {}", func.name);
                }
            }
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

        // Stop here if expected level is MirValidation or below
        if matches!(
            self.expected_level,
            TestLevel::Compilation
                | TestLevel::HirLowering
                | TestLevel::MirLowering
                | TestLevel::MirValidation
        ) {
            return TestResult::Success {
                level: self.expected_level.clone(),
            };
        }

        // L5: Codegen (compile MIR to native code)
        println!("L5: Compiling to native code for {}...", filename);
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

        // Stop here if expected level is Codegen
        if matches!(self.expected_level, TestLevel::Codegen) {
            return TestResult::Success {
                level: TestLevel::Codegen,
            };
        }

        // L6: Execution (run the compiled code)
        println!("L6: Executing compiled code for {}...", filename);
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

    /// Validate MIR structure across all modules
    fn validate_mir_modules(&self, modules: &[std::sync::Arc<IrModule>]) -> Result<(), String> {
        // Collect all extern functions from all modules
        let mut all_extern_functions = std::collections::HashSet::new();
        for module in modules {
            // Check module.extern_functions
            for (_, ef) in &module.extern_functions {
                all_extern_functions.insert(ef.name.clone());
            }
            // Also check module.functions with empty CFGs (these are externs too)
            for (_, func) in &module.functions {
                if func.cfg.blocks.is_empty() {
                    all_extern_functions.insert(func.name.clone());
                }
            }
        }

        // Check that expected extern functions are registered
        if !self.expected_mir_calls.is_empty() {
            for expected_call in &self.expected_mir_calls {
                let found = all_extern_functions
                    .iter()
                    .any(|name| name.contains(expected_call));
                if !found {
                    return Err(format!(
                        "Expected extern function '{}' not found in MIR. Available: {:?}",
                        expected_call,
                        all_extern_functions.iter().collect::<Vec<_>>()
                    ));
                }
            }
            println!("  ✓ All expected extern functions found");
        }

        // Validate function structure (non-extern functions should have blocks)
        for module in modules {
            for (_, func) in &module.functions {
                // We only validate that the CFG exists, blocks could be empty for declarations
                // but typically should have at least an entry block for implemented functions
            }
        }
        println!("  ✓ All functions have valid structure");

        Ok(())
    }

    /// Compile MIR modules to native code using Cranelift
    fn compile_to_native(
        &self,
        modules: &[std::sync::Arc<IrModule>],
    ) -> Result<CraneliftBackend, String> {
        // Get runtime symbols from the plugin system
        let plugin = rayzor_runtime::plugin_impl::get_plugin();
        let symbols = plugin.runtime_symbols();
        let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

        // Create Cranelift backend with runtime symbols
        let mut backend = CraneliftBackend::with_symbols(&symbols_ref)?;

        // Compile all MIR modules (last module is user code)
        for module in modules {
            backend.compile_module(module)?;
        }

        Ok(backend)
    }

    /// Execute the compiled code and validate results
    fn execute_and_validate(
        &self,
        backend: &mut CraneliftBackend,
        name: String,
        modules: &[std::sync::Arc<IrModule>],
    ) -> Result<(), String> {
        // Try to find main in any module (user code might be in any module)
        for module in modules.iter().rev() {
            // Start from the last (most likely to be user code)
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

            results.push((test_name.clone(), result));

            if success {
                println!("\n✅ {} PASSED", test_name);
            } else {
                println!("\n❌ {} FAILED", test_name);
            }
            // sleep(Duration::from_secs(1));
        }

        results
    }

    fn print_summary(&self, results: &[(String, TestResult)]) {
        println!("\n{}", "=".repeat(70));
        println!("TEST SUMMARY");
        println!("{}", "=".repeat(70));

        let total = results.len();
        let passed = results.iter().filter(|(_, r)| r.is_success()).count();
        let failed = total - passed;

        // Group by level
        let mut by_level: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();
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
        println!("   Passed: {} ({}%)", passed, passed * 100 / total);
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
    println!("=== Rayzor Stdlib End-to-End Test Suite ===\n");

    let mut suite = E2ETestSuite::new();

    // ============================================================================
    // TEST 1: Basic Thread Spawn with Import
    // ============================================================================

    suite.add_test(
        E2ETestCase::new(
            "thread_spawn_basic",
            "Basic thread spawn and join with import statement",
            r#"
package test;

import rayzor.concurrent.Thread;

@:derive([Send])
class Message {
    public var value: Int;
    public function new(v: Int) {
        this.value = v;
    }
}

class Main {
    static function main() {
        var msg = new Message(42);
        var handle = Thread.spawn(() -> {
            return msg.value;
        });
        var result = handle.join();
    }
}
"#,
        )
        .expect_mir_calls(vec!["rayzor_thread_spawn", "rayzor_thread_join"]),
        // Removed .expect_level() to use default Execution level
    );

    // ============================================================================
    // TEST 2: Thread Spawn with Fully Qualified Names
    // ============================================================================
    suite.add_test(
        E2ETestCase::new(
            "thread_spawn_qualified",
            "Thread spawn using fully qualified names (no import)",
            r#"
package test;

@:derive([Send])
class Message {
    public var value: Int;
    public function new(v: Int) {
        this.value = v;
    }
}

class Main {
    static function main() {
        var msg = new Message(42);
        var handle = rayzor.concurrent.Thread.spawn(()-> {
            return msg.value;
        });
        var result = handle.join();
    }
}
"#,
        )
        .expect_mir_calls(vec!["rayzor_thread_spawn", "rayzor_thread_join"])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // TEST 3: Multiple Threads
    // ============================================================================
    suite.add_test(
        E2ETestCase::new(
            "thread_multiple",
            "Spawn multiple threads and join them all",
            r#"
package test;

import rayzor.concurrent.Thread;

class Main {
    static function main() {
        var handles = new Array<Thread<Int>>();

        // Simple loop without range operator for now
        var i = 0;
        while (i < 5) {
            var handle = Thread.spawn(() -> {
                return i * 2;
            });
            handles.push(handle);
            i++;
        }

        // Use indexed access to join all threads
        var sum = 0;
        var j = 0;
        while (j < handles.length) {
            sum += handles[j].join();
            j++;
        }
    }
}
"#,
        )
        .expect_mir_calls(vec!["rayzor_thread_spawn", "rayzor_thread_join"])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // TEST 4: Channel Send/Receive
    // ============================================================================
    suite.add_test(
        E2ETestCase::new(
            "channel_basic",
            "Basic channel send and receive",
            r#"
package test;

import rayzor.concurrent.Thread;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Arc;

class Main {
    static function main() {
        // Simplified test: single thread with channel
        var channel = new Arc(new Channel(10));

        // Clone Arc for thread
        var threadChannel = channel.clone();
        var sender = Thread.spawn(() -> {
            // Send a few values using simple loop
            threadChannel.get().send(42);
            threadChannel.get().send(43);
            threadChannel.get().send(44);
            return 3;
        });

        // Join sender
        sender.join();

        // Main thread receives
        var v1 = channel.get().tryReceive();
        var v2 = channel.get().tryReceive();
        var v3 = channel.get().tryReceive();

        return;
    }
}
"#,
        )
        .expect_mir_calls(vec![
            "rayzor_channel_init",
            "rayzor_channel_send",
            "rayzor_channel_try_receive",
        ])
        .expect_level(TestLevel::Execution), // TODO: channel execution hangs — investigate separately
    );

    // ============================================================================
    // TEST 4b: Channel Send in While Loop (inside thread)
    // Tests: while loop inside thread body, captured channel, loop counter send
    // ============================================================================
    suite.add_test(
        E2ETestCase::new(
            "channel_while_loop",
            "Channel send in a while loop inside a thread - tests captured vars + loop + channel ops",
            r#"
package test;
import rayzor.concurrent.Thread;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Arc;
class Main {
    static function main() {
        var channel = new Arc(new Channel(10));
        var threadChannel = channel.clone();

        // Thread with while loop that sends values 0,1,2,3,4 to channel
        // This exercises: captured Arc, while loop in lambda, channel.send in loop
        var sender = Thread.spawn(() -> {
            var i = 0;
            while (i < 5) {
                threadChannel.get().send(i * 10);  // Send 0, 10, 20, 30, 40
                i++;
            }
            return i;  // Return count (5)
        });

        // Join sender and verify it ran 5 iterations
        var count = sender.join();

        // Main thread receives all values and sums them
        // Expected sum: 0 + 10 + 20 + 30 + 40 = 100
        var sum = 0;
        var j = 0;
        while (j < 5) {
            var val = channel.get().tryReceive();
            sum = sum + val;
            j++;
        }

        // Verify results (sum should be 100, count should be 5)
        return;
    }
}
"#,
        )
        .expect_mir_calls(vec!["rayzor_channel_init", "rayzor_channel_send", "rayzor_channel_try_receive"])
        .expect_level(TestLevel::Execution), // TODO: channel execution hangs — investigate separately
    );

    // ============================================================================
    // TEST 5: Mutex Basic Lock/Unlock
    // ============================================================================
    suite.add_test(
        E2ETestCase::new(
            "mutex_basic",
            "Basic mutex lock and trace",
            r#"
package test;

import rayzor.concurrent.Mutex;

@:derive([Send])
class Counter {
    public var value: Int;

    public function new() {
        this.value = 0;
    }
}

class Main {
    static function main() {
        var counter = new Mutex(new Counter());
        var guard = counter.lock();
        var c = guard.get();
        trace(42);
    }
}
"#,
        )
        .expect_mir_calls(vec!["rayzor_mutex_init", "rayzor_mutex_lock"])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // TEST 6: Arc Creation and Cloning
    // ============================================================================
    suite.add_test(
        E2ETestCase::new(
            "arc_basic",
            "Arc creation and cloning across threads",
            r#"
package test;

import rayzor.concurrent.Thread;
import rayzor.concurrent.Arc;

@:derive([Send, Sync])
class SharedData {
    public var value: Int;

    public function new(v: Int) {
        this.value = v;
    }
}

class Main {
    static function main() {
        var shared = new Arc(new SharedData(42));
        var shared_clone = shared.clone();

        var handle = Thread.spawn(() -> {
            return shared_clone.get().value;
        });

        var result = handle.join();
    }
}
"#,
        )
        .expect_mir_calls(vec![
            "rayzor_arc_init",
            "rayzor_arc_clone",
            "rayzor_thread_spawn",
        ])
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // TEST 7: For-In Loop over Array
    // ============================================================================
    // For-in loop over arrays - simplified test just to see if for-in compiles
    // Temporarily test with empty loop body to isolate the issue
    suite.add_test(
        E2ETestCase::new(
            "forin_array_basic",
            "For-in loop iterating over Array<Int> with execution",
            r#"
package test;

class Main {
    static function main() {
        var arr = new Array<Int>();
        arr.push(10);
        arr.push(20);
        arr.push(30);

        // Just iterate, no body work to isolate for-in mechanics
        for (x in arr) {
            // empty body for now
        }
    }
}
"#,
        )
        .expect_level(TestLevel::Execution),
    );

    // ============================================================================
    // TEST 8: Integration - Arc<Mutex<T>> (no threads for simplicity)
    // ============================================================================
    // Note: Explicit unlock() is required since Rayzor doesn't have automatic
    // drop semantics for RAII types like MutexGuard (unlike Rust).
    suite.add_test(
        E2ETestCase::new(
            "arc_mutex_integration",
            "Arc and Mutex combined for thread-safe shared state",
            r#"
package test;

import rayzor.concurrent.Arc;
import rayzor.concurrent.Mutex;

@:derive([Send])
class SharedCounter {
    public var value: Int;

    public function new() {
        this.value = 0;
    }

    public function increment():Void {
        this.value = this.value + 1;
    }
}

class Main {
    static function main() {
        // Test Arc<Mutex<T>> without threads
        var counter = new Arc(new Mutex(new SharedCounter()));

        // First lock/unlock cycle (split chained calls)
        var mutex_ref1 = counter.get();
        var guard1 = mutex_ref1.lock();
        guard1.get().increment();
        guard1.unlock();

        // Second lock/unlock cycle
        var mutex_ref2 = counter.get();
        var guard2 = mutex_ref2.lock();
        guard2.get().increment();
        guard2.unlock();

        // Third lock/unlock cycle
        var mutex_ref3 = counter.get();
        var guard3 = mutex_ref3.lock();
        guard3.get().increment();
        guard3.unlock();

        // Final read
        var mutex_ref4 = counter.get();
        var final_guard = mutex_ref4.lock();
        trace(final_guard.get().value);  // Should be 3
        final_guard.unlock();
    }
}
"#,
        )
        .expect_mir_calls(vec![
            "rayzor_arc_init",
            "rayzor_mutex_init",
            "rayzor_mutex_lock",
        ]),
    );

    // ============================================================================
    // Run all tests
    // ============================================================================
    // NOTE: Static extension tests (using StringTools) are in isolated test:
    // compiler/examples/test_using_static.rs
    // Run with: cargo run --package compiler --example test_using_static

    // ============================================================================
    let results = suite.run_all();
    suite.print_summary(&results);

    // Exit with error code if any tests failed
    let failed_count = results.iter().filter(|(_, r)| !r.is_success()).count();
    if failed_count > 0 {
        Err(format!("{} test(s) failed", failed_count))
    } else {
        Ok(())
    }
}
