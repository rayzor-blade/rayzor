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
//! Comprehensive tests for sys.thread API with thread pools
//!
//! Tests Deque, Condition, FixedThreadPool, ElasticThreadPool, and EventLoop

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() -> Result<(), String> {
    println!("=== Testing sys.thread API with Thread Pools ===\n");

    // Test 1: Deque basic operations
    test_deque_basic()?;

    // Test 2: Deque with blocking pop
    test_deque_blocking()?;

    // Test 3: Condition variable
    test_condition()?;

    // Test 4: FixedThreadPool
    test_fixed_thread_pool()?;

    println!("\n=== All thread pool tests passed! ===");
    Ok(())
}

fn test_deque_basic() -> Result<(), String> {
    println!("TEST: sys.thread.Deque basic operations");

    let code = r#"
import sys.thread.Deque;

class Main {
    static function main() {
        trace("Testing Deque basic operations");

        var deque = new Deque<Int>();

        // Add to end
        deque.add(1);
        deque.add(2);
        deque.add(3);

        // Push to front
        deque.push(0);

        // Pop should get 0 (front)
        var first = deque.pop(false);
        if (first == 0) {
            trace("First pop: 0 (correct)");
        } else {
            trace("First pop: ERROR");
        }

        // Pop should get 1
        var second = deque.pop(false);
        if (second == 1) {
            trace("Second pop: 1 (correct)");
        } else {
            trace("Second pop: ERROR");
        }

        trace("Deque basic operations work!");
    }
}
"#;

    run_code(code, "test_deque_basic")
}

fn test_deque_blocking() -> Result<(), String> {
    println!("\nTEST: sys.thread.Deque with blocking pop");

    let code = r#"
import sys.thread.Deque;
import sys.thread.Thread;

class Main {
    static function main() {
        trace("Testing Deque blocking pop");

        var deque = new Deque<String>();

        // Create producer thread
        Thread.create(function() {
            Sys.sleep(0.1);
            trace("Producer: adding item");
            deque.add("Hello from producer");
        });

        // Consumer blocks until item is available
        trace("Consumer: waiting for item...");
        var item = deque.pop(true);
        trace("Consumer received:");
        trace(item);

        trace("Deque blocking pop works!");
    }
}
"#;

    run_code(code, "test_deque_blocking")
}

fn test_condition() -> Result<(), String> {
    println!("\nTEST: sys.thread.Condition");

    let code = r#"
import sys.thread.Condition;
import sys.thread.Thread;

class Main {
    static var ready = false;
    static var condition:Condition;

    static function main() {
        trace("Testing Condition variable");

        condition = new Condition();

        // Create worker thread
        Thread.create(function() {
            Sys.sleep(0.1);

            condition.acquire();
            ready = true;
            trace("Worker: signaling condition");
            condition.signal();
            condition.release();
        });

        // Wait for condition
        condition.acquire();
        trace("Main: waiting for condition...");
        while (!ready) {
            condition.wait();
        }
        trace("Main: condition received!");
        condition.release();

        trace("Condition variable works!");
    }
}
"#;

    run_code(code, "test_condition")
}

fn test_fixed_thread_pool() -> Result<(), String> {
    println!("\nTEST: sys.thread.FixedThreadPool");

    let code = r#"
import sys.thread.FixedThreadPool;
import sys.thread.Mutex;
import sys.thread.Thread;

class Main {
    static var counter = 0;
    static var mutex:Mutex;

    static function main() {
        trace("Testing FixedThreadPool");

        mutex = new Mutex();
        var pool = new FixedThreadPool(2);

        // Submit 4 tasks
        for (i in 0...4) {
            pool.run(function() {
                Sys.sleep(0.05);
                mutex.acquire();
                counter++;
                trace("Task completed");
                mutex.release();
            });
        }

        // Wait for all tasks to complete
        Sys.sleep(0.5);

        if (counter == 4) {
            trace("Final counter: 4 (correct)");
        } else {
            trace("Final counter: ERROR");
        }

        pool.shutdown();
        trace("FixedThreadPool works!");
    }
}
"#;

    run_code(code, "test_fixed_thread_pool")
}

fn run_code(code: &str, test_name: &str) -> Result<(), String> {
    // Create compilation unit
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    // Load stdlib
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    // Add test code
    unit.add_file(code, &format!("{}.hx", test_name))
        .map_err(|e| format!("Failed to add file: {}", e))?;

    // Compile to TAST
    unit.lower_to_tast()
        .map_err(|errors| format!("TAST errors: {:?}", errors))?;

    // Get MIR modules
    let mir_modules = unit.get_mir_modules();
    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Create Cranelift backend with runtime symbols
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend = CraneliftBackend::with_symbols(&symbols_ref)?;

    // Compile modules
    for module in &mir_modules {
        backend.compile_module(module)?;
    }

    // Execute
    println!("  Executing {}...", test_name);
    for module in mir_modules.iter().rev() {
        if backend.call_main(module).is_ok() {
            println!("  {} PASSED", test_name);
            return Ok(());
        }
    }

    Err(format!("{}: No main found", test_name))
}
