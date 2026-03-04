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
//! Array iterator end-to-end test suite
//!
//! Tests arr.iterator(), arr.keyValueIterator(), and for-in over iterators.

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};

fn run_test(name: &str, code: &str) -> bool {
    println!("\n{}", "=".repeat(60));
    println!("TEST: {}", name);
    println!("{}", "=".repeat(60));

    let mut unit = CompilationUnit::new(CompilationConfig::fast());

    if let Err(e) = unit.load_stdlib() {
        println!("  FAIL: load_stdlib: {}", e);
        return false;
    }

    if let Err(e) = unit.add_file(code, &format!("{}.hx", name)) {
        println!("  FAIL: add_file: {}", e);
        return false;
    }

    if let Err(e) = unit.lower_to_tast() {
        println!("  FAIL: TAST: {:?}", e);
        return false;
    }

    let mir_modules = unit.get_mir_modules();
    if mir_modules.is_empty() {
        println!("  FAIL: No MIR modules");
        return false;
    }

    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend = match CraneliftBackend::with_symbols(&symbols_ref) {
        Ok(b) => b,
        Err(e) => {
            println!("  FAIL: Backend: {}", e);
            return false;
        }
    };

    for module in &mir_modules {
        if let Err(e) = backend.compile_module(module) {
            println!("  FAIL: Compile: {}", e);
            return false;
        }
    }

    for module in mir_modules.iter().rev() {
        if backend.call_main(module).is_ok() {
            println!("  PASS");
            return true;
        }
    }

    println!("  FAIL: No main found");
    false
}

fn main() {
    let mut passed = 0;
    let mut total = 0;

    // Test 1: arr.iterator() manual usage
    total += 1;
    if run_test(
        "iterator_manual",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30];
        var it = arr.iterator();
        while (it.hasNext()) {
            trace(it.next());
        }
        trace("done");
    }
}
"#,
    ) {
        passed += 1;
    }

    // Test 2: for-in over arr.iterator()
    total += 1;
    if run_test(
        "iterator_for_in",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30];
        for (x in arr.iterator()) {
            trace(x);
        }
        trace("done");
    }
}
"#,
    ) {
        passed += 1;
    }

    // Test 3: arr.keyValueIterator() manual usage
    total += 1;
    if run_test(
        "kv_iterator_manual",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30];
        var kvit = arr.keyValueIterator();
        while (kvit.hasNext()) {
            var kv = kvit.next();
            trace(kv.key);
            trace(kv.value);
        }
        trace("done");
    }
}
"#,
    ) {
        passed += 1;
    }

    // Test 4: Existing for-in still works (regression)
    total += 1;
    if run_test(
        "for_in_regression",
        r#"
class Main {
    static function main() {
        var arr = [10, 20, 30];
        for (x in arr) {
            trace(x);
        }
        trace("done");
    }
}
"#,
    ) {
        passed += 1;
    }

    // Test 5: User-defined iterator class
    total += 1;
    if run_test(
        "user_iterator",
        r#"
class Countdown {
    var current:Int;
    public function new(start:Int) {
        current = start;
    }
    public function hasNext():Bool {
        return current > 0;
    }
    public function next():Int {
        current = current - 1;
        return current + 1;
    }
    public function iterator():Countdown {
        return this;
    }
}

class Main {
    static function main() {
        for (x in new Countdown(3)) {
            trace(x);
        }
        trace("done");
    }
}
"#,
    ) {
        passed += 1;
    }

    // Test 6: Empty array iterator
    total += 1;
    if run_test(
        "empty_iterator",
        r#"
class Main {
    static function main() {
        var arr:Array<Int> = [];
        var it = arr.iterator();
        while (it.hasNext()) {
            trace(it.next());
        }
        trace("done");
    }
}
"#,
    ) {
        passed += 1;
    }

    println!("\n{}", "=".repeat(60));
    println!("RESULTS: {}/{} passed", passed, total);
    println!("{}", "=".repeat(60));

    if passed < total {
        std::process::exit(1);
    }
}
