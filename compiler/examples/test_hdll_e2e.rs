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
//! End-to-end test for HDLL plugin loading via hlp_ introspection.
//!
//! This test verifies:
//! 1. HdllPlugin::load_with_introspection correctly loads a C library
//! 2. hlp_ symbols are introspected for type signatures
//! 3. Function pointers are valid and callable
//! 4. CompilerPlugin trait produces correct method_mappings and extern declarations
//! 5. discover_and_load_hdlls correctly detects @:hlNative metadata in parsed ASTs
//!
//! Prerequisites: testmath.hdll must be compiled from testmath.c
//!   cc -shared -o compiler/tests/hdll_fixtures/testmath.hdll compiler/tests/hdll_fixtures/testmath.c

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::compiler_plugin::CompilerPlugin;
use compiler::stdlib::IrTypeDescriptor;
use compiler::stdlib::hdll_plugin::HdllPlugin;
use std::path::Path;

fn main() {
    println!("=== HDLL Integration Test ===\n");

    let hdll_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/hdll_fixtures/testmath.hdll");

    if !hdll_path.exists() {
        eprintln!("ERROR: testmath.hdll not found at {}", hdll_path.display());
        eprintln!(
            "Build it with: cc -shared -o {} tests/hdll_fixtures/testmath.c",
            hdll_path.display()
        );
        std::process::exit(1);
    }

    test_introspection_loading(&hdll_path);
    test_function_pointers(&hdll_path);
    test_compiler_plugin_trait(&hdll_path);
    test_discover_hl_native_metadata();
    test_haxe_calls_hdll();

    println!("\n=== All HDLL tests passed! ===");
}

fn test_introspection_loading(hdll_path: &Path) {
    println!("TEST: HdllPlugin::load_with_introspection");

    let methods = vec![("add", true), ("multiply", true), ("sqrt_approx", true)];

    let plugin = HdllPlugin::load_with_introspection(hdll_path, "testmath", "TestMath", &methods)
        .expect("Failed to load testmath.hdll via introspection");

    assert_eq!(plugin.function_count(), 3, "Should have 3 functions");
    assert!(plugin.has_class("TestMath"), "Should have TestMath class");

    let symbols = plugin.get_symbols();
    assert_eq!(symbols.len(), 3, "Should have 3 symbols");

    // Verify symbol names follow lib_method convention
    let symbol_names: Vec<&str> = symbols.iter().map(|(name, _)| *name).collect();
    assert!(
        symbol_names.contains(&"testmath_add"),
        "Should have testmath_add"
    );
    assert!(
        symbol_names.contains(&"testmath_multiply"),
        "Should have testmath_multiply"
    );
    assert!(
        symbol_names.contains(&"testmath_sqrt_approx"),
        "Should have testmath_sqrt_approx"
    );

    println!(
        "  PASSED: Loaded {} functions via hlp_ introspection",
        plugin.function_count()
    );
}

fn test_function_pointers(hdll_path: &Path) {
    println!("\nTEST: Function pointer validity");

    let methods = vec![("add", true), ("multiply", true), ("sqrt_approx", true)];
    let plugin =
        HdllPlugin::load_with_introspection(hdll_path, "testmath", "TestMath", &methods).unwrap();

    let symbols = plugin.get_symbols();

    // Test testmath_add(3, 4) -> 7
    let add_ptr = symbols
        .iter()
        .find(|(n, _)| *n == "testmath_add")
        .unwrap()
        .1;
    let add_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(add_ptr) };
    let result = add_fn(3, 4);
    assert_eq!(result, 7, "add(3, 4) should be 7, got {}", result);
    println!("  PASSED: testmath_add(3, 4) = {}", result);

    // Test testmath_multiply(5, 6) -> 30
    let mul_ptr = symbols
        .iter()
        .find(|(n, _)| *n == "testmath_multiply")
        .unwrap()
        .1;
    let mul_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(mul_ptr) };
    let result = mul_fn(5, 6);
    assert_eq!(result, 30, "multiply(5, 6) should be 30, got {}", result);
    println!("  PASSED: testmath_multiply(5, 6) = {}", result);

    // Test testmath_sqrt_approx(16.0) ~= 4.0
    let sqrt_ptr = symbols
        .iter()
        .find(|(n, _)| *n == "testmath_sqrt_approx")
        .unwrap()
        .1;
    let sqrt_fn: extern "C" fn(f64) -> f64 = unsafe { std::mem::transmute(sqrt_ptr) };
    let result = sqrt_fn(16.0);
    assert!(
        (result - 4.0).abs() < 0.0001,
        "sqrt_approx(16.0) should be ~4.0, got {}",
        result
    );
    println!("  PASSED: testmath_sqrt_approx(16.0) = {:.6}", result);
}

fn test_compiler_plugin_trait(hdll_path: &Path) {
    println!("\nTEST: CompilerPlugin trait implementation");

    let methods = vec![("add", true), ("multiply", true), ("sqrt_approx", true)];
    let plugin =
        HdllPlugin::load_with_introspection(hdll_path, "testmath", "TestMath", &methods).unwrap();

    // Test name()
    assert_eq!(plugin.name(), "testmath");

    // Test method_mappings()
    let mappings = plugin.method_mappings();
    assert_eq!(mappings.len(), 3, "Should have 3 method mappings");

    // Verify the add method mapping
    let add_mapping = mappings
        .iter()
        .find(|(sig, _)| sig.method == "add")
        .expect("Should have 'add' method mapping");
    assert_eq!(add_mapping.0.class, "TestMath");
    assert_eq!(add_mapping.0.is_static, true);
    assert_eq!(add_mapping.0.param_count, 2);
    assert_eq!(add_mapping.1.runtime_name, "testmath_add");
    assert_eq!(add_mapping.1.has_return, true);
    assert_eq!(add_mapping.1.param_types.as_ref().map(|t| t.len()), Some(2));

    // Verify param types for add (should be I32, I32)
    let param_types = add_mapping.1.param_types.as_ref().unwrap();
    assert_eq!(param_types[0], IrTypeDescriptor::I32);
    assert_eq!(param_types[1], IrTypeDescriptor::I32);
    assert_eq!(add_mapping.1.return_type, Some(IrTypeDescriptor::I32));

    // Verify sqrt_approx mapping
    let sqrt_mapping = mappings
        .iter()
        .find(|(sig, _)| sig.method == "sqrt_approx")
        .expect("Should have 'sqrt_approx' method mapping");
    assert_eq!(sqrt_mapping.0.param_count, 1);
    let param_types = sqrt_mapping.1.param_types.as_ref().unwrap();
    assert_eq!(param_types[0], IrTypeDescriptor::F64);
    assert_eq!(sqrt_mapping.1.return_type, Some(IrTypeDescriptor::F64));

    // Test priority
    assert_eq!(
        plugin.priority(),
        10,
        "HDLL plugins should have priority 10"
    );

    println!("  PASSED: method_mappings() produces correct signatures and runtime calls");

    // Test declare_externs (just verify it doesn't panic)
    let mut builder = compiler::ir::mir_builder::MirBuilder::new("test");
    plugin.declare_externs(&mut builder);
    let module = builder.finish();
    let extern_funcs: Vec<_> = module
        .functions
        .values()
        .filter(|f| f.cfg.blocks.is_empty()) // Extern functions have no blocks
        .collect();
    assert!(
        extern_funcs.len() >= 3,
        "Should declare at least 3 extern functions"
    );
    println!(
        "  PASSED: declare_externs() creates {} extern function declarations",
        extern_funcs.len()
    );
}

fn test_discover_hl_native_metadata() {
    println!("\nTEST: @:hlNative metadata detection in parsed AST");

    // Create a compilation unit with @:hlNative class
    let mut config = CompilationConfig::default();
    config.load_stdlib = false; // Skip stdlib for this test
    config.hdll_search_paths =
        vec![std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/hdll_fixtures")];

    let mut unit = CompilationUnit::new(config);

    // Add a Haxe file with @:hlNative metadata (standard Haxe pattern with stub bodies)
    let haxe_code = r#"
@:hlNative("testmath")
class TestMath {
    public static function add(a:Int, b:Int):Int { return 0; }
    public static function multiply(a:Int, b:Int):Int { return 0; }
}

class Main {
    static function main() {
        trace("hello");
    }
}
"#;

    unit.add_file(haxe_code, "test_hlnative.hx")
        .expect("Failed to add file");

    // discover_and_load_hdlls is called automatically inside lower_to_tast,
    // but we can also call it explicitly to verify HDLL detection
    unit.discover_and_load_hdlls();

    // Verify the HDLL symbols were loaded
    let hdll_symbols = unit.get_hdll_symbols();
    assert!(
        hdll_symbols.len() >= 2,
        "Should have loaded at least 2 HDLL symbols, got {}",
        hdll_symbols.len()
    );

    let symbol_names: Vec<&str> = hdll_symbols.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        symbol_names.contains(&"testmath_add"),
        "Should have testmath_add symbol, got: {:?}",
        symbol_names
    );
    assert!(
        symbol_names.contains(&"testmath_multiply"),
        "Should have testmath_multiply symbol, got: {:?}",
        symbol_names
    );

    println!(
        "  PASSED: discover_and_load_hdlls() found {} HDLL symbols from @:hlNative metadata",
        hdll_symbols.len()
    );
}

fn test_haxe_calls_hdll() {
    println!("\nTEST: Full pipeline - Haxe code calling @:hlNative functions");

    // Create compilation config with HDLL search paths
    let mut config = CompilationConfig::fast();
    config.hdll_search_paths =
        vec![std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/hdll_fixtures")];

    let mut unit = CompilationUnit::new(config);

    // Load stdlib (required for trace, Int, etc.)
    unit.load_stdlib().expect("Failed to load stdlib");

    // Add Haxe code that calls @:hlNative functions (extern class, bodyless methods)
    let haxe_code = r#"
@:hlNative("testmath")
extern class TestMath {
    public static function add(a:Int, b:Int):Int;
    public static function multiply(a:Int, b:Int):Int;
}

class Main {
    static function main() {
        var sum = TestMath.add(3, 4);
        trace(sum);

        var product = TestMath.multiply(5, 6);
        trace(product);

        var combined = TestMath.add(TestMath.multiply(2, 3), TestMath.add(10, 20));
        trace(combined);
    }
}
"#;

    unit.add_file(haxe_code, "test_hlnative_pipeline.hx")
        .expect("Failed to add file");

    // Step 1: Compile to TAST (triggers discover_and_load_hdlls automatically)
    let typed_files = unit.lower_to_tast().expect("TAST lowering failed");
    println!(
        "  Step 1: TAST lowering succeeded ({} files)",
        typed_files.len()
    );

    // Verify HDLL symbols were discovered during TAST lowering
    let hdll_symbols = unit.get_hdll_symbols();
    assert!(
        hdll_symbols.len() >= 2,
        "HDLL symbols should have been loaded during TAST lowering, got {}",
        hdll_symbols.len()
    );
    println!(
        "  Step 2: HDLL discovery found {} symbols",
        hdll_symbols.len()
    );

    // Step 3: Get MIR modules
    let mir_modules = unit.get_mir_modules();
    assert!(
        !mir_modules.is_empty(),
        "Should have at least one MIR module"
    );
    println!(
        "  Step 3: MIR lowering succeeded ({} modules)",
        mir_modules.len()
    );

    // Step 4: Create Cranelift backend with merged runtime + HDLL symbols
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let runtime_symbols = plugin.runtime_symbols();

    let mut all_symbols: Vec<(&str, *const u8)> =
        runtime_symbols.iter().map(|(n, p)| (*n, *p)).collect();
    for (name, ptr) in hdll_symbols {
        all_symbols.push((name.as_str(), *ptr));
    }
    println!(
        "  Step 4: Collected {} runtime + {} HDLL = {} total symbols",
        runtime_symbols.len(),
        hdll_symbols.len(),
        all_symbols.len()
    );

    // Step 5: Compile all MIR modules
    let mut backend =
        CraneliftBackend::with_symbols(&all_symbols).expect("Failed to create Cranelift backend");

    for module in &mir_modules {
        backend
            .compile_module(module)
            .expect("Failed to compile MIR module");
    }
    println!("  Step 5: Cranelift compilation succeeded");

    // Step 6: Execute the compiled code
    println!("  Step 6: Executing compiled Haxe code...");
    let mut executed = false;
    for module in mir_modules.iter().rev() {
        if let Ok(()) = backend.call_main(module) {
            executed = true;
            break;
        }
    }
    assert!(executed, "Failed to execute main in any module");

    // If we get here, the full pipeline worked:
    //   Haxe @:hlNative("testmath") -> discover HDLL -> introspect hlp_ symbols
    //   -> build StdlibMapping -> MIR lowers TestMath.add() to extern testmath_add
    //   -> Cranelift links testmath_add to C function pointer -> execution succeeds
    println!("  PASSED: Full pipeline execution with @:hlNative HDLL calls succeeded!");
    println!("    - TestMath.add(3, 4) should have printed: 7");
    println!("    - TestMath.multiply(5, 6) should have printed: 30");
    println!(
        "    - TestMath.add(TestMath.multiply(2, 3), TestMath.add(10, 20)) should have printed: 36"
    );
}
