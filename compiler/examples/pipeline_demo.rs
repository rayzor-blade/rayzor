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
//! Demo of the Haxe compilation pipeline
//!
//! This example demonstrates the complete Haxe -> AST -> TAST pipeline.

use compiler::pipeline::{
    HaxeCompilationPipeline, PipelineConfig, TargetPlatform, compile_haxe_file,
};

fn main() {
    println!("=== Haxe Compilation Pipeline Demo ===\n");

    // Example 1: Simple class
    let simple_haxe = r#"
        class Hello {
            static function main() {
                trace("Hello, World!");
            }
        }
    "#;

    println!("1. Compiling simple Haxe class:");
    println!("{}", simple_haxe);

    let result1 = compile_haxe_file("Hello.hx", simple_haxe);
    print_compilation_result(&result1);

    // Example 2: More complex Haxe with modern features
    let complex_haxe = r#"
        package examples;
        
        import haxe.ds.Map;
        using StringTools;
        
        @:final
        class Calculator {
            private var history:Array<String> = [];
            
            public function add(a:Float, b:Float):Float {
                var result = a + b;
                history.push('$a + $b = $result');
                return result;
            }
            
            public function getHistory():Array<String> {
                return history.copy();
            }
        }
        
        enum Operation {
            Add(a:Float, b:Float);
            Multiply(a:Float, b:Float);
        }
        
        abstract Vec2(Array<Float>) from Array<Float> {
            public var x(get, never):Float;
            public var y(get, never):Float;
            
            function get_x():Float return this[0];
            function get_y():Float return this[1];
            
            @:op(A + B)
            public function add(other:Vec2):Vec2 {
                return [x + other.x, y + other.y];
            }
        }
        
        class Main {
            static function main() {
                var calc = new Calculator();
                var result = calc.add(2.5, 3.7);
                trace('Result: $result');
                
                for (entry in calc.getHistory()) {
                    trace('History: $entry');
                }
                
                var v1:Vec2 = [1.0, 2.0];
                var v2:Vec2 = [3.0, 4.0];
                var sum = v1 + v2;
                trace('Vector sum: $sum');
            }
        }
    "#;

    println!("\n2. Compiling complex Haxe with modern features:");
    println!("{}", complex_haxe);

    let result2 = compile_haxe_file("Calculator.hx", complex_haxe);
    print_compilation_result(&result2);

    // Example 3: Custom pipeline configuration
    println!("\n3. Using custom pipeline configuration:");

    let config = PipelineConfig {
        strict_type_checking: true,
        enable_lifetime_analysis: true,
        enable_ownership_analysis: false,
        enable_borrow_checking: true,
        enable_hot_reload: false,
        optimization_level: 1,
        collect_statistics: true,
        max_errors: 20,
        target_platform: TargetPlatform::CraneliftJIT,
        enable_colored_errors: true,
        enable_semantic_analysis: true,
        enable_hir_lowering: true,
        enable_hir_optimization: false,
        enable_hir_validation: true,
        enable_mir_lowering: true,
        enable_mir_optimization: false,
        enable_flow_sensitive_analysis: true,
        enable_enhanced_flow_analysis: false,
        enable_memory_safety_analysis: false,
        enable_macro_expansion: true,
    };

    let mut pipeline = HaxeCompilationPipeline::with_config(config);
    let result3 = pipeline.compile_file("Custom.hx", simple_haxe);

    // println!("Pipeline configuration:");
    // println!("  Target: {:?}", pipeline.config.target_platform);
    // println!("  Strict type checking: {}", pipeline.config.strict_type_checking);
    // println!("  Lifetime analysis: {}", pipeline.config.enable_lifetime_analysis);

    print_compilation_result(&result3);

    println!("\n=== Pipeline Demo Complete ===");
}

fn print_compilation_result(result: &compiler::pipeline::CompilationResult) {
    println!("Compilation Results:");
    println!("  ✓ Files processed: {}", result.stats.files_processed);
    println!("  ✓ Total LOC: {}", result.stats.total_loc);
    println!("  ✓ Parse time: {}μs", result.stats.parse_time_us);
    println!("  ✓ Lowering time: {}μs", result.stats.lowering_time_us);
    println!("  ✓ Total time: {}μs", result.stats.total_time_us);
    println!("  ✓ TAST files generated: {}", result.typed_files.len());

    if result.errors.is_empty() {
        println!("  ✅ No errors!");
    } else {
        println!("  ❌ Errors: {}", result.errors.len());
        for (i, error) in result.errors.iter().enumerate() {
            println!(
                "    {}. {} ({}:{}) - {:?}",
                i + 1,
                error.message,
                error.location.line,
                error.location.column,
                error.category
            );
        }
    }

    if !result.warnings.is_empty() {
        println!("  ⚠️  Warnings: {}", result.warnings.len());
        for (i, warning) in result.warnings.iter().enumerate() {
            println!(
                "    {}. {} - {:?}",
                i + 1,
                warning.message,
                warning.category
            );
        }
    }

    // Show TAST details if available
    for typed_file in &result.typed_files {
        println!(
            "  📄 TAST Analysis for '{}':",
            typed_file.metadata.file_path
        );
        println!("     Functions: {}", typed_file.functions.len());
        println!("     Classes: {}", typed_file.classes.len());
        println!("     Interfaces: {}", typed_file.interfaces.len());
        println!("     Enums: {}", typed_file.enums.len());
        println!("     Abstracts: {}", typed_file.abstracts.len());
        println!("     Module fields: {}", typed_file.module_fields.len());
        println!("     Imports: {}", typed_file.imports.len());

        if let Some(package) = &typed_file.metadata.package_name {
            println!("     Package: {}", package);
        }
    }
}
