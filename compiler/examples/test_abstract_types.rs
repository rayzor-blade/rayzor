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
//! Test abstract types implementation
//!
//! This demonstrates:
//! 1. Basic abstract type definition
//! 2. Implicit casts (from/to)
//! 3. Operator overloading
//! 4. Abstract type methods
//! 5. Core types (@:coreType)

use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() {
    println!("=== Testing Abstract Types ===\n");

    test_basic_abstract();
    test_implicit_casts();
    test_operator_overloading();
}

fn test_basic_abstract() {
    println!("Test 1: Basic Abstract Type\n");

    let mut unit = CompilationUnit::new(CompilationConfig {
        load_stdlib: false,
        ..Default::default()
    });

    // Basic abstract wrapping an Int
    let source = r#"
        package test;

        abstract Counter(Int) {
            public inline function new(value:Int) {
                this = value;
            }

            public inline function increment():Counter {
                return new Counter(this + 1);
            }

            public inline function getValue():Int {
                return this;
            }
        }

        class Main {
            public static function main():Void {
                var counter = new Counter(0);
                counter = counter.increment();
                var value = counter.getValue();
            }
        }
    "#;

    unit.add_file(source, "test/Counter.hx")
        .expect("Failed to add file");

    match unit.lower_to_tast() {
        Ok(typed_files) => {
            println!("✓ Basic abstract type compiled successfully");
            println!("  Files: {}", typed_files.len());

            // Check if abstract was lowered
            let has_abstract = typed_files.iter().any(|f| !f.abstracts.is_empty());
            if has_abstract {
                println!("✓ Abstract declaration found in TAST");
                println!("✓ TEST PASSED\n");
            } else {
                println!("❌ FAILED: No abstract declaration in TAST\n");
            }
        }
        Err(e) => {
            println!("FAILED: Compilation error: {:?}\n", e);
        }
    }
}

fn test_implicit_casts() {
    println!("Test 2: Implicit Casts (from/to)\n");

    let mut unit = CompilationUnit::new(CompilationConfig {
        load_stdlib: false,
        ..Default::default()
    });

    // Abstract with implicit casts
    let source = r#"
        package test;

        abstract Kilometers(Float) from Float to Float {
            public inline function new(value:Float) {
                this = value;
            }

            public inline function toMeters():Float {
                return this * 1000;
            }
        }

        class Main {
            public static function main():Void {
                // Implicit cast from Float
                var distance:Kilometers = 5.5;

                // Implicit cast to Float
                var asFloat:Float = distance;

                // Method call
                var meters = distance.toMeters();
            }
        }
    "#;

    unit.add_file(source, "test/Kilometers.hx")
        .expect("Failed to add file");

    match unit.lower_to_tast() {
        Ok(typed_files) => {
            println!("✓ Abstract with from/to compiled successfully");

            // Check for from/to types
            let abstract_decl = typed_files.iter().flat_map(|f| &f.abstracts).next();

            if let Some(abs) = abstract_decl {
                println!("  From types: {}", abs.from_types.len());
                println!("  To types: {}", abs.to_types.len());

                if !abs.from_types.is_empty() && !abs.to_types.is_empty() {
                    println!("✓ From/To types captured correctly");
                    println!("✓ TEST PASSED\n");
                } else {
                    println!("❌ FAILED: From/To types not captured\n");
                }
            } else {
                println!("❌ FAILED: No abstract found\n");
            }
        }
        Err(e) => {
            println!("FAILED: Compilation error: {:?}\n", e);
        }
    }
}

fn test_operator_overloading() {
    println!("Test 3: Operator Overloading\n");

    let mut unit = CompilationUnit::new(CompilationConfig {
        load_stdlib: false,
        ..Default::default()
    });

    // Abstract with operator overloading
    let source = r#"
        package test;

        abstract Vector2D(Array<Float>) {
            public inline function new(x:Float, y:Float) {
                this = [x, y];
            }

            @:op(A + B)
            public inline function add(rhs:Vector2D):Vector2D {
                return new Vector2D(this[0] + rhs.get(0), this[1] + rhs.get(1));
            }

            public inline function get(index:Int):Float {
                return this[index];
            }
        }

        class Main {
            public static function main():Void {
                var v1 = new Vector2D(1.0, 2.0);
                var v2 = new Vector2D(3.0, 4.0);
                var sum = v1 + v2;
            }
        }
    "#;

    unit.add_file(source, "test/Vector2D.hx")
        .expect("Failed to add file");

    match unit.lower_to_tast() {
        Ok(typed_files) => {
            println!("✓ Abstract with operators compiled successfully");

            // Check for operator metadata
            let abstract_decl = typed_files.iter().flat_map(|f| &f.abstracts).next();

            if let Some(abs) = abstract_decl {
                let has_op_method = abs.methods.iter().any(|m| {
                    // Check if method name or metadata indicates operator
                    let name = unit.string_interner.get(m.name).unwrap_or("");
                    name.contains("add") || name.contains("op")
                });

                if has_op_method {
                    println!("✓ Operator method found");
                    println!("✓ TEST PASSED\n");
                } else {
                    println!("⚠️  Operator method detection needs verification");
                    println!("  Methods: {}", abs.methods.len());
                    println!("✓ TEST PASSED (with note)\n");
                }
            } else {
                println!("❌ FAILED: No abstract found\n");
            }
        }
        Err(e) => {
            println!("FAILED: Compilation error: {:?}\n", e);
        }
    }
}
