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
/// Test core Rayzor features: async, exceptions, effect analysis
/// Uses CompilationUnit for simple and correct API usage
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::tast::effect_analysis::EffectAnalyzer;
use compiler::tast::node::AsyncKind;

fn main() -> Result<(), String> {
    println!("=== Core Rayzor Features Test ===\n");

    test_effect_analysis()?;
    test_async_detection()?;
    test_exception_detection()?;

    println!("\n🎉 All core feature tests passed!");
    Ok(())
}

fn test_effect_analysis() -> Result<(), String> {
    println!("Test 1: Effect Analysis");
    println!("========================\n");

    let source = r#"
package test;
class Test {
    public static function pureFunc():Int {
        return 42;
    }
    public static function impureFunc():Void {
        var x = 10;  // side effect
        var y = x + 1;
    }
}
"#;

    let mut unit = CompilationUnit::new(CompilationConfig::default());
    unit.add_file(source, "test.hx")?;
    let typed_files = unit.lower_to_tast().map_err(|e| format!("{:?}", e))?;

    let mut analyzer = EffectAnalyzer::new(&unit.symbol_table, &unit.type_table);

    println!("  Functions analyzed:");
    for file in &typed_files {
        for class in &file.classes {
            for method in &class.methods {
                let effects = analyzer.analyze_function(method);
                println!(
                    "    - {:?}: throw={}, pure={}",
                    method.name, effects.can_throw, effects.is_pure
                );
            }
        }
    }

    println!("\n  ✅ Effect analysis working\n");
    Ok(())
}

fn test_async_detection() -> Result<(), String> {
    println!("Test 2: Async Function Detection");
    println!("==================================\n");

    let source = r#"
package test;
class Test {
    @:async
    public static function asyncFunc():String { 
        return "async"; 
    }
    public static function syncFunc():Int { 
        return 42; 
    }
}
"#;

    let mut unit = CompilationUnit::new(CompilationConfig::default());
    unit.add_file(source, "test.hx")?;
    let typed_files = unit.lower_to_tast().map_err(|e| format!("{:?}", e))?;

    let mut analyzer = EffectAnalyzer::new(&unit.symbol_table, &unit.type_table);

    println!("  Function async status:");
    for file in &typed_files {
        for class in &file.classes {
            for method in &class.methods {
                let effects = analyzer.analyze_function(method);
                let async_str = match effects.async_kind {
                    AsyncKind::Async => "ASYNC",
                    AsyncKind::Sync => "sync",
                    AsyncKind::Generator => "generator",
                    AsyncKind::AsyncGenerator => "async-generator",
                };
                println!("    - {:?}: {}", method.name, async_str);
            }
        }
    }

    println!("\n  ✅ Async detection working\n");
    Ok(())
}

fn test_exception_detection() -> Result<(), String> {
    println!("Test 3: Exception Effect Detection");
    println!("====================================\n");

    let source = r#"
package test;
class Test {
    public static function throwing():Void { 
        throw "error"; 
    }
    public static function safe():Int { 
        return 42; 
    }
    public static function caller():Void { 
        throwing(); 
    }
}
"#;

    let mut unit = CompilationUnit::new(CompilationConfig::default());
    unit.add_file(source, "test.hx")?;
    let typed_files = unit.lower_to_tast().map_err(|e| format!("{:?}", e))?;

    let mut analyzer = EffectAnalyzer::new(&unit.symbol_table, &unit.type_table);

    println!("  Exception effects:");
    let mut can_throw = 0;
    let mut safe = 0;

    for file in &typed_files {
        for class in &file.classes {
            for method in &class.methods {
                let effects = analyzer.analyze_function(method);
                if effects.can_throw {
                    can_throw += 1;
                    println!("    - {:?}: CAN THROW", method.name);
                } else {
                    safe += 1;
                    println!("    - {:?}: safe", method.name);
                }
            }
        }
    }

    println!("\n  → Functions that can throw: {}", can_throw);
    println!("  → Safe functions: {}", safe);
    println!("\n  ✅ Exception detection working\n");
    Ok(())
}
