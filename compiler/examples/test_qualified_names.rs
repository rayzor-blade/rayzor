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
// Test that qualified names are being populated correctly during AST lowering

use compiler::tast::{
    StringInterner, SymbolTable, TypeTable, ast_lowering::AstLowering, scopes::ScopeTree,
};
use parser::haxe_parser::parse_haxe_file;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let source = r#"
package com.example;

class MyClass {
    public function myMethod():Int {
        return 42;
    }
}

interface IMyInterface {
    function interfaceMethod():Int;
}

enum MyEnum {
    A;
    B;
}
    "#;

    println!("=== Testing Qualified Names ===\n");
    println!("Source code length: {} bytes\n", source.len());

    // Step 1: Parse Haxe source
    println!("1. Parsing Haxe source...");
    let ast = match parse_haxe_file("test.hx", source, false) {
        Ok(ast) => {
            println!("   ✓ Successfully parsed");
            ast
        }
        Err(e) => {
            eprintln!("   ✗ Parse error: {}", e);
            return;
        }
    };

    // Step 2: Lower AST to TAST
    println!("\n2. Lowering AST to Typed AST (TAST)...");

    // Create necessary infrastructure
    let mut string_interner = StringInterner::new();
    let mut symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let mut scope_tree = ScopeTree::new(compiler::tast::ScopeId::from_raw(0));
    let mut namespace_resolver = compiler::tast::namespace::NamespaceResolver::new();
    let mut import_resolver = compiler::tast::namespace::ImportResolver::new();

    // Create AST lowering context
    let mut ast_lowering = AstLowering::new(
        &mut string_interner,
        std::rc::Rc::new(std::cell::RefCell::new(
            compiler::tast::StringInterner::new(),
        )),
        &mut symbol_table,
        &type_table,
        &mut scope_tree,
        &mut namespace_resolver,
        &mut import_resolver,
    );

    let typed_file = match ast_lowering.lower_file(&ast) {
        Ok(tast) => {
            println!("   ✓ Successfully lowered to TAST");
            tast
        }
        Err(error) => {
            eprintln!("   ✗ TAST lowering error: {:?}", error);
            return;
        }
    };

    // Step 3: Check qualified names
    println!("\n3. Checking qualified names...");

    let mut found_class = false;
    let mut found_interface = false;
    let mut found_enum = false;
    let mut found_method = false;

    for symbol in symbol_table.all_symbols() {
        if let Some(qualified_name) = symbol.qualified_name {
            let name_str = string_interner.get(qualified_name).unwrap_or("<unknown>");
            let symbol_name_str = string_interner.get(symbol.name).unwrap_or("<unknown>");

            println!("   Symbol '{}' (kind: {:?})", symbol_name_str, symbol.kind);
            println!("      Qualified name: '{}'", name_str);

            // Check for expected qualified names (exact matches to avoid false positives)
            if name_str == "com.example.MyClass" {
                found_class = true;
                println!("      ✓ Class qualified name correct");
            }

            if name_str == "com.example.MyClass.myMethod" {
                found_method = true;
                println!("      ✓ Method qualified name correct");
            }

            if name_str == "com.example.IMyInterface" {
                found_interface = true;
                println!("      ✓ Interface qualified name correct");
            }

            if name_str == "com.example.MyEnum" {
                found_enum = true;
                println!("      ✓ Enum qualified name correct");
            }
        }
    }

    println!("\n4. Summary:");
    println!(
        "   Class qualified name: {}",
        if found_class {
            "✓ Found"
        } else {
            "✗ Not found"
        }
    );
    println!(
        "   Method qualified name: {}",
        if found_method {
            "✓ Found"
        } else {
            "✗ Not found"
        }
    );
    println!(
        "   Interface qualified name: {}",
        if found_interface {
            "✓ Found"
        } else {
            "✗ Not found"
        }
    );
    println!(
        "   Enum qualified name: {}",
        if found_enum {
            "✓ Found"
        } else {
            "✗ Not found"
        }
    );

    if found_class && found_method && found_interface && found_enum {
        println!("\n🎉 SUCCESS: All qualified names are correct!");
    } else {
        println!("\n⚠ PARTIAL: Some qualified names were not found or incorrect");
    }
}
