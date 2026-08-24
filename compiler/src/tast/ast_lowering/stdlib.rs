//! Loading the standard library.

use super::*;
use crate::tast::node::HasSourceLocation;
use crate::tast::{core::*, node::MemoryEffects, node::*, type_resolution, *};
use parser::{
    AbstractDecl, BinaryOp, BlockElement, ClassDecl, ClassField, ClassFieldKind, EnumConstructor,
    EnumDecl, Expr, ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata,
    Modifier, ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

impl<'a> AstLowering<'a> {
    /// Load standard library types from source files
    pub(crate) fn load_standard_library(&mut self) -> LoweringResult<()> {
        use crate::tast::stdlib_loader::{StdLibConfig, StdLibLoader};
        use std::path::PathBuf;

        // IMPORTANT: Save current package context and clear it for stdlib loading
        // Standard library symbols should NOT inherit the current file's package prefix
        let saved_package = self.context.current_package.take();

        // Configure standard library paths
        let mut config = StdLibConfig::default();
        config.std_paths = vec![
            // Look for standard library files relative to the compiler crate
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("haxe-std"),
        ];
        config.default_imports = vec![
            "StdTypes.hx".to_string(),
            "String.hx".to_string(),
            "Array.hx".to_string(),
            "Iterator.hx".to_string(),
            "Std.hx".to_string(),
            "Type.hx".to_string(),
        ];

        let mut loader = StdLibLoader::new(config);

        // Load default imports (top-level types)
        let std_files = loader.load_default_imports();

        // Process each standard library file
        for std_file in std_files {
            // First pass: pre-register all types so they can reference each other
            for declaration in &std_file.declarations {
                self.pre_register_declaration(declaration)?;
            }
        }

        // Second pass: fully lower the standard library declarations to register methods
        // This is critical for things like Array.push to be available
        let std_files = loader.load_default_imports();
        for std_file in std_files {
            for declaration in &std_file.declarations {
                // Fully lower each declaration to register all methods and fields
                match declaration {
                    TypeDeclaration::Class(class_decl) => {
                        // Check if this is the Array class
                        if class_decl.name == "Array" {
                            // println!("DEBUG: Processing Array class from standard library");
                            // println!("DEBUG: Array has {} fields", class_decl.fields.len());
                            for field in &class_decl.fields {
                                if let ClassFieldKind::Function(func) = &field.kind {
                                    // println!("DEBUG: Array method: {}", func.name);
                                }
                            }
                        }

                        // Lower the class to register all its methods
                        let _ = self.lower_class_declaration(class_decl);
                    }
                    TypeDeclaration::Interface(interface_decl) => {
                        let _ = self.lower_interface_declaration(interface_decl);
                    }
                    TypeDeclaration::Enum(enum_decl) => {
                        let _ = self.lower_enum_declaration(enum_decl);
                    }
                    TypeDeclaration::Typedef(typedef_decl) => {
                        let _ = self.lower_typedef_declaration(typedef_decl);
                    }
                    TypeDeclaration::Abstract(abstract_decl) => {
                        let _ = self.lower_abstract_declaration(abstract_decl);
                    }
                    _ => {
                        // Package declarations etc.
                    }
                }
            }
        }

        // Register root-level stdlib classes that need to be resolvable by name
        // before their full definitions load, so references like `Std.int()` or
        // `Math.sin()` resolve regardless of load order.
        for type_name in TOPLEVEL_STDLIB_CLASSES {
            let interned_name = self.context.intern_string(type_name);

            // Skip if this class was already registered (e.g., from loading Std.hx).
            // Creating a duplicate symbol would shadow the fully-lowered one that has
            // methods and fields registered, breaking static method resolution.
            if let Some(_existing) = self
                .context
                .symbol_table
                .lookup_symbol(ScopeId::first(), interned_name)
            {
                continue;
            }

            let builtin_symbol = self
                .context
                .symbol_table
                .create_class_in_scope(interned_name, ScopeId::first());

            // Update qualified name
            self.context.update_symbol_qualified_name(builtin_symbol);

            // Add to root scope for global resolution
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(builtin_symbol, interned_name);
        }

        // Register built-in global functions
        let builtin_functions = [
            ("trace", vec!["Dynamic"], "Void"), // trace(value: Dynamic): Void
        ];

        for (func_name, param_types, return_type) in builtin_functions {
            let func_name_interned = self.context.intern_string(func_name);

            // Create parameter types
            let mut param_type_ids = Vec::new();
            for param_type_name in param_types {
                let param_type_id = match param_type_name {
                    "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                    "Int" => self.context.type_table.borrow().int_type(),
                    "String" => self.context.type_table.borrow().string_type(),
                    "Float" => self.context.type_table.borrow().float_type(),
                    "Bool" => self.context.type_table.borrow().bool_type(),
                    "Void" => self.context.type_table.borrow().void_type(),
                    _ => self.context.type_table.borrow().dynamic_type(),
                };
                param_type_ids.push(param_type_id);
            }

            // Create return type
            let return_type_id = match return_type {
                "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                "Int" => self.context.type_table.borrow().int_type(),
                "String" => self.context.type_table.borrow().string_type(),
                "Float" => self.context.type_table.borrow().float_type(),
                "Bool" => self.context.type_table.borrow().bool_type(),
                "Void" => self.context.type_table.borrow().void_type(),
                _ => self.context.type_table.borrow().dynamic_type(),
            };

            // Create function type (not just return type)
            let function_type_id = self
                .context
                .type_table
                .borrow_mut()
                .create_function_type(param_type_ids, return_type_id);

            // Create function symbol manually with proper scope
            use crate::tast::{
                LifetimeId, Mutability, SourceLocation, Symbol, SymbolFlags, SymbolKind, Visibility,
            };

            let func_symbol_id = SymbolId::from_raw(self.context.symbol_table.len() as u32);
            let func_symbol = Symbol {
                id: func_symbol_id,
                name: func_name_interned,
                kind: SymbolKind::Function,
                type_id: function_type_id, // Use function type, not return type
                scope_id: ScopeId::first(),
                lifetime_id: LifetimeId::invalid(),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                definition_location: SourceLocation::unknown(),
                is_used: false,
                is_exported: false,
                documentation: None,
                flags: SymbolFlags::NONE,
                package_id: None,
                qualified_name: None,
                native_name: None,
                frameworks: None,
                c_includes: None,
                c_sources: None,
                c_libs: None,
                js_import: None,
            };

            // Add symbol to symbol table
            self.context.symbol_table.add_symbol(func_symbol);

            // Add to root scope for global resolution
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(func_symbol_id, func_name_interned);
        }

        // Restore the original package context
        self.context.current_package = saved_package;

        Ok(())
    }
}
