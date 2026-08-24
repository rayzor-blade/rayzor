//! Metadata: annotations, derives and memory effects.

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
    /// Extract SymbolFlags and optional native name from metadata entries.
    /// Shared across class, abstract, and other type declarations.
    pub(crate) fn extract_metadata_flags(
        &mut self,
        meta_list: &[parser::haxe_ast::Metadata],
        symbol_id: SymbolId,
    ) -> crate::tast::symbols::SymbolFlags {
        use crate::tast::symbols::SymbolFlags;

        let mut flags = SymbolFlags::NONE;
        for meta in meta_list {
            let name = meta.name.strip_prefix(':').unwrap_or(&meta.name);
            match name {
                "generic" => flags = flags.union(SymbolFlags::GENERIC),
                "final" => flags = flags.union(SymbolFlags::FINAL),
                "forward" => flags = flags.union(SymbolFlags::FORWARD),
                "extern" => flags = flags.union(SymbolFlags::EXTERN),
                "keep" => flags = flags.union(SymbolFlags::KEEP),
                "native" => {
                    flags = flags.union(SymbolFlags::NATIVE);
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::String(native_str) = &first_param.kind {
                            let native_interned = self.context.string_interner.intern(&native_str);
                            if let Some(sym) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                                sym.native_name = Some(native_interned);
                            }
                        }
                    }
                }
                "cstruct" => {
                    flags = flags.union(SymbolFlags::CSTRUCT);
                    let no_mangle = meta.params.iter().any(|p| {
                        matches!(&p.kind, parser::haxe_ast::ExprKind::Ident(s) if s == "NoMangle")
                    });
                    if no_mangle {
                        flags = flags.union(SymbolFlags::NO_MANGLE);
                    }
                }
                "gpuStruct" => {
                    flags = flags.union(SymbolFlags::GPU_STRUCT);
                }
                "shader" => {
                    flags = flags.union(SymbolFlags::SHADER);
                }
                "no_mangle" => flags = flags.union(SymbolFlags::NO_MANGLE),
                "notNull" => flags = flags.union(SymbolFlags::NOT_NULL),
                "async" => flags = flags.union(SymbolFlags::ASYNC),
                "export" => flags = flags.union(SymbolFlags::WASM_EXPORT),
                "autoDeref" => flags = flags.union(SymbolFlags::AUTO_DEREF),
                "frameworks" | "cInclude" | "cSource" | "clib" => {
                    // @:frameworks(["Accelerate"]), @:cInclude(["vendor/stb"]), @:cSource(["lib.c"])
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::Array(elements) = &first_param.kind {
                            let mut names = Vec::new();
                            for elem in elements {
                                if let parser::haxe_ast::ExprKind::String(s) = &elem.kind {
                                    names.push(self.context.string_interner.intern(s));
                                }
                            }
                            if !names.is_empty() {
                                if let Some(sym) =
                                    self.context.symbol_table.get_symbol_mut(symbol_id)
                                {
                                    match name {
                                        "frameworks" => sym.frameworks = Some(names),
                                        "cInclude" => sym.c_includes = Some(names),
                                        "cSource" => sym.c_sources = Some(names),
                                        "clib" => sym.c_libs = Some(names),
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
                "jsImport" => {
                    // @:jsImport("module") on a class — sets JS host module for all methods.
                    // Methods inherit the module and use @:jsMethod for their import name.
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::String(module_name) = &first_param.kind {
                            let module_interned = self.context.string_interner.intern(module_name);
                            if let Some(sym) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                                // Store module name; methods inherit via propagate_js_import
                                sym.js_import = Some((module_interned, module_interned));
                            }
                        }
                    }
                }
                "jsMethod" => {
                    // @:jsMethod("function-name") on a method in a @:jsImport class.
                    // Sets the JS function name within the class's module.
                    // Also marks as native so the method is resolvable by the compiler.
                    flags = flags.union(SymbolFlags::NATIVE);
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::String(func_name) = &first_param.kind {
                            let func_interned = self.context.string_interner.intern(func_name);
                            if let Some(sym) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                                // Set native name for extern method resolution
                                sym.native_name = Some(func_interned);
                                // Store function name with a placeholder module —
                                // propagate_js_import will fill in the real module from the class.
                                let placeholder =
                                    self.context.string_interner.intern("__jsMethod__");
                                sym.js_import = Some((placeholder, func_interned));
                            }
                        }
                    }
                }
                "jsFunction" => {
                    // @:jsFunction("module", "function") on a standalone extern function.
                    flags = flags.union(SymbolFlags::NATIVE);
                    if meta.params.len() >= 2 {
                        if let (
                            parser::haxe_ast::ExprKind::String(module_name),
                            parser::haxe_ast::ExprKind::String(func_name),
                        ) = (&meta.params[0].kind, &meta.params[1].kind)
                        {
                            let module_interned = self.context.string_interner.intern(module_name);
                            let func_interned = self.context.string_interner.intern(func_name);
                            if let Some(sym) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                                sym.native_name = Some(func_interned);
                                sym.js_import = Some((module_interned, func_interned));
                            }
                        }
                    }
                }
                "jsGet" => {
                    // @:jsGet("property") on a field getter — maps to JS property read.
                    flags = flags.union(SymbolFlags::NATIVE);
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::String(prop_name) = &first_param.kind {
                            let getter_name = format!("get-{}", prop_name);
                            let func_interned = self.context.string_interner.intern(&getter_name);
                            if let Some(sym) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                                sym.native_name = Some(func_interned);
                                let placeholder =
                                    self.context.string_interner.intern("__jsMethod__");
                                sym.js_import = Some((placeholder, func_interned));
                            }
                        }
                    }
                }
                "jsSet" => {
                    // @:jsSet("property") on a field setter — maps to JS property write.
                    flags = flags.union(SymbolFlags::NATIVE);
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::String(prop_name) = &first_param.kind {
                            let setter_name = format!("set-{}", prop_name);
                            let func_interned = self.context.string_interner.intern(&setter_name);
                            if let Some(sym) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                                sym.native_name = Some(func_interned);
                                let placeholder =
                                    self.context.string_interner.intern("__jsMethod__");
                                sym.js_import = Some((placeholder, func_interned));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        flags
    }

    /// Extract memory safety annotations from metadata
    pub(crate) fn extract_memory_annotations(
        &self,
        metadata: &[parser::Metadata],
    ) -> Vec<crate::tast::MemoryAnnotation> {
        metadata
            .iter()
            .filter_map(|meta| {
                // Parse parameters if present (e.g., @:safety(strict=true))
                if !meta.params.is_empty() {
                    let params = self.parse_metadata_params(&meta.params);
                    crate::tast::MemoryAnnotation::from_metadata_with_params(&meta.name, &params)
                } else {
                    crate::tast::MemoryAnnotation::from_metadata_name(&meta.name)
                }
            })
            .collect()
    }

    /// Extract derived traits from @:derive metadata
    /// Example: @:derive([Clone, Copy]) or @:derive(Clone)
    pub(crate) fn extract_derived_traits(
        &self,
        class_decl: &parser::ClassDecl,
    ) -> Vec<crate::tast::DerivedTrait> {
        let trait_names = class_decl.get_derive_traits();

        let mut derived_traits = Vec::new();
        for trait_name in trait_names {
            if let Some(trait_) = crate::tast::DerivedTrait::from_str(&trait_name) {
                derived_traits.push(trait_);
            } else {
                warn!(
                    "Warning: Unknown derived trait '{}' in @:derive",
                    trait_name
                );
            }
        }

        // Validate trait dependencies (e.g., Eq requires PartialEq)
        let mut missing_deps = Vec::new();
        for trait_ in &derived_traits {
            for required in trait_.requires() {
                if !derived_traits.contains(&required) {
                    missing_deps.push((trait_.as_str(), required.as_str()));
                }
            }
        }

        if !missing_deps.is_empty() {
            warn!(
                "Warning: Missing required trait dependencies for class '{}':",
                class_decl.name
            );
            for (trait_, required) in missing_deps {
                warn!("  - {} requires {}", trait_, required);
            }
        }

        // Check if class has @:rc or @:arc - these require Clone
        let has_rc = class_decl
            .meta
            .iter()
            .any(|m| m.name == "rc" || m.name == "arc");
        if has_rc && !derived_traits.contains(&crate::tast::DerivedTrait::Clone) {
            eprintln!(
                "ERROR: Class '{}' has @:rc/@:arc metadata but does not derive Clone",
                class_decl.name
            );
            eprintln!("  Reference counted types must be Clone to support shared ownership");
            eprintln!("  Add @:derive(Clone) to fix this error");

            // Auto-add Clone for RC types to prevent compilation errors
            // User will see the warning above
            eprintln!("  Note: Automatically adding Clone trait for @:rc class");
            derived_traits.push(crate::tast::DerivedTrait::Clone);
        }

        // If Copy is derived, automatically add Clone (Copy implies Clone)
        if derived_traits.contains(&crate::tast::DerivedTrait::Copy)
            && !derived_traits.contains(&crate::tast::DerivedTrait::Clone)
        {
            derived_traits.push(crate::tast::DerivedTrait::Clone);
        }

        derived_traits
    }

    /// Parse metadata parameters from expressions
    /// Converts Expr nodes to String values (positional parameters)
    /// e.g., @:safety(true) -> ["true"], @:author("Name") -> ["Name"]
    fn parse_metadata_params(&self, params: &[parser::Expr]) -> Vec<String> {
        params
            .iter()
            .filter_map(|expr| {
                match &expr.kind {
                    // Boolean literals
                    parser::ExprKind::Bool(b) => Some(b.to_string()),
                    // Integer literals
                    parser::ExprKind::Int(n) => Some(n.to_string()),
                    // String literals
                    parser::ExprKind::String(s) => Some(s.clone()),
                    // Identifiers (e.g., true, false, null)
                    parser::ExprKind::Ident(name) => Some(name.clone()),
                    // Float literals
                    parser::ExprKind::Float(f) => Some(f.to_string()),
                    // Skip other expression types
                    _ => None,
                }
            })
            .collect()
    }

    /// Process @:overload metadata to extract method overload signatures
    pub(crate) fn process_overload_metadata(
        &mut self,
        metadata: &[parser::Metadata],
    ) -> LoweringResult<Vec<MethodOverload>> {
        let mut overload_signatures = Vec::new();

        for meta in metadata {
            if meta.name == "overload" {
                // @:overload(param1:Type1, param2:Type2 -> ReturnType)
                // For now, we'll implement a simplified version that parses function signature strings
                if meta.params.len() == 1 {
                    if let parser::ExprKind::String(signature_str) = &meta.params[0].kind {
                        // Parse the signature string to extract types
                        // This is a simplified implementation - a full parser would be more robust
                        if let Some(overload) =
                            self.parse_overload_signature(signature_str, &meta.span)?
                        {
                            overload_signatures.push(overload);
                        }
                    }
                }
            }
        }

        Ok(overload_signatures)
    }

    /// Process @:op metadata for operator overloading
    /// Extracts operator expressions like "A + B", "A * B", etc.
    pub(crate) fn process_operator_metadata(
        &mut self,
        metadata: &[parser::Metadata],
    ) -> LoweringResult<Vec<(String, Vec<String>)>> {
        let mut operator_metadata = Vec::new();

        for meta in metadata {
            if meta.name == "op" {
                // @:op(A + B) - operator expression is the first parameter
                if !meta.params.is_empty() {
                    // Extract the operator expression as a string
                    let operator_expr = self.expr_to_string(&meta.params[0]);

                    // Store the operator expression and any additional parameters
                    let additional_params: Vec<String> = meta.params[1..]
                        .iter()
                        .map(|e| self.expr_to_string(e))
                        .collect();

                    operator_metadata.push((operator_expr, additional_params));
                }
            }
        }

        Ok(operator_metadata)
    }

    /// Parse a function signature string from @:overload metadata
    fn parse_overload_signature(
        &mut self,
        signature: &str,
        span: &parser::Span,
    ) -> LoweringResult<Option<MethodOverload>> {
        use crate::tast::node::MethodOverload;

        // Simple signature parsing: "param1:Type1, param2:Type2 -> ReturnType"
        // Split on "->" to separate parameters from return type
        if let Some(arrow_pos) = signature.find("->") {
            let params_part = signature[..arrow_pos].trim();
            let return_part = signature[arrow_pos + 2..].trim();

            // Parse parameter types
            let mut parameter_types = Vec::new();
            if !params_part.is_empty() {
                for param in params_part.split(',') {
                    let param = param.trim();
                    if let Some(colon_pos) = param.find(':') {
                        let type_part = param[colon_pos + 1..].trim();
                        // Convert string type name to TypeId
                        if let Ok(type_id) = self.resolve_type_by_name(type_part) {
                            parameter_types.push(type_id);
                        } else {
                            // Use Dynamic as fallback for unresolved types
                            parameter_types.push(self.context.type_table.borrow().dynamic_type());
                        }
                    }
                }
            }

            // Parse return type
            let return_type = if let Ok(type_id) = self.resolve_type_by_name(return_part) {
                type_id
            } else {
                self.context.type_table.borrow().dynamic_type()
            };

            Ok(Some(MethodOverload {
                parameter_types,
                return_type,
                source_location: self.context.create_location_from_span(*span),
            }))
        } else {
            // No return type specified, treat as function with no parameters returning Void
            Ok(Some(MethodOverload {
                parameter_types: Vec::new(),
                return_type: self.context.type_table.borrow().void_type(),
                source_location: self.context.create_location_from_span(*span),
            }))
        }
    }
}
