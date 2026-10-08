//! Construction and per-file context.

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
    pub fn new(
        string_interner: &'a mut StringInterner,
        string_interner_rc: Rc<RefCell<StringInterner>>,
        symbol_table: &'a mut SymbolTable,
        type_table: &'a RefCell<TypeTable>,
        scope_tree: &'a mut ScopeTree,
        namespace_resolver: &'a mut super::namespace::NamespaceResolver,
        import_resolver: &'a mut super::namespace::ImportResolver,
    ) -> Self {
        let root_scope = ScopeId::first(); // Use first scope as root
        let context = LoweringContext::new(
            string_interner,
            string_interner_rc,
            symbol_table,
            type_table,
            scope_tree,
            root_scope,
            namespace_resolver,
            import_resolver,
        );

        Self {
            context,
            resolution_state: TypeResolutionState::default(),
            class_methods: BTreeMap::new(),
            class_fields: BTreeMap::new(),
            retrying_failed_attempt: false,
            incomplete_classes: std::collections::BTreeSet::new(),
            retryable_import: false,
            current_module_types: std::collections::BTreeSet::new(),
            unknown_return_slots: std::collections::BTreeSet::new(),
            class_parents: BTreeMap::new(),
            skip_stdlib_loading: false,
            static_sig_index: None,
            skip_pre_registration: false,
            collected_errors: Vec::new(),
            empty_array_inferred: std::collections::BTreeMap::new(),
            null_inferred: std::collections::BTreeSet::new(),
            null_read_in_closure: std::collections::BTreeSet::new(),
            null_decl_depth: std::collections::BTreeMap::new(),
            closure_depth: 0,
            rest_symbol_cache: None,
            map_first_uses: Vec::new(),
            empty_array_used_uncertain: std::collections::BTreeSet::new(),
            using_modules: Vec::new(),
            unresolved_usings: Vec::new(),
            type_usings: BTreeMap::new(),
            pending_usings: Vec::new(),
            // (class_fields will be seeded below if global_class_fields provided)
            in_static_method: false,
            lowering_callee: false,
            class_type_params: BTreeMap::new(),
            class_constructor_symbols: BTreeMap::new(),
            inferred_param_types: BTreeMap::new(),
            expected_lambda_params_stack: Vec::new(),
            expected_arg_type_stack: Vec::new(),
            suppress_callee_hint: false,
            deferred_macro_expander: None,
            deferred_macro_registry: None,
            generic_build_results: BTreeMap::new(),
            generic_build_active: std::collections::BTreeSet::new(),
            generic_build_engine: None,
            generic_build_resolving: std::collections::BTreeSet::new(),
            generic_build_monomorphs: std::collections::BTreeSet::new(),
            const_generic_templates: BTreeMap::new(),
            const_generic_instances: BTreeMap::new(),
            const_generic_origins: BTreeMap::new(),
            generated_declarations: Vec::new(),
            generated_type_names: std::collections::BTreeSet::new(),
            macro_probe_depth: 0,
            macro_private_access: false,
            class_value_bindings: BTreeMap::new(),
            deferred_macro_calls: BTreeMap::new(),
            abstract_casts: BTreeMap::new(),
            abstract_from_methods: BTreeMap::new(),
        }
    }

    /// Set whether to skip internal stdlib loading (for CompilationUnit)
    pub fn set_skip_stdlib_loading(&mut self, skip: bool) {
        self.skip_stdlib_loading = skip;
    }

    /// Hand over the expander and the call sites it deferred. Lowering
    /// re-expands each site when it reaches it, with itself as the typer.
    pub fn set_deferred_macros(
        &mut self,
        expander: &'a std::cell::RefCell<crate::macro_system::MacroExpander>,
        deferred: Vec<crate::macro_system::expander::DeferredMacroCall>,
    ) {
        self.deferred_macro_expander = Some(expander);
        self.deferred_macro_registry = Some(expander.borrow().registry().clone());
        self.generic_build_engine = expander.borrow().generic_build_engine().map(Rc::new);
        for call in deferred {
            self.deferred_macro_calls
                .insert((call.span.start, call.span.end), call.name);
        }
    }

    /// Set whether to skip pre-registration pass (for CompilationUnit with two-pass compilation)
    pub fn set_skip_pre_registration(&mut self, skip: bool) {
        self.skip_pre_registration = skip;
    }

    /// Share the program-wide declared-static-signature index (from
    /// CompilationUnit) for on-demand typing of cross-file statics.
    pub fn set_static_sig_index(
        &mut self,
        index: std::rc::Rc<RefCell<crate::tast::sig_index::StaticSigIndex>>,
    ) {
        self.static_sig_index = Some(index);
    }

    /// Get all collected errors from both context and collected_errors
    pub fn get_all_errors(&self) -> Vec<LoweringError> {
        let mut all_errors = Vec::new();
        all_errors.extend(self.context.errors.clone());
        all_errors.extend(self.collected_errors.clone());
        all_errors
    }

    /// Set the package context explicitly (used for stdlib loading with "haxe" package)
    pub fn set_package_context(&mut self, package_name: &str) {
        let package_path: Vec<_> = package_name
            .split('.')
            .map(|s| self.context.string_interner.intern(s))
            .collect();
        let package_id = self
            .context
            .namespace_resolver
            .get_or_create_package(package_path);
        self.context.current_package = Some(package_id);
    }

    /// Set the package context from a parsed package path (Vec of path segments)
    pub fn set_package_from_parts(&mut self, parts: &[String]) {
        if !parts.is_empty() {
            self.set_package_context(&parts.join("."));
        }
    }

    /// Clear the current package context (used for root scope)
    pub fn clear_package_context(&mut self) {
        self.context.clear_package_context();
    }

    /// Get the errors collected during lowering
    pub fn get_errors(&self) -> &[LoweringError] {
        &self.context.errors
    }
}
