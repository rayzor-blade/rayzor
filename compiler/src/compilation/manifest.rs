//! Restoring declarations from the BLADE symbol manifest: what a cached
//! compile registers instead of lowering the stdlib from source.

use super::*;

impl CompilationUnit {

    // === BLADE Symbol Loading Methods ===

    /// Load pre-compiled stdlib symbols from .bsym manifest
    /// Returns true if symbols were loaded successfully
    pub fn load_stdlib_symbols(&mut self) -> bool {
        let manifest_path = PathBuf::from(".rayzor/blade/stdlib/stdlib.bsym");
        let manifest = if manifest_path.exists() {
            load_symbol_manifest(&manifest_path)
        } else {
            debug!(
                "[BLADE] No symbol manifest at {}; using the bundled one",
                manifest_path.display()
            );
            load_symbol_manifest_from_bytes(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/stdlib.bsym"
            )))
        };

        match manifest {
            Ok(manifest) => {
                info!(
                    "[BLADE] Loading {} modules from symbol manifest",
                    manifest.modules.len()
                );
                self.register_symbols_from_manifest(&manifest);
                // Also register builtin globals like 'trace' that aren't in the manifest
                self.register_builtin_globals();
                true
            }
            Err(e) => {
                debug!("[BLADE] Failed to load symbol manifest: {}", e);
                false
            }
        }
    }


    /// Register built-in global symbols like 'trace' that aren't in the BLADE manifest
    pub(crate) fn register_builtin_globals(&mut self) {
        use crate::tast::{
            LifetimeId, Mutability, SourceLocation, Symbol, SymbolFlags, SymbolKind, Visibility,
        };

        // Register built-in global functions
        let builtin_functions = [
            ("trace", vec!["Dynamic"], "Void"), // trace(value: Dynamic): Void
        ];

        for (func_name, param_types, return_type_str) in builtin_functions {
            let func_name_interned = self.string_interner.intern(func_name);

            // Create parameter types
            let param_type_ids: Vec<TypeId> = param_types
                .iter()
                .map(|param_type_name| match *param_type_name {
                    "Dynamic" => self.type_table.borrow().dynamic_type(),
                    "Int" => self.type_table.borrow().int_type(),
                    "String" => self.type_table.borrow().string_type(),
                    "Float" => self.type_table.borrow().float_type(),
                    "Bool" => self.type_table.borrow().bool_type(),
                    "Void" => self.type_table.borrow().void_type(),
                    _ => self.type_table.borrow().dynamic_type(),
                })
                .collect();

            // Create return type
            let return_type_id = match return_type_str {
                "Dynamic" => self.type_table.borrow().dynamic_type(),
                "Int" => self.type_table.borrow().int_type(),
                "String" => self.type_table.borrow().string_type(),
                "Float" => self.type_table.borrow().float_type(),
                "Bool" => self.type_table.borrow().bool_type(),
                "Void" => self.type_table.borrow().void_type(),
                _ => self.type_table.borrow().dynamic_type(),
            };

            // Create function type
            let function_type_id = self
                .type_table
                .borrow_mut()
                .create_function_type(param_type_ids, return_type_id);

            // Create function symbol
            let func_symbol_id = SymbolId::from_raw(self.symbol_table.len() as u32);
            let func_symbol = Symbol {
                id: func_symbol_id,
                name: func_name_interned,
                kind: SymbolKind::Function,
                type_id: function_type_id,
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
            self.symbol_table.add_symbol(func_symbol);

            // Add to root scope for global resolution
            if let Some(scope) = self.scope_tree.get_scope_mut(ScopeId::first()) {
                scope.add_symbol(func_symbol_id, func_name_interned);
            }

            trace!("[BLADE] Registered builtin: {}", func_name);
        }
    }


    /// Register all symbols from a loaded manifest
    pub(crate) fn register_symbols_from_manifest(&mut self, manifest: &BladeSymbolManifest) {
        let mut total_classes = 0;
        let mut total_enums = 0;
        let mut total_aliases = 0;
        let mut total_abstracts = 0;
        let mut total_methods = 0;

        // A manifest records module paths relative to the standard library it was
        // built from, so it does not depend on where that tree lived at build
        // time. Resolve them against this run's stdlib root to get the same paths
        // the namespace resolver produces.
        let stdlib_root = self
            .config
            .stdlib_paths
            .iter()
            .find(|path| path.exists())
            .cloned();

        // Two passes over the unit. A class's members are registered only
        // after every class in the manifest exists, so a signature naming a
        // sibling resolves to that sibling's type rather than to a placeholder
        // — which it did whenever the sibling happened to be declared later.
        let mut declared: Vec<(&BladeClassInfo, DeclaredClass)> = Vec::new();

        for module in &manifest.modules {
            // Mark this file as "loaded" so load_import_file_recursive will skip it
            // This prevents redundant re-parsing of files whose symbols are already cached
            let source_path = match &stdlib_root {
                Some(root) => root.join(&module.source_path),
                None => PathBuf::from(&module.source_path),
            };
            self.namespace_resolver.mark_file_loaded(source_path);

            for class_info in &module.types.classes {
                total_methods += class_info.methods.len() + class_info.static_methods.len();
                declared.push((class_info, self.declare_class_from_blade(class_info)));
                total_classes += 1;
            }
            for enum_info in &module.types.enums {
                self.register_enum_from_blade(enum_info);
                total_enums += 1;
            }
            for alias_info in &module.types.type_aliases {
                self.register_type_alias_from_blade(alias_info);
                total_aliases += 1;
            }
            for abstract_info in &module.types.abstracts {
                let method_count = abstract_info.methods.len() + abstract_info.static_methods.len();
                self.register_abstract_from_blade(abstract_info);
                total_abstracts += 1;
                total_methods += method_count;
            }
        }

        for (class_info, declaration) in &declared {
            self.define_class_members_from_blade(class_info, declaration);
        }

        debug!("[BLADE] Registered {} classes, {} enums, {} aliases, {} abstracts ({} methods) from manifest",
            total_classes, total_enums, total_aliases, total_abstracts, total_methods);
    }


    /// Index a packaged manifest type under its short name for signature
    /// resolution. First registration wins, so the mapping is stable per run.
    pub(crate) fn index_manifest_short_name(
        &mut self,
        package: &[String],
        short_name: InternedString,
        symbol_id: SymbolId,
    ) {
        if !package.is_empty() {
            self.manifest_types_by_short_name
                .entry(short_name)
                .or_insert(symbol_id);
        }
    }


    /// Mint a TypeParameter symbol and TypeId for each declared name.
    pub(crate) fn register_manifest_type_params(&mut self, names: &[String]) -> BTreeMap<String, TypeId> {
        let mut params = BTreeMap::new();
        for name in names {
            let interned = self.string_interner.intern(name);
            let symbol = self.symbol_table.create_type_parameter(interned, vec![]);
            let type_id = self.type_table.borrow_mut().create_type_parameter(
                symbol,
                vec![],
                crate::tast::core::Variance::Invariant,
            );
            params.insert(name.clone(), type_id);
        }
        params
    }


    /// Register a class from BLADE symbol info
    /// Publish a class's identity: symbol, scope, class type, aliases,
    /// short-name index and type parameters.
    ///
    /// Nothing here resolves another declaration, so every class in a unit can
    /// be declared before any signature is resolved — which is what stops a
    /// signature naming a sibling from restoring as a placeholder purely
    /// because that sibling was declared later.
    pub(crate) fn declare_class_from_blade(&mut self, class_info: &BladeClassInfo) -> DeclaredClass {
        let short_name = self.string_interner.intern(&class_info.name);
        let qualified_name = if class_info.package.is_empty() {
            class_info.name.clone()
        } else {
            format!("{}.{}", class_info.package.join("."), class_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create a scope for the class members
        let class_scope = self.scope_tree.create_scope(Some(ScopeId::first()));

        // Create class symbol using the existing helper method
        let root_name = manifest_root_name(&class_info.package, short_name, qualified_interned);
        let symbol_id = self
            .symbol_table
            .create_class_in_scope(root_name, ScopeId::first());
        self.index_manifest_short_name(&class_info.package, short_name, symbol_id);

        // Update symbol metadata including the class scope
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.name = short_name;
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
            sym.scope_id = class_scope; // Set the scope where members are registered
            if class_info.is_extern {
                sym.flags = sym.flags.union(SymbolFlags::EXTERN);
            }
            if class_info.is_final {
                sym.flags = sym.flags.union(SymbolFlags::FINAL);
            }
            if class_info.is_abstract {
                sym.flags = sym.flags.union(SymbolFlags::ABSTRACT);
            }
            if let Some(ref native) = class_info.native_name {
                sym.flags = sym.flags.union(SymbolFlags::NATIVE);
                let native_interned = self.string_interner.intern(native);
                sym.native_name = Some(native_interned);
            }
        }

        // Create class type
        let class_type = self
            .type_table
            .borrow_mut()
            .create_class_type(symbol_id, vec![]);

        // Update symbol with type
        self.symbol_table.update_symbol_type(symbol_id, class_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(class_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        // A generic class's parameters, so `Channel<T>.receive():T` restores as
        // a type parameter rather than an unresolved placeholder. Callers
        // recover the ordered ids from the symbol table to substitute the
        // arguments at each instantiation.
        let type_params = self.register_manifest_type_params(&class_info.type_params);
        if !type_params.is_empty() {
            let ordered: Vec<TypeId> = class_info
                .type_params
                .iter()
                .filter_map(|name| type_params.get(name).copied())
                .collect();
            self.symbol_table.set_class_type_params(symbol_id, ordered);
        }
        DeclaredClass {
            symbol_id,
            class_scope,
            type_params,
        }
    }


    /// Register a class's members, once every declaration in the unit exists.
    pub(crate) fn define_class_members_from_blade(
        &mut self,
        class_info: &BladeClassInfo,
        declared: &DeclaredClass,
    ) {
        let symbol_id = declared.symbol_id;
        let class_scope = declared.class_scope;
        let type_params = declared.type_params.clone();
        let qualified_name = if class_info.package.is_empty() {
            class_info.name.clone()
        } else {
            format!("{}.{}", class_info.package.join("."), class_info.name)
        };
        let outer_type_params = std::mem::replace(&mut self.manifest_type_params, type_params);

        // Register instance methods
        for method in &class_info.methods {
            self.register_method_from_blade(method, symbol_id, class_scope, false);
        }

        // Register static methods
        for method in &class_info.static_methods {
            self.register_method_from_blade(method, symbol_id, class_scope, true);
        }

        // Register constructor if present, and record it as the class's
        // constructor. Inference of a generic class's type arguments matches
        // the declared constructor parameters against the call's arguments, so
        // a restored class whose constructor is registered but not recorded
        // leaves `new Channel(value)` with nothing to infer `T` from.
        if let Some(ctor) = &class_info.constructor {
            let ctor_symbol = self.register_method_from_blade(ctor, symbol_id, class_scope, false);
            self.symbol_table
                .set_class_constructor(symbol_id, ctor_symbol);
        }

        // Register fields, and seed the same per-class field table that a
        // fresh AstLowering pass would export. BLADE symbol restore registers
        // fields into the class scope, but static field lowering consults
        // `global_class_fields` when a later file is lowered. Without this,
        // manifest-restored externs such as Math expose methods but not
        // constants like Math.POSITIVE_INFINITY.
        let mut restored_fields = Vec::new();
        for field in &class_info.fields {
            let field_symbol = self.register_field_from_blade(field, symbol_id, class_scope);
            let field_name = self.string_interner.intern(&field.name);
            restored_fields.push((field_name, field_symbol, field.is_static));
        }

        // Register static fields
        for field in &class_info.static_fields {
            let field_symbol = self.register_field_from_blade(field, symbol_id, class_scope);
            let field_name = self.string_interner.intern(&field.name);
            restored_fields.push((field_name, field_symbol, field.is_static));
        }

        if !restored_fields.is_empty() {
            self.global_class_fields
                .entry(symbol_id)
                .or_insert_with(|| restored_fields.clone());
        }

        self.manifest_type_params = outer_type_params;

        trace!(
            "[BLADE] Registered class: {} ({} methods, {} fields) in scope {:?}",
            qualified_name,
            class_info.methods.len() + class_info.static_methods.len(),
            class_info.fields.len() + class_info.static_fields.len(),
            class_scope
        );
    }


    /// Register a method from BLADE info into a class scope
    pub(crate) fn register_method_from_blade(
        &mut self,
        method: &BladeMethodInfo,
        class_symbol: SymbolId,
        class_scope: ScopeId,
        is_static: bool,
    ) -> SymbolId {
        let method_name = self.string_interner.intern(&method.name);
        let class_qualified_name = self.symbol_table.get_symbol(class_symbol).and_then(|sym| {
            sym.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(sym.name))
                .map(|s| s.to_string())
        });

        // Create the function symbol
        let method_symbol = self
            .symbol_table
            .create_function_in_scope(method_name, class_scope);

        // Parse parameter types and return type to create a function type
        let param_types: Vec<TypeId> = method
            .params
            .iter()
            .map(|p| self.resolve_blade_type_or_dynamic(p.param_type.as_ref()))
            .collect();
        let return_type = self.resolve_blade_type_or_dynamic(method.return_type.as_ref());

        // Create function type
        let func_type = self
            .type_table
            .borrow_mut()
            .create_type(TypeKind::Function {
                params: param_types,
                return_type,
                effects: crate::tast::core::FunctionEffects::default(),
            });

        // Resolve native_name from the cached `@:native` metadata before we
        // open the borrow, so we can intern the string without aliasing
        // `self.symbol_table`.
        let native_name_interned = method
            .native_name
            .as_ref()
            .map(|n| self.string_interner.intern(n));

        // Update symbol with type and flags
        if let Some(sym) = self.symbol_table.get_symbol_mut(method_symbol) {
            sym.type_id = func_type;
            if is_static {
                sym.flags = sym.flags.union(SymbolFlags::STATIC);
            }
            if method.is_inline {
                sym.flags = sym.flags.union(SymbolFlags::INLINE);
            }
            if !method.is_public {
                sym.visibility = crate::tast::symbols::Visibility::Private;
            }
            if let Some(class_name) = &class_qualified_name {
                let method_qualified_name = self
                    .string_interner
                    .intern(&format!("{}.{}", class_name, method.name));
                sym.qualified_name = Some(method_qualified_name);
            }
            // Restore `@:native("foo")` from the BLADE cache. Without this,
            // stdlib runtime mappings have to be keyed by Haxe method name
            // (defeating the purpose of `@:native`), and FFI symbol lookup
            // through `sym.native_name` always finds None for cached types.
            if let Some(native_interned) = native_name_interned {
                sym.native_name = Some(native_interned);
                sym.flags = sym.flags.union(SymbolFlags::NATIVE);
            }
        }

        // Add to scope, updating both symbol list and name lookup cache.
        if let Some(scope) = self.scope_tree.get_scope_mut(class_scope) {
            scope.add_symbol(method_symbol, method_name);
        }

        method_symbol
    }


    /// Register a field from BLADE info into a class scope
    pub(crate) fn register_field_from_blade(
        &mut self,
        field: &crate::ir::blade::BladeFieldInfo,
        class_symbol: SymbolId,
        class_scope: ScopeId,
    ) -> SymbolId {
        let field_name = self.string_interner.intern(&field.name);

        // Create the field symbol
        let field_symbol = self.symbol_table.create_field(field_name);

        // Record which class owns the field. Saving a module writes a
        // property out under its owning class and drops any it cannot
        // attribute, so a field registered without this loses its accessors
        // the moment the module is cached.
        if let Some(owner) = self.symbol_table.get_symbol(class_symbol) {
            let owner_name = owner
                .qualified_name
                .or(Some(owner.name))
                .and_then(|n| self.string_interner.get(n))
                .map(|n| n.to_string());
            if let Some(owner_name) = owner_name {
                self.import_field_class_names
                    .insert(field_symbol, owner_name);
            }
        }

        // Parse field type
        let field_type = self.resolve_blade_type_or_dynamic(field.field_type.as_ref());

        // A property reads and writes through methods. Restoring it as a plain
        // field leaves an access lowering to a load from a slot the class does
        // not have, which surfaces as the field not existing at all.
        if let Some(property) = &field.property {
            let accessor = |access: &bsym::BladeAccess, interner: &mut StringInterner| match access
            {
                bsym::BladeAccess::Default => crate::tast::PropertyAccessor::Default,
                bsym::BladeAccess::Null => crate::tast::PropertyAccessor::Null,
                bsym::BladeAccess::Never => crate::tast::PropertyAccessor::Never,
                bsym::BladeAccess::Dynamic => crate::tast::PropertyAccessor::Dynamic,
                bsym::BladeAccess::Method(name) => {
                    crate::tast::PropertyAccessor::Method(interner.intern(name))
                }
            };
            let getter = accessor(&property.getter, &mut self.string_interner);
            let setter = accessor(&property.setter, &mut self.string_interner);
            self.import_property_access_map.insert(
                field_symbol,
                // Restored entries drive accessor dispatch. The storage bit is
                // read from the declaration when a layout is built, and a
                // restored class has none.
                crate::tast::PropertyAccessInfo {
                    getter,
                    setter,
                    is_var: false,
                },
            );
        }

        // Update symbol with type and flags
        if let Some(sym) = self.symbol_table.get_symbol_mut(field_symbol) {
            sym.type_id = field_type;
            sym.scope_id = class_scope;
            if field.is_static {
                sym.flags = sym.flags.union(SymbolFlags::STATIC);
            }
            if field.is_final {
                sym.mutability = crate::tast::symbols::Mutability::Immutable;
            }
            if !field.is_public {
                sym.visibility = crate::tast::symbols::Visibility::Private;
            }
        }

        // Add to scope (using add_symbol to update both symbols list and lookup cache)
        if let Some(scope) = self.scope_tree.get_scope_mut(class_scope) {
            scope.add_symbol(field_symbol, field_name);
        }

        field_symbol
    }


    /// Register an enum from BLADE symbol info
    pub(crate) fn register_enum_from_blade(&mut self, enum_info: &BladeEnumInfo) -> SymbolId {
        let short_name = self.string_interner.intern(&enum_info.name);
        let qualified_name = if enum_info.package.is_empty() {
            enum_info.name.clone()
        } else {
            format!("{}.{}", enum_info.package.join("."), enum_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create enum symbol using the existing helper method
        let root_name = manifest_root_name(&enum_info.package, short_name, qualified_interned);
        let symbol_id = self
            .symbol_table
            .create_enum_in_scope(root_name, ScopeId::first());
        self.index_manifest_short_name(&enum_info.package, short_name, symbol_id);

        // Update symbol metadata
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.name = short_name;
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
            if enum_info.is_extern {
                sym.flags = sym.flags.union(SymbolFlags::EXTERN);
            }
        }

        // Create type parameters for generic enums (e.g., Option<T>, Result<T, E>)
        let type_param_map = self.register_manifest_type_params(&enum_info.type_params);
        let type_param_ids: Vec<TypeId> = enum_info
            .type_params
            .iter()
            .filter_map(|name| type_param_map.get(name).copied())
            .collect();

        // Create enum type with type parameters
        let enum_type = self
            .type_table
            .borrow_mut()
            .create_enum_type(symbol_id, type_param_ids);

        // Update symbol with type
        self.symbol_table.update_symbol_type(symbol_id, enum_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(enum_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        // Register enum variants in root scope so they can be resolved
        // during pattern matching and constructor calls
        for variant in &enum_info.variants {
            let variant_name = self.string_interner.intern(&variant.name);
            // A packaged enum's constructors are no more ambient than the enum
            // itself: `Some`/`None` belong to `haxe.ds.Option`, and a bare root
            // slot here takes the name a user enum's own arm needs.
            let variant_root_name = if enum_info.package.is_empty() {
                variant_name
            } else {
                self.string_interner
                    .intern(&format!("{}.{}", qualified_name, variant.name))
            };
            let variant_symbol = self.symbol_table.create_enum_variant_in_scope(
                variant_root_name,
                ScopeId::first(),
                symbol_id,
            );
            if variant_root_name != variant_name {
                if let Some(sym) = self.symbol_table.get_symbol_mut(variant_symbol) {
                    sym.name = variant_name;
                    sym.flags = sym.flags.union(SymbolFlags::QUALIFIED_ONLY);
                }
            }

            // For generic enum variants whose params reference type parameters,
            // create a Function type with proper TypeParameter TypeIds so
            // resolve_field_type() can substitute them (e.g., T → Int).
            // Only set this for variants where ALL params resolve to known type params;
            // non-generic variants (like TClass(c:Class<Dynamic>)) must NOT get a
            // Function type with invalid params, as that corrupts field type resolution.
            if !variant.params.is_empty() && !type_param_map.is_empty() {
                let param_type_ids: Vec<_> = variant
                    .params
                    .iter()
                    .map(|p| {
                        blade_type_param_name(p.param_type.as_ref())
                            .and_then(|name| type_param_map.get(name).copied())
                            .unwrap_or(TypeId::invalid())
                    })
                    .collect();
                // Only set if all params resolved to valid type parameter TypeIds
                if param_type_ids.iter().all(|id| id.is_valid()) {
                    let fn_type = self
                        .type_table
                        .borrow_mut()
                        .create_function_type(param_type_ids, enum_type);
                    self.symbol_table
                        .update_symbol_type(variant_symbol, fn_type);
                }
            }

            // Add variant to root scope for global resolution
            self.scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(variant_symbol, variant_name);
        }

        trace!(
            "[BLADE] Registered enum: {} ({} variants)",
            qualified_name,
            enum_info.variants.len()
        );

        symbol_id
    }


    /// Pre-register type declarations from default stdlib files (e.g. StdTypes.hx).
    /// This is lightweight: it parses the files and registers enum/class symbols
    /// into the symbol table without full TAST lowering, preserving lazy stdlib performance.
    /// Register a type alias from BLADE symbol info
    pub(crate) fn register_type_alias_from_blade(&mut self, alias_info: &BladeTypeAliasInfo) -> SymbolId {
        let short_name = self.string_interner.intern(&alias_info.name);
        let qualified_name = if alias_info.package.is_empty() {
            alias_info.name.clone()
        } else {
            format!("{}.{}", alias_info.package.join("."), alias_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create type alias symbol using the existing helper method
        let root_name = manifest_root_name(&alias_info.package, short_name, qualified_interned);
        let symbol_id = self
            .symbol_table
            .create_type_alias_in_scope(root_name, ScopeId::first());
        self.index_manifest_short_name(&alias_info.package, short_name, symbol_id);

        // Update symbol metadata
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.name = short_name;
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
        }

        // A generic alias's own parameters, so `KeyValueIterator<K,V> =
        // Iterator<{key:K, value:V}>` restores with K and V as type parameters
        // rather than unresolved names, and keeps them on the node — which is
        // what binds an argument at each use.
        let alias_params = self.register_manifest_type_params(&alias_info.type_params);
        let ordered: Vec<TypeId> = alias_info
            .type_params
            .iter()
            .filter_map(|name| alias_params.get(name).copied())
            .collect();
        let outer_params = std::mem::replace(&mut self.manifest_type_params, alias_params);
        // Parse the target type string and create appropriate TypeId
        let target_type = self.resolve_blade_type(&alias_info.target_type);
        self.manifest_type_params = outer_params;

        // Create type alias type
        let alias_type = self
            .type_table
            .borrow_mut()
            .create_type(TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args: ordered,
            });

        // Update symbol with type
        self.symbol_table.update_symbol_type(symbol_id, alias_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(alias_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        trace!(
            "[BLADE] Registered type alias: {} -> {:?}",
            qualified_name,
            alias_info.target_type
        );

        symbol_id
    }


    /// Register an abstract type from BLADE symbol info
    pub(crate) fn register_abstract_from_blade(&mut self, abstract_info: &BladeAbstractInfo) -> SymbolId {
        let short_name = self.string_interner.intern(&abstract_info.name);
        let qualified_name = if abstract_info.package.is_empty() {
            abstract_info.name.clone()
        } else {
            format!("{}.{}", abstract_info.package.join("."), abstract_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create a scope for the abstract's methods
        let abstract_scope = self.scope_tree.create_scope(Some(ScopeId::first()));

        // Create abstract symbol using the existing helper method
        let root_name = manifest_root_name(&abstract_info.package, short_name, qualified_interned);
        let symbol_id = self
            .symbol_table
            .create_abstract_in_scope(root_name, ScopeId::first());
        self.index_manifest_short_name(&abstract_info.package, short_name, symbol_id);

        // The abstract's own parameters, for the same reason an alias needs
        // them: `Map<K,V>`'s underlying `IMap<K,V>` names them.
        let abstract_params = self.register_manifest_type_params(&abstract_info.type_params);
        let ordered_abstract: Vec<TypeId> = abstract_info
            .type_params
            .iter()
            .filter_map(|name| abstract_params.get(name).copied())
            .collect();
        let outer_abstract_params =
            std::mem::replace(&mut self.manifest_type_params, abstract_params);
        // Parse the underlying type
        let underlying_type = self.resolve_blade_type(&abstract_info.underlying_type);
        self.manifest_type_params = outer_abstract_params;

        // Update symbol metadata including the abstract scope
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.name = short_name;
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
            sym.scope_id = abstract_scope; // Set the scope where methods are registered
            if let Some(ref native) = abstract_info.native_name {
                sym.flags = sym.flags.union(SymbolFlags::NATIVE);
                let native_interned = self.string_interner.intern(native);
                sym.native_name = Some(native_interned);
            }
        }

        // Create abstract type
        let abstract_type = self
            .type_table
            .borrow_mut()
            .create_type(TypeKind::Abstract {
                symbol_id,
                underlying: Some(underlying_type),
                type_args: ordered_abstract,
            });

        // Update symbol with type
        self.symbol_table
            .update_symbol_type(symbol_id, abstract_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(abstract_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        // Register instance methods
        for method in &abstract_info.methods {
            self.register_method_from_blade(method, symbol_id, abstract_scope, false);
        }

        // Register static methods
        for method in &abstract_info.static_methods {
            self.register_method_from_blade(method, symbol_id, abstract_scope, true);
        }

        trace!(
            "[BLADE] Registered abstract: {} ({} methods) in scope {:?}",
            qualified_name,
            abstract_info.methods.len() + abstract_info.static_methods.len(),
            abstract_scope
        );

        symbol_id
    }


    /// Parse a type string (e.g., "Array<Int>", "String", "Null<Float>") and return a TypeId
    /// Resolve a type the manifest recorded.
    ///
    /// The manifest stores the shape of a type, so this only has names left to
    /// resolve — no syntax is re-derived, and a structure, an intersection or a
    /// nested function type arrives intact instead of as a placeholder.
    pub(crate) fn resolve_blade_type(&mut self, ty: &bsym::BladeType) -> TypeId {
        use bsym::BladeType;
        match ty {
            BladeType::Path {
                package,
                name,
                sub,
                params,
            } => self.resolve_blade_path(package, name, sub.as_deref(), params),
            BladeType::Function { params, ret } => {
                let params: Vec<TypeId> =
                    params.iter().map(|p| self.resolve_blade_type(p)).collect();
                let ret = self.resolve_blade_type(ret);
                self.type_table
                    .borrow_mut()
                    .create_function_type(params, ret)
            }
            BladeType::Optional(inner) => {
                let inner = self.resolve_blade_type(inner);
                self.type_table.borrow_mut().create_optional_type(inner)
            }
            BladeType::Anonymous { fields } => {
                let fields: Vec<crate::tast::core::AnonymousField> = fields
                    .iter()
                    .map(|f| crate::tast::core::AnonymousField {
                        name: self.string_interner.intern(&f.name),
                        type_id: self.resolve_blade_type(&f.ty),
                        is_public: true,
                        optional: matches!(f.ty, bsym::BladeType::Optional(_)),
                    })
                    .collect();
                self.type_table
                    .borrow_mut()
                    .create_type(TypeKind::Anonymous { fields })
            }
            // The left side is what a value of an intersection is dispatched
            // on; the constraint the right side adds has no representation.
            BladeType::Intersection { left, .. } => self.resolve_blade_type(left),
            // A type the source left to inference is not a type this pass can
            // name, and `Dynamic` is what an unconstrained value already is.
            BladeType::Wildcard => self.type_table.borrow().dynamic_type(),
        }
    }


    /// A declared type, or `Dynamic` where the source annotated none.
    pub(crate) fn resolve_blade_type_or_dynamic(&mut self, ty: Option<&bsym::BladeType>) -> TypeId {
        match ty {
            Some(ty) => self.resolve_blade_type(ty),
            None => self.type_table.borrow().dynamic_type(),
        }
    }


    pub(crate) fn resolve_blade_path(
        &mut self,
        package: &[String],
        name: &str,
        sub: Option<&str>,
        params: &[bsym::BladeType],
    ) -> TypeId {
        // A parameter of the type being registered shadows everything else.
        if package.is_empty() && sub.is_none() && params.is_empty() {
            if let Some(type_id) = self.manifest_type_params.get(name) {
                return *type_id;
            }
            match name {
                "Int" => return self.type_table.borrow().int_type(),
                "Float" => return self.type_table.borrow().float_type(),
                "Bool" => return self.type_table.borrow().bool_type(),
                "String" => return self.type_table.borrow().string_type(),
                "Void" => return self.type_table.borrow().void_type(),
                "Dynamic" => return self.type_table.borrow().dynamic_type(),
                _ => {}
            }
        }

        // `Null<T>` and `Array<T>` have representations of their own rather
        // than being ordinary classes carrying a type argument.
        if package.is_empty() && sub.is_none() && params.len() == 1 {
            match name {
                "Null" => {
                    let inner = self.resolve_blade_type(&params[0]);
                    return self.type_table.borrow_mut().create_optional_type(inner);
                }
                "Array" => {
                    let element = self.resolve_blade_type(&params[0]);
                    return self.type_table.borrow_mut().create_array_type(element);
                }
                _ => {}
            }
        }

        let args: Vec<TypeId> = params.iter().map(|p| self.resolve_blade_type(p)).collect();

        // Qualified first: a bare name is ambiguous between the library's
        // declaration and a user's, and the manifest recorded which was meant.
        // The unqualified case is most of them, and borrows rather than builds
        // a name — this runs once per type in every signature in the library.
        let qualified = (!package.is_empty() || sub.is_some()).then(|| {
            let mut qualified = String::new();
            for part in package {
                qualified.push_str(part);
                qualified.push('.');
            }
            qualified.push_str(name);
            if let Some(sub) = sub {
                qualified.push('.');
                qualified.push_str(sub);
            }
            qualified
        });
        let lookup = qualified.as_deref().unwrap_or(name);

        let symbol = self
            .lookup_type_symbol(lookup)
            .or_else(|| self.lookup_type_symbol(sub.unwrap_or(name)));

        let Some(symbol_id) = symbol else {
            let name = self.string_interner.intern(lookup);
            return self
                .type_table
                .borrow_mut()
                .create_type(TypeKind::Placeholder { name });
        };

        // A TypeAlias, Abstract or Enum already OWNS a type; wrapping its
        // symbol in a Class type synthesises one no lookup can resolve back.
        let own_type = self
            .symbol_table
            .get_symbol(symbol_id)
            .filter(|s| {
                matches!(
                    s.kind,
                    crate::tast::symbols::SymbolKind::TypeAlias
                        | crate::tast::symbols::SymbolKind::Abstract
                        | crate::tast::symbols::SymbolKind::Enum
                )
            })
            .map(|s| s.type_id)
            .filter(|t| t.is_valid());
        if let Some(t) = own_type {
            if args.is_empty() {
                return t;
            }
        }

        self.type_table
            .borrow_mut()
            .create_class_type(symbol_id, args)
    }


    pub(crate) fn parse_type_string(&mut self, type_str: &str) -> TypeId {
        let type_str = type_str.trim();

        // A parameter of the type being registered shadows everything else.
        if let Some(type_id) = self.manifest_type_params.get(type_str) {
            return *type_id;
        }

        // Handle primitives
        match type_str {
            "Int" => return self.type_table.borrow().int_type(),
            "Float" => return self.type_table.borrow().float_type(),
            "Bool" => return self.type_table.borrow().bool_type(),
            "String" => return self.type_table.borrow().string_type(),
            "Void" => return self.type_table.borrow().void_type(),
            "Dynamic" => return self.type_table.borrow().dynamic_type(),
            _ => {}
        }

        // Handle Null<T>
        if let Some(inner) = type_str
            .strip_prefix("Null<")
            .and_then(|s| s.strip_suffix(">"))
        {
            let inner_type = self.parse_type_string(inner);
            return self
                .type_table
                .borrow_mut()
                .create_optional_type(inner_type);
        }

        // Handle Array<T>
        if let Some(inner) = type_str
            .strip_prefix("Array<")
            .and_then(|s| s.strip_suffix(">"))
        {
            let element_type = self.parse_type_string(inner);
            return self.type_table.borrow_mut().create_array_type(element_type);
        }

        // Handle function types: (A, B) -> C
        if type_str.starts_with("(") {
            if let Some((params_str, return_str)) = type_str.split_once(") -> ") {
                let params_str = params_str.trim_start_matches('(');
                let params: Vec<TypeId> = if params_str.is_empty() {
                    vec![]
                } else {
                    self.parse_type_list(params_str)
                };
                let return_type = self.parse_type_string(return_str);
                return self
                    .type_table
                    .borrow_mut()
                    .create_function_type(params, return_type);
            }
        }

        // Handle generic types: ClassName<T, U>
        // Need to find the matching close bracket, not just the last '>'
        if let Some(open) = type_str.find('<') {
            // Find the matching closing bracket
            let mut depth = 0;
            let mut close = None;
            for (i, ch) in type_str.char_indices() {
                match ch {
                    '<' => depth += 1,
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(close) = close {
                if open < close {
                    let base_name = &type_str[..open];
                    let args_str = &type_str[open + 1..close];
                    let type_args = self.parse_type_list(args_str);

                    // Look up the base type
                    if let Some(symbol_id) = self.lookup_type_symbol(base_name) {
                        return self
                            .type_table
                            .borrow_mut()
                            .create_class_type(symbol_id, type_args);
                    }
                }
            }
        }

        // Simple class/enum name
        if let Some(symbol_id) = self.lookup_type_symbol(type_str) {
            // A TypeAlias, Abstract or Enum already OWNS a type; return it
            // instead of wrapping the symbol in a Class type. Wrapping would
            // synthesise a bogus Class { symbol_id: <non-class-symbol> }: every
            // downstream `resolve_type_to_class_symbol` then returns None and
            // method dispatch silently falls through to the wrong class's
            // same-named method (e.g., `bytes.set(...)` jumping into
            // `VecI32.set`), while a `@:coreType` abstract like SIMD4f loses its
            // vector representation and lowers as a pointer.
            let own_type = self
                .symbol_table
                .get_symbol(symbol_id)
                .filter(|s| {
                    matches!(
                        s.kind,
                        crate::tast::symbols::SymbolKind::TypeAlias
                            | crate::tast::symbols::SymbolKind::Abstract
                            | crate::tast::symbols::SymbolKind::Enum
                    )
                })
                .map(|s| s.type_id)
                .filter(|t| t.is_valid());
            if let Some(t) = own_type {
                return t;
            }
            return self
                .type_table
                .borrow_mut()
                .create_class_type(symbol_id, vec![]);
        }

        // Create a placeholder for unresolved types
        let name = self.string_interner.intern(type_str);
        self.type_table
            .borrow_mut()
            .create_type(TypeKind::Placeholder { name })
    }


    /// Parse a comma-separated list of types, handling nested generics
    pub(crate) fn parse_type_list(&mut self, types_str: &str) -> Vec<TypeId> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in types_str.chars() {
            match ch {
                '<' => {
                    depth += 1;
                    current.push(ch);
                }
                '>' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        result.push(self.parse_type_string(trimmed));
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        // Don't forget the last type
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            result.push(self.parse_type_string(trimmed));
        }

        result
    }


    /// Look up a type symbol by name (checks short name in global scope)
    pub(crate) fn lookup_type_symbol(&self, name: &str) -> Option<SymbolId> {
        // Try short name lookup in global scope first.
        let interned = self.string_interner.intern(name);
        if let Some(symbol) = self.symbol_table.lookup_symbol(ScopeId::first(), interned) {
            return Some(symbol.id);
        }

        // Dotted names (e.g. "rayzor.Bytes"): split into bare short name and
        // verify the resolved symbol's qualified_name matches. Without this,
        // BLADE-preloaded typedefs like `haxe.io.Bytes = rayzor.Bytes` resolve
        // their target to a Placeholder because the symbol is registered as
        // bare name "Bytes".
        if let Some(last_dot) = name.rfind('.') {
            let short = &name[last_dot + 1..];
            let short_interned = self.string_interner.intern(short);
            if let Some(symbol) = self
                .symbol_table
                .lookup_symbol(ScopeId::first(), short_interned)
            {
                let qname_matches = symbol
                    .qualified_name
                    .and_then(|qn| self.string_interner.get(qn))
                    .map(|qn| qn == name)
                    .unwrap_or(false);
                if qname_matches {
                    return Some(symbol.id);
                }
            }
        }

        // Packaged manifest types are published under their qualified name, so
        // a manifest signature naming one by its short name resolves through
        // the manifest index rather than the root scope.
        self.manifest_types_by_short_name.get(&interned).copied()
    }


    /// Register type system symbols from BladeTypeInfo (for cache restore).
    /// Returns a mapping of class names to their fresh IDs for map reconstruction.
    pub(crate) fn register_symbols_from_type_info(
        &mut self,
        symbols: &BladeTypeInfo,
    ) -> BTreeMap<String, (crate::tast::SymbolId, crate::tast::TypeId, ScopeId)> {
        let mut class_map = BTreeMap::new();

        // Declare every class before registering any member, so a signature
        // naming a sibling resolves to that sibling rather than to a
        // placeholder — the order they appear in the module is not a
        // statement about which may refer to which.
        let declared: Vec<DeclaredClass> = symbols
            .classes
            .iter()
            .map(|class_info| self.declare_class_from_blade(class_info))
            .collect();
        for (class_info, declaration) in symbols.classes.iter().zip(&declared) {
            self.define_class_members_from_blade(class_info, declaration);
        }

        for (class_info, declaration) in symbols.classes.iter().zip(&declared) {
            let symbol_id = declaration.symbol_id;
            let qualified_name = if class_info.package.is_empty() {
                class_info.name.clone()
            } else {
                format!("{}.{}", class_info.package.join("."), class_info.name)
            };
            // Get the type ID and scope ID we just created
            if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
                let type_id = sym.type_id;
                let scope_id = sym.scope_id;
                // Insert both qualified name (haxe.Exception) and simple name (Exception)
                // so BLADE field entries using either convention can be restored
                if !class_info.package.is_empty() {
                    class_map.insert(class_info.name.clone(), (symbol_id, type_id, scope_id));
                }
                class_map.insert(qualified_name, (symbol_id, type_id, scope_id));
            }
        }

        for enum_info in &symbols.enums {
            self.register_enum_from_blade(enum_info);
        }

        for alias_info in &symbols.type_aliases {
            self.register_type_alias_from_blade(alias_info);
        }

        for abstract_info in &symbols.abstracts {
            self.register_abstract_from_blade(abstract_info);
        }

        class_map
    }
}
