//! Registers class, interface, abstract and alias metadata before any body is lowered.

use super::*;
use crate::ir::drop_analysis::{DropBehavior, DropPointAnalyzer, DropPoints};
use crate::ir::hir::*;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, EnvironmentLayout, FunctionKind,
    FunctionSignatureBuilder, IrBasicBlock, IrBlockId, IrBuilder, IrEnumVariant, IrField,
    IrFunction, IrFunctionId, IrFunctionSignature, IrGlobal, IrGlobalId, IrId, IrInstruction,
    IrLocal, IrModule, IrParameter, IrPhiNode, IrSourceLocation, IrTerminator, IrType, IrTypeDef,
    IrTypeDefId, IrTypeDefinition, IrValue, Linkage, UnaryOp,
};
use crate::stdlib::{IrTypeDescriptor, MethodSignature, StdlibMapping};
use crate::tast::symbols::SymbolFlags;
use crate::tast::{
    InternedString, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeKind,
    TypeTable,
};
use log::{debug, trace, warn};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

impl<'a> HirToMirContext<'a> {
    pub(crate) fn register_class_metadata(&mut self, type_id: TypeId, class: &HirClass) {
        if std::env::var_os("RAYZOR_E0100_DEBUG").is_some() {
            let cname = self
                .symbol_table
                .get_symbol(class.symbol_id)
                .and_then(|sym| self.string_interner.get(sym.name))
                .unwrap_or("<?>");
            eprintln!("[reg-class] {} (type_id={:?})", cname, type_id);
        }
        // Store class TypeId → SymbolId mapping (survives type_table overwrites)
        self.class_type_to_symbol.insert(type_id, class.symbol_id);

        // Register class as struct type
        let typedef_id = self.builder.module.alloc_typedef_id();

        let mut fields = Vec::new();
        // Index 0 is reserved for the object header (__type_id: i64)
        // This enables runtime type identification for downcasting, Dynamic field access, Type.getClass()
        fields.push(IrField {
            name: "__type_id".to_string(),
            ty: IrType::I64,
            offset: None,
        });
        let mut field_index = 1u32; // User fields start at index 1
        let mut own_instance_fields = Vec::new();

        let mut walk = vec![class.symbol_id];
        self.collect_inherited_fields(
            class.extends,
            class.extends_symbol,
            type_id,
            &mut fields,
            &mut field_index,
            &mut walk,
        );

        // Then add this class's own fields

        for field in &class.fields {
            // Static fields should be stored as globals, not instance fields
            if field.is_static {
                let global_id = self.builder.module.alloc_global_id();
                let field_name = self.string_interner.get(field.name).unwrap_or("<unknown>");
                let class_name = self.string_interner.get(class.name).unwrap_or("<unknown>");

                let initializer = if let Some(ref init_expr) = field.init {
                    let constant_init = self.try_evaluate_constant_init(init_expr);
                    if constant_init.is_none() {
                        // Non-constant static field initializers must run through __init__
                        // so Haxe-style `static var x = new Foo()` works without manual setup.
                        self.dynamic_globals
                            .push((field.symbol_id, init_expr.clone()));
                    }
                    constant_init
                } else {
                    None
                };

                let global_ty = self.refine_global_type_from_initializer(
                    self.convert_type(field.ty),
                    initializer.as_ref(),
                );

                let ir_global = IrGlobal {
                    id: global_id,
                    name: format!("{}.{}", class_name, field_name),
                    symbol_id: field.symbol_id,
                    ty: global_ty,
                    initializer,
                    mutable: !field.is_final,
                    linkage: Linkage::Internal,
                    alignment: None,
                    source_location: IrSourceLocation::unknown(),
                };

                self.builder.module.add_global(ir_global);
                self.global_symbol_map.insert(field.symbol_id, global_id);
                debug!(
                    "[STATIC FIELD] Registered static field {}.{} ({:?}) as global {:?}",
                    class_name, field_name, field.symbol_id, global_id
                );
                continue; // Don't add to instance fields
            }

            if let Some(ref property_info) = field.property_access {
                self.property_access_map
                    .insert(field.symbol_id, property_info.clone());

                // Properties with non-Default getters have no backing storage —
                // skip field_index_map and struct layout for them.
                // But still record the class name for BLADE cache serialization
                // so the property accessor can be restored from cache.
                if !matches!(property_info.getter, crate::tast::PropertyAccessor::Default) {
                    let class_name_str = self
                        .symbol_table
                        .get_symbol(class.symbol_id)
                        .and_then(|sym| {
                            sym.qualified_name.and_then(|n| self.string_interner.get(n))
                        })
                        .or_else(|| self.string_interner.get(class.name))
                        .unwrap_or("<unknown>");
                    self.field_class_names
                        .insert(field.symbol_id, class_name_str.to_string());
                    continue;
                }
            }

            // Store field index mapping for field access lowering (instance fields only)
            {
                let fn_str = self.string_interner.get(field.name).unwrap_or("?");
                let cn_str = self.string_interner.get(class.name).unwrap_or("?");
                if cn_str.contains("Constraint")
                    || cn_str.contains("Scale")
                    || cn_str.contains("Binary")
                {}
            }
            self.field_index_map
                .insert(field.symbol_id, (type_id, field_index));
            own_instance_fields.push((field.symbol_id, field.ty, field_index));
            // Publish (class name, field name) -> field_index to the global
            // name-keyed registry that the fallback path consumes: the same
            // field carries different SymbolIds per lowering context, so the
            // SymbolId-keyed map above misses cross-context lookups. Same
            // index space as the primary lookup.
            {
                let (qn, cn) = self
                    .symbol_table
                    .get_symbol(class.symbol_id)
                    .map(|sym| {
                        (
                            sym.qualified_name.and_then(|q| self.string_interner.get(q)),
                            self.string_interner.get(sym.name),
                        )
                    })
                    .unwrap_or((None, None));
                let fname = self.string_interner.get(field.name);
                if let (Some(cn), Some(fname)) = (cn, fname) {
                    record_class_field(qn, cn, fname, field_index);
                }
            }

            // Store qualified class name for BLADE cache serialization
            let class_name_str = self
                .symbol_table
                .get_symbol(class.symbol_id)
                .and_then(|sym| sym.qualified_name.and_then(|n| self.string_interner.get(n)))
                .or_else(|| self.string_interner.get(class.name))
                .unwrap_or("<unknown>");
            self.field_class_names
                .insert(field.symbol_id, class_name_str.to_string());

            fields.push(IrField {
                name: self
                    .string_interner
                    .get(field.name)
                    .unwrap_or("<unknown>")
                    .to_string(),
                ty: self.convert_type(field.ty),
                offset: None,
            });

            field_index += 1;
        }

        let class_runtime_id = self.deterministic_class_type_id(class.symbol_id);
        let typedef = IrTypeDef {
            id: typedef_id,
            name: self
                .string_interner
                .get(class.name)
                .unwrap_or("<unknown>")
                .to_string(),
            type_id,
            runtime_type_id: class_runtime_id,
            definition: IrTypeDefinition::Struct {
                fields,
                packed: false,
            },
            source_location: IrSourceLocation::unknown(),
            super_type_id: class.extends,
        };

        self.builder.module.add_type(typedef);

        // Record allocation size: field_index is the next available index,
        // so total slots = field_index (includes header at index 0).
        let alloc_size = (field_index as u64 * 8).max(16);
        self.class_alloc_sizes.insert(class.symbol_id, alloc_size);
        // Also store by qualified class name for cross-compilation-context lookups.
        // TypeIds are unstable across compilation units, but class names are stable.
        let class_name_for_alloc = self
            .symbol_table
            .get_symbol(class.symbol_id)
            .and_then(|sym| sym.qualified_name.and_then(|n| self.string_interner.get(n)))
            .or_else(|| self.string_interner.get(class.name))
            .unwrap_or("<unknown>");
        self.class_alloc_sizes_by_name
            .insert(class_name_for_alloc.to_string(), alloc_size);
        // Struct-init object literals need the concrete storage layout even
        // when the class has no derive metadata. This map used to be populated
        // only for derived classes, so `{ field: value }` targeting an ordinary
        // `@:structInit` class fell back to anonymous-object allocation and was
        // later read through class-field GEPs -- two incompatible layouts.
        self.class_instance_fields
            .entry(class.symbol_id)
            .or_insert(own_instance_fields);

        // Build interface vtables for each implemented interface AND their
        // transitive parent interfaces. For each interface method, find the
        // matching class method by name.
        //
        // Collect all interface symbols (direct + transitive parents)
        let mut all_iface_symbols: Vec<SymbolId> = Vec::new();
        for &iface_type_id in &class.implements {
            let iface_symbol = {
                let type_table = self.type_table;
                type_table.get(iface_type_id).and_then(|t| {
                    if let TypeKind::Interface { symbol_id, .. } = &t.kind {
                        Some(*symbol_id)
                    } else {
                        None
                    }
                })
            };
            if let Some(iface_sym) = iface_symbol {
                if !all_iface_symbols.contains(&iface_sym) {
                    all_iface_symbols.push(iface_sym);
                }
                // Recursively add parent interfaces
                let mut stack = vec![iface_sym];
                while let Some(current) = stack.pop() {
                    if let Some(parents) = self.interface_extends.get(&current).cloned() {
                        for parent in parents {
                            if !all_iface_symbols.contains(&parent) {
                                all_iface_symbols.push(parent);
                                stack.push(parent);
                            }
                        }
                    }
                }
            }
        }

        // Build vtable for each interface (direct and inherited)
        for iface_sym in all_iface_symbols {
            if let Some(method_names) = self.resolve_interface_method_names(iface_sym) {
                let mut vtable_entries = Vec::new();
                for iface_method_name in &method_names {
                    let class_method_sym = class
                        .methods
                        .iter()
                        .find(|m| m.function.name == *iface_method_name)
                        .map(|m| m.function.symbol_id);

                    if let Some(method_sym) = class_method_sym {
                        vtable_entries.push(method_sym);
                    }
                }
                self.interface_vtables
                    .insert((class.symbol_id, iface_sym), vtable_entries);
            }
        }

        // Inherit interface vtables from parent class.
        // If a parent implements IDrawable, Button (which extends Widget) also implements IDrawable.
        // Build vtables for inherited interfaces using the child's methods (which may override).
        if let Some(extends_type_id) = class.extends {
            let parent_symbol = {
                let type_table = self.type_table;
                type_table
                    .get(extends_type_id)
                    .and_then(|t| {
                        if let TypeKind::Class { symbol_id, .. } = &t.kind {
                            Some(*symbol_id)
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        // Fallback for a canonicalised parent TypeId
                        // (`TypeId::from_raw(parent_sym.as_raw())` produced in
                        // `tast_to_hir.rs::extends_canonical`); without it the
                        // child inherits none of the parent's vtables.
                        let candidate = SymbolId::from_raw(extends_type_id.as_raw());
                        self.symbol_table.get_symbol(candidate).and_then(|sym| {
                            use crate::tast::SymbolKind;
                            if matches!(sym.kind, SymbolKind::Class) {
                                Some(candidate)
                            } else {
                                None
                            }
                        })
                    })
            };
            if let Some(parent_sym) = parent_symbol {
                let parent_iface_keys: Vec<(SymbolId, SymbolId)> = self
                    .interface_vtables
                    .keys()
                    .filter(|(cs, _)| *cs == parent_sym)
                    .cloned()
                    .collect();
                for (_, iface_sym) in parent_iface_keys {
                    if self
                        .interface_vtables
                        .contains_key(&(class.symbol_id, iface_sym))
                    {
                        continue;
                    }
                    // Build vtable using child's methods (with overrides) or parent's vtable entry
                    if let Some(method_names) = self.interface_method_names.get(&iface_sym).cloned()
                    {
                        let parent_vtable = self
                            .interface_vtables
                            .get(&(parent_sym, iface_sym))
                            .cloned()
                            .unwrap_or_default();
                        let mut vtable_entries = Vec::new();
                        for (i, iface_method_name) in method_names.iter().enumerate() {
                            // Prefer child's own method (override)
                            let child_method = class
                                .methods
                                .iter()
                                .find(|m| m.function.name == *iface_method_name)
                                .map(|m| m.function.symbol_id);
                            if let Some(sym) = child_method {
                                vtable_entries.push(sym);
                            } else if i < parent_vtable.len() {
                                vtable_entries.push(parent_vtable[i]);
                            }
                        }
                        self.interface_vtables
                            .insert((class.symbol_id, iface_sym), vtable_entries);
                    }
                }
            }
        }

        // Register class method symbols for iterator protocol lookup
        for method in &class.methods {
            self.class_method_symbols.insert(
                (class.symbol_id, method.function.name),
                method.function.symbol_id,
            );
        }

        // Record method names and overrides for virtual dispatch
        for method in &class.methods {
            self.class_method_by_name.insert(
                (class.symbol_id, method.function.name),
                method.function.symbol_id,
            );
            if method.is_override {
                self.override_methods
                    .insert((class.symbol_id, method.function.name));
            }
            if method.is_abstract {
                self.abstract_methods
                    .insert((class.symbol_id, method.function.name));
            }
        }

        let parent_symbol = class.extends_symbol.or_else(|| {
            // HIR not produced by tast_to_hir (older/bundled modules): fall back
            // to the type_table lookup ONLY. No raw-bit reinterpretation.
            class.extends.and_then(|tid| {
                self.type_table.get(tid).and_then(|t| match &t.kind {
                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                })
            })
        });
        if let Some(parent_sym) = parent_symbol {
            // A class is never its own parent; recording that cycles the
            // parent-chain walks.
            if parent_sym != class.symbol_id {
                self.class_parent_map.insert(class.symbol_id, parent_sym);
            }
        }

        // Pre-scan class metadata for `@:move` / `@:shared` so the Clone
        // derive branch below can pick the right Tier B extern (deep-copy
        // vs atomic-refcount) and so the move-tracking branch later in
        // this block can suppress `MarkMoved` / `CheckLive` emission for
        // `@:shared` classes.
        let mut has_move_attr = false;
        let mut has_shared_attr = false;
        // RAYZOR_DEBUG_MOVECLASS lists every class this context lowers. A
        // class missing from the list was never lowered here, which is a
        // different problem from an annotation that was read and ignored.
        if std::env::var_os("RAYZOR_DEBUG_MOVECLASS").is_some() {
            eprintln!(
                "[moveclass] lowering {:?} metadata={:?}",
                self.string_interner.get(class.name),
                class
                    .metadata
                    .iter()
                    .filter_map(|a| self.string_interner.get(a.name))
                    .collect::<Vec<_>>()
            );
        }
        for attr in &class.metadata {
            if let Some(attr_name) = self.string_interner.get(attr.name) {
                match attr_name {
                    "move" => has_move_attr = true,
                    "shared" => has_shared_attr = true,
                    _ => {}
                }
            }
        }

        // `@:move` and `@:shared` express opposite ownership models
        // (compile-time linear move vs runtime atomic sharing) and must never
        // co-occur on the same class. Compilation continues after the warning
        // with `@:shared` winning: move-tracking suppressed, arc_clone used.
        if has_move_attr && has_shared_attr {
            let span = diagnostics::SourceSpan::new(
                diagnostics::SourcePosition::new(0, 0, 0),
                diagnostics::SourcePosition::new(0, 1, 1),
                diagnostics::FileId::new(0),
            );
            let class_name = self.string_interner.get(class.name).unwrap_or("?");
            let diagnostic = diagnostics::DiagnosticBuilder::warning(
                format!(
                    "class `{}` carries both `@:move` and `@:shared` — these are mutually exclusive (move = compile-time linear ownership, shared = runtime atomic refcount). `@:shared` takes precedence; remove `@:move` to silence this warning.",
                    class_name
                ),
                span.clone(),
            )
            .code("W0030")
            .label(span, "conflicting memory annotations")
            .help("remove one of `@:move` or `@:shared` from the class declaration")
            .build();
            self.diagnostics.push(diagnostic);
        }

        // Populate derive trait sets from HirClass metadata
        {
            use crate::tast::DerivedTrait;
            for trait_ in &class.derived_traits {
                match trait_ {
                    DerivedTrait::PartialEq | DerivedTrait::Eq => {
                        self.derive_partial_eq_classes.insert(class.symbol_id);
                    }
                    DerivedTrait::PartialOrd | DerivedTrait::Ord => {
                        self.derive_partial_ord_classes.insert(class.symbol_id);
                    }
                    DerivedTrait::Hash => {
                        self.derive_hash_classes.insert(class.symbol_id);
                    }
                    DerivedTrait::Clone => {
                        self.derive_clone_classes.insert(class.symbol_id);
                        // Tier B: extern classes route to a named runtime fn,
                        // keyed on the fully-qualified name from the class-level
                        // `@:native("rayzor::ds::Tensor")` metadata (`::`
                        // normalised to `.`). Keying on the bare `class.name`
                        // would also catch a same-named extern class in another
                        // package and route its clone to the wrong symbol.
                        if class.is_extern {
                            let mut fqn: Option<String> = None;
                            for attr in &class.metadata {
                                let attr_name = self.string_interner.get(attr.name).unwrap_or("");
                                if attr_name != "native" {
                                    continue;
                                }
                                if let Some(first_arg) = attr.args.first() {
                                    if let crate::ir::hir::HirAttributeArg::Literal(
                                        crate::ir::hir::HirLiteral::String(s),
                                    ) = first_arg
                                    {
                                        if let Some(raw) = self.string_interner.get(*s) {
                                            fqn = Some(raw.replace("::", "."));
                                            break;
                                        }
                                    }
                                }
                            }
                            // The runtime clone extern name follows a convention,
                            // so any `@:native("rayzor::ds::X")` extern class
                            // with `@:derive([Clone])` works without a change
                            // here: take the last FQN component, lowercase it,
                            // prepend `rayzor_` and append `_clone` (or
                            // `_arc_clone` when `@:shared` is also set), e.g.
                            // `rayzor.ds.Tensor` → `rayzor_tensor_clone`. A name
                            // with no matching runtime symbol surfaces as a link
                            // error at call time.
                            let derived = fqn.as_deref().and_then(|s| {
                                s.rsplit('.').next().map(|tail| {
                                    let lower = tail.to_ascii_lowercase();
                                    if has_shared_attr {
                                        format!("rayzor_{}_arc_clone", lower)
                                    } else {
                                        format!("rayzor_{}_clone", lower)
                                    }
                                })
                            });
                            if let Some(name) = derived {
                                let leaked: &'static str = Box::leak(name.into_boxed_str());
                                self.derive_clone_extern_fns.insert(class.symbol_id, leaked);
                            }
                        }
                    }
                    DerivedTrait::Copy => {
                        self.derive_copy_classes.insert(class.symbol_id);
                    }
                    DerivedTrait::Drop => {
                        self.derive_drop_classes.insert(class.symbol_id);
                    }
                    DerivedTrait::Debug => {
                        self.derive_debug_classes.insert(class.symbol_id);
                        // Check for @:debugFormat("pattern") on the class
                        if let Some(ref fmt) = class.debug_format {
                            self.debug_format_strings
                                .insert(class.symbol_id, fmt.clone());
                        }
                    }
                    DerivedTrait::Default => {
                        self.derive_default_classes.insert(class.symbol_id);
                    }
                    _ => {}
                }
            }

            // `@:move`-annotated classes drive use-after-move diagnostics: MIR
            // lowering emits `MarkMoved` / `CheckLive` against their bindings.
            // `@:shared` wins when both are present — an Arc refcount makes
            // aliasing safe by construction and `.clone()` returns the SAME
            // pointer, so move enforcement would be spurious and incorrect.
            if has_move_attr && !has_shared_attr {
                self.derive_move_classes.insert(class.symbol_id);
            }
            if has_shared_attr {
                self.derive_shared_classes.insert(class.symbol_id);
            }
            // Publish under the class's name too, so an importer — which lowers
            // its own classes only and never sees this one's metadata — still
            // enforces the annotation.
            if has_move_attr || has_shared_attr {
                let (qn, cn) = self
                    .symbol_table
                    .get_symbol(class.symbol_id)
                    .map(|sym| {
                        (
                            sym.qualified_name.and_then(|q| self.string_interner.get(q)),
                            self.string_interner.get(sym.name),
                        )
                    })
                    .unwrap_or((None, None));
                if let Some(cn) = cn {
                    record_move_class(qn, cn, has_move_attr && !has_shared_attr);
                }
            }

            // Check for @:manualDrop metadata
            for attr in &class.metadata {
                if let Some(attr_name) = self.string_interner.get(attr.name) {
                    if attr_name == "manualDrop" {
                        self.manual_drop_classes.insert(class.symbol_id);
                    }
                }
            }

            // Build instance field list for derive codegen (non-static fields only)
            let needs_field_list = self.derive_partial_eq_classes.contains(&class.symbol_id)
                || self.derive_partial_ord_classes.contains(&class.symbol_id)
                || self.derive_hash_classes.contains(&class.symbol_id)
                || self.derive_clone_classes.contains(&class.symbol_id)
                || self.derive_copy_classes.contains(&class.symbol_id)
                || self.derive_debug_classes.contains(&class.symbol_id)
                || self.derive_default_classes.contains(&class.symbol_id);

            if needs_field_list {
                let mut instance_fields = Vec::new();
                // field_index starts at 1 (0 = type_id header)
                let mut idx = 1u32;
                // Include inherited fields (walk parent chain)
                if let Some(extends_ty) = class.extends {
                    self.collect_parent_instance_fields(extends_ty, &mut instance_fields, &mut idx);
                }
                // Then this class's own fields
                for field in &class.fields {
                    if !field.is_static {
                        if let Some(ref prop) = field.property_access {
                            if !matches!(prop.getter, crate::tast::PropertyAccessor::Default) {
                                continue; // Skip computed properties
                            }
                        }
                        instance_fields.push((field.symbol_id, field.ty, idx));
                        idx += 1;

                        // Store field default expression: @:default(value) takes priority, then field initializer
                        if let Some(ref default_expr) = field.metadata_default {
                            self.field_default_exprs
                                .insert(field.symbol_id, default_expr.clone());
                        } else if let Some(ref init_expr) = field.init {
                            self.field_default_exprs
                                .insert(field.symbol_id, init_expr.clone());
                        }
                    }
                }
                self.class_instance_fields
                    .insert(class.symbol_id, instance_fields);
            }
        }
    }

    pub(crate) fn register_interface_metadata(
        &mut self,
        type_id: TypeId,
        interface: &HirInterface,
    ) {
        // Interfaces are represented as method tables
        let typedef_id = self.builder.module.alloc_typedef_id();

        // Collect parent interface SymbolIds and their methods
        let mut parent_symbols = Vec::new();
        let mut all_method_names: Vec<InternedString> = Vec::new();

        for &parent_type_id in &interface.extends {
            // Recover the parent's SymbolId via `get_interface_symbol` (which
            // name-matches when the extends clause lands as a cross-module
            // Placeholder) and pull its methods through the drift-tolerant
            // resolver. A raw type_table lookup drops the base methods when the
            // parent type or SymbolId drifted across modules, which shifts every
            // dispatch slot — the fat-pointer builder and call sites index this
            // list by position.
            if let Some(psym) = self.get_interface_symbol(parent_type_id) {
                parent_symbols.push(psym);
                if let Some(parent_methods) = self.resolve_interface_method_names(psym) {
                    for m in parent_methods {
                        if !all_method_names.contains(&m) {
                            all_method_names.push(m);
                        }
                    }
                }
            }
        }

        // Add own methods after inherited ones, and store return types
        for method in &interface.methods {
            if !all_method_names.contains(&method.name) {
                all_method_names.push(method.name);
            }
            self.interface_method_return_types
                .insert((interface.symbol_id, method.name), method.return_type);
        }

        // Store extends relationships for transitive vtable building
        self.interface_extends
            .insert(interface.symbol_id, parent_symbols);

        let fields: Vec<IrField> = all_method_names
            .iter()
            .map(|name| IrField {
                name: self
                    .string_interner
                    .get(*name)
                    .unwrap_or("<unknown>")
                    .to_string(),
                ty: IrType::Ptr(Box::new(IrType::Function {
                    params: vec![IrType::Any],
                    return_type: Box::new(IrType::Any),
                    varargs: false,
                })),
                offset: None,
            })
            .collect();

        let interface_runtime_id =
            self.deterministic_iface_or_enum_type_id(interface.symbol_id, "iface");
        let typedef = IrTypeDef {
            id: typedef_id,
            name: self
                .string_interner
                .get(interface.name)
                .unwrap_or("<unknown>")
                .to_string(),
            type_id,
            runtime_type_id: interface_runtime_id,
            definition: IrTypeDefinition::Struct {
                fields,
                packed: false,
            },
            source_location: IrSourceLocation::unknown(),
            super_type_id: None,
        };

        self.builder.module.add_type(typedef);

        // Store method ordering for vtable construction (includes inherited methods)
        self.interface_method_names
            .insert(interface.symbol_id, all_method_names);
    }

    pub(crate) fn register_abstract_metadata(
        &mut self,
        type_id: TypeId,
        abstract_decl: &HirAbstract,
    ) {
        // Abstract types are type aliases with additional constraints
        let typedef_id = self.builder.module.alloc_typedef_id();

        let typedef = IrTypeDef {
            id: typedef_id,
            name: self
                .string_interner
                .get(abstract_decl.name)
                .unwrap_or("<unknown>")
                .to_string(),
            type_id,
            runtime_type_id: None,
            definition: IrTypeDefinition::Alias {
                aliased_type: IrType::Any, // TODO: Get underlying type
            },
            source_location: IrSourceLocation::unknown(),
            super_type_id: None,
        };

        self.builder.module.add_type(typedef);

        // Store @:from/@:to conversion rules keyed by abstract qualified_name.
        // We use qualified_name (or native_name for extern abstracts, or plain name as fallback)
        // because HIR and type_table have different SymbolIds/TypeIds for the same abstract.
        // The name is the only stable identifier shared between both.
        let rule_key = self
            .symbol_table
            .get_symbol(abstract_decl.symbol_id)
            .and_then(|sym| sym.qualified_name.or(sym.native_name).or(Some(sym.name)))
            .unwrap_or(abstract_decl.name);

        if !abstract_decl.from_rules.is_empty() {
            self.abstract_from_rules
                .insert(rule_key, abstract_decl.from_rules.clone());
        }
        if !abstract_decl.to_rules.is_empty() {
            self.abstract_to_rules
                .insert(rule_key, abstract_decl.to_rules.clone());
        }

        // Store @:forward rules if present
        if !abstract_decl.forward_fields.is_empty()
            || self
                .symbol_table
                .get_symbol(abstract_decl.symbol_id)
                .map_or(false, |s| s.flags.is_forward())
        {
            self.abstract_forward_rules.insert(
                abstract_decl.symbol_id,
                (
                    abstract_decl.underlying,
                    abstract_decl.forward_fields.clone(),
                ),
            );
        }
    }

    pub(crate) fn register_alias_metadata(&mut self, type_id: TypeId, alias: &HirTypeAlias) {
        // An alias to an anonymous struct registers that struct's fields in
        // typedef_field_map so field access works on the alias (e.g. FileStat).
        let type_table = self.type_table;
        if let Some(aliased_info) = type_table.get(alias.aliased_type) {
            if let TypeKind::Anonymous { fields } = &aliased_info.kind {
                // All fields are 8 bytes for consistent boxing/sizing.
                for (index, field) in fields.iter().enumerate() {
                    // Keyed by (typedef_type_id, field_name) so lookup still
                    // works when field access creates new symbols.
                    self.typedef_field_map
                        .insert((type_id, field.name), index as u32);

                    // Also try to register any existing symbols with this name
                    let field_symbol = self
                        .symbol_table
                        .symbols_of_kind(crate::tast::symbols::SymbolKind::Field)
                        .into_iter()
                        .find(|s| s.name == field.name)
                        .map(|s| s.id);

                    if let Some(field_sym_id) = field_symbol {
                        self.field_index_map
                            .insert(field_sym_id, (type_id, index as u32));
                    }
                }

                // Also create an IrTypeDef with struct fields for proper layout info
                let typedef_id = self.builder.module.alloc_typedef_id();

                let ir_fields: Vec<IrField> = fields
                    .iter()
                    .enumerate()
                    .map(|(idx, f)| {
                        let field_name = self
                            .string_interner
                            .get(f.name)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("field_{}", idx));
                        IrField {
                            name: field_name,
                            ty: self.convert_type(f.type_id),
                            offset: Some((idx * 8) as u32), // 8 bytes per field
                        }
                    })
                    .collect();

                let typedef = IrTypeDef {
                    id: typedef_id,
                    name: self
                        .string_interner
                        .get(alias.name)
                        .unwrap_or("<unknown>")
                        .to_string(),
                    type_id,
                    runtime_type_id: None,
                    definition: IrTypeDefinition::Struct {
                        fields: ir_fields,
                        packed: false,
                    },
                    source_location: IrSourceLocation::unknown(),
                    super_type_id: None,
                };

                self.builder.module.add_type(typedef);
                return;
            }
        }

        // Not an anonymous struct - just register as simple alias
        let typedef_id = self.builder.module.alloc_typedef_id();

        let typedef = IrTypeDef {
            id: typedef_id,
            name: self
                .string_interner
                .get(alias.name)
                .unwrap_or("<unknown>")
                .to_string(),
            type_id,
            runtime_type_id: None,
            definition: IrTypeDefinition::Alias {
                aliased_type: IrType::Any, // TODO: Convert aliased TypeId to IrType
            },
            source_location: IrSourceLocation::unknown(),
            super_type_id: None,
        };

        self.builder.module.add_type(typedef);
    }
}
