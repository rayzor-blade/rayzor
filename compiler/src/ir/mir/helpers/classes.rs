//! Class identity and inheritance: qualified names, field sets, parent chains.

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
    /// Qualified name of a class symbol, the only identity stable across
    /// compilation contexts.
    pub(crate) fn class_qualified_name(&self, symbol: SymbolId) -> Option<String> {
        self.symbol_table.get_symbol(symbol).and_then(|s| {
            s.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(s.name))
                .map(|s| s.to_string())
        })
    }

    /// Compare two class names for the owner-class property guard, tolerating
    /// qualified-vs-bare differences across compilation contexts (e.g.
    /// `haxe.io.Bytes` vs `Bytes`, `StringBuf` vs `haxe.StringBuf`). Matches
    /// on the trailing name segment.
    pub(crate) fn class_names_match(a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let a_last = a.rsplit('.').next().unwrap_or(a);
        let b_last = b.rsplit('.').next().unwrap_or(b);
        a_last == b_last
    }

    /// Recursively collect fields from parent classes.
    ///
    /// `parent_symbol` is the declaration's `extends_symbol` where known. Prefer
    /// it over anything derived from `parent_type`: `HirClass::extends` is a
    /// SymbolId punned into a TypeId, so resolving the parent through the type
    /// table can land on an unrelated class. The parent set decided here becomes
    /// the child's field indices AND its allocation size, so a wrong parent
    /// silently corrupts the heap.
    pub(crate) fn collect_inherited_fields(
        &mut self,
        parent_type: Option<TypeId>,
        parent_symbol_hint: Option<SymbolId>,
        child_type: TypeId,
        fields: &mut Vec<IrField>,
        field_index: &mut u32,
        walk: &mut Vec<SymbolId>,
    ) {
        // A declaration-named parent wins over a punned-TypeId lookup.
        if let Some(hint) = parent_symbol_hint {
            if let Some(parent_class) = self.current_hir_types.values().find_map(|d| match d {
                HirTypeDecl::Class(c) if c.symbol_id == hint => Some(c),
                _ => None,
            }) {
                self.add_parent_fields(parent_class, child_type, fields, field_index, walk);
                return;
            }
        }
        if let Some(parent_type_id) = parent_type {
            // Try direct lookup first
            if let Some(HirTypeDecl::Class(parent_class)) =
                self.current_hir_types.get(&parent_type_id)
            {
                self.add_parent_fields(parent_class, child_type, fields, field_index, walk);
            } else {
                // TypeId mismatch - the extends field might use instance type while
                // hir_module.types uses declaration type. Search by matching class type.
                let type_table_symbol = self.type_table.get(parent_type_id).and_then(|t| match &t
                    .kind
                {
                    crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                });
                if let Some(parent_symbol) = parent_symbol_hint.or(type_table_symbol).as_ref() {
                    for (decl_type_id, type_decl) in self.current_hir_types.iter() {
                        if let HirTypeDecl::Class(class) = type_decl {
                            if class.symbol_id == *parent_symbol {
                                self.add_parent_fields(
                                    class,
                                    child_type,
                                    fields,
                                    field_index,
                                    walk,
                                );
                                return;
                            }
                        }
                    }

                    // Cross-module parent class fallback:
                    // Collect parent fields from field_index_map (seeded from external modules).
                    // These already include inherited fields from grandparent classes.
                    let parent_sym = *parent_symbol;
                    let parent_name = self.class_qualified_name(parent_sym);
                    let mut parent_fields: Vec<(SymbolId, u32)> = Vec::new();
                    for (field_sym, (class_ty, idx)) in &self.field_index_map {
                        let id_match = *class_ty == parent_type_id
                            || self.class_type_to_symbol.get(class_ty) == Some(&parent_sym);
                        if !id_match {
                            continue;
                        }
                        // Both id tests above are raw-id coincidences across
                        // context-local spaces. Where the field records its
                        // declaring class, that NAME is the stable identity —
                        // use it to reject another class's fields, which would
                        // otherwise land in this object's layout and size.
                        if let (Some(owner), Some(expected)) = (
                            self.field_class_names.get(field_sym),
                            parent_name.as_deref(),
                        ) {
                            if !Self::class_names_match(owner, expected) {
                                continue;
                            }
                        }
                        parent_fields.push((*field_sym, *idx));
                    }
                    parent_fields.sort_by_key(|(_, idx)| *idx);

                    for (field_sym, _orig_idx) in &parent_fields {
                        let field_name = self
                            .symbol_table
                            .get_symbol(*field_sym)
                            .and_then(|s| self.string_interner.get(s.name))
                            .unwrap_or("<unknown>")
                            .to_string();
                        let field_ir_type = self
                            .symbol_table
                            .get_symbol(*field_sym)
                            .map(|s| self.convert_type(s.type_id))
                            .unwrap_or(IrType::I64);

                        // Re-map this field to the child class with correct index
                        self.field_index_map
                            .insert(*field_sym, (child_type, *field_index));

                        fields.push(IrField {
                            name: field_name,
                            ty: field_ir_type,
                            offset: None,
                        });

                        *field_index += 1;
                    }
                }
            }
        }
    }

    /// Collect parent instance fields for derive codegen (PartialEq/PartialOrd/Hash).
    /// Builds a simple (SymbolId, TypeId, gep_index) list by walking the parent chain.
    pub(crate) fn collect_parent_instance_fields(
        &self,
        parent_type_id: TypeId,
        fields: &mut Vec<(SymbolId, TypeId, u32)>,
        idx: &mut u32,
    ) {
        if let Some(HirTypeDecl::Class(parent_class)) = self.current_hir_types.get(&parent_type_id)
        {
            // Recurse to grandparent first
            if let Some(gp_ty) = parent_class.extends {
                self.collect_parent_instance_fields(gp_ty, fields, idx);
            }
            // Then parent's own fields
            for field in &parent_class.fields {
                if !field.is_static {
                    if let Some(ref prop) = field.property_access {
                        if !prop.has_backing_storage() {
                            continue;
                        }
                    }
                    fields.push((field.symbol_id, field.ty, *idx));
                    *idx += 1;
                }
            }
        }
    }

    /// Add parent class fields to the field list
    pub(crate) fn add_parent_fields(
        &mut self,
        parent_class: &HirClass,
        child_type: TypeId,
        fields: &mut Vec<IrField>,
        field_index: &mut u32,
        walk: &mut Vec<SymbolId>,
    ) {
        // A class cannot be its own ancestor. `extends` is followed by TypeId, and
        // TypeIds are context-local — declaring a new type shifts them — so the guard
        // keys on SymbolId, which is stable across those shifts.
        if walk.contains(&parent_class.symbol_id) {
            return;
        }
        walk.push(parent_class.symbol_id);

        // First, recursively collect grandparent fields
        self.collect_inherited_fields(
            parent_class.extends,
            parent_class.extends_symbol,
            child_type,
            fields,
            field_index,
            walk,
        );

        // Then add parent's own fields
        let parent_class_name = self
            .symbol_table
            .get_symbol(parent_class.symbol_id)
            .and_then(|sym| sym.qualified_name.and_then(|n| self.string_interner.get(n)))
            .or_else(|| self.string_interner.get(parent_class.name))
            .unwrap_or("<unknown>")
            .to_string();
        for parent_field in &parent_class.fields {
            // Skip static fields — they're globals, not instance fields
            if parent_field.is_static {
                continue;
            }

            // Skip computed property fields, which have no backing storage
            if let Some(ref prop) = parent_field.property_access {
                if !prop.has_backing_storage() {
                    continue;
                }
            }

            // Map parent field symbol to child class's type with the correct index
            self.field_index_map
                .insert(parent_field.symbol_id, (child_type, *field_index));
            self.field_class_names
                .insert(parent_field.symbol_id, parent_class_name.clone());

            fields.push(IrField {
                name: self
                    .string_interner
                    .get(parent_field.name)
                    .unwrap_or("<unknown>")
                    .to_string(),
                ty: self.convert_type(parent_field.ty),
                offset: None,
            });

            *field_index += 1;
        }
    }

    pub(crate) fn parent_chain(&self, start: SymbolId) -> Vec<SymbolId> {
        let mut chain = Vec::new();
        let mut seen: BTreeSet<SymbolId> = BTreeSet::new();
        seen.insert(start);
        let mut current = start;
        while let Some(&parent) = self.class_parent_map.get(&current) {
            if !seen.insert(parent) {
                // Unreachable while the parent symbol is recorded directly
                // (`extends_symbol`); kept because an unbounded walk must never be
                // able to hang the compiler. Warn rather than loop or exit silently.
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "\x1b[33mWarning:\x1b[0m [W0021] class parent chain from {:?} \
                         revisits {:?}; truncating the walk.",
                        start, parent
                    );
                }
                break;
            }
            chain.push(parent);
            current = parent;
        }
        chain
    }

    /// Check if `source_type` is a subclass of `target_type` by walking the extends chain.
    /// Uses SymbolId-based lookup to handle TAST/HIR TypeId mismatch.
    pub(crate) fn is_subclass_of(&self, source_type: TypeId, target_type: TypeId) -> bool {
        let target_sym = {
            let type_table = self.type_table;
            match type_table.get(target_type).map(|ti| &ti.kind) {
                Some(TypeKind::Class { symbol_id, .. }) => *symbol_id,
                _ => return false,
            }
        };

        // Get source SymbolId and walk the class hierarchy
        let source_sym = {
            let type_table = self.type_table;
            match type_table.get(source_type).map(|ti| &ti.kind) {
                Some(TypeKind::Class { symbol_id, .. }) => *symbol_id,
                _ => return false,
            }
        };

        let mut current_sym = source_sym;
        let mut visited = BTreeSet::new();
        while visited.insert(current_sym) {
            let parent_type = self.current_hir_types.values().find_map(|decl| {
                if let crate::ir::hir::HirTypeDecl::Class(class) = decl {
                    if class.symbol_id == current_sym {
                        class.extends
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            match parent_type {
                Some(parent_type_id) => {
                    // Parent SymbolId from type_table, falling back to the
                    // canonicalised `TypeId::from_raw(symbol_id.as_raw())` from
                    // `tast_to_hir.rs::extends_canonical`: that synthetic parent
                    // TypeId does not resolve through `type_table`, so without the
                    // fallback `Cat is Animal` reports false.
                    let parent_sym = {
                        let type_table = self.type_table;
                        type_table
                            .get(parent_type_id)
                            .and_then(|ti| match &ti.kind {
                                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                _ => None,
                            })
                            .or_else(|| {
                                let candidate = SymbolId::from_raw(parent_type_id.as_raw());
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
                    let Some(parent_sym) = parent_sym else {
                        return false;
                    };
                    if parent_sym == target_sym {
                        return true;
                    }
                    current_sym = parent_sym;
                }
                None => return false,
            }
        }
        false
    }

    /// Get all fields for a given TypeId. Uses cached reverse index when available,
    /// falls back to linear scan of field_index_map otherwise.
    pub(crate) fn fields_for_type(&self, type_id: TypeId) -> Vec<(SymbolId, u32)> {
        if let Some(ref cache) = self.fields_by_type_cache {
            return cache.get(&type_id).cloned().unwrap_or_default();
        }
        // Fallback: linear scan (before cache is built)
        self.field_index_map
            .iter()
            .filter(|(_, (class_ty, _))| *class_ty == type_id)
            .map(|(sym, (_, idx))| (*sym, *idx))
            .collect()
    }

    /// Check if any class in field_index_map has a field with the same name.
    pub(crate) fn field_exists_in_any_class(&self, field: SymbolId) -> bool {
        // Check by SymbolId first
        if self.field_index_map.contains_key(&field) {
            return true;
        }
        // Check by name
        let field_name = match self.symbol_table.get_symbol(field).map(|s| s.name) {
            Some(name) => name,
            None => return false,
        };
        for (sym, _) in &self.field_index_map {
            if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                if sym_info.name == field_name {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn extract_class_native_name(&self, decl: &HirTypeDecl) -> Option<String> {
        if let HirTypeDecl::Class(cls) = decl {
            for attr in &cls.metadata {
                let attr_name = self.string_interner.get(attr.name).unwrap_or("");
                if attr_name == "native" {
                    if let Some(crate::ir::hir::HirAttributeArg::Literal(
                        crate::ir::hir::HirLiteral::String(s),
                    )) = attr.args.first()
                    {
                        if let Some(native_str) = self.string_interner.get(*s) {
                            return Some(self.canonical_class_spelling(native_str));
                        }
                    }
                }
            }
            // No @:native: the declaration's own name. A simple name that
            // unambiguously identifies one registered class resolves to its
            // key; an ambiguous one is left alone, since picking among
            // classes is dispatch's decision, not a name's.
            return self.string_interner.get(cls.name).map(|name| {
                self.stdlib_mapping
                    .unique_class_key(name)
                    .map(|key| key.as_str())
                    .unwrap_or(name)
                    .to_string()
            });
        }
        None
    }

    /// A class-name spelling normalized for use as a receiver hint: a
    /// registered stdlib class resolves to the key it is registered under,
    /// anything else keeps its dotted form. Hints produced here are compared
    /// against mapping keys downstream, so producing the registered spelling
    /// at the source keeps those comparisons exact.
    pub(crate) fn canonical_class_spelling(&self, name: &str) -> String {
        let dotted = name.replace("::", ".");
        match self.stdlib_mapping.get_class_static_str(&dotted) {
            Some(key) => key.to_string(),
            None => dotted,
        }
    }

    /// Check whether a method symbol is declared on the class or one of its parents.
    /// This avoids ambiguous bare-name fallback when multiple classes share names
    /// like `toString`, especially across lazy-loaded import modules.
    pub(crate) fn symbol_belongs_to_class_hierarchy(
        &self,
        method_symbol: SymbolId,
        class_symbol: SymbolId,
    ) -> bool {
        let method_scope = match self.symbol_table.get_symbol(method_symbol) {
            Some(sym) => sym.scope_id,
            None => return false,
        };

        let mut current = Some(class_symbol);
        let mut visited = BTreeSet::new();
        while let Some(cls) = current {
            if !visited.insert(cls) {
                break;
            }
            let cls_scope = self.symbol_table.get_symbol(cls).map(|s| s.scope_id);
            if cls_scope == Some(method_scope) {
                return true;
            }
            current = self.class_parent_map.get(&cls).copied();
        }

        false
    }

    /// Build or rebuild the fields_by_type reverse index from field_index_map.
    pub(crate) fn rebuild_fields_by_type_cache(&mut self) {
        let mut cache: BTreeMap<TypeId, Vec<(SymbolId, u32)>> = BTreeMap::new();
        for (field_sym, (class_ty, idx)) in &self.field_index_map {
            cache.entry(*class_ty).or_default().push((*field_sym, *idx));
        }
        self.fields_by_type_cache = Some(cache);
    }

    pub(crate) fn func_not_other_class(&self, func_id: IrFunctionId, class_short: &str) -> bool {
        let ir_qn = self
            .builder
            .module
            .functions
            .get(&func_id)
            .and_then(|f| f.qualified_name.clone());
        let qn = ir_qn.or_else(|| self.symbol_qualified_name_for_func(func_id));
        match qn.as_deref() {
            Some(q) => {
                let parts: Vec<&str> = q.split('.').collect();
                if parts.len() >= 2 {
                    parts[parts.len() - 2] == class_short
                } else {
                    true
                }
            }
            None => true,
        }
    }

    pub(crate) fn is_class_symbol_expr(&self, expr: &HirExpr) -> bool {
        match &expr.kind {
            HirExprKind::Variable { symbol, .. } => self
                .symbol_table
                .get_symbol(*symbol)
                .map(|sym| {
                    matches!(
                        sym.kind,
                        crate::tast::symbols::SymbolKind::Class
                            | crate::tast::symbols::SymbolKind::Abstract
                            | crate::tast::symbols::SymbolKind::TypeAlias
                    )
                })
                .unwrap_or(false),
            HirExprKind::Cast { expr: inner, .. } => self.is_class_symbol_expr(inner),
            _ => false,
        }
    }

    pub(crate) fn extract_underlying_class_symbol(&self, expr: &HirExpr) -> Option<SymbolId> {
        match &expr.kind {
            HirExprKind::New { class_type, .. } => self.get_class_symbol(*class_type),
            HirExprKind::Variable { symbol, .. } => {
                let sym_type = self.symbol_table.get_symbol(*symbol)?.type_id;
                self.get_class_symbol(sym_type)
            }
            // Don't recurse through a Cast whose target is an interface: the Cast
            // handler (class→interface case) already wrapped the inner class value
            // in a fat pointer, and recursing lets the outer `lower_expression`
            // post-process wrap it a second time, producing fat_ptr{fat_ptr, fn_ptr}
            // with `this` bound to the inner fat pointer.
            HirExprKind::Cast {
                expr: inner,
                target,
                ..
            } => {
                if self.get_interface_symbol(*target).is_some() {
                    None
                } else {
                    self.extract_underlying_class_symbol(inner)
                }
            }
            HirExprKind::Block(block) => block
                .expr
                .as_ref()
                .and_then(|e| self.extract_underlying_class_symbol(e)),
            HirExprKind::If {
                then_expr,
                else_expr,
                ..
            } => self
                .extract_underlying_class_symbol(then_expr)
                .or_else(|| self.extract_underlying_class_symbol(else_expr)),
            _ => None,
        }
    }

    /// Recover the concrete class SymbolId of a call argument whose static
    /// type was promoted to an interface (so `get_class_symbol(arg.ty)`
    /// returns None) but whose value register still holds a RAW, unwrapped
    /// class object. The class name is tracked on the object's register by the
    /// `New` lowering (`register_class_hints`) and, for variables, propagated
    /// to `monomorphized_var_types`. Returns None when the class can't be
    /// recovered (then wrapping is skipped and the value is passed as-is).
    pub(crate) fn recover_arg_concrete_class(
        &self,
        arg_expr: &HirExpr,
        arg_reg: IrId,
    ) -> Option<SymbolId> {
        // Recover only when the arg's static type isn't a resolvable concrete Class:
        // the typechecker promoted it to the interface, or cross-module it arrives
        // Unknown/Placeholder. Both leave the register holding a raw class object.
        // A concrete Class (handled by the primary path) and a primitive (never
        // wrappable) must not be guessed at from the New/variable name below.
        let arg_kind_ok = {
            let tt = self.type_table;
            match tt.get(arg_expr.ty).map(|t| &t.kind) {
                // Interface-promoted, or cross-module-unresolved: recover.
                Some(TypeKind::Interface { .. })
                | Some(TypeKind::Unknown)
                | Some(TypeKind::Placeholder { .. })
                | Some(TypeKind::Dynamic) => true,
                // Concrete class or anything else: don't guess here.
                _ => false,
            }
        };
        if !arg_kind_ok {
            return None;
        }

        // 0) Direct `new C()` arg — recover the class by name from the New node
        //    (its class_type may be Unknown/Placeholder cross-module, but its
        //    class_name survives). Also chase a wrapping Cast.
        if let Some(sid) = self.recover_cast_src_class_by_name(arg_expr) {
            return Some(sid);
        }
        // 1) Register-level class hint (set on raw `new C()` object pointers).
        if let Some(name) = self.register_class_hints.get(&arg_reg).cloned() {
            if let Some(sid) = self.lookup_class_symbol_by_name(&name) {
                return Some(sid);
            }
        }
        // 2) Variable → concrete class via the symbol's mapped register hint
        //    or its monomorphized-type record.
        if let HirExprKind::Variable { symbol, .. } = &arg_expr.kind {
            if let Some(sym_reg) = self.symbol_map.get(symbol).copied() {
                if let Some(name) = self.register_class_hints.get(&sym_reg).cloned() {
                    if let Some(sid) = self.lookup_class_symbol_by_name(&name) {
                        return Some(sid);
                    }
                }
            }
            if let Some(name) = self.monomorphized_var_types.get(symbol).cloned() {
                if let Some(sid) = self.lookup_class_symbol_by_name(&name) {
                    return Some(sid);
                }
            }
        }
        None
    }

    /// Recover the source class SymbolId of a `Cast(inner → Interface)` when
    /// `inner.ty` didn't resolve to a Class in this MIR-lowering context
    /// (cross-module: the class arrives as Unknown/Placeholder). Uses the inner
    /// expression's class NAME — the `New` node's `class_name`, or a variable's
    /// resolved class — and looks it up by name in this context's symbol table.
    pub(crate) fn recover_cast_src_class_by_name(&self, inner: &HirExpr) -> Option<SymbolId> {
        // Fast path: the type already resolves to a class.
        if let Some(sid) = self.get_class_symbol(inner.ty) {
            return Some(sid);
        }
        match &inner.kind {
            HirExprKind::New {
                class_name,
                class_type,
                ..
            } => {
                // Placeholder class_type may still carry the name.
                let name = class_name
                    .and_then(|n| self.string_interner.get(n))
                    .or_else(|| {
                        self.type_table.get(*class_type).and_then(|t| {
                            if let TypeKind::Placeholder { name } = &t.kind {
                                self.string_interner.get(*name)
                            } else {
                                None
                            }
                        })
                    })?;
                self.lookup_class_symbol_by_name(name)
            }
            HirExprKind::Variable { symbol, .. } => self
                .monomorphized_var_types
                .get(symbol)
                .and_then(|name| self.lookup_class_symbol_by_name(name)),
            HirExprKind::Cast { expr: e, .. } => self.recover_cast_src_class_by_name(e),
            _ => None,
        }
    }

    /// Derive the fully-qualified class name of a `new C()` (or a wrapping
    /// Cast/variable) argument, for the name-based interface fat-ptr wrap when
    /// the class isn't resolvable in this context. Prefers the class's own
    /// qualified name if the symbol is present; otherwise qualifies the bare
    /// `new` name with THIS module's package (same-package classes share it —
    /// e.g. `new Foo()` in package `a.b` → `a.b.Foo`).
    pub(crate) fn new_arg_class_fqn(&self, arg_expr: &HirExpr) -> Option<String> {
        let bare = match &arg_expr.kind {
            HirExprKind::New { class_name, .. } => {
                class_name.and_then(|n| self.string_interner.get(n))?
            }
            HirExprKind::Cast { expr, .. } => return self.new_arg_class_fqn(expr),
            _ => return None,
        };
        // Already qualified?
        if bare.contains('.') {
            return Some(bare.to_string());
        }
        // If the class symbol is resolvable, use its qualified name.
        if let Some(sid) = self.lookup_class_symbol_by_name(bare) {
            if let Some(sym) = self.symbol_table.get_symbol(sid) {
                if let Some(qn) = sym.qualified_name.and_then(|n| self.string_interner.get(n)) {
                    return Some(qn.to_string());
                }
            }
        }
        // Qualify with the enclosing class's package (a same-package sibling class
        // shares it). Prefer the enclosing class's qualified name over
        // `current_module`, which may be truncated and yield the wrong package: that
        // produces a thunk name which does not dedupe with the real one at merge.
        let enclosing_pkg = self
            .current_class_symbol
            .and_then(|s| self.symbol_table.get_symbol(s))
            .and_then(|s| s.qualified_name.and_then(|n| self.string_interner.get(n)))
            .and_then(|qn| qn.rsplit_once('.').map(|(p, _)| p.to_string()))
            .or_else(|| {
                self.current_module
                    .as_deref()
                    .and_then(|m| m.rsplit_once('.').map(|(p, _)| p.to_string()))
            });
        match enclosing_pkg {
            Some(pkg) if !pkg.is_empty() => Some(format!("{}.{}", pkg, bare)),
            _ => Some(bare.to_string()),
        }
    }

    /// Qualified constructor key (`<class_fqn>.new`) for a cross-module `new`.
    /// Prefers the recovered class symbol's qualified name; falls back to the
    /// bare class name qualified by the current module's package.
    pub(crate) fn cross_module_constructor_fqn_key(
        &self,
        actual_symbol_id: Option<SymbolId>,
        final_class_name: Option<&str>,
    ) -> Option<String> {
        let fqn = actual_symbol_id
            .and_then(|sid| self.symbol_table.get_symbol(sid))
            .and_then(|s| {
                s.qualified_name
                    .and_then(|n| self.string_interner.get(n))
                    .map(|s| s.to_string())
            })
            .or_else(|| {
                let bare = final_class_name?;
                if bare.contains('.') {
                    return Some(bare.to_string());
                }
                let pkg = self
                    .current_class_symbol
                    .and_then(|s| self.symbol_table.get_symbol(s))
                    .and_then(|s| s.qualified_name.and_then(|n| self.string_interner.get(n)))
                    .and_then(|qn| qn.rsplit_once('.').map(|(p, _)| p.to_string()));
                match pkg {
                    Some(p) if !p.is_empty() => Some(format!("{}.{}", p, bare)),
                    _ => Some(bare.to_string()),
                }
            })?;
        Some(format!("{}.new", fqn))
    }

    /// Class key for a `new C(x)` no constructor path resolved, when the parsed
    /// declaration says `C` is a constructible class. `Some` means emit
    /// alloc + the named forward-ref stub; `None` keeps the caller's fallback.
    /// The key comes from `cross_module_constructor_fqn_key`, which also names
    /// the stub, so a "yes" promises the fixup pass has something to bind —
    /// an unmatched stub SIGILLs rather than no-opping.
    pub(crate) fn unlowered_constructor_class(
        &self,
        actual_symbol_id: Option<SymbolId>,
        final_class_name: Option<&str>,
        argc: usize,
    ) -> Option<String> {
        if crate::debug_flags::no_xmodule_ctor() {
            return None;
        }
        let index = self.static_sig_index.as_ref()?.clone();
        let key = self.cross_module_constructor_fqn_key(actual_symbol_id, final_class_name)?;
        let class_fqn = key.strip_suffix(".new")?.to_string();
        let mut index = index.borrow_mut();
        let arity = index.declared_constructor_arity(&class_fqn)?;
        // Fewer declared params than the call passes means we resolved the wrong class.
        (arity >= argc).then_some(class_fqn)
    }

    /// Best-effort class/type name for a field-access receiver type. Handles
    /// Class (qualified or bare name), Placeholder (unresolved cross-module
    /// extern, carries the name), and TypeAlias. Returns None for anonymous /
    /// primitive / genuinely-unknown receivers (the caller then skips the
    /// owner-class guard rather than guessing).
    pub(crate) fn receiver_type_class_name(&self, receiver_ty: TypeId) -> Option<String> {
        let type_table = self.type_table;
        match type_table.get(receiver_ty).map(|t| &t.kind) {
            Some(TypeKind::Class { symbol_id, .. }) => self
                .symbol_table
                .get_symbol(*symbol_id)
                .and_then(|s| {
                    s.qualified_name
                        .and_then(|n| self.string_interner.get(n))
                        .or_else(|| self.string_interner.get(s.name))
                })
                .map(|s| s.to_string()),
            Some(TypeKind::Placeholder { name }) => {
                self.string_interner.get(*name).map(|s| s.to_string())
            }
            Some(TypeKind::TypeAlias { target_type, .. }) => {
                self.receiver_type_class_name(*target_type)
            }
            _ => None,
        }
    }

    /// Class name a static factory method returns, for tagging its result
    /// register so later member access resolves even when the binding's type
    /// stayed unresolved cross-module. Returns the receiver's `length`/`get`/…
    /// owner class (bare name) for the reference classes that expose a factory.
    pub(crate) fn static_factory_return_class(
        &self,
        class_name: &str,
        method: &str,
    ) -> Option<String> {
        let bare = class_name.rsplit('.').next().unwrap_or(class_name);
        let bare = bare.strip_prefix("haxe_io_").unwrap_or(bare);
        match (bare, method) {
            ("Bytes", "ofString" | "alloc" | "ofData" | "ofHex") => Some("Bytes".to_string()),
            _ => None,
        }
    }

    /// Return the "effective" HIR type for a receiver expression — i.e.
    /// the type that should drive dispatch decisions. Currently this
    /// only differs from `expr.ty` when the receiver is a bare
    /// `Variable` whose binding picked up a cross-context iface return
    /// type that TAST couldn't resolve (see
    /// `var_concrete_overrides`). Returns `None` if no override
    /// applies, so callers can keep their existing `args[0].ty`
    /// path as the default.
    pub(crate) fn effective_receiver_type(&self, receiver: &HirExpr) -> Option<TypeId> {
        if let HirExprKind::Variable { symbol, .. } = &receiver.kind {
            self.var_concrete_overrides.get(symbol).copied()
        } else {
            None
        }
    }

    /// Set register_class_hints on a call result if the HIR return type is a class.
    /// This enables method dispatch on cross-module return values
    /// (e.g., `a.add(b).toString()` where `add` returns Point2D across packages).
    pub(crate) fn set_class_hint_for_return(&mut self, result_reg: IrId, hir_return_type: TypeId) {
        let type_info = self.type_table.get(hir_return_type);
        if let Some(ti) = type_info {
            if let TypeKind::Class { symbol_id, .. } = &ti.kind {
                if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                    let name = sym
                        .qualified_name
                        .and_then(|qn| self.string_interner.get(qn))
                        .or_else(|| self.string_interner.get(sym.name))
                        .unwrap_or("?");
                    self.register_class_hints
                        .insert(result_reg, name.to_string());
                }
            }
        }
    }
}
