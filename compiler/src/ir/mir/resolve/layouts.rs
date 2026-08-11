//! Computes and caches C-struct and GPU-struct field layouts.

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
    /// Get or compute the CStruct layout for a class type
    pub(crate) fn get_or_compute_cstruct_layout(
        &mut self,
        type_id: TypeId,
    ) -> Option<CStructLayout> {
        if let Some(layout) = self.cstruct_layouts.get(&type_id) {
            return Some(layout.clone());
        }

        // Look up class info from type table
        let (symbol_id, class_name, no_mangle) = {
            let type_table = self.type_table;
            let type_info = type_table.get(type_id)?;
            let symbol_id = type_info.symbol_id()?;
            let sym = self.symbol_table.get_symbol(symbol_id)?;
            let name = self.string_interner.get(sym.name)?.to_string();
            let no_mangle = sym.flags.is_no_mangle();
            (symbol_id, name, no_mangle)
        };

        // Get fields from field_index_map for this type (O(1) via reverse index).
        let mut fields_with_index = self.fields_for_type(type_id);
        // Fallback: if no fields found by type_id, match fields by name.
        // TypeIds can differ between expression types and registration types,
        // so we find fields whose name matches our class's field names.
        if fields_with_index.is_empty() {
            // Get the class name to match against field_index_map entries
            let class_name_interned = self.symbol_table.get_symbol(symbol_id).map(|s| s.name);

            // Find all unique class type_ids in field_index_map
            let field_class_tys: std::collections::BTreeSet<TypeId> =
                self.field_index_map.values().map(|(ty, _)| *ty).collect();

            // For each class type_id, check if its fields belong to our class
            // by checking field names against what the class should have
            for candidate_ty in &field_class_tys {
                // Check if this candidate type has the same class name/symbol
                let is_our_class = self.current_hir_types.iter().any(|(tid, decl)| {
                    if *tid != *candidate_ty {
                        return false;
                    }
                    if let HirTypeDecl::Class(c) = decl {
                        c.symbol_id == symbol_id
                    } else {
                        false
                    }
                });
                if is_our_class {
                    for (field_sym, (class_ty, idx)) in &self.field_index_map {
                        if class_ty == candidate_ty {
                            fields_with_index.push((*field_sym, *idx));
                        }
                    }
                    break;
                }
            }
        }
        fields_with_index.sort_by_key(|(_, idx)| *idx);

        let c_name = if no_mangle {
            class_name.clone()
        } else {
            // Simple mangling: package_ClassName
            class_name.replace('.', "_")
        };

        let mut byte_offset: u32 = 0;
        let mut max_align: u32 = 1;
        let mut layout_fields = Vec::new();

        let mut dep_cdefs: Vec<String> = Vec::new();

        for (field_sym_id, _idx) in &fields_with_index {
            let sym = self.symbol_table.get_symbol(*field_sym_id)?;
            let field_name = self.string_interner.get(sym.name)?.to_string();
            let ir_type = self.convert_type(sym.type_id);

            // Resolve field type to determine C type mapping.
            // Check the original Haxe type (before IR conversion) for rich type info.
            let (size, align, c_type, ir_type) =
                self.resolve_cstruct_field_type(sym.type_id, ir_type, &mut dep_cdefs);

            // Align offset
            if align > 0 {
                byte_offset = (byte_offset + align - 1) & !(align - 1);
            }
            if align > max_align {
                max_align = align;
            }

            layout_fields.push(CStructFieldLayout {
                symbol_id: *field_sym_id,
                name: field_name,
                byte_offset,
                ir_type,
                c_type,
            });

            byte_offset += size;
        }

        // Pad total size to alignment
        if max_align > 0 {
            byte_offset = (byte_offset + max_align - 1) & !(max_align - 1);
        }

        // Build cdef string — own typedef only
        let mut own_cdef = format!("typedef struct {{ ");
        for f in &layout_fields {
            own_cdef.push_str(&format!("{} {}; ", f.c_type, f.name));
        }
        own_cdef.push_str(&format!("}} {};\n", c_name));

        // Full cdef includes dependency typedefs first, then own typedef
        let mut full_cdef = String::new();
        for dep in &dep_cdefs {
            full_cdef.push_str(dep);
        }
        full_cdef.push_str(&own_cdef);

        let layout = CStructLayout {
            fields: layout_fields,
            total_size: byte_offset,
            alignment: max_align,
            c_name,
            dep_cdefs,
            own_cdef,
            cdef_string: full_cdef,
        };

        self.cstruct_layouts.insert(type_id, layout.clone());
        Some(layout)
    }

    /// Resolve a field's Haxe type to C type info for @:cstruct layout.
    ///
    /// Returns (size_bytes, alignment, c_type_string, ir_type).
    /// For nested @:cstruct types, recursively computes their layout and adds
    /// dependency typedefs to `dep_cdefs`.
    pub(crate) fn resolve_cstruct_field_type(
        &mut self,
        haxe_type_id: TypeId,
        ir_type: IrType,
        dep_cdefs: &mut Vec<String>,
    ) -> (u32, u32, String, IrType) {
        // Check the Haxe-level type for rich info (Ptr, Usize, nested cstruct, etc.)
        let type_info = {
            let type_table = self.type_table;
            type_table.get(haxe_type_id).map(|t| (t.kind.clone(), t.id))
        };

        if let Some((kind, _tid)) = type_info {
            match &kind {
                // Abstract types: Ptr<T>, Ref<T>, Box<T>, Usize, CString
                TypeKind::Abstract {
                    symbol_id,
                    type_args,
                    ..
                } => {
                    let abstract_name = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|s| self.string_interner.get(s.name))
                        .map(|s| s.to_string())
                        .unwrap_or_default();

                    match abstract_name.as_str() {
                        "Ptr" => {
                            // Ptr<T> → T* if T is @:cstruct, else void*
                            let c_type = if let Some(inner_tid) = type_args.first() {
                                self.cstruct_ptr_c_type(*inner_tid, dep_cdefs)
                            } else {
                                "void*".to_string()
                            };
                            return (8, 8, c_type, IrType::I64);
                        }
                        "Ref" => {
                            // Ref<T> → const T* if T is @:cstruct, else const void*
                            let c_type = if let Some(inner_tid) = type_args.first() {
                                let inner = self.cstruct_ptr_c_type(*inner_tid, dep_cdefs);
                                if inner.ends_with('*') {
                                    format!("const {}", inner)
                                } else {
                                    format!("const {}*", inner)
                                }
                            } else {
                                "const void*".to_string()
                            };
                            return (8, 8, c_type, IrType::I64);
                        }
                        "Box" => {
                            // Box<T> → T* (owned pointer)
                            let c_type = if let Some(inner_tid) = type_args.first() {
                                self.cstruct_ptr_c_type(*inner_tid, dep_cdefs)
                            } else {
                                "void*".to_string()
                            };
                            return (8, 8, c_type, IrType::I64);
                        }
                        "Usize" => {
                            return (8, 8, "size_t".to_string(), IrType::I64);
                        }
                        "CString" => {
                            return (8, 8, "char*".to_string(), IrType::I64);
                        }
                        _ => {} // Fall through to IR-based matching
                    }
                }
                // Class types: check for CString or nested @:cstruct
                TypeKind::Class { symbol_id, .. } => {
                    // Check if this is CString (extern abstract stored as Class)
                    if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                        let is_cstring = sym
                            .qualified_name
                            .and_then(|n| self.string_interner.get(n))
                            .map(|s| s == "rayzor.CString")
                            .unwrap_or(false);
                        if is_cstring {
                            return (8, 8, "char*".to_string(), IrType::I64);
                        }
                    }

                    let is_nested_cstruct = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .map(|s| s.flags.is_cstruct())
                        .unwrap_or(false);
                    if is_nested_cstruct {
                        // Recursively compute nested layout
                        if let Some(nested_layout) =
                            self.get_or_compute_cstruct_layout(haxe_type_id)
                        {
                            // Add nested cdef as dependency (if not already present)
                            if !dep_cdefs.contains(&nested_layout.cdef_string) {
                                // Add nested deps first (transitive)
                                for dep in &nested_layout.dep_cdefs {
                                    if !dep_cdefs.contains(dep) {
                                        dep_cdefs.push(dep.clone());
                                    }
                                }
                                // Then add nested struct's own typedef
                                let own_cdef = {
                                    let mut s = format!("typedef struct {{ ");
                                    for f in &nested_layout.fields {
                                        s.push_str(&format!("{} {}; ", f.c_type, f.name));
                                    }
                                    s.push_str(&format!("}} {};\n", nested_layout.c_name));
                                    s
                                };
                                if !dep_cdefs.contains(&own_cdef) {
                                    dep_cdefs.push(own_cdef);
                                }
                            }
                            let c_type = nested_layout.c_name.clone();
                            let size = nested_layout.total_size;
                            let align = nested_layout.alignment;
                            // Nested cstruct is embedded inline — IR type is a byte array
                            return (size, align, c_type, IrType::I64);
                        }
                    }
                }
                _ => {}
            }
        }

        // Fallback: use IR type for primitives
        match &ir_type {
            IrType::F64 => (8, 8, "double".to_string(), ir_type),
            IrType::I64 => (8, 8, "long".to_string(), ir_type),
            IrType::I32 => (8, 8, "long".to_string(), IrType::I64), // Promote to i64
            IrType::Bool => (4, 4, "int".to_string(), ir_type),
            _ => (8, 8, "long".to_string(), ir_type),
        }
    }

    /// Get or compute the GPU struct layout for a @:gpuStruct class type.
    ///
    /// GPU structs use 4-byte floats (not 8-byte like C doubles), 4-byte ints,
    /// and 4-byte bools. This matches Metal/CUDA/WGSL struct layout conventions.
    pub(crate) fn get_or_compute_gpu_struct_layout(
        &mut self,
        type_id: TypeId,
    ) -> Option<GpuStructLayout> {
        if let Some(layout) = self.gpu_struct_layouts.get(&type_id) {
            return Some(layout.clone());
        }

        // Look up class info from type table
        let (symbol_id, class_name) = {
            let type_table = self.type_table;
            let type_info = type_table.get(type_id)?;
            let symbol_id = type_info.symbol_id()?;
            let sym = self.symbol_table.get_symbol(symbol_id)?;
            let name = self.string_interner.get(sym.name)?.to_string();
            (symbol_id, name)
        };

        // Get fields from reverse index (O(1) via cache)
        let mut fields_with_index = self.fields_for_type(type_id);
        // Fallback: match by class symbol_id via HirTypeDecl
        if fields_with_index.is_empty() {
            let field_class_tys: std::collections::BTreeSet<TypeId> =
                self.field_index_map.values().map(|(ty, _)| *ty).collect();
            for candidate_ty in &field_class_tys {
                let is_our_class = self.current_hir_types.iter().any(|(tid, decl)| {
                    if *tid != *candidate_ty {
                        return false;
                    }
                    if let HirTypeDecl::Class(c) = decl {
                        c.symbol_id == symbol_id
                    } else {
                        false
                    }
                });
                if is_our_class {
                    for (field_sym, (class_ty, idx)) in &self.field_index_map {
                        if class_ty == candidate_ty {
                            fields_with_index.push((*field_sym, *idx));
                        }
                    }
                    break;
                }
            }
        }
        fields_with_index.sort_by_key(|(_, idx)| *idx);

        // Use simple class name (last segment) for MSL struct name
        let msl_name = class_name
            .split('.')
            .last()
            .unwrap_or(&class_name)
            .to_string();

        let mut byte_offset: u32 = 0;
        let mut max_align: u32 = 1;
        let mut layout_fields = Vec::new();
        let mut dep_typedefs: Vec<String> = Vec::new();

        for (field_sym_id, _idx) in &fields_with_index {
            let sym = self.symbol_table.get_symbol(*field_sym_id)?;
            let field_name = self.string_interner.get(sym.name)?.to_string();
            let ir_type = self.convert_type(sym.type_id);

            let (size, align, msl_type, gpu_ir_type) =
                self.resolve_gpu_struct_field_type(sym.type_id, ir_type, &mut dep_typedefs);

            // Align offset
            if align > 0 {
                byte_offset = (byte_offset + align - 1) & !(align - 1);
            }
            if align > max_align {
                max_align = align;
            }

            // Map GPU type to WGSL vertex format code
            let vertex_format = match (&gpu_ir_type, size) {
                (IrType::F32, 4) => 0,  // Float32
                (IrType::F32, 8) => 1,  // Float32x2
                (IrType::F32, 12) => 2, // Float32x3
                (IrType::F32, 16) => 3, // Float32x4
                (IrType::I32, _) => 4,  // Sint32
                _ => 0,                 // default Float32
            };

            layout_fields.push(GpuStructFieldLayout {
                symbol_id: *field_sym_id,
                name: field_name,
                byte_offset,
                ir_type: gpu_ir_type,
                msl_type,
                size,
                vertex_format,
            });

            byte_offset += size;
        }

        // Pad total size to alignment
        if max_align > 0 {
            byte_offset = (byte_offset + max_align - 1) & !(max_align - 1);
        }

        // Build MSL typedef string
        let mut msl_typedef = format!("struct {} {{ ", msl_name);
        for f in &layout_fields {
            msl_typedef.push_str(&format!("{} {}; ", f.msl_type, f.name));
        }
        msl_typedef.push_str("};\n");

        let layout = GpuStructLayout {
            fields: layout_fields,
            total_size: byte_offset,
            alignment: max_align,
            msl_name,
            msl_typedef,
            dep_typedefs,
        };

        self.gpu_struct_layouts.insert(type_id, layout.clone());
        Some(layout)
    }

    /// Resolve a field's Haxe type to GPU type info for @:gpuStruct layout.
    ///
    /// Key difference from cstruct: Float → f32 (4 bytes), not f64 (8 bytes).
    /// Returns (size_bytes, alignment, msl_type_string, ir_type).
    pub(crate) fn resolve_gpu_struct_field_type(
        &mut self,
        haxe_type_id: TypeId,
        ir_type: IrType,
        dep_typedefs: &mut Vec<String>,
    ) -> (u32, u32, String, IrType) {
        let type_info = {
            let type_table = self.type_table;
            type_table.get(haxe_type_id).map(|t| (t.kind.clone(), t.id))
        };

        if let Some((kind, _tid)) = type_info {
            match &kind {
                // Nested @:gpuStruct — embed inline
                TypeKind::Class { symbol_id, .. } => {
                    let is_nested_gpu_struct = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .map(|s| s.flags.is_gpu_struct())
                        .unwrap_or(false);
                    if is_nested_gpu_struct {
                        if let Some(nested) = self.get_or_compute_gpu_struct_layout(haxe_type_id) {
                            // Add nested typedef as dependency
                            if !dep_typedefs.contains(&nested.msl_typedef) {
                                for dep in &nested.dep_typedefs {
                                    if !dep_typedefs.contains(dep) {
                                        dep_typedefs.push(dep.clone());
                                    }
                                }
                                dep_typedefs.push(nested.msl_typedef.clone());
                            }
                            let msl_type = nested.msl_name.clone();
                            let size = nested.total_size;
                            let align = nested.alignment;
                            return (size, align, msl_type, IrType::I64);
                        }
                    }
                }
                _ => {}
            }
        }

        // GPU type mapping: Float→f32 (4 bytes), Int→i32 (4 bytes), Bool→u32 (4 bytes)
        match &ir_type {
            IrType::F64 => (4, 4, "float".to_string(), IrType::F32), // Float → GPU f32
            IrType::F32 => (4, 4, "float".to_string(), IrType::F32),
            IrType::I64 => (4, 4, "int".to_string(), IrType::I32), // Int → GPU i32
            IrType::I32 => (4, 4, "int".to_string(), IrType::I32),
            IrType::Bool => (4, 4, "uint".to_string(), IrType::I32), // Bool → GPU u32
            _ => (4, 4, "int".to_string(), IrType::I32),
        }
    }
}
