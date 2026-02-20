//! Monomorphization Pass for Generic Types
//!
//! This module implements lazy monomorphization - generating specialized versions
//! of generic functions and types on-demand when concrete type arguments are used.
//!
//! ## Strategy
//!
//! 1. **Lazy Instantiation**: Generate specializations only when actually used
//! 2. **Caching**: Use MonoKey (function + type_args) to cache generated instances
//! 3. **Type Substitution**: Replace TypeVar with concrete types throughout the function
//! 4. **Name Mangling**: Generate unique names like `Container_Int`, `Container_String`
//!
//! ## Integration Points
//!
//! - Called during MIR-to-MIR transformation before codegen
//! - Uses SymbolFlags::GENERIC to identify monomorphizable types
//! - Rewrites CallDirect instructions that target generic functions

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::functions::{IrFunctionSignature, IrParameter};
use super::modules::IrModule;
use super::{
    IrBasicBlock, IrBlockId, IrControlFlowGraph, IrFunction, IrFunctionId, IrInstruction,
    IrTerminator, IrType, IrValue,
};

/// Key for caching monomorphized function instances
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MonoKey {
    /// The original generic function ID
    pub generic_func: IrFunctionId,
    /// Concrete type arguments used for instantiation
    pub type_args: Vec<IrType>,
}

impl MonoKey {
    pub fn new(generic_func: IrFunctionId, type_args: Vec<IrType>) -> Self {
        Self {
            generic_func,
            type_args,
        }
    }

    /// Generate a mangled name for this instantiation
    pub fn mangled_name(&self, base_name: &str) -> String {
        if self.type_args.is_empty() {
            return base_name.to_string();
        }

        let type_suffix: Vec<String> = self
            .type_args
            .iter()
            .map(|ty| Self::mangle_type(ty))
            .collect();

        format!("{}__{}", base_name, type_suffix.join("_"))
    }

    /// Mangle a type into a name-safe string
    fn mangle_type(ty: &IrType) -> String {
        match ty {
            IrType::Void => "void".to_string(),
            IrType::Bool => "bool".to_string(),
            IrType::I8 => "i8".to_string(),
            IrType::I16 => "i16".to_string(),
            IrType::I32 => "i32".to_string(),
            IrType::I64 => "i64".to_string(),
            IrType::U8 => "u8".to_string(),
            IrType::U16 => "u16".to_string(),
            IrType::U32 => "u32".to_string(),
            IrType::U64 => "u64".to_string(),
            IrType::F32 => "f32".to_string(),
            IrType::F64 => "f64".to_string(),
            IrType::String => "String".to_string(),
            IrType::Ptr(inner) => format!("Ptr{}", Self::mangle_type(inner)),
            IrType::Ref(inner) => format!("Ref{}", Self::mangle_type(inner)),
            IrType::Array(elem, size) => format!("Arr{}x{}", Self::mangle_type(elem), size),
            IrType::Slice(elem) => format!("Slice{}", Self::mangle_type(elem)),
            IrType::Struct { name, .. } => name.replace("::", "_"),
            IrType::Union { name, .. } => name.replace("::", "_"),
            IrType::Opaque { name, .. } => name.replace("::", "_"),
            IrType::Function { .. } => "Fn".to_string(),
            IrType::TypeVar(name) => name.clone(),
            IrType::Generic { base, type_args } => {
                let base_name = Self::mangle_type(base);
                let args: Vec<String> = type_args.iter().map(Self::mangle_type).collect();
                format!("{}__{}", base_name, args.join("_"))
            }
            IrType::Any => "Any".to_string(),
            IrType::Vector { element, count } => {
                format!("Vec{}x{}", Self::mangle_type(element), count)
            }
        }
    }
}

/// Statistics for monomorphization pass
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonomorphizationStats {
    /// Number of generic functions found
    pub generic_functions_found: usize,
    /// Number of instantiations created
    pub instantiations_created: usize,
    /// Number of cache hits (reused existing instantiation)
    pub cache_hits: usize,
    /// Number of call sites rewritten
    pub call_sites_rewritten: usize,
    /// Types that were monomorphized
    pub monomorphized_types: Vec<String>,
}

/// The monomorphization engine
pub struct Monomorphizer {
    /// Cache of generated specializations: MonoKey -> specialized function ID
    instances: HashMap<MonoKey, IrFunctionId>,

    /// Mapping from type parameter names to concrete types (current substitution context)
    substitution_map: HashMap<String, IrType>,

    /// Next available function ID for new instantiations
    next_func_id: u32,

    /// Substitution maps used for each instantiation.
    /// Needed for transitive monomorphization: when a specialized function calls
    /// another function that has type_param_tag_fixups, we propagate the same
    /// substitution_map to create a specialized version of the callee.
    instantiation_sub_maps: HashMap<IrFunctionId, HashMap<String, IrType>>,

    /// Newly created functions from transitive monomorphization, to be inserted
    /// into the module after `instantiate_with_sub_map` returns (avoids borrow issues).
    pending_transitive_funcs: Vec<IrFunction>,

    /// Statistics
    stats: MonomorphizationStats,
}

impl Monomorphizer {
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
            substitution_map: HashMap::new(),
            next_func_id: 10000, // Start high to avoid conflicts
            instantiation_sub_maps: HashMap::new(),
            pending_transitive_funcs: Vec::new(),
            stats: MonomorphizationStats::default(),
        }
    }

    /// Get statistics about the monomorphization pass
    pub fn stats(&self) -> &MonomorphizationStats {
        &self.stats
    }

    /// Run monomorphization on an entire module
    ///
    /// This will:
    /// 1. Identify all generic functions (those with type_params)
    /// 2. Find all call sites that use generic functions with concrete type args
    /// 3. Generate specialized versions and rewrite call sites
    pub fn monomorphize_module(&mut self, module: &mut IrModule) {
        // Phase 1: Identify generic functions
        let generic_funcs: Vec<IrFunctionId> = module
            .functions
            .iter()
            .filter(|(_, func)| !func.signature.type_params.is_empty())
            .map(|(id, _)| *id)
            .collect();

        self.stats.generic_functions_found = generic_funcs.len();

        if generic_funcs.is_empty() {
            return; // No generic functions to monomorphize
        }

        // Phase 2: Collect all instantiation requests
        let instantiation_requests = self.collect_instantiation_requests(module, &generic_funcs);

        // Phase 3: Generate specialized functions
        let mut new_functions: Vec<IrFunction> = Vec::new();
        for (key, call_sites) in &instantiation_requests {
            if let Some(generic_func) = module.functions.get(&key.generic_func) {
                let specialized = self.instantiate(generic_func, &key.type_args);
                new_functions.push(specialized);
            }
        }

        // Phase 4: Add new functions to module
        for func in new_functions {
            module.functions.insert(func.id, func);
        }

        // Phase 5: Rewrite call sites
        self.rewrite_call_sites(module, &instantiation_requests);

        // Phase 6: Transitive monomorphization
        // Specialized functions may call other functions that have type_param_tag_fixups
        // (e.g., set__String_i32 calls setLoop which has fixups for Reflect.compare).
        // Propagate the substitution map to create specialized versions of those callees.
        self.propagate_transitive_fixups(module);
    }

    /// Collect all places where generic functions are called with concrete types
    fn collect_instantiation_requests(
        &self,
        module: &IrModule,
        generic_funcs: &[IrFunctionId],
    ) -> HashMap<MonoKey, Vec<CallSiteLocation>> {
        let mut requests: HashMap<MonoKey, Vec<CallSiteLocation>> = HashMap::new();

        for (func_id, function) in &module.functions {
            for (block_id, block) in &function.cfg.blocks {
                for (inst_idx, inst) in block.instructions.iter().enumerate() {
                    if let Some((target_func, type_args)) =
                        self.extract_generic_call(inst, generic_funcs, function)
                    {
                        let key = MonoKey::new(target_func, type_args);
                        let location = CallSiteLocation {
                            function_id: *func_id,
                            block_id: *block_id,
                            instruction_index: inst_idx,
                        };
                        requests.entry(key).or_default().push(location);
                    }
                }
            }
        }

        requests
    }

    /// Extract generic call information from an instruction
    fn extract_generic_call(
        &self,
        inst: &IrInstruction,
        generic_funcs: &[IrFunctionId],
        _context_func: &IrFunction,
    ) -> Option<(IrFunctionId, Vec<IrType>)> {
        match inst {
            IrInstruction::CallDirect {
                func_id, type_args, ..
            } => {
                if generic_funcs.contains(func_id) && !type_args.is_empty() {
                    Some((*func_id, type_args.clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Generate a specialized version of a generic function
    pub fn instantiate(&mut self, generic_func: &IrFunction, type_args: &[IrType]) -> IrFunction {
        let key = MonoKey::new(generic_func.id, type_args.to_vec());

        // Check cache
        if let Some(&existing_id) = self.instances.get(&key) {
            self.stats.cache_hits += 1;
            // Return a dummy - the real function is already in the module
            // This shouldn't happen in normal flow since we check before calling
            let mut dummy = generic_func.clone();
            dummy.id = existing_id;
            return dummy;
        }

        // Build substitution map: type_param_name -> concrete_type
        self.substitution_map.clear();
        for (param, arg) in generic_func.signature.type_params.iter().zip(type_args) {
            self.substitution_map
                .insert(param.name.clone(), arg.clone());
        }

        // Clone and specialize the function
        let new_id = IrFunctionId(self.next_func_id);
        self.next_func_id += 1;

        let mut specialized = generic_func.clone();
        specialized.id = new_id;
        specialized.name = key.mangled_name(&generic_func.name);

        // Clear type params - this is now a concrete function
        specialized.signature.type_params.clear();

        // Substitute types in signature
        specialized.signature = self.substitute_signature(&specialized.signature);

        // Substitute types in locals
        for (_, local) in specialized.locals.iter_mut() {
            local.ty = self.substitute_type(&local.ty);
        }

        // Substitute types in register_types
        let mut new_register_types = HashMap::new();
        for (id, ty) in &specialized.register_types {
            new_register_types.insert(*id, self.substitute_type(ty));
        }
        specialized.register_types = new_register_types;

        // Substitute types in CFG instructions
        self.substitute_cfg(&mut specialized.cfg);

        // Process type parameter tag fixups: replace placeholder const values
        // with concrete type tags based on the substitution map
        self.apply_type_param_tag_fixups(&mut specialized);

        // Cache the result
        self.instances.insert(key.clone(), new_id);
        self.instantiation_sub_maps
            .insert(new_id, self.substitution_map.clone());
        self.stats.instantiations_created += 1;
        self.stats
            .monomorphized_types
            .push(specialized.name.clone());

        specialized
    }

    /// Substitute types in a function signature
    fn substitute_signature(&self, sig: &IrFunctionSignature) -> IrFunctionSignature {
        IrFunctionSignature {
            parameters: sig
                .parameters
                .iter()
                .map(|p| IrParameter {
                    name: p.name.clone(),
                    ty: self.substitute_type(&p.ty),
                    reg: p.reg,
                    by_ref: p.by_ref,
                })
                .collect(),
            return_type: self.substitute_type(&sig.return_type),
            calling_convention: sig.calling_convention,
            can_throw: sig.can_throw,
            type_params: Vec::new(), // Cleared - now concrete
            uses_sret: sig.uses_sret,
        }
    }

    /// Recursively substitute type variables with concrete types
    fn substitute_type(&self, ty: &IrType) -> IrType {
        match ty {
            IrType::TypeVar(name) => self
                .substitution_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| ty.clone()),
            IrType::Ptr(inner) => IrType::Ptr(Box::new(self.substitute_type(inner))),
            IrType::Ref(inner) => IrType::Ref(Box::new(self.substitute_type(inner))),
            IrType::Array(elem, size) => IrType::Array(Box::new(self.substitute_type(elem)), *size),
            IrType::Slice(elem) => IrType::Slice(Box::new(self.substitute_type(elem))),
            IrType::Function {
                params,
                return_type,
                varargs,
            } => IrType::Function {
                params: params.iter().map(|p| self.substitute_type(p)).collect(),
                return_type: Box::new(self.substitute_type(return_type)),
                varargs: *varargs,
            },
            IrType::Struct { name, fields } => IrType::Struct {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|f| super::types::StructField {
                        name: f.name.clone(),
                        ty: self.substitute_type(&f.ty),
                        offset: f.offset,
                    })
                    .collect(),
            },
            IrType::Union { name, variants } => IrType::Union {
                name: name.clone(),
                variants: variants
                    .iter()
                    .map(|v| super::types::UnionVariant {
                        name: v.name.clone(),
                        tag: v.tag,
                        fields: v.fields.iter().map(|f| self.substitute_type(f)).collect(),
                    })
                    .collect(),
            },
            IrType::Generic { base, type_args } => {
                let new_base = self.substitute_type(base);
                let new_args: Vec<IrType> =
                    type_args.iter().map(|a| self.substitute_type(a)).collect();

                // If all type args are now concrete, we could potentially
                // resolve this to a concrete type, but for now keep as Generic
                IrType::Generic {
                    base: Box::new(new_base),
                    type_args: new_args,
                }
            }
            // Primitive types pass through unchanged
            _ => ty.clone(),
        }
    }

    /// Substitute types in all CFG instructions
    fn substitute_cfg(&self, cfg: &mut IrControlFlowGraph) {
        for (_, block) in cfg.blocks.iter_mut() {
            for inst in block.instructions.iter_mut() {
                self.substitute_instruction(inst);
            }
            self.substitute_terminator(&mut block.terminator);
        }
    }

    /// Substitute types in a single instruction
    fn substitute_instruction(&self, inst: &mut IrInstruction) {
        match inst {
            IrInstruction::Alloc { ty, .. } => {
                *ty = self.substitute_type(ty);
            }
            IrInstruction::Load { ty, .. } => {
                *ty = self.substitute_type(ty);
            }
            IrInstruction::Cast { from_ty, to_ty, .. } => {
                *from_ty = self.substitute_type(from_ty);
                *to_ty = self.substitute_type(to_ty);
            }
            IrInstruction::BitCast { ty, .. } => {
                *ty = self.substitute_type(ty);
            }
            IrInstruction::CallDirect { type_args, .. } => {
                // Substitute type args
                for arg in type_args.iter_mut() {
                    *arg = self.substitute_type(arg);
                }
            }
            IrInstruction::GetElementPtr { ty, .. } => {
                *ty = self.substitute_type(ty);
            }
            // Other instructions don't have type fields that need substitution
            _ => {}
        }
    }

    /// Substitute types in a terminator
    fn substitute_terminator(&self, _term: &mut IrTerminator) {
        // Most terminators don't have type information
        // Add cases here if needed
    }

    /// Process type parameter tag fixups after monomorphization.
    ///
    /// During HIR-to-MIR lowering, when a generic function calls Reflect.compare with
    /// type-erased arguments, a placeholder const 0 is emitted for the type tag.
    /// After monomorphization resolves the concrete types, this method replaces the
    /// placeholder with the correct type tag based on the substitution map.
    fn apply_type_param_tag_fixups(&self, func: &mut IrFunction) {
        if func.type_param_tag_fixups.is_empty() {
            return;
        }

        let fixups: Vec<_> = func.type_param_tag_fixups.drain(..).collect();
        for (reg_id, type_param_name) in &fixups {
            let concrete_type = match self.substitution_map.get(type_param_name) {
                Some(ty) => ty.clone(),
                None => continue,
            };

            let type_tag = Self::ir_type_to_type_tag(&concrete_type);

            for block in func.cfg.blocks.values_mut() {
                for inst in block.instructions.iter_mut() {
                    if let IrInstruction::Const { dest, value } = inst {
                        if *dest == *reg_id {
                            *value = IrValue::I32(type_tag);
                        }
                    }
                }
            }
        }
    }

    /// Map an IrType to a runtime type tag for Reflect.compare_typed.
    /// Tags: 1=Int, 2=Bool, 4=Float, 5=String
    fn ir_type_to_type_tag(ty: &IrType) -> i32 {
        match ty {
            IrType::I32 | IrType::I64 | IrType::I8 | IrType::I16 => 1, // TYPE_INT
            IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64 => 1, // TYPE_INT
            IrType::Bool => 2,                                         // TYPE_BOOL
            IrType::F32 | IrType::F64 => 4,                            // TYPE_FLOAT
            IrType::String => 5,                                       // TYPE_STRING
            IrType::Ptr(inner) if matches!(**inner, IrType::U8) => 5,  // Ptr(U8) = String
            IrType::Ptr(_) => 6,                                       // Reference/Object
            _ => 1,                                                    // Default: Int
        }
    }

    /// Propagate substitution maps transitively through call chains.
    ///
    /// When a specialized function (e.g., set__String_i32) calls another function
    /// that has unresolved type_param_tag_fixups (e.g., setLoop with fixups for "K"),
    /// create a specialized version of the callee using the caller's substitution map.
    fn propagate_transitive_fixups(&mut self, module: &mut IrModule) {
        // Collect IDs of functions with direct type_param_tag_fixups
        let direct_fixups: HashSet<IrFunctionId> = module
            .functions
            .iter()
            .filter(|(_, f)| !f.type_param_tag_fixups.is_empty())
            .map(|(id, _)| *id)
            .collect();

        if direct_fixups.is_empty() {
            return;
        }

        // Compute transitive closure: include functions that call (transitively)
        // any function with fixups. E.g., if compare() has fixups and setLoop()
        // calls compare(), then setLoop() is also in the set.
        let mut funcs_with_fixups = direct_fixups.clone();
        loop {
            let mut added = false;
            for (fid, func) in &module.functions {
                if funcs_with_fixups.contains(fid) {
                    continue;
                }
                let calls_fixup_func = func.cfg.blocks.values().any(|block| {
                    block.instructions.iter().any(|inst| {
                        if let IrInstruction::CallDirect {
                            func_id: callee, ..
                        } = inst
                        {
                            funcs_with_fixups.contains(callee)
                        } else {
                            false
                        }
                    })
                });
                if calls_fixup_func {
                    funcs_with_fixups.insert(*fid);
                    added = true;
                }
            }
            if !added {
                break;
            }
        }

        // Worklist: start with all initially monomorphized functions
        let mut worklist: Vec<IrFunctionId> = self.instantiation_sub_maps.keys().copied().collect();
        let mut processed: HashSet<IrFunctionId> = HashSet::new();

        while let Some(func_id) = worklist.pop() {
            if processed.contains(&func_id) {
                continue;
            }
            processed.insert(func_id);

            let sub_map = match self.instantiation_sub_maps.get(&func_id) {
                Some(m) => m.clone(),
                None => continue,
            };

            // Scan this function's body for calls to functions with fixups
            let func = match module.functions.get(&func_id) {
                Some(f) => f.clone(),
                None => continue,
            };

            let mut rewrites: Vec<(IrBlockId, usize, IrFunctionId)> = vec![];

            for (block_id, block) in &func.cfg.blocks {
                for (inst_idx, inst) in block.instructions.iter().enumerate() {
                    if let IrInstruction::CallDirect {
                        func_id: callee_id,
                        type_args,
                        ..
                    } = inst
                    {
                        // Only handle calls without type_args (already-monomorphized calls
                        // with type_args are handled by the standard flow)
                        if type_args.is_empty() && funcs_with_fixups.contains(callee_id) {
                            // Specialize this callee with our substitution map
                            if let Some(callee) = module.functions.get(callee_id).cloned() {
                                let (new_id, is_new) =
                                    self.instantiate_with_sub_map(&callee, &sub_map);
                                rewrites.push((*block_id, inst_idx, new_id));
                                // Only insert newly created functions into the module
                                if is_new {
                                    // Drain pending funcs created by instantiate_with_sub_map
                                    let pending: Vec<_> =
                                        self.pending_transitive_funcs.drain(..).collect();
                                    for f in pending {
                                        module.functions.insert(f.id, f);
                                    }
                                    worklist.push(new_id);
                                }
                            }
                        }
                    }
                }
            }

            // Apply rewrites
            if !rewrites.is_empty() {
                if let Some(func) = module.functions.get_mut(&func_id) {
                    for (block_id, inst_idx, new_callee_id) in &rewrites {
                        if let Some(block) = func.cfg.blocks.get_mut(block_id) {
                            if let Some(inst) = block.instructions.get_mut(*inst_idx) {
                                if let IrInstruction::CallDirect { func_id, .. } = inst {
                                    *func_id = *new_callee_id;
                                    self.stats.call_sites_rewritten += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Create a specialized version of a function using an explicit substitution map.
    ///
    /// Unlike `instantiate()` which derives the substitution map from the function's
    /// type_params and type_args, this method accepts a pre-built substitution map.
    /// This is needed for transitive monomorphization where the callee doesn't have
    /// type_params in its signature (inherited from the enclosing generic class).
    ///
    /// Returns `Some((specialized_func, is_new))` — `is_new` is true when the function
    /// was freshly created, false on cache hit (caller should NOT insert into module).
    fn instantiate_with_sub_map(
        &mut self,
        generic_func: &IrFunction,
        sub_map: &HashMap<String, IrType>,
    ) -> (IrFunctionId, bool) {
        // Build deterministic type_args from sorted substitution map for caching
        let mut sorted_entries: Vec<_> = sub_map.iter().collect();
        sorted_entries.sort_by_key(|(k, _)| (*k).clone());
        let type_args: Vec<IrType> = sorted_entries.iter().map(|(_, v)| (*v).clone()).collect();

        let key = MonoKey::new(generic_func.id, type_args);

        // Check cache — return existing ID, do NOT create a dummy clone
        if let Some(&existing_id) = self.instances.get(&key) {
            self.stats.cache_hits += 1;
            return (existing_id, false);
        }

        // Set the substitution map
        self.substitution_map = sub_map.clone();

        let new_id = IrFunctionId(self.next_func_id);
        self.next_func_id += 1;

        let mut specialized = generic_func.clone();
        specialized.id = new_id;
        specialized.name = key.mangled_name(&generic_func.name);

        // Clear any type params (shouldn't have any, but be safe)
        specialized.signature.type_params.clear();

        // Substitute types in register_types
        let mut new_register_types = HashMap::new();
        for (id, ty) in &specialized.register_types {
            new_register_types.insert(*id, self.substitute_type(ty));
        }
        specialized.register_types = new_register_types;

        // Substitute types in locals
        for (_, local) in specialized.locals.iter_mut() {
            local.ty = self.substitute_type(&local.ty);
        }

        // Substitute types in CFG
        self.substitute_cfg(&mut specialized.cfg);

        // Resolve type_param_tag_fixups using the substitution map
        self.apply_type_param_tag_fixups(&mut specialized);

        // Cache
        self.instances.insert(key, new_id);
        self.instantiation_sub_maps.insert(new_id, sub_map.clone());
        self.stats.instantiations_created += 1;
        self.stats
            .monomorphized_types
            .push(specialized.name.clone());

        // Store the newly created specialized function for caller to insert
        self.pending_transitive_funcs.push(specialized);

        (new_id, true)
    }

    /// Rewrite call sites to use specialized functions
    fn rewrite_call_sites(
        &mut self,
        module: &mut IrModule,
        requests: &HashMap<MonoKey, Vec<CallSiteLocation>>,
    ) {
        for (key, locations) in requests {
            if let Some(&specialized_id) = self.instances.get(key) {
                for loc in locations {
                    if let Some(func) = module.functions.get_mut(&loc.function_id) {
                        if let Some(block) = func.cfg.blocks.get_mut(&loc.block_id) {
                            if let Some(inst) = block.instructions.get_mut(loc.instruction_index) {
                                if let IrInstruction::CallDirect {
                                    func_id, type_args, ..
                                } = inst
                                {
                                    *func_id = specialized_id;
                                    type_args.clear(); // No longer generic
                                    self.stats.call_sites_rewritten += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Default for Monomorphizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Location of a call site in the IR
#[derive(Debug, Clone)]
struct CallSiteLocation {
    function_id: IrFunctionId,
    block_id: IrBlockId,
    instruction_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_key_mangling() {
        let key = MonoKey::new(IrFunctionId(1), vec![IrType::I32, IrType::String]);
        assert_eq!(key.mangled_name("Container"), "Container__i32_String");
    }

    #[test]
    fn test_mono_key_nested_types() {
        let key = MonoKey::new(IrFunctionId(1), vec![IrType::Ptr(Box::new(IrType::I32))]);
        assert_eq!(key.mangled_name("Process"), "Process__Ptri32");
    }

    #[test]
    fn test_type_substitution() {
        let mut mono = Monomorphizer::new();
        mono.substitution_map.insert("T".to_string(), IrType::I32);

        let original = IrType::TypeVar("T".to_string());
        let substituted = mono.substitute_type(&original);
        assert_eq!(substituted, IrType::I32);
    }

    #[test]
    fn test_nested_type_substitution() {
        let mut mono = Monomorphizer::new();
        mono.substitution_map
            .insert("T".to_string(), IrType::String);

        let original = IrType::Ptr(Box::new(IrType::TypeVar("T".to_string())));
        let substituted = mono.substitute_type(&original);
        assert_eq!(substituted, IrType::Ptr(Box::new(IrType::String)));
    }
}
