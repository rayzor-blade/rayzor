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
use std::collections::{BTreeMap, BTreeSet};

use super::functions::{IrFunctionSignature, IrParameter};
use super::modules::IrModule;
use super::{
    IrBasicBlock, IrBlockId, IrControlFlowGraph, IrFunction, IrFunctionId, IrId, IrInstruction,
    IrTerminator, IrType, IrValue,
};

/// How a generic body moves a T value across an erased slot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WordSite {
    /// `bitcast i64 -> T`.
    Decode,
    /// `bitcast T -> i64`.
    Encode,
    /// A T register as this call argument.
    Arg(usize),
    /// A T register as this native field call argument, which takes the
    /// value's raw bits.
    NativeArg(usize),
    /// A cast that converts a T register from its erased i64 form; it
    /// converts from the bound type instead.
    Retype,
}

/// Key for caching monomorphized function instances
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
            IrType::Dynamic => "Dynamic".to_string(),
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
            IrType::NullInt => "NullInt".to_string(),
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
    instances: BTreeMap<MonoKey, IrFunctionId>,

    /// Mapping from type parameter names to concrete types (current substitution context)
    substitution_map: BTreeMap<String, IrType>,

    /// Next available function ID for new instantiations
    next_func_id: u32,

    /// Substitution maps used for each instantiation.
    /// Needed for transitive monomorphization: when a specialized function calls
    /// another function that has type_param_tag_fixups, we propagate the same
    /// substitution_map to create a specialized version of the callee.
    instantiation_sub_maps: BTreeMap<IrFunctionId, BTreeMap<String, IrType>>,

    /// Newly created functions from transitive monomorphization, to be inserted
    /// into the module after `instantiate_with_sub_map` returns (avoids borrow issues).
    pending_transitive_funcs: Vec<IrFunction>,

    /// Statistics
    stats: MonomorphizationStats,

    /// Word decoders for erased returns, declared per module.
    word_decoders: Option<super::erased_word::Decoders>,

    /// Runtime calls that store or load a field in its native form, so a T
    /// value crossing them is not a word.
    native_field_calls: BTreeSet<IrFunctionId>,
}

impl Monomorphizer {
    pub fn new() -> Self {
        Self {
            instances: BTreeMap::new(),
            substitution_map: BTreeMap::new(),
            next_func_id: 10000, // Start high to avoid conflicts
            instantiation_sub_maps: BTreeMap::new(),
            pending_transitive_funcs: Vec::new(),
            stats: MonomorphizationStats::default(),
            word_decoders: None,
            native_field_calls: BTreeSet::new(),
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
        // Seed the instance-id counter ABOVE every id already in THIS module
        // instead of the fixed 10_000 from `new()`. The import renumber
        // (compilation.rs `renumber_and_push_import_mir`) shifts every
        // function in import module N by `import_base = 100_000 + N*10_000`
        // (stride 10_000). A fixed start of 10_000 makes a monomorphized
        // instance in module N renumber to `base(N)+10_000 == base(N+1)` —
        // the slot-0 (typically the `malloc` extern) of the *next* import
        // module — which the final merge then silently overwrites. Symptom:
        // `new WorkerPool()`'s heap-alloc CallDirect (to WorkerPool's malloc,
        // renumbered to e.g. 230000) was clobbered by BalancedTree's
        // `get_height__i64_i64` instance (10000 + base 220000 = 230000), so
        // the allocation dispatched into get_height, returned a small int
        // (a tree height) as the object pointer, and SIGSEGV'd. Seeding from
        // the live max keeps instance ids inside this module's own
        // [0, stride) band so they never alias another module's base after
        // renumbering. (`module.functions` may contain ids inserted directly
        // — Phase 4 below does exactly that — so consult the keys, not just
        // `next_function_id`.)
        let max_existing = module
            .functions
            .keys()
            .chain(module.extern_functions.keys())
            .map(|id| id.0)
            .max()
            .unwrap_or(0)
            .max(module.next_function_id);
        self.next_func_id = max_existing.saturating_add(1);

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
        self.word_decoders = super::erased_word::Decoders::declare(module);
        // Instances take ids after the decoders just declared.
        self.next_func_id = self.next_func_id.max(module.next_function_id);
        const NATIVE_FIELD_CALLS: &[&str] = &[
            "rayzor_anon_set_field_by_index",
            "rayzor_anon_set_field",
            "rayzor_anon_set_struct_field",
            "rayzor_anon_get_field_by_index",
            "rayzor_anon_get_field",
            "rayzor_anon_get_struct_field",
        ];
        self.native_field_calls = module
            .all_functions()
            .filter(|(_, name)| NATIVE_FIELD_CALLS.contains(name))
            .map(|(id, _)| id)
            .chain(
                module
                    .external_function_names
                    .iter()
                    .filter(|(_, name)| NATIVE_FIELD_CALLS.contains(&name.as_str()))
                    .map(|(id, _)| *id),
            )
            .collect();

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
        self.encode_template_call_args(module);
        self.adapt_specialized_call_returns(module);
        self.fold_generic_value_adapters(module);
        Self::widen_generic_box_floats(module);

        // Keep the module's id allocator above the instances we just minted.
        // Phase 4 / transitive monomorphization insert directly into
        // `module.functions` without going through `add_function`, so without
        // this a later `new_function_id()`/`add_function()` could hand out an
        // id that aliases a monomorphized instance.
        module.next_function_id = module.next_function_id.max(self.next_func_id);
    }

    /// Collect all places where generic functions are called with concrete types
    fn collect_instantiation_requests(
        &self,
        module: &IrModule,
        generic_funcs: &[IrFunctionId],
    ) -> BTreeMap<MonoKey, Vec<CallSiteLocation>> {
        let mut requests: BTreeMap<MonoKey, Vec<CallSiteLocation>> = BTreeMap::new();

        for (func_id, function) in &module.functions {
            for (block_id, block) in &function.cfg.blocks {
                for (inst_idx, inst) in block.instructions.iter().enumerate() {
                    if let Some((target_func, type_args)) =
                        self.extract_generic_call(inst, generic_funcs, function, module)
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

    /// Extract generic call information from an instruction.
    /// When explicit type_args are empty, infers concrete types from argument register types.
    fn extract_generic_call(
        &self,
        inst: &IrInstruction,
        generic_funcs: &[IrFunctionId],
        context_func: &IrFunction,
        module: &IrModule,
    ) -> Option<(IrFunctionId, Vec<IrType>)> {
        match inst {
            IrInstruction::CallDirect {
                func_id,
                type_args,
                args,
                ..
            } => {
                if !generic_funcs.contains(func_id) {
                    return None;
                }
                let dbg_mono = std::env::var("RAYZOR_DUMP_MONO").is_ok();
                // Use explicit type_args if available
                if !type_args.is_empty() {
                    // A template's call binds only after the caller is specialized.
                    if type_args.iter().any(contains_type_param) {
                        return None;
                    }
                    if dbg_mono {
                        if let Some(c) = module.functions.get(func_id) {
                            eprintln!(
                                "[mono] call->{:?} {} EXPLICIT type_args={:?}",
                                func_id, c.name, type_args
                            );
                        }
                    }
                    return Some((*func_id, type_args.clone()));
                }
                // Infer type_args from argument register types
                let callee = module.functions.get(func_id)?;
                if dbg_mono {
                    let ptys: Vec<_> = callee
                        .signature
                        .parameters
                        .iter()
                        .map(|p| p.ty.clone())
                        .collect();
                    eprintln!(
                        "[mono] call->{:?} {} NO explicit type_args; type_params={:?} param_tys={:?}",
                        func_id,
                        callee.name,
                        callee
                            .signature
                            .type_params
                            .iter()
                            .map(|t| t.name.clone())
                            .collect::<Vec<_>>(),
                        ptys
                    );
                }
                let mut inferred_args = Vec::new();
                for type_param in &callee.signature.type_params {
                    let mut found = false;
                    for (i, sig_param) in callee.signature.parameters.iter().enumerate() {
                        if let IrType::TypeVar(ref name) = sig_param.ty {
                            if name == &type_param.name && i < args.len() {
                                if let Some(arg_ty) = context_func.register_types.get(&args[i]) {
                                    if !matches!(arg_ty, IrType::TypeVar(_)) {
                                        inferred_args.push(arg_ty.clone());
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // An erased method-level generic never spells its type
                    // variable in a parameter, but its body still reads the
                    // erasure through a tag this pass resolves, and the
                    // argument says what that tag is. One variable only, and
                    // only when the erased slots agree: with two of them
                    // nothing says which argument belongs to which variable.
                    if !found
                        && callee.signature.type_params.len() == 1
                        && !callee.type_param_tag_fixups.is_empty()
                    {
                        let mut agreed: Option<IrType> = None;
                        let mut conflict = false;
                        for (i, sig_param) in callee.signature.parameters.iter().enumerate() {
                            if sig_param.ty != IrType::I64 || i >= args.len() {
                                continue;
                            }
                            // One slot known and another not is not agreement.
                            // `eq(12, t.get())` over `T:(Float)` reads Int off
                            // the literal while the call returns float bits,
                            // and the tag that comes out compares them unequal.
                            let ty =
                                match crate::ir::inlining::concrete_arg_type(context_func, args[i])
                                {
                                    Some(ty) => ty,
                                    None => {
                                        conflict = true;
                                        break;
                                    }
                                };
                            match &agreed {
                                Some(prev) if *prev != ty => conflict = true,
                                Some(_) => {}
                                None => agreed = Some(ty),
                            }
                        }
                        if !conflict {
                            if let Some(ty) = agreed {
                                if dbg_mono {
                                    eprintln!(
                                        "[mono]   -> inferred '{}' = {:?} from an erased slot",
                                        type_param.name, ty
                                    );
                                }
                                inferred_args.push(ty);
                                found = true;
                            }
                        }
                    }
                    // No argument spells the variable (`Enum<T>`, a result-only
                    // `T`), so Haxe leaves it unbound. Specialize it as the
                    // erased slot the template already lowers it to, rather
                    // than call a template that is stubbed. Not for a variable
                    // the body reads a tag for: erasure says nothing there.
                    if !found
                        && references_type_params(callee)
                        && !callee
                            .signature
                            .parameters
                            .iter()
                            .any(|p| p.ty.type_param_name() == Some(type_param.name.as_str()))
                        && (!callee
                            .type_param_tag_fixups
                            .iter()
                            .any(|(_, name)| *name == type_param.name)
                            || self.erasure_safe_tag_fixups(callee, &type_param.name, module))
                    {
                        if dbg_mono {
                            eprintln!(
                                "[mono]   -> '{}' is unbound; erased to I64",
                                type_param.name
                            );
                        }
                        inferred_args.push(IrType::I64);
                        found = true;
                    }
                    if !found && type_param.constraints.iter().any(|bound| bound == "{}") {
                        inferred_args.push(IrType::Ptr(Box::new(IrType::Void)));
                        found = true;
                    }
                    if !found {
                        // Can't infer all type params — skip
                        if dbg_mono {
                            eprintln!(
                                "[mono]   -> cannot infer type_param '{}' (no TypeVar param matches); SKIP specialization",
                                type_param.name
                            );
                        }
                        return None;
                    }
                }
                if !inferred_args.is_empty() {
                    Some((*func_id, inferred_args))
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
        Self::bind_object_constraints(generic_func, &mut self.substitution_map);

        // Clone and specialize the function
        let new_id = IrFunctionId(self.next_func_id);
        self.next_func_id += 1;

        let mut specialized = generic_func.clone();
        specialized.id = new_id;
        specialized.name = key.mangled_name(&generic_func.name);

        // Substitute types in signature
        specialized.signature = self.substitute_signature(&specialized.signature);

        // Substitute types in locals
        for (_, local) in specialized.locals.iter_mut() {
            local.ty = self.substitute_type(&local.ty);
        }

        // Substitute types in register_types
        let mut new_register_types = BTreeMap::new();
        for (id, ty) in &specialized.register_types {
            new_register_types.insert(*id, self.substitute_type(ty));
        }
        specialized.register_types = new_register_types;

        // A parameter's register holds what its slot holds, and an erased slot
        // stays I64 whatever the type argument is. Substituting the register
        // types alone can put a concrete type on a register the ABI still
        // passes as bits.
        for param in &specialized.signature.parameters {
            specialized
                .register_types
                .insert(param.reg, param.ty.clone());
        }

        // Substitute types in CFG instructions
        let crossings = self.word_crossings(generic_func);
        self.substitute_cfg(&mut specialized.cfg);
        self.apply_word_crossings(&mut specialized, crossings);

        self.adapt_erased_returns(&mut specialized);

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
            type_params: sig
                .type_params
                .iter()
                .filter(|parameter| {
                    self.substitution_map
                        .get(&parameter.name)
                        .is_none_or(contains_type_param)
                })
                .cloned()
                .collect(),
            uses_sret: sig.uses_sret,
        }
    }

    /// Where a generic body moves a T value between an i64 slot and its
    /// type-variable register: the bitcasts in and out, and a T register
    /// passed straight to a call, whose parameter is an erased slot. Empty
    /// unless `RAYZOR_ERASED_WORDS` is set, when the slot holds a word.
    fn word_crossings(&self, function: &IrFunction) -> Vec<(IrBlockId, usize, WordSite, String)> {
        if self.word_decoders.is_none() {
            return Vec::new();
        }
        let slot = |ty: Option<&IrType>| matches!(ty, Some(IrType::I64 | IrType::U64));
        // A register produced by, or only consumed by, a native field call
        // holds the field's native form.
        let mut native_defs = BTreeSet::new();
        let mut uses: BTreeMap<IrId, (usize, usize)> = BTreeMap::new();
        for block in function.cfg.blocks.values() {
            for instruction in &block.instructions {
                let native = matches!(instruction,
                    IrInstruction::CallDirect { func_id, .. } if self.native_field_calls.contains(func_id));
                if native && let Some(dest) = instruction.dest() {
                    native_defs.insert(dest);
                }
                for used in instruction.uses() {
                    let entry = uses.entry(used).or_default();
                    entry.0 += 1;
                    if native {
                        entry.1 += 1;
                    }
                }
            }
        }
        let native_only = |reg: &IrId| {
            uses.get(reg)
                .is_some_and(|(all, native)| all == native && *all > 0)
        };
        let mut out = Vec::new();
        for (id, block) in &function.cfg.blocks {
            for (index, instruction) in block.instructions.iter().enumerate() {
                match instruction {
                    IrInstruction::BitCast { dest, src, .. }
                    | IrInstruction::Cast { dest, src, .. }
                        if native_defs.contains(src)
                            || (native_only(dest)
                                && matches!(
                                    function.register_types.get(src),
                                    Some(IrType::TypeVar(_))
                                )) => {}
                    IrInstruction::BitCast { src, ty, .. } => {
                        match (function.register_types.get(src), ty) {
                            (source, IrType::TypeVar(name)) if slot(source) => {
                                out.push((*id, index, WordSite::Decode, name.clone()));
                            }
                            (Some(IrType::TypeVar(name)), IrType::I64 | IrType::U64) => {
                                out.push((*id, index, WordSite::Encode, name.clone()));
                            }
                            _ => {}
                        }
                    }
                    IrInstruction::Cast {
                        src,
                        from_ty: IrType::I64 | IrType::U64,
                        to_ty,
                        ..
                    } if !matches!(to_ty, IrType::TypeVar(_))
                        && let Some(IrType::TypeVar(name)) = function.register_types.get(src) =>
                    {
                        out.push((*id, index, WordSite::Retype, name.clone()));
                    }
                    IrInstruction::Cast { from_ty, to_ty, .. } => match (from_ty, to_ty) {
                        (IrType::I64 | IrType::U64, IrType::TypeVar(name)) => {
                            out.push((*id, index, WordSite::Decode, name.clone()));
                        }
                        (IrType::TypeVar(name), IrType::I64 | IrType::U64) => {
                            out.push((*id, index, WordSite::Encode, name.clone()));
                        }
                        _ => {}
                    },
                    IrInstruction::CallDirect { args, func_id, .. } => {
                        let native = self.native_field_calls.contains(func_id);
                        for (position, arg) in args.iter().enumerate() {
                            if let Some(IrType::TypeVar(name)) = function.register_types.get(arg) {
                                let site = if native {
                                    WordSite::NativeArg(position)
                                } else {
                                    WordSite::Arg(position)
                                };
                                out.push((*id, index, site, name.clone()));
                            }
                        }
                    }
                    // A dispatched method's erased parameter stays a slot.
                    IrInstruction::CallIndirect {
                        args,
                        signature: IrType::Function { params, .. },
                        ..
                    } => {
                        for (position, arg) in args.iter().enumerate() {
                            if let Some(IrType::TypeVar(name)) = function.register_types.get(arg)
                                && matches!(params.get(position), Some(IrType::I64 | IrType::U64))
                            {
                                out.push((*id, index, WordSite::Arg(position), name.clone()));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        out
    }

    /// Where T is bound to a scalar (or Dynamic), a slot-to-T bitcast decodes
    /// the word, and a T-to-slot bitcast or a T argument encodes one.
    fn apply_word_crossings(
        &self,
        function: &mut IrFunction,
        crossings: Vec<(IrBlockId, usize, WordSite, String)>,
    ) {
        let mut by_block: BTreeMap<IrBlockId, BTreeMap<usize, Vec<(WordSite, IrType)>>> =
            BTreeMap::new();
        for (block, index, site, name) in crossings {
            if let Some(bound) = self.substitution_map.get(&name).filter(|ty| {
                super::erased_word::is_word_scalar(ty) || **ty == IrType::dynamic_ptr()
            }) {
                by_block
                    .entry(block)
                    .or_default()
                    .entry(index)
                    .or_default()
                    .push((site, bound.clone()));
            }
        }
        if by_block.is_empty() {
            return;
        }
        let decoders = self.word_decoders.as_ref();
        let mut regs = super::erased_word::Regs::new(function);
        let mut encode = |regs: &mut super::erased_word::Regs,
                          out: &mut Vec<IrInstruction>,
                          src,
                          bound: &IrType| {
            if *bound == IrType::dynamic_ptr() {
                regs.encode_dynamic(decoders, out, src)
            } else {
                regs.encode(out, src, bound)
            }
        };
        for (block_id, sites) in by_block {
            let Some(block) = regs.function().cfg.blocks.get_mut(&block_id) else {
                continue;
            };
            let old = std::mem::take(&mut block.instructions);
            let mut out = Vec::with_capacity(old.len() + sites.len() * 4);
            for (index, mut instruction) in old.into_iter().enumerate() {
                let Some(here) = sites.get(&index) else {
                    out.push(instruction);
                    continue;
                };
                for (site, bound) in here {
                    match (site, &mut instruction) {
                        (
                            WordSite::Arg(position),
                            IrInstruction::CallDirect { args, .. }
                            | IrInstruction::CallIndirect { args, .. },
                        ) => {
                            if let Some(arg) = args.get_mut(*position) {
                                *arg = encode(&mut regs, &mut out, *arg, bound);
                            }
                        }
                        (WordSite::NativeArg(position), IrInstruction::CallDirect { args, .. }) => {
                            if let Some(arg) = args.get_mut(*position) {
                                *arg = regs.raw(&mut out, *arg, bound);
                            }
                        }
                        (
                            WordSite::Decode | WordSite::Encode,
                            IrInstruction::BitCast { dest, src, .. }
                            | IrInstruction::Cast { dest, src, .. },
                        ) => {
                            let (dest, src) = (*dest, *src);
                            let value = if *site == WordSite::Decode {
                                regs.decode(decoders, &mut out, src, bound)
                            } else {
                                encode(&mut regs, &mut out, src, bound)
                            };
                            let ty = if *site == WordSite::Decode {
                                bound.clone()
                            } else {
                                IrType::I64
                            };
                            regs.function().register_types.insert(dest, ty);
                            instruction = IrInstruction::Copy { dest, src: value };
                        }
                        (
                            WordSite::Retype,
                            IrInstruction::Cast {
                                dest,
                                src,
                                from_ty,
                                to_ty,
                            },
                        ) => {
                            if bound == to_ty {
                                instruction = IrInstruction::Copy {
                                    dest: *dest,
                                    src: *src,
                                };
                            } else {
                                *from_ty = bound.clone();
                            }
                        }
                        _ => {}
                    }
                }
                out.push(instruction);
            }
            if let Some(block) = regs.function().cfg.blocks.get_mut(&block_id) {
                block.instructions = out;
            }
        }
    }

    /// A generic return takes its concrete ABI from the erased slot's bits.
    fn adapt_erased_returns(&self, function: &mut IrFunction) {
        let ty = function.signature.return_type.clone();
        if ty == IrType::Void {
            for block in function.cfg.blocks.values_mut() {
                if matches!(block.terminator, IrTerminator::Return { .. }) {
                    block.terminator = IrTerminator::Return { value: None };
                }
            }
            return;
        }
        if !matches!(
            ty,
            IrType::F64
                | IrType::F32
                | IrType::String
                | IrType::Ptr(_)
                | IrType::I32
                | IrType::Bool
        ) {
            return;
        }
        let returns: Vec<_> = function
            .cfg
            .blocks
            .iter()
            .filter_map(|(id, block)| match &block.terminator {
                IrTerminator::Return { value: Some(value) }
                    if function.register_types.get(value) == Some(&IrType::I64) =>
                {
                    Some((*id, *value))
                }
                _ => None,
            })
            .collect();
        let mut regs = super::erased_word::Regs::new(function);
        for (id, returned) in returns {
            let mut out = Vec::new();
            let value = if super::erased_word::is_word_scalar(&ty) || ty == IrType::dynamic_ptr() {
                regs.decode(self.word_decoders.as_ref(), &mut out, returned, &ty)
            } else {
                let dest = regs.function().alloc_reg();
                regs.function().register_types.insert(dest, ty.clone());
                out.push(IrInstruction::Cast {
                    dest,
                    src: returned,
                    from_ty: IrType::I64,
                    to_ty: ty.clone(),
                });
                dest
            };
            if let Some(block) = regs.function().cfg.blocks.get_mut(&id) {
                block.instructions.extend(out);
                block.terminator = IrTerminator::Return { value: Some(value) };
            }
        }
    }

    /// Slot readers and raw array writes do not require a concrete type binding.
    fn erasure_safe_tag_fixups(
        &self,
        function: &IrFunction,
        name: &str,
        module: &IrModule,
    ) -> bool {
        function
            .type_param_tag_fixups
            .iter()
            .filter(|(_, parameter)| parameter == name)
            .all(|(tag, _)| {
                // A word access's tag carries the element-kind hint only a
                // specialization can supply.
                let word = rayzor_runtime::type_system::ValueTag::Word as i32;
                let word_tag = function.cfg.blocks.values().any(|block| {
                    block.instructions.iter().any(|instruction| {
                        matches!(
                            instruction,
                            IrInstruction::Const { dest, value: IrValue::I32(v) }
                                if dest == tag && v & 0xff == word
                        )
                    })
                });
                !word_tag
                    && function.cfg.blocks.values().any(|block| {
                        block
                            .instructions
                            .iter()
                            .any(|instruction| instruction.uses().contains(tag))
                    })
                    && function.cfg.blocks.values().all(|block| {
                        block.instructions.iter().all(|instruction| {
                            if !instruction.uses().contains(tag) {
                                return true;
                            }
                            let IrInstruction::CallDirect { func_id, args, .. } = instruction
                            else {
                                return false;
                            };
                            let callee = module
                                .extern_functions
                                .get(func_id)
                                .map(|function| function.name.as_str())
                                .or_else(|| {
                                    module
                                        .functions
                                        .get(func_id)
                                        .map(|function| function.name.as_str())
                                });
                            callee.is_some_and(|callee| match callee {
                                "haxe_array_init_element_tag" | "haxe_unbox_generic_return" => {
                                    args.get(1) == Some(tag)
                                }
                                "haxe_array_get_typed" => args.get(2) == Some(tag),
                                "haxe_array_push_typed" => args.get(2) == Some(tag),
                                "haxe_adapt_generic_value" => {
                                    args.get(1) == Some(tag) || args.get(2) == Some(tag)
                                }
                                _ => false,
                            })
                        })
                    })
            })
    }

    /// A specialization's `haxe_box_generic_ptr` value is the i64 word of the
    /// slot: a Float as its bits, a Single first widened to the Float it boxes as.
    fn widen_generic_box_floats(module: &mut IrModule) {
        let boxes: BTreeSet<_> = module
            .extern_functions
            .iter()
            .filter(|(_, function)| function.name == "haxe_box_generic_ptr")
            .map(|(id, _)| *id)
            .collect();
        if boxes.is_empty() {
            return;
        }
        for function in module.functions.values_mut() {
            let mut next = function.next_reg_id;
            for reg in function.register_types.keys() {
                next = next.max(reg.0 + 1);
            }
            let block_ids: Vec<_> = function.cfg.blocks.keys().copied().collect();
            for block_id in block_ids {
                let mut index = 0;
                loop {
                    let Some(block) = function.cfg.blocks.get(&block_id) else {
                        break;
                    };
                    let Some(instruction) = block.instructions.get(index) else {
                        break;
                    };
                    let value = match instruction {
                        IrInstruction::CallDirect { func_id, args, .. }
                            if boxes.contains(func_id) && !args.is_empty() =>
                        {
                            args[0]
                        }
                        _ => {
                            index += 1;
                            continue;
                        }
                    };
                    let mut adapters = Vec::new();
                    let mut word = value;
                    match function.register_types.get(&value) {
                        Some(IrType::F32) => {
                            let wide = IrId::new(next);
                            let bits = IrId::new(next + 1);
                            next += 2;
                            adapters.push(IrInstruction::Cast {
                                dest: wide,
                                src: value,
                                from_ty: IrType::F32,
                                to_ty: IrType::F64,
                            });
                            adapters.push(IrInstruction::BitCast {
                                dest: bits,
                                src: wide,
                                ty: IrType::I64,
                            });
                            function.register_types.insert(wide, IrType::F64);
                            function.register_types.insert(bits, IrType::I64);
                            word = bits;
                        }
                        Some(IrType::F64) => {
                            let bits = IrId::new(next);
                            next += 1;
                            adapters.push(IrInstruction::BitCast {
                                dest: bits,
                                src: value,
                                ty: IrType::I64,
                            });
                            function.register_types.insert(bits, IrType::I64);
                            word = bits;
                        }
                        _ => {}
                    }
                    let inserted = adapters.len();
                    let Some(block) = function.cfg.blocks.get_mut(&block_id) else {
                        break;
                    };
                    if let Some(IrInstruction::CallDirect { args, .. }) =
                        block.instructions.get_mut(index)
                    {
                        args[0] = word;
                    }
                    for (offset, adapter) in adapters.into_iter().enumerate() {
                        block.instructions.insert(index + offset, adapter);
                    }
                    index += inserted + 1;
                }
            }
            function.next_reg_id = function.next_reg_id.max(next);
        }
    }

    /// Concrete specializations retain raw slots without a runtime adapter call.
    fn fold_generic_value_adapters(&self, module: &mut IrModule) {
        let adapters: BTreeSet<_> = module
            .functions
            .iter()
            .map(|(id, function)| (id, &function.name))
            .chain(
                module
                    .extern_functions
                    .iter()
                    .map(|(id, function)| (id, &function.name)),
            )
            .filter(|(_, name)| name.as_str() == "haxe_adapt_generic_value")
            .map(|(id, _)| *id)
            .collect();
        for function in module.functions.values_mut() {
            let unresolved: BTreeSet<_> = function
                .type_param_tag_fixups
                .iter()
                .map(|(reg, _)| *reg)
                .collect();
            let tags: BTreeMap<_, _> = function
                .cfg
                .blocks
                .values()
                .flat_map(|block| &block.instructions)
                .filter_map(|instruction| match instruction {
                    IrInstruction::Const {
                        dest,
                        value: IrValue::I32(tag),
                    } if !unresolved.contains(dest) => Some((*dest, *tag)),
                    _ => None,
                })
                .collect();
            for block in function.cfg.blocks.values_mut() {
                for instruction in &mut block.instructions {
                    let IrInstruction::CallDirect {
                        dest: Some(dest),
                        func_id,
                        args,
                        ..
                    } = instruction
                    else {
                        continue;
                    };
                    if !adapters.contains(func_id) || args.len() != 4 {
                        continue;
                    }
                    let (Some(source), Some(target)) = (tags.get(&args[1]), tags.get(&args[2]))
                    else {
                        continue;
                    };
                    let dynamic = rayzor_runtime::type_system::ValueTag::Dynamic as i32;
                    let word = rayzor_runtime::type_system::ValueTag::Word as i32;
                    if source & 0xff == word || target & 0xff == word {
                        continue;
                    }
                    if source == target || (*source != dynamic && *target != dynamic) {
                        *instruction = IrInstruction::Copy {
                            dest: *dest,
                            src: args[0],
                        };
                    }
                }
            }
        }
    }

    /// Preserve the caller's slot type while reading the specialized return ABI.
    pub(crate) fn adapt_specialized_call_returns(&self, module: &mut IrModule) {
        let returns: BTreeMap<_, _> = self.call_return_types(module);
        let mut parameters: BTreeMap<_, _> = module
            .functions
            .iter()
            .filter(|(_, function)| function.signature.type_params.is_empty())
            .map(|(id, function)| {
                (
                    *id,
                    function
                        .signature
                        .parameters
                        .iter()
                        .map(|p| p.ty.clone())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        for (id, function) in &module.extern_functions {
            parameters.insert(
                *id,
                function
                    .signature
                    .parameters
                    .iter()
                    .map(|p| p.ty.clone())
                    .collect(),
            );
        }
        // Under RAYZOR_ERASED_WORDS an instance's scalar result reaching an
        // erased slot of another instance crosses as its word; a concrete
        // caller reads the slot's raw bits.
        let instances: BTreeSet<IrFunctionId> = if self.word_decoders.is_some() {
            self.instances.values().copied().collect()
        } else {
            BTreeSet::new()
        };
        for function in module.functions.values_mut() {
            let caller_is_instance = instances.contains(&function.id);
            let mut word_results: BTreeMap<IrId, IrType> = BTreeMap::new();
            let erased_float_returns: BTreeSet<_> = function
                .cfg
                .blocks
                .values()
                .flat_map(|block| &block.instructions)
                .filter_map(|instruction| match instruction {
                    IrInstruction::CallDirect {
                        dest: Some(dest),
                        func_id,
                        ..
                    } if returns.get(func_id) == Some(&IrType::F64)
                        && function
                            .register_types
                            .get(dest)
                            .is_some_and(contains_type_param) =>
                    {
                        Some(*dest)
                    }
                    _ => None,
                })
                .collect();
            let mut next = function
                .register_types
                .keys()
                .map(|reg| reg.0 + 1)
                .max()
                .unwrap_or(function.next_reg_id)
                .max(function.next_reg_id);
            for block in function.cfg.blocks.values_mut() {
                let mut instructions = Vec::new();
                for mut instruction in std::mem::take(&mut block.instructions) {
                    if let IrInstruction::CallDirect { func_id, args, .. } = &mut instruction
                        && let Some(params) = parameters.get(func_id)
                    {
                        for (arg, param) in args.iter_mut().zip(params) {
                            let actual = function.register_types.get(arg).cloned();
                            if matches!(param, IrType::I64 | IrType::Ptr(_))
                                && matches!(actual, Some(IrType::F64 | IrType::I32 | IrType::Bool))
                            {
                                let bits = IrId::new(next);
                                next += 1;
                                instructions.push(if actual == Some(IrType::F64) {
                                    IrInstruction::BitCast {
                                        dest: bits,
                                        src: *arg,
                                        ty: param.clone(),
                                    }
                                } else {
                                    IrInstruction::Cast {
                                        dest: bits,
                                        src: *arg,
                                        from_ty: actual.unwrap(),
                                        to_ty: param.clone(),
                                    }
                                });
                                function.register_types.insert(bits, param.clone());
                                *arg = bits;
                            }
                        }
                    }
                    if let IrInstruction::Cast {
                        dest,
                        src,
                        from_ty,
                        to_ty,
                    } = &instruction
                        && contains_type_param(from_ty)
                        && *to_ty == IrType::F64
                        && erased_float_returns.contains(src)
                    {
                        instruction = IrInstruction::BitCast {
                            dest: *dest,
                            src: *src,
                            ty: IrType::F64,
                        };
                    }
                    let conversion = if let IrInstruction::CallDirect {
                        dest: Some(dest),
                        func_id,
                        ..
                    } = &mut instruction
                    {
                        returns.get(func_id).and_then(|returned| {
                            let expected = function.register_types.get(dest)?.clone();
                            let expected = if contains_type_param(&expected) {
                                IrType::I64
                            } else {
                                expected
                            };
                            function.register_types.insert(*dest, expected.clone());
                            if expected == *returned || *returned == IrType::Void {
                                return None;
                            }
                            let source = IrId::new(next);
                            next += 1;
                            let target = *dest;
                            *dest = source;
                            function.register_types.insert(source, returned.clone());
                            if expected == IrType::I64
                                && caller_is_instance
                                && instances.contains(func_id)
                                && super::erased_word::is_word_scalar(returned)
                            {
                                word_results.insert(target, returned.clone());
                            }
                            Some(
                                if matches!(
                                    (returned, &expected),
                                    (IrType::F64, IrType::I64 | IrType::U64)
                                        | (IrType::I64 | IrType::U64, IrType::F64)
                                ) {
                                    IrInstruction::BitCast {
                                        dest: target,
                                        src: source,
                                        ty: expected,
                                    }
                                } else {
                                    IrInstruction::Cast {
                                        dest: target,
                                        src: source,
                                        from_ty: returned.clone(),
                                        to_ty: expected,
                                    }
                                },
                            )
                        })
                    } else {
                        None
                    };
                    instructions.push(instruction);
                    if let Some(conversion) = conversion {
                        instructions.push(conversion);
                    }
                }
                block.instructions = instructions;
            }
            function.next_reg_id = next;
            if !word_results.is_empty() {
                Self::encode_word_results(function, &word_results);
            }
        }
    }

    /// Replace each widening of an instance result in `results` with the
    /// result's word.
    fn encode_word_results(function: &mut IrFunction, results: &BTreeMap<IrId, IrType>) {
        let mut regs = super::erased_word::Regs::new(function);
        let ids: Vec<_> = regs.function().cfg.blocks.keys().copied().collect();
        for id in ids {
            let Some(block) = regs.function().cfg.blocks.get_mut(&id) else {
                continue;
            };
            let old = std::mem::take(&mut block.instructions);
            let mut out = Vec::with_capacity(old.len());
            for instruction in old {
                match &instruction {
                    IrInstruction::BitCast { dest, src, .. }
                    | IrInstruction::Cast { dest, src, .. }
                        if results.contains_key(dest) =>
                    {
                        let (dest, src) = (*dest, *src);
                        let word = regs.encode(&mut out, src, &results[&dest]);
                        out.push(IrInstruction::Copy { dest, src: word });
                    }
                    _ => out.push(instruction),
                }
            }
            if let Some(block) = regs.function().cfg.blocks.get_mut(&id) {
                block.instructions = out;
            }
        }
    }

    fn call_return_types(&self, module: &IrModule) -> BTreeMap<IrFunctionId, IrType> {
        module
            .functions
            .iter()
            .filter(|(_, function)| !contains_type_param(&function.signature.return_type))
            .map(|(id, function)| (*id, function.signature.return_type.clone()))
            .collect()
    }

    /// Recursively substitute type variables with concrete types
    fn substitute_type(&self, ty: &IrType) -> IrType {
        match ty {
            IrType::TypeVar(name) => self
                .substitution_map
                .get(name)
                .map(|bound| {
                    if matches!(bound, IrType::Ptr(inner) if matches!(&**inner, IrType::Opaque { name, .. } if name == "__haxe_int64")) {
                        IrType::I64
                    } else {
                        bound.clone()
                    }
                })
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
            for phi in &mut block.phi_nodes {
                phi.ty = self.substitute_type(&phi.ty);
            }
            for inst in block.instructions.iter_mut() {
                self.substitute_instruction(inst);
            }
            self.substitute_terminator(&mut block.terminator);
        }
    }

    /// Call-site arguments retain semantic descriptors independently of the slot ABI.
    fn substitute_type_argument(&self, ty: &IrType) -> IrType {
        match ty {
            IrType::TypeVar(name) => self
                .substitution_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| ty.clone()),
            IrType::Ptr(inner) => IrType::Ptr(Box::new(self.substitute_type_argument(inner))),
            IrType::Array(inner, count) => {
                IrType::Array(Box::new(self.substitute_type_argument(inner)), *count)
            }
            _ => self.substitute_type(ty),
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
                    *arg = self.substitute_type_argument(arg);
                }
            }
            IrInstruction::CallIndirect { signature, .. } => {
                *signature = self.substitute_type(signature);
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
        if func.type_param_tag_fixups.is_empty() && func.type_param_metadata_fixups.is_empty() {
            return;
        }

        let fixups = std::mem::take(&mut func.type_param_tag_fixups);
        for (reg_id, type_param_name) in &fixups {
            let concrete_type = match self.substitution_map.get(type_param_name) {
                Some(ty) if !contains_type_param(ty) => ty.clone(),
                _ => {
                    func.type_param_tag_fixups
                        .push((*reg_id, type_param_name.clone()));
                    continue;
                }
            };

            let type_tag = concrete_type.value_tag() as i32;

            for block in func.cfg.blocks.values_mut() {
                for inst in block.instructions.iter_mut() {
                    if let IrInstruction::Const { dest, value } = inst {
                        if *dest == *reg_id {
                            *value = super::erased_word::resolve_tag_fixup(value, type_tag);
                        }
                    }
                }
            }
        }
        for (reg, name) in std::mem::take(&mut func.type_param_metadata_fixups) {
            let Some(concrete) = self
                .substitution_map
                .get(&name)
                .filter(|ty| !contains_type_param(ty))
            else {
                func.type_param_metadata_fixups.push((reg, name));
                continue;
            };
            let metadata = concrete.value_metadata();
            for block in func.cfg.blocks.values_mut() {
                for instruction in &mut block.instructions {
                    if let IrInstruction::Const { dest, value } = instruction {
                        if *dest == reg {
                            *value = IrValue::U64(metadata);
                        }
                    }
                }
            }
        }
    }

    /// Map an IrType to a runtime type tag for Reflect.compare_typed.
    /// Tags: 1=Int, 2=Bool, 4=Float, 5=String, 6=Reference/Dynamic
    ///
    /// A string is `String` or `Ptr(String)`; `Ptr(U8)` is the boxed
    /// DynamicValue, which is what a `Null<Int>` type argument lowers to.
    /// Calling either of those the other's tag makes the runtime read a box
    /// header as a HaxeString, or two string addresses as raw integers.
    /// Propagate substitution maps transitively through call chains.
    ///
    /// When a specialized function (e.g., set__String_i32) calls another function
    /// that has unresolved type_param_tag_fixups (e.g., setLoop with fixups for "K"),
    /// create a specialized version of the callee using the caller's substitution map.
    fn propagate_transitive_fixups(&mut self, module: &mut IrModule) {
        // Collect IDs of functions with direct type_param_tag_fixups
        let direct_fixups: BTreeSet<IrFunctionId> = module
            .functions
            .iter()
            .filter(|(_, f)| !f.type_param_tag_fixups.is_empty())
            .map(|(id, _)| *id)
            .collect();

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
        let mut processed: BTreeSet<IrFunctionId> = BTreeSet::new();
        let mut closure_clones: BTreeSet<(IrFunctionId, IrFunctionId)> = BTreeSet::new();

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
                        args,
                        ..
                    } = inst
                    {
                        // Handle calls to functions with fixups OR unmonomorphized
                        // generic functions. Monomorphized clones (created in Phase 3)
                        // may still carry stale erased type_args (e.g., [I64, I64])
                        // from before Phase 5 cleared them. We must propagate regardless
                        // so transitive fixups get correct types.
                        //
                        // Also catch calls to generic functions that still have type_params
                        // but no fixups (e.g., balance() called from setLoop()). These
                        // need monomorphization too, even though they have no fixups.
                        let callee_is_generic = module
                            .functions
                            .get(callee_id)
                            .map_or(false, |f| !f.signature.type_params.is_empty());
                        if funcs_with_fixups.contains(callee_id) || callee_is_generic {
                            if let Some(callee) = module.functions.get(callee_id).cloned() {
                                // A callee with its OWN type parameters
                                // (`StringBuf.add<T>`) binds them from the
                                // arguments it is handed; the caller's `T` is
                                // another type that happens to share the name.
                                let own_map = if callee_is_generic {
                                    self.extract_generic_call(inst, &[*callee_id], &func, module)
                                        .map(|(_, type_args)| {
                                            callee
                                                .signature
                                                .type_params
                                                .iter()
                                                .map(|tp| tp.name.clone())
                                                .zip(type_args)
                                                .collect::<BTreeMap<String, IrType>>()
                                        })
                                } else {
                                    None
                                };
                                let callee_map = match own_map.as_ref() {
                                    Some(map) => map,
                                    None if callee_is_generic && !type_args.is_empty() => continue,
                                    // A concrete callee inherits the caller's bindings
                                    // only as a method of the same class: elsewhere a
                                    // type parameter of the same name is another one.
                                    None if !callee_is_generic && !same_class(&func, &callee) => {
                                        continue;
                                    }
                                    None => &sub_map,
                                };
                                if callee_is_generic
                                    && !callee.signature.type_params.iter().any(|parameter| {
                                        callee_map
                                            .get(&parameter.name)
                                            .is_some_and(|ty| !contains_type_param(ty))
                                    })
                                {
                                    continue;
                                }
                                let (new_id, is_new) =
                                    self.instantiate_with_sub_map(&callee, callee_map);
                                rewrites.push((*block_id, inst_idx, new_id));
                                // Only insert newly created functions into the module
                                if is_new {
                                    // Drain pending funcs created by instantiate_with_sub_map
                                    let pending =
                                        std::mem::take(&mut self.pending_transitive_funcs);
                                    for f in pending {
                                        module.functions.insert(f.id, f);
                                    }
                                    worklist.push(new_id);
                                }
                            }
                        }
                    } else if let IrInstruction::MakeClosure {
                        func_id: target, ..
                    } = inst
                    {
                        if let Some(closure) = module.functions.get(target).cloned()
                            && !closure.signature.type_params.is_empty()
                        {
                            let (new_id, is_new) =
                                self.instantiate_with_sub_map(&closure, &sub_map);
                            rewrites.push((*block_id, inst_idx, new_id));
                            closure_clones.insert((*target, new_id));
                            if is_new {
                                for function in std::mem::take(&mut self.pending_transitive_funcs) {
                                    module.functions.insert(function.id, function);
                                }
                                worklist.push(new_id);
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
                                if let IrInstruction::CallDirect {
                                    func_id, type_args, ..
                                } = inst
                                {
                                    *func_id = *new_callee_id;
                                    type_args.clear(); // Clear stale erased type_args
                                    self.stats.call_sites_rewritten += 1;
                                } else if let IrInstruction::MakeClosure { func_id, .. } = inst {
                                    *func_id = *new_callee_id;
                                    self.stats.call_sites_rewritten += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        self.register_closure_clones(module, &closure_clones);
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
        sub_map: &BTreeMap<String, IrType>,
    ) -> (IrFunctionId, bool) {
        let mut sub_map = sub_map.clone();
        Self::bind_object_constraints(generic_func, &mut sub_map);
        // Declared parameters use the same order as direct instantiation requests.
        let type_args: Vec<IrType> = if generic_func.signature.type_params.is_empty() {
            sub_map.values().cloned().collect()
        } else {
            generic_func
                .signature
                .type_params
                .iter()
                .map(|parameter| {
                    sub_map
                        .get(&parameter.name)
                        .cloned()
                        .unwrap_or_else(|| IrType::TypeVar(parameter.name.clone()))
                })
                .collect()
        };

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

        // Substitute types in signature (params + return type)
        specialized.signature = self.substitute_signature(&specialized.signature);

        // Substitute types in register_types
        let mut new_register_types = BTreeMap::new();
        for (id, ty) in &specialized.register_types {
            new_register_types.insert(*id, self.substitute_type(ty));
        }
        specialized.register_types = new_register_types;

        // A parameter's register holds what its slot holds, and an erased slot
        // stays I64 whatever the type argument is. Substituting the register
        // types alone can put a concrete type on a register the ABI still
        // passes as bits.
        for param in &specialized.signature.parameters {
            specialized
                .register_types
                .insert(param.reg, param.ty.clone());
        }

        // Substitute types in locals
        for (_, local) in specialized.locals.iter_mut() {
            local.ty = self.substitute_type(&local.ty);
        }

        // Substitute types in CFG
        let crossings = self.word_crossings(generic_func);
        self.substitute_cfg(&mut specialized.cfg);
        self.apply_word_crossings(&mut specialized, crossings);

        self.adapt_erased_returns(&mut specialized);

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

    /// Object bounds supply a representation even without a concrete caller binding.
    fn bind_object_constraints(function: &IrFunction, bindings: &mut BTreeMap<String, IrType>) {
        for parameter in &function.signature.type_params {
            if parameter.constraints.iter().any(|bound| bound == "{}") {
                bindings
                    .entry(parameter.name.clone())
                    .or_insert_with(|| IrType::Ptr(Box::new(IrType::Void)));
            }
        }
    }

    /// Point each call at its instance, handing an erased slot the bits.
    ///
    /// A slot the callee declares `I64` takes a float as its BITS -- the body
    /// reads it through the tag this pass resolved. Inlining already coerces
    /// there; a call that survives as a call had nobody to do it, and passing
    /// an f64 register into an integer slot is a register class Cranelift
    /// refuses to emit.
    fn rewrite_call_sites(
        &mut self,
        module: &mut IrModule,
        requests: &BTreeMap<MonoKey, Vec<CallSiteLocation>>,
    ) {
        let mut param_tys: BTreeMap<IrFunctionId, (Vec<IrType>, IrType)> = BTreeMap::new();
        let mut float_bindings = BTreeSet::new();
        let mut erased_params: BTreeMap<IrFunctionId, Vec<usize>> = BTreeMap::new();
        let mut by_block: BTreeMap<(IrFunctionId, IrBlockId), Vec<(usize, IrFunctionId)>> =
            BTreeMap::new();
        for (key, locations) in requests {
            let specialized_id = match self.instances.get(key) {
                Some(id) => *id,
                None => continue,
            };
            if key.type_args == vec![IrType::F64] {
                float_bindings.insert(specialized_id);
            }
            if let Some(f) = module.functions.get(&specialized_id) {
                erased_params.insert(specialized_id, f.erased_params.clone());
                param_tys.entry(specialized_id).or_insert_with(|| {
                    (
                        f.signature
                            .parameters
                            .iter()
                            .map(|p| p.ty.clone())
                            .collect(),
                        f.signature.return_type.clone(),
                    )
                });
            }
            for loc in locations {
                by_block
                    .entry((loc.function_id, loc.block_id))
                    .or_default()
                    .push((loc.instruction_index, specialized_id));
            }
        }

        for ((func_id, block_id), mut sites) in by_block {
            // Descending: inserting a coercion shifts only the sites after it,
            // and those are already done.
            sites.sort_by_key(|a| std::cmp::Reverse(a.0));
            let func = match module.functions.get_mut(&func_id) {
                Some(f) => f,
                None => continue,
            };
            let mut next = func.next_reg_id;
            for reg in func.register_types.keys() {
                next = next.max(reg.0 + 1);
            }
            let widened = if self.word_decoders.is_some() {
                super::erased_word::widened_scalars(func)
            } else {
                BTreeMap::new()
            };
            for (index, specialized_id) in sites {
                let (ptys, _) = match param_tys.get(&specialized_id) {
                    Some(p) => p,
                    None => continue,
                };
                let args: Vec<IrId> = match func
                    .cfg
                    .blocks
                    .get(&block_id)
                    .and_then(|b| b.instructions.get(index))
                {
                    Some(IrInstruction::CallDirect { args, .. }) => args.clone(),
                    _ => continue,
                };
                let mut coercions: Vec<IrInstruction> = Vec::new();
                let mut new_args = args.clone();
                for (i, arg) in args.iter().enumerate() {
                    if ptys.get(i) != Some(&IrType::I64) {
                        continue;
                    }
                    // Under RAYZOR_ERASED_WORDS a scalar reaching an erased slot
                    // crosses as its word: a Float always, as no other i64
                    // parameter takes one; an Int or Bool where the parameter
                    // is declared at a bare type parameter.
                    let declared = erased_params
                        .get(&specialized_id)
                        .is_some_and(|params| params.contains(&i));
                    // A scalar the lowering already widened to the slot is
                    // encoded from its source.
                    let source = match func.register_types.get(arg) {
                        Some(ty) if super::erased_word::is_word_scalar(ty) => {
                            Some((*arg, ty.clone()))
                        }
                        _ => widened.get(arg).cloned(),
                    };
                    let word_arg = match source.as_ref().map(|(_, ty)| ty) {
                        Some(IrType::F64 | IrType::F32) => true,
                        Some(IrType::I32) => declared || float_bindings.contains(&specialized_id),
                        Some(IrType::Bool) => declared,
                        _ => false,
                    };
                    // A box reaching a declared slot crosses as its Dynamic
                    // view: a scalar box opens, a reference is itself.
                    if declared
                        && matches!(func.register_types.get(arg), Some(IrType::Ptr(_)))
                        && let Some(decoders) = self.word_decoders.as_ref()
                    {
                        func.next_reg_id = func.next_reg_id.max(next);
                        let mut regs = super::erased_word::Regs::new(func);
                        new_args[i] = regs.encode_dynamic(Some(decoders), &mut coercions, *arg);
                        next = regs.function().next_reg_id;
                        continue;
                    }
                    if self.word_decoders.is_some()
                        && word_arg
                        && let Some((mut src, mut ty)) = source
                    {
                        if float_bindings.contains(&specialized_id) && ty == IrType::I32 {
                            let wide = IrId::new(next);
                            next += 1;
                            coercions.push(IrInstruction::Cast {
                                dest: wide,
                                src,
                                from_ty: IrType::I32,
                                to_ty: IrType::F64,
                            });
                            func.register_types.insert(wide, IrType::F64);
                            (src, ty) = (wide, IrType::F64);
                        }
                        func.next_reg_id = func.next_reg_id.max(next);
                        let mut regs = super::erased_word::Regs::new(func);
                        new_args[i] = regs.encode(&mut coercions, src, &ty);
                        next = regs.function().next_reg_id;
                        continue;
                    }
                    if float_bindings.contains(&specialized_id)
                        && func.register_types.get(arg) == Some(&IrType::I32)
                    {
                        let wide = IrId::new(next);
                        let bits = IrId::new(next + 1);
                        next += 2;
                        coercions.push(IrInstruction::Cast {
                            dest: wide,
                            src: *arg,
                            from_ty: IrType::I32,
                            to_ty: IrType::F64,
                        });
                        coercions.push(IrInstruction::BitCast {
                            dest: bits,
                            src: wide,
                            ty: IrType::I64,
                        });
                        func.register_types.insert(wide, IrType::F64);
                        func.register_types.insert(bits, IrType::I64);
                        new_args[i] = bits;
                        continue;
                    }
                    match func.register_types.get(arg) {
                        Some(IrType::F64) => {
                            let bits = IrId::new(next);
                            next += 1;
                            coercions.push(IrInstruction::BitCast {
                                dest: bits,
                                src: *arg,
                                ty: IrType::I64,
                            });
                            func.register_types.insert(bits, IrType::I64);
                            new_args[i] = bits;
                        }
                        Some(IrType::F32) => {
                            let wide = IrId::new(next);
                            next += 1;
                            let bits = IrId::new(next);
                            next += 1;
                            coercions.push(IrInstruction::Cast {
                                dest: wide,
                                src: *arg,
                                from_ty: IrType::F32,
                                to_ty: IrType::F64,
                            });
                            coercions.push(IrInstruction::BitCast {
                                dest: bits,
                                src: wide,
                                ty: IrType::I64,
                            });
                            func.register_types.insert(wide, IrType::F64);
                            func.register_types.insert(bits, IrType::I64);
                            new_args[i] = bits;
                        }
                        _ => {}
                    }
                }
                let block = match func.cfg.blocks.get_mut(&block_id) {
                    Some(b) => b,
                    None => continue,
                };
                if let Some(IrInstruction::CallDirect {
                    func_id: callee,
                    type_args,
                    args,
                    ..
                }) = block.instructions.get_mut(index)
                {
                    *callee = specialized_id;
                    type_args.clear();
                    *args = new_args;
                    self.stats.call_sites_rewritten += 1;
                }
                for (k, inst) in coercions.into_iter().enumerate() {
                    block.instructions.insert(index + k, inst);
                }
            }
            func.next_reg_id = next;
        }
    }
}

impl Default for Monomorphizer {
    fn default() -> Self {
        Self::new()
    }
}

fn contains_type_param(ty: &IrType) -> bool {
    match ty {
        IrType::TypeVar(_) => true,
        IrType::Ptr(inner)
        | IrType::Ref(inner)
        | IrType::Array(inner, _)
        | IrType::Slice(inner) => contains_type_param(inner),
        IrType::Vector { element, .. } => contains_type_param(element),
        IrType::Function {
            params,
            return_type,
            ..
        } => params.iter().any(contains_type_param) || contains_type_param(return_type),
        IrType::Struct { fields, .. } => fields.iter().any(|field| contains_type_param(&field.ty)),
        IrType::Union { variants, .. } => variants
            .iter()
            .any(|variant| variant.fields.iter().any(contains_type_param)),
        IrType::Generic { base, type_args } => {
            contains_type_param(base) || type_args.iter().any(contains_type_param)
        }
        _ => false,
    }
}

/// Whether a generic function's body still depends on its type parameters.
///
/// A template is stubbed because its body cannot be compiled until the
/// parameters are known. That only holds while something in it still refers
/// to them: a body whose values all travel in concrete slots is already
/// monomorphic, whatever its declaration says, and compiling it is what lets
/// a caller reach it at all.
pub fn references_type_params(function: &IrFunction) -> bool {
    !function.type_param_tag_fixups.is_empty() || mentions_type_vars(function)
}

/// Whether the signature or a local is typed by a type parameter.
fn mentions_type_vars(function: &IrFunction) -> bool {
    fn is_var(ty: &IrType) -> bool {
        match ty {
            IrType::TypeVar(_) => true,
            IrType::Ptr(inner) => is_var(inner),
            IrType::Function {
                params,
                return_type,
                ..
            } => params.iter().any(is_var) || is_var(return_type),
            _ => false,
        }
    }
    function.signature.parameters.iter().any(|p| is_var(&p.ty))
        || is_var(&function.signature.return_type)
        || function.locals.values().any(|l| is_var(&l.ty))
}

/// Under word slots, a template tied to its type parameters only by tags that
/// read a word runs unspecialized: a word describes itself. A class's
/// interface thunk calls such a template, as no binding reaches it.
pub fn template_runs_erased(function: &IrFunction) -> bool {
    if !crate::debug_flags::erased_words()
        || function.type_param_tag_fixups.is_empty()
        || mentions_type_vars(function)
    {
        return false;
    }
    let word = rayzor_runtime::type_system::ValueTag::Word as i32;
    function.type_param_tag_fixups.iter().all(|(tag, _)| {
        function.cfg.blocks.values().any(|block| {
            block.instructions.iter().any(|instruction| {
                matches!(
                    instruction,
                    IrInstruction::Const { dest, value: IrValue::I32(v) }
                        if dest == tag && v & 0xff == word
                )
            })
        })
    })
}

impl Monomorphizer {
    /// Register each specialized closure target. A slot or word caller goes
    /// through entries built from the clone's own signature; a Dynamic or
    /// packed caller through the template's, as the clone captures what the
    /// template captures and the template's body takes erased values.
    fn register_closure_clones(
        &mut self,
        module: &mut IrModule,
        clones: &BTreeSet<(IrFunctionId, IrFunctionId)>,
    ) {
        if clones.is_empty() {
            return;
        }
        let registrar = |name: &str| {
            module
                .functions
                .values()
                .map(|f| (f.id, f.name.as_str()))
                .chain(
                    module
                        .extern_functions
                        .values()
                        .map(|f| (f.id, f.name.as_str())),
                )
                .find(|(_, n)| *n == name)
                .map(|(id, _)| id)
        };
        let register_entries = registrar("haxe_closure_register_entries");
        let register_word = registrar("haxe_closure_register_word_entry");
        let register_packed = registrar("haxe_closure_register_packed_entry");
        let mut by_template: BTreeMap<IrFunctionId, Vec<ClonedEntries>> = BTreeMap::new();
        for (template, clone) in clones {
            if template == clone {
                continue;
            }
            let slot = self.build_clone_entry(module, *clone, false);
            let word = match slot {
                Some(_) if self.word_decoders.is_some() => {
                    self.build_clone_entry(module, *clone, true)
                }
                _ => None,
            };
            by_template
                .entry(*template)
                .or_default()
                .push(ClonedEntries {
                    clone: *clone,
                    slot,
                    word,
                });
        }
        let mut registered: BTreeSet<IrFunctionId> = BTreeSet::new();
        for function in module.functions.values_mut() {
            let mut regs = super::erased_word::Regs::new(function);
            let blocks: Vec<IrBlockId> = regs.function().cfg.blocks.keys().copied().collect();
            for block_id in blocks {
                let instructions = std::mem::take(
                    &mut regs
                        .function()
                        .cfg
                        .blocks
                        .get_mut(&block_id)
                        .unwrap()
                        .instructions,
                );
                // A registrar takes each record as `*u8`, bitcast from the ref.
                let mut refs: BTreeMap<IrId, (IrFunctionId, Option<IrType>)> = BTreeMap::new();
                let mut out = Vec::with_capacity(instructions.len());
                for inst in instructions {
                    match &inst {
                        IrInstruction::FunctionRef { dest, func_id } => {
                            refs.insert(*dest, (*func_id, None));
                        }
                        IrInstruction::BitCast { dest, src, ty } => {
                            if let Some((func_id, None)) = refs.get(src).cloned() {
                                refs.insert(*dest, (func_id, Some(ty.clone())));
                            }
                        }
                        _ => {}
                    }
                    let found = match &inst {
                        IrInstruction::CallDirect { func_id, args, .. } => args
                            .first()
                            .and_then(|target| refs.get(target))
                            .and_then(|(template, cast)| {
                                Some((*func_id, by_template.get(template)?, cast.clone()))
                            }),
                        _ => None,
                    };
                    out.push(inst.clone());
                    let Some((registrar, cloned, cast)) = found else {
                        continue;
                    };
                    for entries in cloned {
                        let mut record = |out: &mut Vec<IrInstruction>, f: Option<IrFunctionId>| {
                            let raw = match f {
                                Some(func_id) => {
                                    let dest = regs.fresh(IrType::I64);
                                    out.push(IrInstruction::FunctionRef { dest, func_id });
                                    dest
                                }
                                None => {
                                    let dest = regs.fresh(IrType::Ptr(Box::new(IrType::Void)));
                                    out.push(IrInstruction::Const {
                                        dest,
                                        value: IrValue::Null,
                                    });
                                    dest
                                }
                            };
                            match &cast {
                                Some(ty) => regs.bitcast(out, raw, ty),
                                None => raw,
                            }
                        };
                        let IrInstruction::CallDirect { args, .. } = &inst else {
                            continue;
                        };
                        let new_args = if Some(registrar) == register_entries {
                            registered.insert(entries.clone);
                            let target = record(&mut out, Some(entries.clone));
                            let slot = record(&mut out, entries.slot);
                            vec![target, slot, args[2]]
                        } else if Some(registrar) == register_word {
                            let Some(word) = entries.word else {
                                continue;
                            };
                            let target = record(&mut out, Some(entries.clone));
                            let word = record(&mut out, Some(word));
                            vec![target, word]
                        } else if Some(registrar) == register_packed {
                            let target = record(&mut out, Some(entries.clone));
                            vec![target, args[1]]
                        } else {
                            continue;
                        };
                        let mut call = inst.clone();
                        if let IrInstruction::CallDirect { args, .. } = &mut call {
                            *args = new_args;
                        }
                        out.push(call);
                    }
                }
                regs.function()
                    .cfg
                    .blocks
                    .get_mut(&block_id)
                    .unwrap()
                    .instructions = out;
            }
        }
        // A template registered by another module's init leaves nothing here
        // to register beside; the clone registers its own entries from this
        // module's.
        let unregistered: Vec<ClonedEntries> = by_template
            .into_values()
            .flatten()
            .filter(|entries| !registered.contains(&entries.clone) && entries.slot.is_some())
            .collect();
        if !unregistered.is_empty() {
            self.register_in_module_init(module, &unregistered);
        }
    }

    /// Register `clones`' slot and word entries from this module's vtable
    /// init, with no dynamic entry.
    fn register_in_module_init(&mut self, module: &mut IrModule, clones: &[ClonedEntries]) {
        let Some(init) = module
            .functions
            .values()
            .find(|f| f.name == "__vtable_init__" && !f.cfg.blocks.is_empty())
            .map(|f| f.id)
        else {
            panic!("no __vtable_init__ to register specialized closure entries in");
        };
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        module.next_function_id = module.next_function_id.max(self.next_func_id);
        let register_entries = super::erased_word::extern_fn_with(
            module,
            "haxe_closure_register_entries",
            vec![ptr_u8.clone(), ptr_u8.clone(), ptr_u8.clone()],
            IrType::Void,
        );
        let register_word = super::erased_word::extern_fn_with(
            module,
            "haxe_closure_register_word_entry",
            vec![ptr_u8.clone(), ptr_u8.clone()],
            IrType::Void,
        );
        self.next_func_id = self.next_func_id.max(module.next_function_id);
        let function = module.functions.get_mut(&init).expect("init exists");
        let exits: Vec<IrBlockId> = function
            .cfg
            .blocks
            .iter()
            .filter(|(_, b)| matches!(b.terminator, IrTerminator::Return { .. }))
            .map(|(id, _)| *id)
            .collect();
        let Some(exit) = exits.first().copied() else {
            panic!("__vtable_init__ has no return");
        };
        let mut regs = super::erased_word::Regs::new(function);
        let mut out = Vec::new();
        let call = |func_id: IrFunctionId, args: Vec<IrId>| IrInstruction::CallDirect {
            dest: None,
            func_id,
            arg_ownership: vec![super::OwnershipMode::Copy; args.len()],
            args,
            type_args: vec![],
            is_tail_call: false,
        };
        for entries in clones {
            let mut record = |out: &mut Vec<IrInstruction>, f: Option<IrFunctionId>| {
                let raw = match f {
                    Some(func_id) => {
                        let dest = regs.fresh(IrType::I64);
                        out.push(IrInstruction::FunctionRef { dest, func_id });
                        dest
                    }
                    None => {
                        let dest = regs.fresh(IrType::Ptr(Box::new(IrType::Void)));
                        out.push(IrInstruction::Const {
                            dest,
                            value: IrValue::Null,
                        });
                        dest
                    }
                };
                regs.bitcast(out, raw, &ptr_u8)
            };
            let target = record(&mut out, Some(entries.clone));
            let slot = record(&mut out, entries.slot);
            let dynamic = record(&mut out, None);
            out.push(call(register_entries, vec![target, slot, dynamic]));
            if let Some(word) = entries.word {
                let target = record(&mut out, Some(entries.clone));
                let word = record(&mut out, Some(word));
                out.push(call(register_word, vec![target, word]));
            }
        }
        let block = regs
            .function()
            .cfg
            .blocks
            .get_mut(&exit)
            .expect("exit block");
        block.instructions.extend(out);
    }

    /// The slot- or word-shaped entry of a specialized closure target,
    /// `(env, i64..) -> i64`, converting through the clone's own parameter
    /// and result types. None when every one already is a slot.
    fn build_clone_entry(
        &mut self,
        module: &mut IrModule,
        clone_id: IrFunctionId,
        word: bool,
    ) -> Option<IrFunctionId> {
        use super::erased_word::is_word_scalar;
        let target = module.functions.get(&clone_id)?;
        let sig = target.signature.clone();
        let name: String = target
            .name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let env_ty = sig.parameters.first()?.ty.clone();
        let declared: Vec<IrType> = sig.parameters[1..].iter().map(|p| p.ty.clone()).collect();
        let is_slot = |ty: &IrType| matches!(ty, IrType::I64 | IrType::Ptr(_) | IrType::Void);
        if declared.iter().all(is_slot) && is_slot(&sig.return_type) {
            return None;
        }
        let null_int_from_slot = declared.contains(&IrType::NullInt).then(|| {
            module.next_function_id = module.next_function_id.max(self.next_func_id);
            let id = super::erased_word::extern_fn(
                module,
                "haxe_null_int_from_slot",
                IrType::I64,
                IrType::NullInt,
            );
            self.next_func_id = self.next_func_id.max(module.next_function_id);
            id
        });
        let mut open = |name: &str| {
            (word && declared.iter().any(|t| matches!(t, IrType::Ptr(_)))).then(|| {
                module.next_function_id = module.next_function_id.max(self.next_func_id);
                let id = super::erased_word::extern_fn(
                    module,
                    name,
                    IrType::I64,
                    IrType::Ptr(Box::new(IrType::U8)),
                );
                self.next_func_id = self.next_func_id.max(module.next_function_id);
                id
            })
        };
        let open_reference = open("haxe_word_open_reference");
        let open_interface = open("haxe_word_open_interface");
        let id = IrFunctionId(self.next_func_id);
        self.next_func_id += 1;
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let slot_param = |name: String, ty: IrType| IrParameter {
            name,
            ty,
            reg: IrId(0),
            by_ref: false,
        };
        let mut parameters = vec![slot_param("env".into(), ptr_u8.clone())];
        parameters.extend((0..declared.len()).map(|i| slot_param(format!("a{i}"), IrType::I64)));
        let signature = IrFunctionSignature {
            parameters,
            return_type: IrType::I64,
            calling_convention: super::CallingConvention::Haxe,
            can_throw: sig.can_throw,
            type_params: vec![],
            uses_sret: false,
        };
        let shape = if word { "word" } else { "slot" };
        let mut entry = IrFunction::new(
            id,
            crate::tast::SymbolId::from_raw(0),
            format!("__closure_{shape}_entry__{name}"),
            signature,
        );
        let inputs: Vec<IrId> = entry.signature.parameters.iter().map(|p| p.reg).collect();
        let decoders = self.word_decoders;
        let mut out = Vec::new();
        let mut regs = super::erased_word::Regs::new(&mut entry);
        let mut args = vec![regs.cast(&mut out, inputs[0], &ptr_u8, &env_ty)];
        for (input, ty) in inputs[1..].iter().zip(&declared) {
            args.push(match ty {
                IrType::NullInt => regs.call(&mut out, null_int_from_slot?, *input, ty),
                _ if word && (is_word_scalar(ty) || *ty == IrType::dynamic_ptr()) => {
                    regs.decode(decoders.as_ref(), &mut out, *input, ty)
                }
                // A reference filled through a Dynamic view holds the
                // instance's box; the parameter takes the instance.
                IrType::Ptr(inner) if word => {
                    let interface = matches!(&**inner,
                        IrType::Opaque { name, .. } if name == "__haxe_interface");
                    let open = if interface {
                        open_interface?
                    } else {
                        open_reference?
                    };
                    let instance = regs.call(&mut out, open, *input, &ptr_u8);
                    regs.cast(&mut out, instance, &ptr_u8, ty)
                }
                _ => regs.from_slot(&mut out, *input, ty),
            });
        }
        let ret = sig.return_type.clone();
        let dest = (ret != IrType::Void).then(|| regs.fresh(ret.clone()));
        out.push(IrInstruction::CallDirect {
            dest,
            func_id: clone_id,
            arg_ownership: vec![super::OwnershipMode::Copy; args.len()],
            args,
            type_args: vec![],
            is_tail_call: false,
        });
        let result = match dest {
            None => regs.constant(&mut out, 0),
            Some(r) if word && is_word_scalar(&ret) => regs.encode(&mut out, r, &ret),
            Some(r) if word && ret == IrType::dynamic_ptr() => {
                regs.encode_dynamic(decoders.as_ref(), &mut out, r)
            }
            Some(r) => regs.to_slot(&mut out, r, &ret),
        };
        let block = entry.cfg.entry_block;
        let block = entry.cfg.blocks.get_mut(&block)?;
        block.instructions = out;
        block.terminator = IrTerminator::Return {
            value: Some(result),
        };
        module.functions.insert(id, entry);
        Some(id)
    }
}

impl Monomorphizer {
    /// Encode each scalar a call still aimed at a generic template passes to
    /// one of its declared erased parameters, and open the reference box a
    /// template's T result may be where the caller takes it as a reference.
    /// A call whose type arguments were not recovered (a callee from another
    /// module) keeps the template, and nothing else crosses its words.
    fn encode_template_call_args(&mut self, module: &mut IrModule) {
        if self.word_decoders.is_none() {
            return;
        }
        let erased: BTreeMap<IrFunctionId, Vec<usize>> = module
            .functions
            .iter()
            .filter(|(_, f)| !f.signature.type_params.is_empty() && !f.erased_params.is_empty())
            .map(|(id, f)| (*id, f.erased_params.clone()))
            .collect();
        let word_results: BTreeSet<IrFunctionId> = module
            .functions
            .iter()
            .filter(|(_, f)| {
                !f.signature.type_params.is_empty()
                    && matches!(f.signature.return_type, IrType::TypeVar(_))
            })
            .map(|(id, _)| *id)
            .collect();
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let mut open = |name: &str| {
            module.next_function_id = module.next_function_id.max(self.next_func_id);
            let id = super::erased_word::extern_fn(module, name, IrType::I64, ptr_u8.clone());
            self.next_func_id = self.next_func_id.max(module.next_function_id);
            id
        };
        let opens = (
            open("haxe_word_open_reference"),
            open("haxe_word_open_interface"),
        );
        // A `*u8` is the runtime's untyped view, which reads a box as Dynamic.
        let opener = |ty: &IrType, (reference, interface): (IrFunctionId, IrFunctionId)| match ty {
            IrType::Ptr(inner) if *ty != IrType::dynamic_ptr() && **inner != IrType::U8 => Some(
                if matches!(&**inner, IrType::Opaque { name, .. } if name == "__haxe_interface") {
                    interface
                } else {
                    reference
                },
            ),
            _ => None,
        };
        for function in module.functions.values_mut() {
            let concrete = function.signature.type_params.is_empty();
            let widened = super::erased_word::widened_scalars(function);
            let blocks: Vec<IrBlockId> = function.cfg.blocks.keys().copied().collect();
            for block_id in blocks {
                let instructions = std::mem::take(
                    &mut function.cfg.blocks.get_mut(&block_id).unwrap().instructions,
                );
                let mut out = Vec::with_capacity(instructions.len());
                for mut inst in instructions {
                    // A template's T result taken at a reference type may be
                    // the instance's Dynamic box; the caller takes the instance.
                    // So does a T word a concrete function casts to one, as
                    // an inlined template's result.
                    if concrete
                        && let IrInstruction::Cast {
                            dest,
                            src,
                            to_ty: ty,
                            ..
                        }
                        | IrInstruction::BitCast { dest, src, ty } = &inst
                        && matches!(function.register_types.get(src), Some(IrType::TypeVar(_)))
                        && let Some(open) = opener(ty, opens)
                    {
                        let (dest, src, ty) = (*dest, *src, ty.clone());
                        let mut regs = super::erased_word::Regs::new(function);
                        let instance = regs.call(&mut out, open, src, &ptr_u8);
                        out.push(IrInstruction::Cast {
                            dest,
                            src: instance,
                            from_ty: ptr_u8.clone(),
                            to_ty: ty,
                        });
                        continue;
                    }
                    let mut opened = None;
                    if let IrInstruction::CallDirect {
                        dest: Some(dest),
                        func_id,
                        ..
                    } = &mut inst
                        && word_results.contains(func_id)
                        && let Some(ty) = function.register_types.get(dest).cloned()
                        && let Some(open) = opener(&ty, opens)
                    {
                        let word = super::erased_word::Regs::new(function).fresh(IrType::I64);
                        opened = Some((*dest, word, ty, open));
                        *dest = word;
                    }
                    if let IrInstruction::CallDirect { func_id, args, .. } = &mut inst
                        && let Some(positions) = erased.get(func_id)
                    {
                        for &i in positions {
                            let Some(arg) = args.get(i).copied() else {
                                continue;
                            };
                            let source = match function.register_types.get(&arg) {
                                Some(ty) if super::erased_word::is_word_scalar(ty) => {
                                    Some((arg, ty.clone()))
                                }
                                _ => widened.get(&arg).cloned(),
                            };
                            if let Some((src, ty)) = source {
                                let mut regs = super::erased_word::Regs::new(function);
                                args[i] = regs.encode(&mut out, src, &ty);
                            }
                        }
                    }
                    out.push(inst);
                    if let Some((dest, word, ty, open)) = opened {
                        let mut regs = super::erased_word::Regs::new(function);
                        let instance = regs.call(&mut out, open, word, &ptr_u8);
                        out.push(IrInstruction::Cast {
                            dest,
                            src: instance,
                            from_ty: ptr_u8.clone(),
                            to_ty: ty,
                        });
                    }
                }
                function.cfg.blocks.get_mut(&block_id).unwrap().instructions = out;
            }
        }
    }
}

/// Whether two functions are methods of one class, by qualified name.
fn same_class(a: &IrFunction, b: &IrFunction) -> bool {
    let class = |f: &IrFunction| {
        f.qualified_name
            .as_deref()
            .and_then(|q| q.rsplit_once('.'))
            .map(|(class, _)| class.to_string())
    };
    matches!((class(a), class(b)), (Some(x), Some(y)) if x == y)
}

/// The entries a specialized closure target registers.
struct ClonedEntries {
    clone: IrFunctionId,
    slot: Option<IrFunctionId>,
    word: Option<IrFunctionId>,
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

    fn two_parameter_template() -> IrFunction {
        let signature = IrFunctionSignature {
            parameters: Vec::new(),
            return_type: IrType::TypeVar("T".into()),
            calling_convention: crate::ir::CallingConvention::Haxe,
            can_throw: false,
            type_params: ["T", "C"]
                .into_iter()
                .map(|name| crate::ir::IrTypeParam {
                    name: name.into(),
                    constraints: Vec::new(),
                })
                .collect(),
            uses_sret: false,
        };
        let mut function = IrFunction::new(
            IrFunctionId(1),
            crate::tast::SymbolId::from_raw(1),
            "evaluate".into(),
            signature,
        );
        let tag = function.alloc_reg();
        function.register_types.insert(tag, IrType::I32);
        function.type_param_tag_fixups.push((tag, "C".into()));
        function.type_param_metadata_fixups.push((tag, "C".into()));
        function
    }

    /// A template whose only type-parameter tie is the tag at `tag_value`.
    fn tagged_template(tag_value: i32) -> IrFunction {
        let mut function = two_parameter_template();
        function.signature.return_type = IrType::Void;
        let tag = function.type_param_tag_fixups[0].0;
        let entry = function.cfg.entry_block;
        let block = function.cfg.blocks.get_mut(&entry).unwrap();
        block.instructions.push(IrInstruction::Const {
            dest: tag,
            value: IrValue::I32(tag_value),
        });
        function
    }

    #[test]
    fn word_tagged_template_runs_erased() {
        if !crate::debug_flags::erased_words() {
            return;
        }
        let word = rayzor_runtime::type_system::ValueTag::Word as i32;
        assert!(template_runs_erased(&tagged_template(word)));
        let int = rayzor_runtime::type_system::ValueTag::Int as i32;
        assert!(!template_runs_erased(&tagged_template(int)));
        assert!(!template_runs_erased(&two_parameter_template()));
    }

    #[test]
    fn transitive_cache_uses_declared_parameter_order() {
        let template = two_parameter_template();
        let mut mono = Monomorphizer::new();
        let direct = mono.instantiate(&template, &[IrType::F64, IrType::Bool]);
        let bindings = BTreeMap::from([("T".into(), IrType::F64), ("C".into(), IrType::Bool)]);
        let (transitive, is_new) = mono.instantiate_with_sub_map(&template, &bindings);
        assert_eq!(transitive, direct.id);
        assert!(!is_new);
    }

    #[test]
    fn partial_specialization_keeps_unbound_parameters_and_fixups() {
        let template = two_parameter_template();
        let mut mono = Monomorphizer::new();
        mono.instantiate_with_sub_map(&template, &BTreeMap::from([("T".into(), IrType::Bool)]));
        let instance = mono.pending_transitive_funcs.pop().unwrap();
        assert_eq!(instance.signature.return_type, IrType::Bool);
        assert_eq!(instance.signature.type_params.len(), 1);
        assert_eq!(instance.signature.type_params[0].name, "C");
        assert_eq!(
            instance.type_param_tag_fixups,
            template.type_param_tag_fixups
        );
        assert_eq!(
            instance.type_param_metadata_fixups,
            template.type_param_metadata_fixups
        );
        assert!(references_type_params(&instance));
    }

    #[test]
    fn partial_specialization_resolves_object_bounded_parameters() {
        let mut template = two_parameter_template();
        template.signature.type_params[1]
            .constraints
            .push("{}".into());
        let mut mono = Monomorphizer::new();
        mono.instantiate_with_sub_map(&template, &BTreeMap::from([("T".into(), IrType::Bool)]));
        let instance = mono.pending_transitive_funcs.pop().unwrap();
        assert_eq!(instance.signature.return_type, IrType::Bool);
        assert!(instance.signature.type_params.is_empty());
        assert!(instance.type_param_tag_fixups.is_empty());
        assert!(instance.type_param_metadata_fixups.is_empty());
        assert!(!references_type_params(&instance));
        assert_eq!(
            mono.instantiation_sub_maps[&instance.id]["C"],
            IrType::Ptr(Box::new(IrType::Void))
        );
    }

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

    /// `make<T>(e:Enum<T>):T` (the enum lowers to an i32 id) and a caller
    /// passing it with no type arguments; returns the callee the call targets.
    fn call_result_only_generic(tag_fixup: bool) -> (IrModule, IrFunctionId) {
        use crate::ir::{CallingConvention, IrTypeParam, OwnershipMode};
        use crate::tast::SymbolId;
        let signature = |ret: IrType, type_params: Vec<IrTypeParam>| IrFunctionSignature {
            parameters: vec![IrParameter {
                name: "e".to_string(),
                ty: IrType::I32,
                reg: IrId::new(0),
                by_ref: false,
            }],
            return_type: ret,
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params,
            uses_sret: false,
        };
        let t = IrTypeParam {
            name: "T".to_string(),
            constraints: Vec::new(),
        };
        let mut template = IrFunction::new(
            IrFunctionId(0),
            SymbolId::from_raw(1),
            "make".to_string(),
            signature(IrType::TypeVar("T".to_string()), vec![t]),
        );
        if tag_fixup {
            let tag = template.alloc_reg();
            template.type_param_tag_fixups.push((tag, "T".to_string()));
        }
        let mut caller = IrFunction::new(
            IrFunctionId(1),
            SymbolId::from_raw(2),
            "caller".to_string(),
            signature(IrType::Ptr(Box::new(IrType::Void)), Vec::new()),
        );
        let arg = caller.signature.parameters[0].reg;
        let dest = caller.alloc_reg();
        caller
            .register_types
            .insert(dest, IrType::TypeVar("T".to_string()));
        let entry = caller.cfg.entry_block;
        caller
            .cfg
            .blocks
            .get_mut(&entry)
            .unwrap()
            .instructions
            .push(IrInstruction::CallDirect {
                dest: Some(dest),
                func_id: IrFunctionId(0),
                args: vec![arg],
                arg_ownership: vec![OwnershipMode::Copy],
                type_args: Vec::new(),
                is_tail_call: false,
            });
        let mut module = IrModule::new("m".to_string(), "m.hx".to_string());
        module.add_function(template);
        module.add_function(caller);
        Monomorphizer::new().monomorphize_module(&mut module);
        let caller = &module.functions[&IrFunctionId(1)];
        let callee = match &caller.cfg.blocks[&caller.cfg.entry_block].instructions[0] {
            IrInstruction::CallDirect { func_id, .. } => *func_id,
            other => panic!("expected a direct call, got {other:?}"),
        };
        (module, callee)
    }

    #[test]
    fn unbound_variable_specialises_erased() {
        let (module, callee) = call_result_only_generic(false);
        assert_ne!(callee, IrFunctionId(0));
        let instance = &module.functions[&callee];
        assert!(instance.signature.type_params.is_empty());
        assert_eq!(instance.signature.return_type, IrType::I64);
        assert!(!references_type_params(instance));
    }

    #[test]
    fn unbound_variable_with_a_tag_stays_on_the_template() {
        let (_, callee) = call_result_only_generic(true);
        assert_eq!(callee, IrFunctionId(0));
    }

    #[test]
    fn generic_call_binds_after_modules_are_merged() {
        use crate::ir::{CallingConvention, IrTypeParam};
        use crate::tast::SymbolId;

        let signature = |name: Option<&str>| IrFunctionSignature {
            parameters: Vec::new(),
            return_type: IrType::I32,
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params: name
                .into_iter()
                .map(|name| IrTypeParam {
                    name: name.into(),
                    constraints: Vec::new(),
                })
                .collect(),
            uses_sret: false,
        };
        let call = |dest, func_id, ty| IrInstruction::CallDirect {
            dest: Some(dest),
            func_id,
            args: Vec::new(),
            arg_ownership: Vec::new(),
            type_args: vec![ty],
            is_tail_call: false,
        };
        let mut leaf = IrFunction::new(
            IrFunctionId(0),
            SymbolId::from_raw(1),
            "tag".into(),
            signature(Some("T")),
        );
        let tag = leaf.alloc_reg();
        leaf.register_types.insert(tag, IrType::I32);
        leaf.type_param_tag_fixups.push((tag, "T".into()));
        let block = leaf.cfg.blocks.get_mut(&leaf.cfg.entry_block).unwrap();
        block.instructions.push(IrInstruction::Const {
            dest: tag,
            value: IrValue::I32(0),
        });
        block.terminator = IrTerminator::Return { value: Some(tag) };
        let mut relay = IrFunction::new(
            IrFunctionId(1),
            SymbolId::from_raw(2),
            "relay".into(),
            signature(Some("U")),
        );
        let result = relay.alloc_reg();
        relay.register_types.insert(result, IrType::I32);
        let block = relay.cfg.blocks.get_mut(&relay.cfg.entry_block).unwrap();
        block
            .instructions
            .push(call(result, leaf.id, IrType::TypeVar("U".into())));
        block.terminator = IrTerminator::Return {
            value: Some(result),
        };

        let mut module = IrModule::new("imports".into(), "imports.hx".into());
        module.add_function(leaf);
        module.add_function(relay);
        Monomorphizer::new().monomorphize_module(&mut module);
        assert_eq!(module.functions.len(), 2);

        let mut entry = IrFunction::new(
            module.alloc_function_id(),
            SymbolId::from_raw(3),
            "entry".into(),
            signature(None),
        );
        let entry_id = entry.id;
        let result = entry.alloc_reg();
        entry.register_types.insert(result, IrType::I32);
        let block = entry.cfg.blocks.get_mut(&entry.cfg.entry_block).unwrap();
        block
            .instructions
            .push(call(result, IrFunctionId(1), IrType::enum_ptr()));
        block.terminator = IrTerminator::Return {
            value: Some(result),
        };
        module.add_function(entry);
        Monomorphizer::new().monomorphize_module(&mut module);

        let callee = |id| {
            let function = &module.functions[&id];
            match &function.cfg.blocks[&function.cfg.entry_block].instructions[0] {
                IrInstruction::CallDirect { func_id, .. } => *func_id,
                other => panic!("expected direct call, got {other:?}"),
            }
        };
        let relay_instance = callee(entry_id);
        assert_ne!(relay_instance, IrFunctionId(1));
        let tag_instance = &module.functions[&callee(relay_instance)];
        assert!(tag_instance.signature.type_params.is_empty());
        assert!(tag_instance.type_param_tag_fixups.is_empty());
        assert!(matches!(
            &tag_instance.cfg.blocks[&tag_instance.cfg.entry_block].instructions[0],
            IrInstruction::Const {
                value: IrValue::I32(11),
                ..
            }
        ));
        assert_eq!(callee(IrFunctionId(1)), IrFunctionId(0));
    }
}
