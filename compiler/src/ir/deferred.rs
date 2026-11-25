//! Function bodies kept serialized until something reads them.
//!
//! A warm start used to decode every function body of the cached program, and
//! a run reads a small part of them. The cache now holds the module as a
//! skeleton -- every function's signature and attributes over a stub body --
//! with each real body as its own record, decoded the first time a reader asks
//! for it. Whole-program facts that were read from every body are computed
//! when the cache is written and stored with the skeleton.
//!
//! The live module is never changed: a pending body is decoded into the
//! reader's own copy (`full_function`, `materialize_all`), because module
//! addresses and decoded functions are shared across threads.

use super::vec_map::VecMap;
use super::{
    IrControlFlowGraph, IrFunction, IrFunctionId, IrId, IrInstruction, IrLocal, IrModule, IrType,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// What a function's body is, apart from its signature and attributes.
#[derive(Serialize, Deserialize)]
struct FunctionBody {
    cfg: IrControlFlowGraph,
    locals: VecMap<IrId, IrLocal>,
    register_types: VecMap<IrId, IrType>,
    type_param_tag_fixups: Vec<(IrId, String)>,
    type_param_metadata_fixups: Vec<(IrId, String)>,
}

impl FunctionBody {
    fn put(self, function: &mut IrFunction) {
        function.cfg = self.cfg;
        function.locals = self.locals;
        function.register_types = self.register_types;
        function.type_param_tag_fixups = self.type_param_tag_fixups;
        function.type_param_metadata_fixups = self.type_param_metadata_fixups;
    }
}

/// Facts about the whole program read from every body, kept with the skeleton.
#[derive(Serialize, Deserialize, Default, Debug)]
pub struct ProgramFacts {
    /// Functions whose address is taken, as `abi::collect_indirect_targets`
    /// finds them.
    pub indirect_targets: BTreeSet<IrFunctionId>,
    /// Functions with vector or atomic instructions, which skip the
    /// interpreter.
    pub uses_simd: BTreeSet<IrFunctionId>,
    /// Functions with a wide vector register, whose placeholders keep those
    /// register types.
    pub wide_vectors: BTreeSet<IrFunctionId>,
}

/// The serialized bodies and where each one is.
pub struct BodyStore {
    bytes: Vec<u8>,
    index: BTreeMap<IrFunctionId, (usize, usize)>,
}

impl BodyStore {
    fn body(&self, id: IrFunctionId) -> Option<FunctionBody> {
        let &(start, len) = self.index.get(&id)?;
        postcard::from_bytes(self.bytes.get(start..start + len)?).ok()
    }
}

/// Which bodies of a module are still in the store.
#[derive(Clone)]
pub struct DeferredBodies {
    store: Arc<BodyStore>,
    pending: BTreeSet<IrFunctionId>,
    pub facts: Arc<ProgramFacts>,
}

impl DeferredBodies {
    /// The facts alone, for a copy whose bodies are all present.
    pub fn facts_only(&self) -> Self {
        Self {
            store: self.store.clone(),
            pending: BTreeSet::new(),
            facts: self.facts.clone(),
        }
    }
}

impl std::fmt::Debug for DeferredBodies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeferredBodies")
            .field("pending", &self.pending.len())
            .field("bytes", &self.store.bytes.len())
            .finish()
    }
}

#[derive(Serialize, Deserialize)]
struct Skeleton {
    module: IrModule,
    facts: ProgramFacts,
    index: Vec<(IrFunctionId, u32, u32)>,
}

/// `FunctionBody` by reference, so writing the cache copies no body; the two
/// serialize alike.
#[derive(Serialize)]
struct BodyRef<'a> {
    cfg: &'a IrControlFlowGraph,
    locals: &'a VecMap<IrId, IrLocal>,
    register_types: &'a VecMap<IrId, IrType>,
    type_param_tag_fixups: &'a Vec<(IrId, String)>,
    type_param_metadata_fixups: &'a Vec<(IrId, String)>,
}

#[derive(Serialize)]
struct SkeletonRef<'a> {
    module: &'a IrModule,
    facts: &'a ProgramFacts,
    index: &'a Vec<(IrFunctionId, u32, u32)>,
}

/// `function` without its body: every other field, over a stub CFG.
fn stub_of(function: &IrFunction) -> IrFunction {
    let IrFunction {
        id,
        symbol_id,
        name,
        qualified_name,
        signature,
        cfg: _,
        locals: _,
        register_types: _,
        attributes,
        kind,
        source_location,
        next_reg_id,
        type_param_tag_fixups: _,
        type_param_metadata_fixups: _,
        erased_params,
        wasm_export,
        js_import,
    } = function;
    IrFunction {
        id: *id,
        symbol_id: *symbol_id,
        name: name.clone(),
        qualified_name: qualified_name.clone(),
        signature: signature.clone(),
        cfg: IrControlFlowGraph::new(),
        locals: Default::default(),
        register_types: Default::default(),
        attributes: attributes.clone(),
        kind: kind.clone(),
        source_location: source_location.clone(),
        next_reg_id: *next_reg_id,
        type_param_tag_fixups: Vec::new(),
        type_param_metadata_fixups: Vec::new(),
        erased_params: erased_params.clone(),
        wasm_export: *wasm_export,
        js_import: js_import.clone(),
    }
}

/// `module` with `functions` in place of its own.
fn with_functions(module: &IrModule, functions: BTreeMap<IrFunctionId, IrFunction>) -> IrModule {
    let IrModule {
        name,
        source_file,
        functions: _,
        globals,
        types,
        string_pool,
        extern_functions,
        metadata,
        next_function_id,
        next_global_id,
        next_typedef_id,
        symbol_to_register,
        register_to_symbol,
        external_function_names,
        import_file_blocks,
        deferred: _,
    } = module;
    IrModule {
        name: name.clone(),
        source_file: source_file.clone(),
        functions,
        globals: globals.clone(),
        types: types.clone(),
        string_pool: string_pool.clone(),
        extern_functions: extern_functions.clone(),
        metadata: metadata.clone(),
        next_function_id: *next_function_id,
        next_global_id: *next_global_id,
        next_typedef_id: *next_typedef_id,
        symbol_to_register: symbol_to_register.clone(),
        register_to_symbol: register_to_symbol.clone(),
        external_function_names: external_function_names.clone(),
        import_file_blocks: import_file_blocks.clone(),
        deferred: None,
    }
}

/// Whether `function` has vector or atomic instructions, which the
/// interpreter cannot run.
pub(crate) fn uses_simd(function: &IrFunction) -> bool {
    function.cfg.blocks.values().any(|block| {
        block.instructions.iter().any(|inst| {
            matches!(
                inst,
                IrInstruction::VectorLoad { .. }
                    | IrInstruction::VectorStore { .. }
                    | IrInstruction::VectorBinOp { .. }
                    | IrInstruction::VectorSplat { .. }
                    | IrInstruction::VectorExtract { .. }
                    | IrInstruction::VectorInsert { .. }
                    | IrInstruction::VectorReduce { .. }
                    | IrInstruction::VectorUnaryOp { .. }
                    | IrInstruction::VectorMinMax { .. }
                    | IrInstruction::VectorDot { .. }
                    | IrInstruction::VectorShuffle { .. }
                    | IrInstruction::VectorConvert { .. }
                    | IrInstruction::VectorNarrow { .. }
                    | IrInstruction::AtomicLoad { .. }
                    | IrInstruction::AtomicStore { .. }
                    | IrInstruction::AtomicRmw { .. }
                    | IrInstruction::AtomicCas { .. }
            )
        })
    })
}

/// The cache bytes for `module`: its skeleton, the facts, and every body.
pub fn split(module: &IrModule) -> Result<Vec<u8>, postcard::Error> {
    let mut facts = ProgramFacts::default();
    super::abi::collect_indirect_targets(module, &mut facts.indirect_targets);
    let mut bodies = Vec::new();
    let mut index = Vec::new();
    let mut functions = BTreeMap::new();
    for (id, function) in &module.functions {
        if function.cfg.blocks.is_empty() {
            functions.insert(*id, function.clone());
            continue;
        }
        if uses_simd(function) {
            facts.uses_simd.insert(*id);
        }
        if function.uses_wide_vectors() {
            facts.wide_vectors.insert(*id);
        }
        let start = bodies.len();
        bodies.extend(postcard::to_allocvec(&BodyRef {
            cfg: &function.cfg,
            locals: &function.locals,
            register_types: &function.register_types,
            type_param_tag_fixups: &function.type_param_tag_fixups,
            type_param_metadata_fixups: &function.type_param_metadata_fixups,
        })?);
        index.push((*id, start as u32, (bodies.len() - start) as u32));
        functions.insert(*id, stub_of(function));
    }
    let skeleton = with_functions(module, functions);
    let head = postcard::to_allocvec(&SkeletonRef {
        module: &skeleton,
        facts: &facts,
        index: &index,
    })?;
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend(head);
    out.extend(bodies);
    Ok(out)
}

/// The module `split` wrote, with every body pending.
pub fn join(bytes: &[u8]) -> Option<IrModule> {
    let head_len = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
    let head = bytes.get(8..8 + head_len)?;
    let skeleton: Skeleton = postcard::from_bytes(head).ok()?;
    let mut module = skeleton.module;
    let index: BTreeMap<IrFunctionId, (usize, usize)> = skeleton
        .index
        .into_iter()
        .map(|(id, start, len)| (id, (start as usize, len as usize)))
        .collect();
    let pending = index.keys().copied().collect();
    module.deferred = Some(DeferredBodies {
        store: Arc::new(BodyStore {
            bytes: bytes.get(8 + head_len..)?.to_vec(),
            index,
        }),
        pending,
        facts: Arc::new(skeleton.facts),
    });
    Some(module)
}

impl IrModule {
    /// Whether `id`'s body is still in the store.
    pub fn body_pending(&self, id: IrFunctionId) -> bool {
        self.deferred
            .as_ref()
            .is_some_and(|d| d.pending.contains(&id))
    }

    /// The facts stored with a skeleton, when this module is one.
    pub fn deferred_facts(&self) -> Option<&ProgramFacts> {
        self.deferred.as_ref().map(|d| d.facts.as_ref())
    }

    /// `id` with its body, decoded from the store when it is pending.
    pub fn full_function(&self, id: IrFunctionId) -> Option<Cow<'_, IrFunction>> {
        let function = self.functions.get(&id)?;
        if !self.body_pending(id) {
            return Some(Cow::Borrowed(function));
        }
        let body = self.deferred.as_ref()?.store.body(id)?;
        let mut full = function.clone();
        body.put(&mut full);
        Some(Cow::Owned(full))
    }

    /// Decode `id`'s body into this copy.
    pub fn materialize(&mut self, id: IrFunctionId) {
        let Some(deferred) = self.deferred.as_mut() else {
            return;
        };
        if !deferred.pending.remove(&id) {
            return;
        }
        if let (Some(body), Some(function)) = (deferred.store.body(id), self.functions.get_mut(&id))
        {
            body.put(function);
        }
    }

    /// Decode every pending body into this copy, which then stands alone.
    pub fn materialize_all(&mut self) {
        let Some(deferred) = self.deferred.take() else {
            return;
        };
        for id in &deferred.pending {
            if let (Some(body), Some(function)) =
                (deferred.store.body(*id), self.functions.get_mut(id))
            {
                body.put(function);
            }
        }
    }
}
