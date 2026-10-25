//! MIR interpreter: the first tier.
//!
//! It runs MIR with the representation compiled code uses, so the two can call
//! each other and share state: a register holds the machine word the Cranelift
//! lowering would hold, typed as that lowering types it; strings, objects,
//! closures and globals live where compiled code keeps them; externs are the
//! runtime's own functions. An interpreted function whose address escapes gets
//! a trampoline, so native code can call it.
//!
//! Decoded functions are cached by the address of the module they came from,
//! so a module must stay at one address while any of its functions can still
//! be called through a trampoline.

mod decode;
mod exec;
mod native;
mod trampoline;

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::ir::{IrFunctionId, IrModule, IrType};

use decode::{Function, Mt, canon, float_bits};

/// A value crossing into or out of the interpreter.
#[derive(Clone, Debug, Default)]
pub enum InterpValue {
    #[default]
    Void,
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Ptr(usize),
    Null,
}

impl InterpValue {
    pub fn to_bool(&self) -> Result<bool, InterpError> {
        Ok(self.to_i64()? != 0)
    }

    pub fn to_i64(&self) -> Result<i64, InterpError> {
        Ok(match *self {
            InterpValue::Void | InterpValue::Null => 0,
            InterpValue::Bool(b) => b as i64,
            InterpValue::I8(n) => n as i64,
            InterpValue::I16(n) => n as i64,
            InterpValue::I32(n) => n as i64,
            InterpValue::I64(n) => n,
            InterpValue::U8(n) => n as i64,
            InterpValue::U16(n) => n as i64,
            InterpValue::U32(n) => n as i64,
            InterpValue::U64(n) => n as i64,
            InterpValue::F32(f) => f as i64,
            InterpValue::F64(f) => f as i64,
            InterpValue::Ptr(p) => p as i64,
        })
    }

    pub fn to_f64(&self) -> Result<f64, InterpError> {
        Ok(match *self {
            InterpValue::F32(f) => f as f64,
            InterpValue::F64(f) => f,
            ref other => other.to_i64()? as f64,
        })
    }

    pub fn to_usize(&self) -> Result<usize, InterpError> {
        Ok(self.to_i64()? as usize)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, InterpValue::F32(_) | InterpValue::F64(_))
    }

    fn to_bits(&self, mt: Mt) -> u64 {
        match (self, mt.is_float()) {
            (InterpValue::F32(f), true) => float_bits(*f as f64, mt),
            (InterpValue::F64(f), true) => float_bits(*f, mt),
            (other, true) => float_bits(other.to_i64().unwrap_or(0) as f64, mt),
            (other, false) => canon(other.to_i64().unwrap_or(0) as u64, mt),
        }
    }

    fn from_bits(bits: u64, ty: &IrType) -> Self {
        match ty {
            IrType::Void => InterpValue::Void,
            IrType::Bool => InterpValue::Bool(bits & 0xff != 0),
            IrType::I8 => InterpValue::I8(bits as i8),
            IrType::I16 => InterpValue::I16(bits as i16),
            IrType::I32 => InterpValue::I32(bits as i32),
            IrType::I64 | IrType::NullInt => InterpValue::I64(bits as i64),
            IrType::U8 => InterpValue::U8(bits as u8),
            IrType::U16 => InterpValue::U16(bits as u16),
            IrType::U32 => InterpValue::U32(bits as u32),
            IrType::U64 => InterpValue::U64(bits),
            IrType::F32 => InterpValue::F32(f32::from_bits(bits as u32)),
            IrType::F64 => InterpValue::F64(f64::from_bits(bits)),
            _ => InterpValue::Ptr(bits as usize),
        }
    }
}

#[derive(Debug)]
pub enum InterpError {
    FunctionNotFound(IrFunctionId),
    StackOverflow,
    TypeError(String),
    RuntimeError(String),
    Panic(String),
    Exception(String),
    /// A loop got hot. The interpreted stack is set aside; compile the
    /// program and `resume` it.
    JitBailout(IrFunctionId),
    /// The set-aside stack stopped where only compiled code can carry on, so
    /// the call has to start over in compiled code.
    NotResumable(IrFunctionId),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpError::FunctionNotFound(id) => write!(f, "Function not found: {:?}", id),
            InterpError::StackOverflow => write!(f, "Stack overflow"),
            InterpError::TypeError(msg) => write!(f, "Type error: {}", msg),
            InterpError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            InterpError::Panic(msg) => write!(f, "Panic: {}", msg),
            InterpError::Exception(msg) => write!(f, "Exception: {}", msg),
            InterpError::JitBailout(id) => write!(f, "JIT bailout requested for {:?}", id),
            InterpError::NotResumable(id) => write!(f, "{:?} cannot resume interpreted", id),
        }
    }
}

impl std::error::Error for InterpError {}

type ResumeKey = (usize, IrFunctionId, crate::ir::IrBlockId);

/// Takes the module address and function the interpreter wants compiled, and
/// the block a frame of it was entering when it turned hot, if one was.
pub(crate) type CompileRequest =
    Box<dyn Fn(usize, IrFunctionId, Option<crate::ir::IrBlockId>) + Send + Sync>;

/// What every thread running one program's interpreted code shares.
pub(crate) struct Program {
    symbols: RwLock<HashMap<String, usize>>,
    decoded: RwLock<HashMap<(usize, IrFunctionId), Arc<Function>>>,
    indirect: RwLock<HashMap<usize, Arc<BTreeSet<IrFunctionId>>>>,
    natives: RwLock<HashMap<IrFunctionId, usize>>,
    /// Set once compiled code is installed: hot loops then transfer to it.
    compiled: AtomicBool,
    /// Compiled resume points by module address, function and loop header,
    /// with the frame each takes.
    resume_points: RwLock<HashMap<ResumeKey, (usize, Arc<crate::ir::osr::OsrLayout>)>>,
    /// Bumped whenever resume points are published, so a function looks again.
    resume_generation: std::sync::atomic::AtomicU64,
    /// Where a function that turns hot is sent to be compiled while the
    /// interpreter carries on. Unset, a hot loop suspends the program instead.
    compile_request: std::sync::OnceLock<CompileRequest>,
    /// Loops that carried on in compiled code.
    transfers: std::sync::atomic::AtomicU64,
    literals: Mutex<HashMap<String, &'static [u8]>>,
    cache: AtomicBool,
}

impl Program {
    fn new() -> Self {
        Self {
            symbols: RwLock::new(HashMap::new()),
            decoded: RwLock::new(HashMap::new()),
            indirect: RwLock::new(HashMap::new()),
            natives: RwLock::new(HashMap::new()),
            compiled: AtomicBool::new(false),
            resume_points: RwLock::new(HashMap::new()),
            resume_generation: std::sync::atomic::AtomicU64::new(0),
            compile_request: std::sync::OnceLock::new(),
            transfers: std::sync::atomic::AtomicU64::new(0),
            literals: Mutex::new(HashMap::new()),
            cache: AtomicBool::new(true),
        }
    }

    pub(crate) fn symbol(&self, name: &str) -> Option<usize> {
        if let Some(&p) = self.symbols.read().unwrap().get(name) {
            return Some(p);
        }
        native::dlsym(name)
    }

    /// The bytes of a string constant, alive for the rest of the process:
    /// the runtime's string literal borrows them. They sit in a real
    /// allocation even when empty, as compiled code's data section does; the
    /// runtime reads a string's data pointer to tell it from an object.
    pub(crate) fn literal(&self, s: &str) -> &'static [u8] {
        let mut literals = self.literals.lock().unwrap();
        if let Some(b) = literals.get(s) {
            return b;
        }
        let mut owned = s.as_bytes().to_vec();
        owned.push(0);
        let stored: &'static [u8] = Box::leak(owned.into_boxed_slice());
        let bytes = &stored[..s.len()];
        literals.insert(s.to_string(), bytes);
        bytes
    }

    pub(crate) fn native(&self, func: IrFunctionId) -> Option<usize> {
        if !self.has_natives() {
            return None;
        }
        self.natives.read().unwrap().get(&func).copied()
    }

    pub(crate) fn resume_point(
        &self,
        module: usize,
        func: IrFunctionId,
        header: crate::ir::IrBlockId,
    ) -> Option<(usize, Arc<crate::ir::osr::OsrLayout>)> {
        self.resume_points
            .read()
            .unwrap()
            .get(&(module, func, header))
            .cloned()
    }

    #[inline(always)]
    pub(crate) fn has_natives(&self) -> bool {
        self.compiled.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub(crate) fn resume_generation(&self) -> u64 {
        self.resume_generation.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub(crate) fn compiles_in_background(&self) -> bool {
        self.compile_request.get().is_some()
    }

    /// Ask for `func` of the module at `module` to be compiled, with a frame
    /// of it at `at`.
    pub(crate) fn request_compile(
        &self,
        module: usize,
        func: IrFunctionId,
        at: Option<crate::ir::IrBlockId>,
    ) {
        if let Some(request) = self.compile_request.get() {
            request(module, func, at);
        }
    }

    /// Compiled code for `func`: trampolines handed out for it jump there from
    /// now on, and new function values point at it directly. Safe from any
    /// thread.
    pub(crate) fn install_native(&self, func: IrFunctionId, code: usize) {
        self.natives.write().unwrap().insert(func, code);
        self.compiled.store(true, Ordering::Release);
        let stubs = trampoline::forward(func, code);
        // The runtime keys closures by code address; a stub handed out before
        // now is the same function as the compiled code.
        if let Some(alias) = self.symbol("haxe_closure_alias_code") {
            let alias: extern "C" fn(usize, usize) = unsafe { std::mem::transmute(alias) };
            for stub in stubs {
                alias(stub, code);
            }
        }
    }

    /// Whether `func` of the module at `module` resumes at `header`.
    pub(crate) fn has_resume_point(
        &self,
        module: usize,
        func: IrFunctionId,
        header: crate::ir::IrBlockId,
    ) -> bool {
        self.resume_points
            .read()
            .unwrap()
            .contains_key(&(module, func, header))
    }

    /// Compiled code that resumes `func` of the module at `module` at
    /// `layout.header`. Safe from any thread.
    pub(crate) fn install_resume_point(
        &self,
        module: usize,
        func: IrFunctionId,
        code: usize,
        layout: crate::ir::osr::OsrLayout,
    ) {
        self.resume_points
            .write()
            .unwrap()
            .insert((module, func, layout.header), (code, Arc::new(layout)));
        self.resume_generation.fetch_add(1, Ordering::AcqRel);
    }

    fn indirect_targets(&self, module: &IrModule) -> Arc<BTreeSet<IrFunctionId>> {
        let key = module as *const IrModule as usize;
        if let Some(s) = self.indirect.read().unwrap().get(&key) {
            return s.clone();
        }
        let set = Arc::new(crate::ir::abi::indirect_targets(std::slice::from_ref(
            module,
        )));
        self.indirect.write().unwrap().insert(key, set.clone());
        set
    }
}

/// One thread's interpreter state.
pub(crate) struct Machine {
    program: Arc<Program>,
    iterations: Cell<u64>,
    max_iterations: Cell<u64>,
    /// Native frames between here and the outermost interpreted call. A
    /// suspension returns through every frame, which it cannot do through them.
    native_depth: Cell<u32>,
    /// The interpreted stack a suspension set aside.
    suspended: std::cell::RefCell<Option<Vec<exec::Frame>>>,
}

impl Machine {
    fn new(program: Arc<Program>) -> Self {
        Self {
            program,
            iterations: Cell::new(0),
            max_iterations: Cell::new(10_000),
            native_depth: Cell::new(0),
            suspended: std::cell::RefCell::new(None),
        }
    }

    fn can_bail(&self) -> bool {
        self.native_depth.get() == 0
    }

    fn suspend(&self, stack: Vec<exec::Frame>) {
        *self.suspended.borrow_mut() = Some(stack);
    }

    fn native<T>(&self, call: impl FnOnce() -> Result<T, String>) -> Result<T, InterpError> {
        let d = self.native_depth.get();
        self.native_depth.set(d + 1);
        // The interpreter's frames hold copies of values whose live copies
        // are in its registers; a collection the callee asks for must not
        // read them as references.
        let (low, high) = (stack_mark(), ENTRY_SP.with(|e| e.get()));
        let _opaque = (low < high).then(|| rayzor_runtime::gc::OpaqueFrames::new(low, high));
        let r = call();
        self.native_depth.set(d);
        r.map_err(InterpError::RuntimeError)
    }

    fn function(&self, module: &IrModule, id: IrFunctionId) -> Result<Arc<Function>, InterpError> {
        let key = (module as *const IrModule as usize, id);
        let cache = self.program.cache.load(Ordering::Relaxed);
        if cache {
            if let Some(f) = self.program.decoded.read().unwrap().get(&key) {
                return Ok(f.clone());
            }
        }
        if !module.functions.contains_key(&id) {
            return Err(InterpError::FunctionNotFound(id));
        }
        let indirect = self.program.indirect_targets(module);
        let f = Arc::new(
            decode::decode(&self.program, module, id, &indirect)
                .map_err(InterpError::RuntimeError)?,
        );
        if cache {
            self.program.decoded.write().unwrap().insert(key, f.clone());
        }
        Ok(f)
    }

    fn function_at(&self, module: usize, id: IrFunctionId) -> Result<Arc<Function>, InterpError> {
        // SAFETY: decoded code names only modules that are still loaded.
        let module = unsafe { &*(module as *const IrModule) };
        self.function(module, id)
    }

    fn function_address(&self, module: usize, func: IrFunctionId) -> Result<usize, InterpError> {
        if let Some(code) = self.program.native(func) {
            return Ok(code);
        }
        trampoline::address(&self.program, module, func).map_err(InterpError::RuntimeError)
    }
}

thread_local! {
    static CURRENT: Cell<*const Machine> = const { Cell::new(std::ptr::null()) };
    /// The native stack address the innermost interpreter activation began at.
    static ENTRY_SP: Cell<usize> = const { Cell::new(0) };
}

/// An address in the calling frame.
#[inline(always)]
fn stack_mark() -> usize {
    let here = 0u8;
    std::hint::black_box(&here) as *const u8 as usize
}

fn current() -> *const Machine {
    CURRENT.with(|c| c.get())
}

fn set_current(m: *const Machine) {
    CURRENT.with(|c| c.set(m));
}

/// The machine native callbacks on this thread re-enter, for one scope, and
/// where on the native stack that activation began.
struct Entered(*const Machine, usize);

impl Entered {
    #[inline(always)]
    fn new(m: &Machine) -> Self {
        let prev = current();
        set_current(m);
        let prev_entry = ENTRY_SP.with(|e| e.replace(stack_mark()));
        Self(prev, prev_entry)
    }
}

impl Drop for Entered {
    fn drop(&mut self) {
        set_current(self.0);
        ENTRY_SP.with(|e| e.set(self.1));
    }
}

/// Run the function behind a trampoline with the arguments a native caller
/// left in `saved`, and put its result where the caller will look.
fn enter_from_native(
    t: &trampoline::Target,
    saved: &mut trampoline::Saved,
) -> Result<(), InterpError> {
    let active = current();
    let fresh;
    let m: &Machine = match unsafe { active.as_ref() } {
        Some(m) if Arc::ptr_eq(&m.program, &t.program) => m,
        _ => {
            fresh = Machine::new(t.program.clone());
            &fresh
        }
    };
    let _entered = Entered::new(m);
    let f = m.function_at(t.module, t.func)?;
    let params = &f.abi.params;
    // Cranelift passes an aarch64 struct-return address in x8.
    let in_x8 = f.abi.sret && cfg!(target_arch = "aarch64");
    let mut args = unsafe {
        native::unplace(
            &params[usize::from(in_x8)..],
            &saved.ints[..8],
            &saved.floats,
            saved.stack,
        )
    };
    if in_x8 {
        args.insert(0, saved.ints[8]);
    }
    let d = m.native_depth.get();
    m.native_depth.set(d + 1);
    let result = exec::call(m, f.clone(), &args);
    m.native_depth.set(d);
    let bits = result?;
    if f.abi.sret {
        saved.ints[0] = args[0];
    } else if let Some(mt) = f.abi.ret {
        if mt.is_float() {
            saved.floats[0] = bits;
        } else {
            saved.ints[0] = bits;
        }
    }
    Ok(())
}

/// The interpreter tier.
pub struct MirInterpreter {
    machine: Machine,
}

impl Default for MirInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl MirInterpreter {
    pub fn new() -> Self {
        Self {
            machine: Machine::new(Arc::new(Program::new())),
        }
    }

    pub fn new_without_cache() -> Self {
        let interp = Self::new();
        interp.machine.program.cache.store(false, Ordering::Relaxed);
        interp
    }

    pub fn set_use_decoded_cache(&mut self, enabled: bool) {
        self.machine.program.cache.store(enabled, Ordering::Relaxed);
        if !enabled {
            self.clear_decoded_cache();
        }
    }

    /// Forget decoded functions, as when the modules behind them change.
    pub fn clear_decoded_cache(&mut self) {
        self.machine.program.decoded.write().unwrap().clear();
        self.machine.program.indirect.write().unwrap().clear();
    }

    pub fn with_symbols(symbols: &[(&str, *const u8)]) -> Self {
        let mut interp = Self::new();
        for (name, ptr) in symbols {
            interp.register_symbol(name, *ptr);
        }
        interp
    }

    pub fn register_symbol(&mut self, name: &str, ptr: *const u8) {
        exec::note_symbol(name, ptr as usize);
        self.machine
            .program
            .symbols
            .write()
            .unwrap()
            .insert(name.to_string(), ptr as usize);
    }

    /// Blocks entered before execution asks for compiled code.
    pub fn set_max_iterations(&mut self, max: u64) {
        self.machine.max_iterations.set(max);
    }

    pub fn reset_iteration_count(&mut self) {
        self.machine.iterations.set(0);
    }

    pub fn iteration_count(&self) -> u64 {
        self.machine.iterations.get()
    }

    /// Compiled code for `func`: trampolines handed out for it jump there from
    /// now on, and new function values point at it directly.
    pub fn install_native(&mut self, func: IrFunctionId, code: *const u8) {
        self.machine.program.install_native(func, code as usize);
    }

    /// Compiled code that resumes `func` of `module` at `layout.header`,
    /// taking the frame `layout` describes. The layout is of the function as
    /// interpreted frames run it, which the module may since have optimized.
    pub fn install_resume_point(
        &mut self,
        module: &IrModule,
        func: IrFunctionId,
        code: *const u8,
        layout: crate::ir::osr::OsrLayout,
    ) {
        self.machine.program.install_resume_point(
            module as *const IrModule as usize,
            func,
            code as usize,
            layout,
        );
    }

    /// What compiled code is published through, from any thread.
    pub(crate) fn program(&self) -> Arc<Program> {
        self.machine.program.clone()
    }

    /// Send functions that turn hot to `request` and keep interpreting,
    /// rather than suspending for a compile. Set once.
    pub(crate) fn compile_in_background(&self, request: CompileRequest) {
        let _ = self.machine.program.compile_request.set(request);
    }

    /// How many interpreted loops have carried on in compiled code.
    pub fn transfers(&self) -> u64 {
        self.machine.program.transfers.load(Ordering::Relaxed)
    }

    /// The functions of the frames a suspension set aside, innermost last.
    pub fn suspended_functions(&self) -> Vec<IrFunctionId> {
        self.machine
            .suspended
            .borrow()
            .iter()
            .flatten()
            .map(|fr| fr.function().id)
            .collect()
    }

    /// Where each frame a suspension set aside stopped, innermost last: the
    /// block, and whether the frame stopped on entering it.
    pub fn suspended_positions(&self) -> Vec<(IrFunctionId, crate::ir::IrBlockId, bool)> {
        self.machine
            .suspended
            .borrow()
            .iter()
            .flatten()
            .map(|fr| {
                let (block, at_entry) = fr.position();
                (fr.function().id, block, at_entry)
            })
            .collect()
    }

    /// Continue the stack a `JitBailout` set aside. Its hot loop carries on in
    /// compiled code, if a resume point for it has been compiled, and the rest
    /// of the stack here; nothing that already ran runs again.
    pub fn resume(&mut self) -> Result<InterpValue, InterpError> {
        let m = &self.machine;
        let Some(stack) = m.suspended.borrow_mut().take() else {
            return Err(InterpError::RuntimeError("nothing to resume".to_string()));
        };
        let (base, top) = (stack[0].function().clone(), stack.last().unwrap());
        if top.at_bail() {
            return Err(InterpError::NotResumable(base.id));
        }
        let _entered = Entered::new(m);
        let bits = exec::resume(m, stack)?;
        Ok(InterpValue::from_bits(bits, &base.return_type))
    }

    pub fn execute(
        &mut self,
        module: &IrModule,
        func_id: IrFunctionId,
        args: Vec<InterpValue>,
    ) -> Result<InterpValue, InterpError> {
        let m = &self.machine;
        let _entered = Entered::new(m);
        let f = m.function(module, func_id)?;
        let mut abi_args = Vec::with_capacity(f.abi.params.len());
        if f.abi.sret {
            // The caller of a struct-returning entry gets the struct's address.
            let buf: &'static mut [u64] = Box::leak(vec![0u64; 64].into_boxed_slice());
            abi_args.push(buf.as_mut_ptr() as u64);
        }
        if f.abi.hidden_env {
            abi_args.push(0);
        }
        for (i, &(_, mt)) in f.params.iter().enumerate() {
            abi_args.push(args.get(i).map(|a| a.to_bits(mt)).unwrap_or(0));
        }
        let bits = exec::call(m, f.clone(), &abi_args)?;
        Ok(InterpValue::from_bits(bits, &f.return_type))
    }
}

#[cfg(test)]
mod tests;
