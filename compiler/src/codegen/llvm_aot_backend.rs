//! LLVM AOT Backend — AOT-specific operations on top of LLVMJitBackend
//!
//! Free functions for cross-compilation, main wrapper generation, and multiple
//! output format support without modifying the JIT code path.

#[cfg(feature = "llvm-backend")]
use inkwell::{
    OptimizationLevel,
    module::Module,
    passes::PassBuilderOptions,
    targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
    },
    values::AsValueRef,
};

#[cfg(feature = "llvm-backend")]
use std::path::Path;

#[cfg(feature = "llvm-backend")]
use std::sync::Once;

#[cfg(feature = "llvm-backend")]
static AOT_INIT: Once = Once::new();

/// Initialize LLVM for AOT compilation (all targets, no MCJIT).
#[cfg(feature = "llvm-backend")]
pub fn init_llvm_aot() {
    AOT_INIT.call_once(|| {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize LLVM native target");
        Target::initialize_all(&InitializationConfig::default());
    });
}

/// The machine an AOT build emits code for: the requested triple, or this
/// host's. A build for this host compiles for its CPU, as the JIT does; a
/// cross build gets a generic CPU, because the machine it will run on cannot
/// be asked. Every target-dependent step takes its machine from here, so the
/// module's layout, the optimizer and the object file agree.
#[cfg(feature = "llvm-backend")]
#[derive(Clone, Debug)]
pub struct AotTarget {
    triple: String,
    native: bool,
}

/// Whether two normalized triples name the same machine. The vendor is
/// skipped when either side leaves it generic: `x86_64-unknown-linux-gnu` and
/// LLVM's own `x86_64-pc-linux-gnu` are one host.
#[cfg(feature = "llvm-backend")]
fn same_machine(a: &str, b: &str) -> bool {
    let a: Vec<&str> = a.split('-').collect();
    let b: Vec<&str> = b.split('-').collect();
    if a.len() != b.len() || a.len() < 3 {
        return a == b;
    }
    let generic = |v: &str| v == "unknown" || v == "pc";
    a[0] == b[0] && (a[1] == b[1] || generic(a[1]) && generic(b[1])) && a[2..] == b[2..]
}

#[cfg(feature = "llvm-backend")]
impl AotTarget {
    pub fn new(requested: Option<&str>) -> AotTarget {
        init_llvm_aot();
        let host = TargetMachine::get_default_triple();
        let Some(requested) = requested else {
            return AotTarget {
                triple: host.as_str().to_string_lossy().into_owned(),
                native: true,
            };
        };
        let normalized = TargetMachine::normalize_triple(&TargetTriple::create(requested))
            .as_str()
            .to_string_lossy()
            .into_owned();
        let host_normalized = TargetMachine::normalize_triple(&host)
            .as_str()
            .to_string_lossy()
            .into_owned();
        AotTarget {
            native: same_machine(&normalized, &host_normalized),
            triple: normalized,
        }
    }

    pub fn triple(&self) -> &str {
        &self.triple
    }

    /// Whether the output runs on this machine, and so may use its CPU.
    pub fn is_native(&self) -> bool {
        self.native
    }

    pub fn target_machine(
        &self,
        reloc_mode: RelocMode,
        opt_level: OptimizationLevel,
    ) -> Result<TargetMachine, String> {
        let triple = TargetTriple::create(&self.triple);
        let target = Target::from_triple(&triple)
            .map_err(|e| format!("No LLVM target for {}: {}", self.triple, e))?;
        let (cpu, features) = if self.native {
            (
                TargetMachine::get_host_cpu_name()
                    .to_string_lossy()
                    .into_owned(),
                TargetMachine::get_host_cpu_features()
                    .to_string_lossy()
                    .into_owned(),
            )
        } else {
            ("generic".to_string(), String::new())
        };
        target
            .create_target_machine(
                &triple,
                &cpu,
                &features,
                opt_level,
                reloc_mode,
                CodeModel::Default,
            )
            .ok_or_else(|| format!("Could not create a target machine for {}", self.triple))
    }

    /// Install the triple and data layout on `module`. Done before any body
    /// is lowered, so type sizes are the target's and not the host's.
    pub fn apply_to_module(&self, module: &Module, machine: &TargetMachine) {
        module.set_triple(&TargetTriple::create(&self.triple));
        module.set_data_layout(&machine.get_target_data().get_data_layout());
    }
}

#[cfg(feature = "llvm-backend")]
fn run_opt_passes(
    module: &Module,
    target_machine: &TargetMachine,
    opt_level: OptimizationLevel,
) -> Result<(), String> {
    if opt_level != OptimizationLevel::None {
        let passes = match opt_level {
            OptimizationLevel::None => "default<O0>",
            OptimizationLevel::Less => "default<O1>",
            OptimizationLevel::Default => "default<O2>",
            OptimizationLevel::Aggressive => "default<O3>",
        };
        let pass_options = PassBuilderOptions::create();
        module
            .run_passes(passes, target_machine, pass_options)
            .map_err(|e| format!("Failed to run optimization passes: {}", e))?;

        // Debug: dump optimized IR if requested
        if std::env::var("RAYZOR_DUMP_LLVM_IR").is_ok() {
            let ir_str = module.print_to_string().to_string();
            if std::fs::write("/tmp/rayzor_aot_opt.ll", &ir_str).is_ok() {
                eprintln!(
                    "=== AOT Optimized LLVM IR saved to /tmp/rayzor_aot_opt.ll ({} bytes) ===",
                    ir_str.len()
                );
            }
        }
    }
    Ok(())
}

/// Run LLVM optimization passes on the module.
/// Call this BEFORE adding main() wrappers or other post-optimization transforms.
#[cfg(feature = "llvm-backend")]
pub fn optimize_module(
    module: &Module,
    target: &AotTarget,
    opt_level: OptimizationLevel,
) -> Result<(), String> {
    if let Err(msg) = module.verify() {
        return Err(format!(
            "LLVM module verification failed: {}",
            msg.to_string()
        ));
    }
    // RelocMode doesn't affect optimization, only codegen
    let target_machine = target.target_machine(RelocMode::Default, opt_level)?;
    run_opt_passes(module, &target_machine, opt_level)
}

/// Compile to object file with configurable target and relocation mode.
/// The module should already be optimized via `optimize_module`.
#[cfg(feature = "llvm-backend")]
pub fn compile_to_object_file(
    module: &Module,
    output_path: &Path,
    target: &AotTarget,
    reloc_mode: RelocMode,
    opt_level: OptimizationLevel,
) -> Result<(), String> {
    target
        .target_machine(reloc_mode, opt_level)?
        .write_to_file(module, FileType::Object, output_path)
        .map_err(|e| format!("Failed to write object file: {}", e))
}

/// Emit LLVM IR text (.ll).
#[cfg(feature = "llvm-backend")]
pub fn emit_llvm_ir(module: &Module, output_path: &Path) -> Result<(), String> {
    if let Err(msg) = module.verify() {
        return Err(format!(
            "LLVM module verification failed: {}",
            msg.to_string()
        ));
    }
    let ir_str = module.print_to_string().to_string();
    std::fs::write(output_path, ir_str).map_err(|e| format!("Failed to write LLVM IR: {}", e))
}

/// Emit LLVM bitcode (.bc).
#[cfg(feature = "llvm-backend")]
pub fn emit_llvm_bitcode(module: &Module, output_path: &Path) -> Result<(), String> {
    if let Err(msg) = module.verify() {
        return Err(format!(
            "LLVM module verification failed: {}",
            msg.to_string()
        ));
    }
    if module.write_bitcode_to_path(output_path) {
        Ok(())
    } else {
        Err("Failed to write LLVM bitcode".to_string())
    }
}

/// Emit native assembly (.s).
/// The module should already be optimized via `optimize_module`.
#[cfg(feature = "llvm-backend")]
pub fn emit_assembly(
    module: &Module,
    output_path: &Path,
    target: &AotTarget,
    opt_level: OptimizationLevel,
) -> Result<(), String> {
    target
        .target_machine(RelocMode::Default, opt_level)?
        .write_to_file(module, FileType::Assembly, output_path)
        .map_err(|e| format!("Failed to write assembly: {}", e))
}

/// Generate a C main() wrapper that calls the Haxe entry point.
///
/// Creates a `main()` wrapper that initializes args, runs startup hooks, and then
/// calls the Haxe entry point. If entry is named "main", renames it to "_haxe_main" first.
#[cfg(feature = "llvm-backend")]
pub fn generate_main_wrapper(
    module: &Module,
    entry_func_name: &str,
    startup_func_names: &[String],
) -> Result<(), String> {
    let entry_func = module.get_function(entry_func_name).ok_or_else(|| {
        format!(
            "Entry function '{}' not found in LLVM module",
            entry_func_name
        )
    })?;

    // Rename if collides with C main
    let actual_name = if entry_func_name == "main" {
        unsafe {
            use std::ffi::CString;
            let new_name = CString::new("_haxe_main").unwrap();
            llvm_sys::core::LLVMSetValueName2(
                entry_func.as_value_ref(),
                new_name.as_ptr(),
                "_haxe_main".len(),
            );
        }
        "_haxe_main"
    } else {
        entry_func_name
    };

    let context = module.get_context();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let i8_ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    let main_fn_type = i32_type.fn_type(&[i32_type.into(), i8_ptr_type.into()], false);
    let main_fn = module.add_function("main", main_fn_type, None);
    let entry_bb = context.append_basic_block(main_fn, "entry");

    let builder = context.create_builder();
    builder.position_at_end(entry_bb);

    // Declare and call rayzor_init_args_from_argv(argc, argv) to initialize Sys.args()
    let init_args_type = context
        .void_type()
        .fn_type(&[i32_type.into(), i8_ptr_type.into()], false);
    let init_args_fn = module.add_function(
        "rayzor_init_args_from_argv",
        init_args_type,
        Some(inkwell::module::Linkage::External),
    );
    let argc_val = main_fn.get_nth_param(0).unwrap();
    let argv_val = main_fn.get_nth_param(1).unwrap();
    builder
        .build_call(init_args_fn, &[argc_val.into(), argv_val.into()], "")
        .map_err(|e| format!("Failed to build call to rayzor_init_args_from_argv: {}", e))?;

    let zero = i64_type.const_int(0, false);

    for startup_name in startup_func_names {
        let startup_fn = module
            .get_function(startup_name)
            .ok_or_else(|| format!("Startup function '{}' not found", startup_name))?;
        builder
            .build_call(startup_fn, &[zero.into()], "")
            .map_err(|e| format!("Failed to build call to {}: {}", startup_name, e))?;
    }

    // Call the Haxe entry point
    let haxe_entry = module
        .get_function(actual_name)
        .ok_or_else(|| format!("Renamed entry function '{}' not found", actual_name))?;
    builder
        .build_call(haxe_entry, &[zero.into()], "")
        .map_err(|e| format!("Failed to build call to entry: {}", e))?;

    builder
        .build_return(Some(&i32_type.const_int(0, false)))
        .map_err(|e| format!("Failed to build return: {}", e))?;

    Ok(())
}

#[cfg(all(test, feature = "llvm-backend"))]
mod tests {
    use super::same_machine;

    #[test]
    fn generic_vendors_name_one_machine() {
        assert!(same_machine(
            "x86_64-unknown-linux-gnu",
            "x86_64-pc-linux-gnu"
        ));
        assert!(same_machine("aarch64-apple-darwin", "aarch64-apple-darwin"));
        assert!(!same_machine(
            "aarch64-unknown-linux-gnu",
            "x86_64-pc-linux-gnu"
        ));
        assert!(!same_machine(
            "x86_64-pc-windows-msvc",
            "x86_64-pc-linux-gnu"
        ));
        assert!(!same_machine(
            "x86_64-unknown-linux-musl",
            "x86_64-pc-linux-gnu"
        ));
    }
}
