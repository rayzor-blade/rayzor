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

#[cfg(feature = "llvm-backend")]
fn create_target_machine(
    target_triple: Option<&str>,
    cpu: &str,
    features: &str,
    reloc_mode: RelocMode,
    opt_level: OptimizationLevel,
) -> Result<TargetMachine, String> {
    let triple = if let Some(triple_str) = target_triple {
        TargetTriple::create(triple_str)
    } else {
        TargetMachine::get_default_triple()
    };

    let target = Target::from_triple(&triple)
        .map_err(|e| format!("Failed to get target for triple: {}", e))?;

    target
        .create_target_machine(
            &triple,
            cpu,
            features,
            opt_level,
            reloc_mode,
            CodeModel::Default,
        )
        .ok_or_else(|| "Failed to create target machine".to_string())
}

/// Get host CPU name and feature string
#[cfg(feature = "llvm-backend")]
fn get_host_cpu_info() -> (String, String) {
    let cpu = TargetMachine::get_host_cpu_name()
        .to_str()
        .unwrap_or("generic")
        .to_string();
    let features = TargetMachine::get_host_cpu_features()
        .to_str()
        .unwrap_or("")
        .to_string();
    (cpu, features)
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
    target_triple: Option<&str>,
    opt_level: OptimizationLevel,
) -> Result<(), String> {
    if let Err(msg) = module.verify() {
        return Err(format!(
            "LLVM module verification failed: {}",
            msg.to_string()
        ));
    }

    let (host_cpu, host_features) = get_host_cpu_info();
    let (cpu, features) = if target_triple.is_some() {
        ("generic", "")
    } else {
        (host_cpu.as_str(), host_features.as_str())
    };
    // RelocMode doesn't affect optimization, only codegen
    let target_machine =
        create_target_machine(target_triple, cpu, features, RelocMode::Default, opt_level)?;

    // Set data layout from the target machine — this provides type size information
    // that the optimizer needs for accurate inlining cost analysis.
    let data_layout = target_machine.get_target_data().get_data_layout();
    module.set_data_layout(&data_layout);

    run_opt_passes(module, &target_machine, opt_level)?;
    Ok(())
}

/// Compile to object file with configurable target and relocation mode.
/// The module should already be optimized via `optimize_module`.
#[cfg(feature = "llvm-backend")]
pub fn compile_to_object_file(
    module: &Module,
    output_path: &Path,
    target_triple: Option<&str>,
    reloc_mode: RelocMode,
    opt_level: OptimizationLevel,
) -> Result<(), String> {
    let (host_cpu, host_features) = get_host_cpu_info();
    let (cpu, features) = if target_triple.is_some() {
        ("generic", "")
    } else {
        (host_cpu.as_str(), host_features.as_str())
    };
    let target_machine =
        create_target_machine(target_triple, cpu, features, reloc_mode, opt_level)?;

    let triple = if let Some(t) = target_triple {
        TargetTriple::create(t)
    } else {
        TargetMachine::get_default_triple()
    };
    module.set_triple(&triple);

    target_machine
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
    target_triple: Option<&str>,
    opt_level: OptimizationLevel,
) -> Result<(), String> {
    let (host_cpu, host_features) = get_host_cpu_info();
    let (cpu, features) = if target_triple.is_some() {
        ("generic", "")
    } else {
        (host_cpu.as_str(), host_features.as_str())
    };
    let target_machine =
        create_target_machine(target_triple, cpu, features, RelocMode::Default, opt_level)?;

    let triple = if let Some(t) = target_triple {
        TargetTriple::create(t)
    } else {
        TargetMachine::get_default_triple()
    };
    module.set_triple(&triple);

    target_machine
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
