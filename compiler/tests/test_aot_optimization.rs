#![cfg(feature = "llvm-backend")]

use compiler::codegen::aot_compiler::{AotCompiler, LlvmOptimizationLevel, OutputFormat};
use compiler::ir::optimization::OptimizationLevel;
use std::path::PathBuf;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "rayzor-aot-options-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn aot_emits_every_retained_body() {
    let work = Scratch::new();
    let mut source = String::from("class Main { static function main() {}\n");
    for i in 0..300 {
        source.push_str(&format!(
            "public static function body_{i}(x:Int):Int return x + {i};\n"
        ));
    }
    source.push('}');
    let input = work.0.join("Main.hx");
    std::fs::write(&input, source).unwrap();
    let output = work.0.join("all.ll");
    AotCompiler {
        opt_level: OptimizationLevel::O0,
        llvm_opt_level: Some(LlvmOptimizationLevel::O0),
        codegen_opt_level: Some(OptimizationLevel::O0),
        output_format: OutputFormat::LlvmIr,
        strip: false,
        enable_cache: false,
        ..Default::default()
    }
    .compile(&[input.to_string_lossy().into_owned()], &output)
    .unwrap();
    let ir = std::fs::read_to_string(output).unwrap();
    assert_eq!(
        ir.lines()
            .filter(|line| line.starts_with("define ") && line.contains("body_"))
            .count(),
        300
    );
    assert!(!ir.contains("via_table"));
}

#[test]
fn size_pipeline_preserves_exception_shields_and_can_emit_baseline_machine_code() {
    use compiler::codegen::llvm_aot_backend::{self, AotTarget};
    use inkwell::{
        attributes::{Attribute, AttributeLoc},
        context::Context,
        targets::RelocMode,
    };
    let work = Scratch::new();
    let context = Context::create();
    let module = context.create_module("independent-levels");
    let target = AotTarget::new(None);
    let machine = target
        .target_machine(RelocMode::PIC, inkwell::OptimizationLevel::None)
        .unwrap();
    target.apply_to_module(&module, &machine);
    let builder = context.create_builder();
    for name in ["ordinary", "shielded"] {
        let f = module.add_function(
            name,
            context
                .i32_type()
                .fn_type(&[context.i32_type().into()], false),
            None,
        );
        builder.position_at_end(context.append_basic_block(f, "entry"));
        let zero = context.i32_type().const_zero();
        let value = builder
            .build_int_mul(
                f.get_first_param().unwrap().into_int_value(),
                zero,
                "multiply",
            )
            .unwrap();
        let value = builder
            .build_int_add(value, context.i32_type().const_int(42, false), "answer")
            .unwrap();
        builder.build_return(Some(&value)).unwrap();
        if name == "shielded" {
            for attr in ["optnone", "noinline"] {
                f.add_attribute(
                    AttributeLoc::Function,
                    context.create_enum_attribute(Attribute::get_named_enum_kind_id(attr), 0),
                );
            }
        }
    }
    llvm_aot_backend::optimize_module_with_level(&module, &target, LlvmOptimizationLevel::MinSize)
        .unwrap();
    let minsize = Attribute::get_named_enum_kind_id("minsize");
    let ordinary = module.get_function("ordinary").unwrap();
    assert!(
        ordinary
            .get_enum_attribute(AttributeLoc::Function, minsize)
            .is_some()
    );
    assert!(
        module
            .get_function("shielded")
            .unwrap()
            .get_enum_attribute(AttributeLoc::Function, minsize)
            .is_none()
    );
    use inkwell::values::AnyValue;
    let optimized = ordinary.print_to_string().to_string();
    assert!(optimized.contains("ret i32 42"), "{optimized}");
    module.verify().unwrap();
    let object = work.0.join("baseline.o");
    llvm_aot_backend::compile_to_object_file(
        &module,
        &object,
        &target,
        RelocMode::PIC,
        inkwell::OptimizationLevel::None,
    )
    .unwrap();
    assert!(std::fs::metadata(object).unwrap().len() > 0);
}
