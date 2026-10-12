use compiler::codegen::CraneliftBackend;
use compiler::pipeline::HaxeCompilationPipeline;

#[test]
fn jit_catches_exceptions_with_registered_runtime_symbols() {
    let source = r#"
class Main {
    public static function statement():Int {
        var result = 0;
        try {
            throw 41;
        } catch (value:Int) {
            result = value + 1;
        }
        return result;
    }

    public static function expression():Int {
        return try {
            statement() + fail();
        } catch (value:Int) {
            value + 2;
        };
    }

    static function fail():Int {
        throw 40;
    }

    static function main() {}
}
"#;
    let mut pipeline = HaxeCompilationPipeline::new();
    let result = pipeline.compile_file("Main.hx", source);
    assert!(result.errors.is_empty(), "{:?}", result.errors);

    let symbols = rayzor_runtime::plugin_impl::get_plugin().runtime_symbols();
    assert!(
        symbols
            .iter()
            .any(|(name, ptr)| *name == "_setjmp" && !ptr.is_null()),
        "the JIT must resolve _setjmp without a process-symbol fallback"
    );
    let mut backend = CraneliftBackend::with_symbols(&symbols).unwrap();
    let module = result
        .mir_modules
        .iter()
        .find(|module| {
            module
                .functions
                .values()
                .any(|function| function.qualified_name.as_deref() == Some("Main.statement"))
        })
        .expect("Main is lowered");
    backend.compile_module(module).unwrap();
    for name in ["Main.statement", "Main.expression"] {
        let (id, _) = module
            .functions
            .iter()
            .find(|(_, function)| function.qualified_name.as_deref() == Some(name))
            .unwrap();
        let ptr = backend.get_function_ptr(*id).unwrap();
        // Both Haxe methods are static, take no arguments and return Int.
        let run: extern "C" fn() -> i32 = unsafe { std::mem::transmute(ptr) };
        assert_eq!(run(), 42, "{name}");
    }
}
