use compiler::pipeline::HaxeCompilationPipeline;
use compiler::tast::{TypedExpressionKind, TypedStatement};

#[test]
fn anonymous_generic_return_keeps_a_type_tag_fixup() {
    let source = r#"
class Main {
    static function field<T>(value:Null<T>):{value:T} return {value:value};
    static function main() {
        var value:Dynamic = 1.5;
        trace(field(value).value);
    }
}
"#;
    let mut pipeline = HaxeCompilationPipeline::new();
    let result = pipeline.compile_file("Main.hx", source);
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    let field = result
        .mir_modules
        .iter()
        .flat_map(|m| m.functions.values())
        .find(|f| f.name == "field")
        .expect("field is lowered");
    assert!(
        field
            .type_param_tag_fixups
            .iter()
            .any(|(_, name)| name == "T"),
        "{:?}",
        field
    );
    let file = &result.typed_files[0];
    let main = file.classes[0]
        .methods
        .iter()
        .find(|method| file.string_interner.borrow().get(method.name) == Some("main"))
        .expect("main is typed");
    let body = match &main.body[0] {
        TypedStatement::Expression { expression, .. } => match &expression.kind {
            TypedExpressionKind::Block { statements, .. } => statements,
            _ => &main.body,
        },
        _ => &main.body,
    };
    let TypedStatement::VarDeclaration { var_type, .. } = &body[0] else {
        panic!("expected the Dynamic declaration: {:?}", main.body);
    };
    let TypedStatement::Expression { expression, .. } = &body[1] else {
        panic!("expected trace: {:?}", main.body);
    };
    let arguments = match &expression.kind {
        TypedExpressionKind::FunctionCall { arguments, .. }
        | TypedExpressionKind::StaticMethodCall { arguments, .. } => arguments,
        other => panic!("expected trace call, got {other:?}"),
    };
    assert_eq!(arguments[0].expr_type, *var_type, "{:?}", arguments[0]);
}
