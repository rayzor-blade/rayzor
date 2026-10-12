use crate::tast::{
    AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeKind, TypeTable,
    namespace::{ImportResolver, NamespaceResolver},
    node::{TypedExpressionKind, TypedStatement},
};
use std::{cell::RefCell, rc::Rc};

#[test]
fn constructor_identity_follows_its_module_in_both_load_orders() {
    let foreign = r#"
        package foreign;
        enum Constant { CFloat(text:String); }
        class Foreign {
            static function main():Void {
                var value = CFloat("foreign");
                var CFloat:String = "shadow";
                var shadow = CFloat;
            }
        }
    "#;
    let local = r#"
        package local;
        enum Constant<T> { CFloat(text:String):Constant<Float>; }
        class Local {
            static function main():Void {
                var value = CFloat("local");
                var CFloat:String = "shadow";
                var shadow = CFloat;
            }
        }
    "#;
    for sources in [[foreign, local], [local, foreign]] {
        let mut interner = StringInterner::new();
        let mut symbols = SymbolTable::new();
        let types = RefCell::new(TypeTable::new());
        let mut scopes = ScopeTree::new(ScopeId::first());
        let mut namespaces = NamespaceResolver::new();
        let mut imports = ImportResolver::new();
        let parsed =
            sources.map(|source| parser::parse_haxe_file("fixture.hx", source, false).unwrap());
        for file in &parsed {
            let mut lowering = AstLowering::new(
                &mut interner,
                Rc::new(RefCell::new(StringInterner::new())),
                &mut symbols,
                &types,
                &mut scopes,
                &mut namespaces,
                &mut imports,
            );
            lowering.set_package_from_parts(&file.package.as_ref().unwrap().path);
            for declaration in &file.declarations {
                lowering.pre_register_declaration(declaration).unwrap();
            }
        }
        for (source, parsed) in sources.into_iter().zip(&parsed) {
            let mut lowering = AstLowering::new(
                &mut interner,
                Rc::new(RefCell::new(StringInterner::new())),
                &mut symbols,
                &types,
                &mut scopes,
                &mut namespaces,
                &mut imports,
            );
            lowering.set_skip_stdlib_loading(true);
            lowering.set_skip_pre_registration(true);
            let typed = lowering.lower_file(parsed).unwrap();
            let TypedStatement::Expression { expression, .. } =
                &typed.classes[0].methods[0].body[0]
            else {
                panic!("expected a method block");
            };
            let TypedExpressionKind::Block { statements, .. } = &expression.kind else {
                panic!("expected a method block");
            };
            let initializer = match &statements[0] {
                TypedStatement::VarDeclaration {
                    initializer: Some(value),
                    ..
                } => value,
                statement => panic!("expected a constructor initializer: {statement:?}"),
            };
            let table = types.borrow();
            let TypeKind::Enum {
                symbol_id,
                type_args,
            } = &table.get(initializer.expr_type).unwrap().kind
            else {
                panic!("expected an enum value");
            };
            assert_eq!(*symbol_id, typed.enums[0].symbol_id);
            if source == local {
                assert_eq!(type_args.len(), 1);
                assert!(matches!(
                    table.get(type_args[0]).unwrap().kind,
                    TypeKind::Float
                ));
            }
            let TypedExpressionKind::FunctionCall { function, .. } = &initializer.kind else {
                panic!("expected a constructor call");
            };
            let TypedExpressionKind::Variable { symbol_id } = &function.kind else {
                panic!("expected a constructor symbol");
            };
            assert_eq!(
                symbols.find_parent_enum_for_constructor(*symbol_id),
                Some(typed.enums[0].symbol_id)
            );
            let TypedStatement::VarDeclaration {
                initializer: Some(shadow),
                ..
            } = &statements[2]
            else {
                panic!("expected a shadowed-name initializer");
            };
            assert!(matches!(
                table.get(shadow.expr_type).unwrap().kind,
                TypeKind::String
            ));
        }
    }
}
