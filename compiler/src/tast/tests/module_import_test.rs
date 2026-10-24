use crate::tast::{
    AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeKind, TypeTable,
    namespace::{ImportResolver, NamespaceResolver},
};
use std::{cell::RefCell, rc::Rc};

/// `import pkg.Mod;` names every type `Mod` declares, ahead of another
/// package's type of the same name, from inside the importing file's classes.
#[test]
fn a_module_import_names_its_subtypes_inside_a_packaged_class() {
    let sources = [
        ("other/Shape.hx", "package other; class Shape {}"),
        (
            "pkg/Mod.hx",
            "package pkg; typedef Mod = {name:String}; typedef Shape = {x:Int};",
        ),
        (
            "app/Main.hx",
            "package app; import pkg.Mod; class Main { static function take(s:Shape):Void {} }",
        ),
    ];
    let mut interner = StringInterner::new();
    let mut symbols = SymbolTable::new();
    let types = RefCell::new(TypeTable::new());
    let mut scopes = ScopeTree::new(ScopeId::first());
    let mut namespaces = NamespaceResolver::new();
    let mut imports = ImportResolver::new();
    let parsed: Vec<_> = sources
        .iter()
        .map(|(name, source)| parser::parse_haxe_file(name, source, false).unwrap())
        .collect();
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
    let mut main = None;
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
        lowering.set_skip_stdlib_loading(true);
        lowering.set_skip_pre_registration(true);
        main = Some(lowering.lower_file(file).unwrap());
    }
    let main = main.unwrap();
    let shape = main.classes[0].methods[0].parameters[0].param_type;
    let table = types.borrow();
    match &table.get(shape).unwrap().kind {
        TypeKind::TypeAlias { symbol_id, .. } => {
            let name = symbols.get_symbol(*symbol_id).unwrap().qualified_name;
            assert_eq!(name.and_then(|n| interner.get(n)), Some("pkg.Shape"));
        }
        TypeKind::Anonymous { fields } => {
            assert!(fields.iter().any(|f| interner.get(f.name) == Some("x")));
        }
        kind => panic!("`Shape` resolved to {kind:?}, not the imported module's typedef"),
    }
}
