//! An abstract class may leave an interface's methods to its subclasses.
//! Each one it does not declare becomes a bodyless (abstract) member, so a
//! call through the class dispatches to the subclass's implementation.

use parser::{Access, ClassFieldKind, HaxeFile, Type, TypeDeclaration};

/// The file with the missing interface members declared, or None when no
/// abstract class needs one. Only a class with no parent: an inherited
/// implementation must not be shadowed by a stub.
pub(crate) fn desugar(file: &HaxeFile) -> Option<HaxeFile> {
    let interfaces: Vec<&parser::InterfaceDecl> = file
        .declarations
        .iter()
        .filter_map(|decl| match decl {
            TypeDeclaration::Interface(iface) => Some(iface),
            _ => None,
        })
        .collect();
    if interfaces.is_empty() {
        return None;
    }
    let mut out = file.clone();
    let mut changed = false;
    for decl in &mut out.declarations {
        let TypeDeclaration::Class(class) = decl else {
            continue;
        };
        if !class.is_abstract || class.extends.is_some() {
            continue;
        }
        let declared = |class: &parser::ClassDecl, name: &str| {
            class
                .fields
                .iter()
                .any(|f| matches!(&f.kind, ClassFieldKind::Function(func) if func.name == name))
        };
        let mut stubs = Vec::new();
        for implemented in &class.implements {
            let Type::Path { path, .. } = implemented else {
                continue;
            };
            let Some(iface) = interfaces.iter().find(|i| i.name == path.name) else {
                continue;
            };
            for field in &iface.fields {
                let ClassFieldKind::Function(func) = &field.kind else {
                    continue;
                };
                if declared(class, &func.name) {
                    continue;
                }
                let mut stub = field.clone();
                if let ClassFieldKind::Function(f) = &mut stub.kind {
                    f.body = None;
                }
                stub.access = Some(Access::Public);
                stubs.push(stub);
            }
        }
        if !stubs.is_empty() {
            class.fields.extend(stubs);
            changed = true;
        }
    }
    changed.then_some(out)
}
