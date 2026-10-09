//! `overload` functions: same-named declarations are renamed apart before
//! lowering, and each group is kept so a call can pick its member from the
//! argument types (`call_overload.rs`).

use parser::{ClassFieldKind, Function, HaxeFile, Modifier, ModuleFieldKind, TypeDeclaration};
use std::collections::BTreeMap;

/// Owner of module-level functions in the group keys.
pub(crate) const MODULE_OWNER: &str = "";

/// One renamed member of a group.
#[derive(Clone)]
pub(crate) struct OverloadCandidate {
    pub name: String,
    pub function: Function,
    pub is_static: bool,
}

/// `(owner type name, function name)` -> its members in declaration order.
pub(crate) type OverloadGroups = BTreeMap<(String, String), Vec<OverloadCandidate>>;

pub(crate) fn mangled(name: &str, index: usize) -> String {
    format!("{name}__ovl{index}")
}

/// The file with every overload group renamed apart, and the groups. None
/// when no owner declares two `overload` functions of one name.
pub(crate) fn desugar(file: &HaxeFile) -> Option<(HaxeFile, OverloadGroups)> {
    // Renaming by declaration order cannot keep an override or an interface
    // member paired with what it overrides; such files are left as written.
    let hierarchical = file.declarations.iter().any(|decl| match decl {
        TypeDeclaration::Class(class) => class.fields.iter().any(|f| {
            f.modifiers.contains(&Modifier::Overload) && f.modifiers.contains(&Modifier::Override)
        }),
        TypeDeclaration::Interface(iface) => iface
            .fields
            .iter()
            .any(|f| f.modifiers.contains(&Modifier::Overload)),
        _ => false,
    });
    if hierarchical {
        return None;
    }
    let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    for decl in &file.declarations {
        if let TypeDeclaration::Class(class) = decl {
            for field in &class.fields {
                if let ClassFieldKind::Function(f) = &field.kind
                    && field.modifiers.contains(&Modifier::Overload)
                {
                    *counts
                        .entry((class.name.clone(), f.name.clone()))
                        .or_default() += 1;
                }
            }
        }
    }
    for field in &file.module_fields {
        if let ModuleFieldKind::Function(f) = &field.kind
            && field.modifiers.contains(&Modifier::Overload)
        {
            *counts
                .entry((MODULE_OWNER.to_string(), f.name.clone()))
                .or_default() += 1;
        }
    }
    // Constructors keep their name: a class has one `new`, and overloaded
    // constructors are not selected here.
    counts.retain(|(_, name), n| *n > 1 && name != "new");
    if counts.is_empty() {
        return None;
    }

    let mut file = file.clone();
    let mut groups = OverloadGroups::new();
    let mut rename = |owner: &str, f: &mut Function, is_static: bool| {
        let key = (owner.to_string(), f.name.clone());
        if !counts.contains_key(&key) {
            return;
        }
        let members = groups.entry(key).or_default();
        f.name = mangled(&f.name, members.len());
        members.push(OverloadCandidate {
            name: f.name.clone(),
            function: f.clone(),
            is_static,
        });
    };
    for decl in &mut file.declarations {
        if let TypeDeclaration::Class(class) = decl {
            let owner = class.name.clone();
            for field in &mut class.fields {
                let overload = field.modifiers.contains(&Modifier::Overload);
                let is_static = field.modifiers.contains(&Modifier::Static);
                if let ClassFieldKind::Function(f) = &mut field.kind
                    && overload
                {
                    rename(&owner, f, is_static);
                }
            }
        }
    }
    for field in &mut file.module_fields {
        let overload = field.modifiers.contains(&Modifier::Overload);
        if let ModuleFieldKind::Function(f) = &mut field.kind
            && overload
        {
            rename(MODULE_OWNER, f, true);
        }
    }
    Some((file, groups))
}
