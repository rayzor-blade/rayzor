//! `overload` functions: same-named declarations are renamed apart before
//! lowering, and each group is kept so a call can pick its member from the
//! argument types (`call_overload.rs`).
//!
//! A member is named after its parameter types, so an interface member and
//! the class member implementing it get the same name. An `override` takes
//! the name of the member it overrides, found by comparing parameter types
//! with the parent's type parameters substituted, so virtual dispatch keeps
//! working through the renamed members.
//!
//! Overloaded constructors become instance methods named per class, beside
//! a constructor without parameters; `new C(args)` builds the instance with
//! that and then runs the selected method on it.

use parser::{
    Access, BlockElement, ClassField, ClassFieldKind, Expr, ExprKind, Function, FunctionParam,
    HaxeFile, Modifier, ModuleFieldKind, Span, Type, TypeDeclaration,
};
use std::collections::{BTreeMap, BTreeSet};

/// Owner of module-level functions in the group keys.
pub(crate) const MODULE_OWNER: &str = "";

/// One renamed member of a group.
#[derive(Clone)]
pub(crate) struct OverloadCandidate {
    pub name: String,
    /// Parameter types are written in the owner's frame: an inherited
    /// member's type parameters are replaced by the owner's arguments.
    pub function: Function,
    pub is_static: bool,
    /// The owner's type parameters, bound at a call from the receiver.
    pub owner_params: Vec<String>,
}

#[derive(Clone, Default)]
pub(crate) struct OverloadGroups {
    /// `(owner type name, function name)` -> every member a call on the
    /// owner can reach: its own first, then inherited ones it does not
    /// override.
    pub groups: BTreeMap<(String, String), Vec<OverloadCandidate>>,
    /// Each class or interface of the file -> the types it extends or
    /// implements, by name.
    pub parents: BTreeMap<String, Vec<String>>,
}

impl OverloadGroups {
    pub(crate) fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// A member as seen from one type, for the names being renamed.
#[derive(Clone)]
struct Member {
    base: String,
    name: String,
    function: Function,
    is_static: bool,
}

struct TypeInfo {
    decl: usize,
    params: Vec<String>,
    superclass: Option<(String, Vec<Type>)>,
    interfaces: Vec<(String, Vec<Type>)>,
    is_interface: bool,
}

pub(crate) fn mangled(name: &str, sig: &str) -> String {
    let mut out = format!("{name}__ovl_");
    for c in sig.chars() {
        match c {
            c if c.is_ascii_alphanumeric() || c == '_' => out.push(c),
            '<' => out.push_str("_of_"),
            ',' => out.push_str("__"),
            '?' => out.push_str("opt_"),
            _ => out.push('_'),
        }
    }
    out
}

/// The file with every overload group renamed apart, and the groups. None
/// when no name is declared `overload` twice.
pub(crate) fn desugar(file: &HaxeFile) -> Option<(HaxeFile, OverloadGroups)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut module_counts: BTreeMap<String, usize> = BTreeMap::new();
    for decl in &file.declarations {
        for field in type_fields(decl) {
            if let ClassFieldKind::Function(f) = &field.kind
                && field.modifiers.contains(&Modifier::Overload)
            {
                *counts.entry(f.name.clone()).or_default() += 1;
            }
        }
    }
    for field in &file.module_fields {
        if let ModuleFieldKind::Function(f) = &field.kind
            && field.modifiers.contains(&Modifier::Overload)
        {
            *module_counts.entry(f.name.clone()).or_default() += 1;
        }
    }
    // Constructors are handled per class by `desugar_constructors`.
    let names: BTreeSet<String> = counts
        .into_iter()
        .filter(|(name, n)| *n > 1 && name != "new")
        .map(|(name, _)| name)
        .collect();
    let module_names: BTreeSet<String> = module_counts
        .into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(name, _)| name)
        .collect();
    let overloaded_constructors = file.declarations.iter().any(|decl| {
        type_fields(decl)
            .iter()
            .filter(|f| f.modifiers.contains(&Modifier::Overload) && constructor(f).is_some())
            .count()
            > 1
    });
    if names.is_empty() && module_names.is_empty() && !overloaded_constructors {
        return None;
    }

    let mut file = file.clone();
    let mut out = OverloadGroups::default();
    let infos = type_infos(&file);
    for (owner, info) in &infos {
        let parents = info
            .superclass
            .iter()
            .chain(&info.interfaces)
            .map(|(name, _)| name.clone())
            .collect();
        out.parents.insert(owner.clone(), parents);
    }

    // Parents before children, so an override finds its parent's names.
    let mut order = Vec::new();
    let mut seen = BTreeSet::new();
    for owner in infos.keys() {
        visit(owner, &infos, &mut seen, &mut order);
    }
    let mut effective: BTreeMap<String, Vec<Member>> = BTreeMap::new();
    for owner in &order {
        let info = &infos[owner];
        let inherit_from = |(parent, args): &(String, Vec<Type>)| {
            let params = infos
                .get(parent)
                .map(|p| p.params.as_slice())
                .unwrap_or(&[]);
            effective
                .get(parent)
                .map(|members| inherit(members, params, args))
                .unwrap_or_default()
        };
        let from_super: Vec<Member> = info.superclass.iter().flat_map(inherit_from).collect();
        let from_ifaces: Vec<Member> = info.interfaces.iter().flat_map(inherit_from).collect();

        let mut own: Vec<Member> = Vec::new();
        let fields = type_fields_mut(&mut file.declarations[info.decl]);
        for field in fields {
            let overload = field.modifiers.contains(&Modifier::Overload);
            let is_override = field.modifiers.contains(&Modifier::Override);
            let is_static = field.modifiers.contains(&Modifier::Static);
            let ClassFieldKind::Function(f) = &mut field.kind else {
                continue;
            };
            if !names.contains(&f.name) {
                continue;
            }
            let base = f.name.clone();
            let name = if overload {
                member_name(f, owner, is_override, &from_super, &from_ifaces, &own)
            } else {
                base.clone()
            };
            f.name = name.clone();
            own.push(Member {
                base,
                name,
                function: f.clone(),
                is_static,
            });
        }

        // A class reaches its superclass's members; an interface the
        // members of every interface it extends.
        let inherited = if info.is_interface {
            from_ifaces
        } else {
            from_super
        };
        let mut members = own;
        for member in inherited {
            let shadowed = members.iter().any(|m| {
                m.name == member.name
                    || (m.base == member.base
                        && sig_text(&m.function.params) == sig_text(&member.function.params))
            });
            if !shadowed {
                members.push(member);
            }
        }
        for base in &names {
            let group: Vec<&Member> = members.iter().filter(|m| m.base == *base).collect();
            if group.len() > 1 || group.iter().any(|m| m.name != m.base) {
                let candidates = group
                    .into_iter()
                    .map(|m| OverloadCandidate {
                        name: m.name.clone(),
                        function: m.function.clone(),
                        is_static: m.is_static,
                        owner_params: info.params.clone(),
                    })
                    .collect();
                out.groups.insert((owner.clone(), base.clone()), candidates);
            }
        }
        effective.insert(owner.clone(), members);
    }
    if overloaded_constructors {
        desugar_constructors(&mut file, &infos, &order, &mut out);
    }

    for field in &mut file.module_fields {
        let overload = field.modifiers.contains(&Modifier::Overload);
        if let ModuleFieldKind::Function(f) = &mut field.kind
            && overload
            && module_names.contains(&f.name)
        {
            let key = (MODULE_OWNER.to_string(), f.name.clone());
            f.name = mangled(&f.name, &sig_text(&f.params));
            out.groups.entry(key).or_default().push(OverloadCandidate {
                name: f.name.clone(),
                function: f.clone(),
                is_static: true,
                owner_params: Vec::new(),
            });
        }
    }
    Some((file, out))
}

/// Rename the overloaded constructors of each class whose constructors are
/// all `overload`, none callable without arguments, and whose ancestors are
/// in this file and either qualify too or declare no constructor. Classes
/// below such a class without a constructor of their own inherit its group.
fn desugar_constructors(
    file: &mut HaxeFile,
    infos: &BTreeMap<String, TypeInfo>,
    order: &[String],
    out: &mut OverloadGroups,
) {
    // Class -> whether it declares the overloads (else it inherits them).
    let mut eligible: BTreeMap<String, bool> = BTreeMap::new();
    for owner in order {
        let info = &infos[owner];
        let parent_ok = match &info.superclass {
            None => true,
            Some((parent, _)) => eligible.contains_key(parent),
        };
        if info.is_interface || !parent_ok {
            continue;
        }
        let ctors: Vec<&ClassField> = type_fields(&file.declarations[info.decl])
            .iter()
            .filter(|f| constructor(f).is_some())
            .collect();
        let qualifies = ctors.len() > 1
            && ctors.iter().all(|f| {
                f.modifiers.contains(&Modifier::Overload)
                    && constructor(f).is_some_and(|func| {
                        func.params
                            .iter()
                            .any(|p| !(p.optional || p.rest || p.default_value.is_some()))
                    })
            });
        if qualifies {
            eligible.insert(owner.clone(), true);
        } else if ctors.is_empty() && info.superclass.is_some() {
            eligible.insert(owner.clone(), false);
        }
    }

    let mut groups: BTreeMap<String, Vec<Member>> = BTreeMap::new();
    for owner in order {
        let Some(&declares) = eligible.get(owner) else {
            continue;
        };
        let info = &infos[owner];
        let mut group = Vec::new();
        if declares {
            for field in type_fields_mut(&mut file.declarations[info.decl]) {
                let ClassFieldKind::Function(f) = &mut field.kind else {
                    continue;
                };
                if f.name != "new" {
                    continue;
                }
                f.name = mangled(&format!("new_{owner}"), &sig_text(&f.params));
                if let Some(body) = &mut f.body {
                    route_super_calls(body);
                }
                group.push(Member {
                    base: "new".to_string(),
                    name: f.name.clone(),
                    function: f.clone(),
                    is_static: false,
                });
            }
        } else if let Some((parent, args)) = &info.superclass {
            let params = infos
                .get(parent)
                .map(|p| p.params.as_slice())
                .unwrap_or(&[]);
            group = groups
                .get(parent)
                .map(|members| inherit(members, params, args))
                .unwrap_or_default();
        }
        if let TypeDeclaration::Class(class) = &mut file.declarations[info.decl] {
            let span = class.span;
            class
                .fields
                .push(empty_constructor(span, info.superclass.is_some()));
        }
        let candidates = group
            .iter()
            .map(|m| OverloadCandidate {
                name: m.name.clone(),
                function: m.function.clone(),
                is_static: false,
                owner_params: info.params.clone(),
            })
            .collect();
        out.groups
            .insert((owner.clone(), "new".to_string()), candidates);
        groups.insert(owner.clone(), group);
    }
}

fn constructor(field: &ClassField) -> Option<&Function> {
    match &field.kind {
        ClassFieldKind::Function(f) if f.name == "new" => Some(f),
        _ => None,
    }
}

/// A constructor body's `super(args)` statements, as `super.new(args)` for
/// the call to pick among the parent's renamed constructors.
fn route_super_calls(body: &mut Expr) {
    let ExprKind::Block(elements) = &mut body.kind else {
        return;
    };
    for element in elements {
        if let BlockElement::Expr(expr) = element
            && let ExprKind::Call { expr: callee, .. } = &mut expr.kind
            && matches!(callee.kind, ExprKind::Super)
        {
            let span = callee.span;
            **callee = Expr {
                kind: ExprKind::Field {
                    expr: Box::new(Expr {
                        kind: ExprKind::Super,
                        span,
                    }),
                    field: "new".to_string(),
                    is_optional: false,
                },
                span,
            };
        }
    }
}

/// `public function new() { super(); }`, or an empty body at the root.
fn empty_constructor(span: Span, call_super: bool) -> ClassField {
    let at = |kind: ExprKind| Expr { kind, span };
    let body = if call_super {
        vec![BlockElement::Expr(at(ExprKind::Call {
            expr: Box::new(at(ExprKind::Super)),
            args: Vec::new(),
        }))]
    } else {
        Vec::new()
    };
    ClassField {
        meta: Vec::new(),
        access: Some(Access::Public),
        modifiers: Vec::new(),
        kind: ClassFieldKind::Function(Function {
            name: "new".to_string(),
            type_params: Vec::new(),
            params: Vec::new(),
            return_type: None,
            body: Some(Box::new(at(ExprKind::Block(body)))),
            span,
        }),
        span,
    }
}

/// The name of an `overload` member: an override's is the one it overrides,
/// an interface member's implementation takes the interface's name, and
/// anything else is named after its parameter types.
fn member_name(
    f: &Function,
    owner: &str,
    is_override: bool,
    from_super: &[Member],
    from_ifaces: &[Member],
    own: &[Member],
) -> String {
    let sig = sig_text(&f.params);
    let same_sig = |members: &[Member]| {
        members
            .iter()
            .find(|m| !m.is_static && m.base == f.name && sig_text(&m.function.params) == sig)
            .map(|m| m.name.clone())
    };
    if is_override {
        if let Some(name) = same_sig(from_super) {
            return name;
        }
        // Parameter types spelled differently: the one parent member of
        // that arity, if there is exactly one.
        let mut arity = from_super.iter().filter(|m| {
            !m.is_static && m.base == f.name && m.function.params.len() == f.params.len()
        });
        if let (Some(m), None) = (arity.next(), arity.next()) {
            return m.name.clone();
        }
        // The overridden member is outside this file, under its own name.
        if !own.iter().any(|m| m.name == f.name) {
            return f.name.clone();
        }
    }
    if let Some(name) = same_sig(from_ifaces) {
        return name;
    }
    let name = mangled(&f.name, &sig);
    if from_super.iter().any(|m| m.name == name) {
        format!("{name}_{owner}")
    } else {
        name
    }
}

/// `members` of a parent seen from a child that names it `Parent<args>`.
/// Statics are not inherited.
fn inherit(members: &[Member], params: &[String], args: &[Type]) -> Vec<Member> {
    let map: BTreeMap<String, Type> = params.iter().cloned().zip(args.iter().cloned()).collect();
    members
        .iter()
        .filter(|m| !m.is_static)
        .map(|m| {
            let mut m = m.clone();
            let mut map = map.clone();
            for tp in &m.function.type_params {
                map.remove(&tp.name);
            }
            if !map.is_empty() {
                for param in &mut m.function.params {
                    if let Some(hint) = &mut param.type_hint {
                        *hint = subst_type(hint, &map);
                    }
                }
            }
            m
        })
        .collect()
}

fn visit(
    owner: &str,
    infos: &BTreeMap<String, TypeInfo>,
    seen: &mut BTreeSet<String>,
    order: &mut Vec<String>,
) {
    if !seen.insert(owner.to_string()) {
        return;
    }
    let Some(info) = infos.get(owner) else {
        return;
    };
    for (parent, _) in info.superclass.iter().chain(&info.interfaces) {
        visit(parent, infos, seen, order);
    }
    order.push(owner.to_string());
}

fn type_infos(file: &HaxeFile) -> BTreeMap<String, TypeInfo> {
    let path_of = |ty: &Type| match ty {
        Type::Path { path, params, .. } => Some((path.name.clone(), params.clone())),
        _ => None,
    };
    let mut infos = BTreeMap::new();
    for (decl, declaration) in file.declarations.iter().enumerate() {
        let info = match declaration {
            TypeDeclaration::Class(class) => TypeInfo {
                decl,
                params: class.type_params.iter().map(|p| p.name.clone()).collect(),
                superclass: class.extends.as_ref().and_then(path_of),
                interfaces: class.implements.iter().filter_map(path_of).collect(),
                is_interface: false,
            },
            TypeDeclaration::Interface(iface) => TypeInfo {
                decl,
                params: iface.type_params.iter().map(|p| p.name.clone()).collect(),
                superclass: None,
                interfaces: iface.extends.iter().filter_map(path_of).collect(),
                is_interface: true,
            },
            _ => continue,
        };
        let name = match declaration {
            TypeDeclaration::Class(class) => class.name.clone(),
            TypeDeclaration::Interface(iface) => iface.name.clone(),
            _ => continue,
        };
        infos.insert(name, info);
    }
    infos
}

fn type_fields(decl: &TypeDeclaration) -> &[ClassField] {
    match decl {
        TypeDeclaration::Class(class) => &class.fields,
        TypeDeclaration::Interface(iface) => &iface.fields,
        _ => &[],
    }
}

fn type_fields_mut(decl: &mut TypeDeclaration) -> &mut [ClassField] {
    match decl {
        TypeDeclaration::Class(class) => &mut class.fields,
        TypeDeclaration::Interface(iface) => &mut iface.fields,
        _ => &mut [],
    }
}

/// The parameter types as written, `?` marking an optional one.
fn sig_text(params: &[FunctionParam]) -> String {
    params
        .iter()
        .map(|p| {
            let ty = p
                .type_hint
                .as_ref()
                .map_or_else(|| "_".to_string(), type_text);
            if p.optional || p.default_value.is_some() {
                format!("?{ty}")
            } else {
                ty
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn type_text(ty: &Type) -> String {
    let list =
        |types: &[Type]| -> String { types.iter().map(type_text).collect::<Vec<_>>().join(",") };
    match ty {
        Type::Path { path, params, .. } => {
            let mut text: String = path.package.iter().map(|p| format!("{p}.")).collect();
            text.push_str(&path.name);
            if let Some(sub) = &path.sub {
                text.push('.');
                text.push_str(sub);
            }
            if !params.is_empty() {
                text.push_str(&format!("<{}>", list(params.as_slice())));
            }
            text
        }
        Type::Function { params, ret, .. } => {
            format!("({})->{}", list(params.as_slice()), type_text(ret))
        }
        Type::Anonymous { fields, .. } => {
            let fields: Vec<String> = fields
                .iter()
                .map(|f| {
                    let opt = if f.optional { "?" } else { "" };
                    format!("{opt}{}:{}", f.name, type_text(&f.type_hint))
                })
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        Type::Optional { inner, .. } => format!("Null<{}>", type_text(inner)),
        Type::Parenthesis { inner, .. } => type_text(inner),
        Type::Intersection { left, right, .. } => {
            format!("{}&{}", type_text(left), type_text(right))
        }
        Type::Wildcard { .. } => "?".to_string(),
        Type::Const { .. } => "const".to_string(),
    }
}

/// `ty` with each bare type-parameter name in `map` replaced.
fn subst_type(ty: &Type, map: &BTreeMap<String, Type>) -> Type {
    let each = |types: &[Type]| -> Vec<Type> { types.iter().map(|t| subst_type(t, map)).collect() };
    match ty {
        Type::Path { path, params, span } => {
            if path.package.is_empty()
                && path.sub.is_none()
                && params.is_empty()
                && let Some(bound) = map.get(&path.name)
            {
                return bound.clone();
            }
            Type::Path {
                path: path.clone(),
                params: each(params.as_slice()),
                span: *span,
            }
        }
        Type::Function { params, ret, span } => Type::Function {
            params: each(params.as_slice()),
            ret: Box::new(subst_type(ret, map)),
            span: *span,
        },
        Type::Anonymous { fields, span } => Type::Anonymous {
            fields: fields
                .iter()
                .map(|f| {
                    let mut f = f.clone();
                    f.type_hint = subst_type(&f.type_hint, map);
                    f
                })
                .collect(),
            span: *span,
        },
        Type::Optional { inner, span } => Type::Optional {
            inner: Box::new(subst_type(inner, map)),
            span: *span,
        },
        Type::Parenthesis { inner, span } => Type::Parenthesis {
            inner: Box::new(subst_type(inner, map)),
            span: *span,
        },
        Type::Intersection { left, right, span } => Type::Intersection {
            left: Box::new(subst_type(left, map)),
            right: Box::new(subst_type(right, map)),
            span: *span,
        },
        Type::Wildcard { .. } | Type::Const { .. } => ty.clone(),
    }
}

/// Whether `ty` names any of `names` as a bare type.
pub(crate) fn mentions(ty: &Type, names: &[&str]) -> bool {
    match ty {
        Type::Path { path, params, .. } => {
            (path.package.is_empty() && names.contains(&path.name.as_str()))
                || params.iter().any(|p| mentions(p, names))
        }
        Type::Function { params, ret, .. } => {
            params.iter().any(|p| mentions(p, names)) || mentions(ret, names)
        }
        Type::Anonymous { fields, .. } => fields.iter().any(|f| mentions(&f.type_hint, names)),
        Type::Optional { inner, .. } | Type::Parenthesis { inner, .. } => mentions(inner, names),
        Type::Intersection { left, right, .. } => mentions(left, names) || mentions(right, names),
        Type::Wildcard { .. } | Type::Const { .. } => false,
    }
}
