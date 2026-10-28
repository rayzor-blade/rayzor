//! Diagnostics a macro probe raises for `switch` patterns, in Haxe's wording:
//! or-pattern bindings and the report of unmatched patterns.

use super::{AstLowering, LoweringError, LoweringResult};
use crate::tast::{
    SymbolKind, TypeId,
    core::TypeKind,
    node::{TypedExpression, TypedExpressionKind},
};
use parser::{Case, Pattern};

/// A pattern as the matrix sees it.
#[derive(Clone)]
enum Pat {
    Any,
    Con(String, Vec<Pat>),
    /// A constant of an open type: it covers one value of infinitely many.
    Lit,
}

struct Ctor {
    name: String,
    args: Vec<TypeId>,
}

/// The values a column can hold.
enum Sig {
    Closed(Vec<Ctor>),
    Open,
    Unknown,
}

impl AstLowering<'_> {
    pub(crate) fn macro_probe_check_patterns(
        &mut self,
        subject: &TypedExpression,
        cases: &[Case],
        has_default: bool,
    ) -> LoweringResult<()> {
        let columns = self.pattern_columns(subject, cases);
        let location = subject.source_location;
        let fail = |message: String| LoweringError::SemanticError { message, location };

        for case in cases {
            if case.patterns.len() > 1
                && case
                    .patterns
                    .iter()
                    .any(|pattern| matches!(pattern, Pattern::Extractor { .. }))
            {
                return Err(fail(
                    "Extractors cannot be combined in an or-pattern".to_string(),
                ));
            }
            for pattern in &case.patterns {
                match (&columns, pattern) {
                    (Columns::Tuple(types), Pattern::Array(parts))
                        if types.len() == parts.len() =>
                    {
                        for (part, ty) in parts.iter().zip(types) {
                            self.pattern_bindings(part, Some(*ty)).map_err(&fail)?;
                        }
                    }
                    (Columns::Single(ty), pattern) => {
                        self.pattern_bindings(pattern, *ty).map_err(&fail)?;
                    }
                    _ => {}
                }
            }
        }
        if has_default {
            return Ok(());
        }
        if let Some(report) = self.unmatched_patterns(&columns, cases) {
            return Err(fail(format!("Unmatched patterns: {report}")));
        }
        Ok(())
    }

    /// The subject's columns: one per element of a tuple subject that every
    /// case destructures, else the subject itself.
    fn pattern_columns(&self, subject: &TypedExpression, cases: &[Case]) -> Columns {
        if let TypedExpressionKind::ArrayLiteral { elements } = &subject.kind
            && cases.iter().all(|case| {
                case.patterns
                    .iter()
                    .all(|p| matches!(p, Pattern::Array(parts) if parts.len() == elements.len()))
            })
        {
            return Columns::Tuple(elements.iter().map(|e| e.expr_type).collect());
        }
        let declared = match &subject.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                self.macro_source_types.get(symbol_id).copied()
            }
            _ => None,
        };
        Columns::Single(
            declared
                .or(self.context.switch_discriminant_type)
                .filter(|ty| ty.is_valid()),
        )
    }

    /// The variables `pattern` binds, in order. Alternatives of an or-pattern
    /// must bind the same variables, each once and at one type.
    fn pattern_bindings(
        &mut self,
        pattern: &Pattern,
        ty: Option<TypeId>,
    ) -> Result<Vec<(String, Option<TypeId>)>, String> {
        let mut bound: Vec<(String, Option<TypeId>)> = Vec::new();
        match pattern {
            Pattern::Var(name) => {
                if !ty.is_some_and(|ty| self.constructor_of(ty, name).is_some()) {
                    bound.push((name.clone(), ty));
                }
            }
            Pattern::Bind { name, pattern } => {
                bound.push((name.clone(), ty));
                bound.extend(self.pattern_bindings(pattern, ty)?);
            }
            Pattern::Constructor { path, params } => {
                let args = ty
                    .map(|ty| self.constructor_arg_types(ty, &path.name))
                    .unwrap_or_default();
                for (i, param) in params.iter().enumerate() {
                    bound.extend(self.pattern_bindings(param, args.get(i).copied())?);
                }
            }
            Pattern::Or(alternatives) => {
                if alternatives
                    .iter()
                    .any(|pattern| matches!(pattern, Pattern::Extractor { .. }))
                {
                    return Err("Extractors cannot be combined in an or-pattern".to_string());
                }
                let sets = alternatives
                    .iter()
                    .map(|alt| self.pattern_bindings(alt, ty))
                    .collect::<Result<Vec<_>, _>>()?;
                let multiple = "must appear exactly once in each sub-pattern";
                if let Some((first, rest)) = sets.split_first() {
                    for set in rest {
                        if let Some((name, _)) =
                            set.iter().find(|(n, _)| !first.iter().any(|(m, _)| m == n))
                        {
                            return Err(format!("Variable {name} {multiple}"));
                        }
                        if let Some((name, _)) =
                            first.iter().find(|(n, _)| !set.iter().any(|(m, _)| m == n))
                        {
                            return Err(format!("Variable {name} {multiple}"));
                        }
                        for (name, later) in set {
                            let earlier =
                                first.iter().find(|(n, _)| n == name).and_then(|(_, t)| *t);
                            if let (Some(earlier), Some(later)) = (earlier, *later)
                                && !self.same_type(earlier, later)
                            {
                                return Err(format!(
                                    "{} should be {}",
                                    self.render_pattern_type(later),
                                    self.render_pattern_type(earlier)
                                ));
                            }
                        }
                    }
                    bound = first.clone();
                }
            }
            Pattern::Array(parts)
            | Pattern::ArrayRest {
                elements: parts, ..
            } => {
                for part in parts {
                    bound.extend(self.pattern_bindings(part, None)?);
                }
            }
            Pattern::Object { fields } => {
                for (_, part) in fields {
                    bound.extend(self.pattern_bindings(part, None)?);
                }
            }
            Pattern::Type { var, .. } => bound.push((var.clone(), None)),
            Pattern::Extractor { .. } | Pattern::Const(_) | Pattern::Null | Pattern::Underscore => {
            }
        }
        for (i, (name, _)) in bound.iter().enumerate() {
            if bound[..i].iter().any(|(earlier, _)| earlier == name) {
                return Err(format!("Variable {name} is bound multiple times"));
            }
        }
        Ok(bound)
    }

    fn render_pattern_type(&self, ty: TypeId) -> String {
        super::macro_defer::render_type(
            ty,
            self.context.type_table,
            self.context.symbol_table,
            self.context.string_interner,
            0,
        )
    }

    /// The patterns no case matches, in Haxe's order and spelling; `None`
    /// when every value is matched or the patterns are beyond this check.
    fn unmatched_patterns(&mut self, columns: &Columns, cases: &[Case]) -> Option<String> {
        let types: Vec<Option<TypeId>> = match columns {
            Columns::Single(ty) => vec![*ty],
            Columns::Tuple(types) => types.iter().map(|t| Some(*t)).collect(),
        };
        // Top-level enum switches may omit null; nested constructor arguments
        // still have to cover every value of their declared nullable type.
        let types: Vec<_> = types
            .into_iter()
            .map(|ty| {
                ty.map(|mut ty| {
                    let table = self.context.type_table.borrow();
                    for _ in 0..32 {
                        ty = Self::resolve_alias_chain(&table, ty);
                        match table.get(ty).map(|ty| &ty.kind) {
                            Some(TypeKind::Optional { inner_type }) => ty = *inner_type,
                            _ => break,
                        }
                    }
                    ty
                })
            })
            .collect();
        let mut rows: Vec<Vec<Pat>> = Vec::new();
        for case in cases {
            // A guarded case may not match, so it covers nothing.
            if case.guard.is_some() {
                continue;
            }
            for pattern in &case.patterns {
                let row_patterns: Vec<&Pattern> = match (columns, pattern) {
                    (Columns::Tuple(_), Pattern::Array(parts)) => parts.iter().collect(),
                    (Columns::Single(_), pattern) => vec![pattern],
                    _ => return None,
                };
                let mut expanded: Vec<Vec<Pat>> = vec![Vec::new()];
                for (part, ty) in row_patterns.into_iter().zip(&types) {
                    let alternatives = self.matrix_patterns(part, *ty)?;
                    expanded = expanded
                        .into_iter()
                        .flat_map(|prefix| {
                            alternatives.iter().map(move |alt| {
                                let mut row = prefix.clone();
                                row.push(alt.clone());
                                row
                            })
                        })
                        .collect();
                }
                rows.extend(expanded);
            }
        }
        let missing = self.missing(&rows, &types)?;
        if missing.is_empty() {
            return None;
        }
        let spelled: Vec<String> = missing
            .into_iter()
            .map(|columns| match columns.len() {
                1 => columns.into_iter().next().unwrap_or_default(),
                _ => {
                    let shown: Vec<String> = columns.into_iter().filter(|c| c != "_").collect();
                    if shown.is_empty() {
                        "_".to_string()
                    } else {
                        shown.join(", ")
                    }
                }
            })
            .collect();
        Some(spelled.join(" | "))
    }

    /// The matrix patterns `pattern` stands for at type `ty`: more than one
    /// for an or-pattern. `None` for a pattern the matrix cannot express.
    fn matrix_patterns(&mut self, pattern: &Pattern, ty: Option<TypeId>) -> Option<Vec<Pat>> {
        match pattern {
            Pattern::Underscore => Some(vec![Pat::Any]),
            Pattern::Var(name) => Some(vec![
                match ty.and_then(|ty| self.constructor_of(ty, name)) {
                    Some(_) => Pat::Con(name.clone(), Vec::new()),
                    None => Pat::Any,
                },
            ]),
            Pattern::Bind { pattern, .. } => self.matrix_patterns(pattern, ty),
            Pattern::Const(expr) => Some(vec![match &expr.kind {
                parser::ExprKind::Bool(value) => Pat::Con(value.to_string(), Vec::new()),
                parser::ExprKind::Ident(name)
                    if ty.is_some_and(|ty| self.constructor_of(ty, name).is_some()) =>
                {
                    Pat::Con(name.clone(), Vec::new())
                }
                _ => Pat::Lit,
            }]),
            Pattern::Null => Some(vec![Pat::Con("null".to_string(), Vec::new())]),
            Pattern::Or(alternatives) => {
                let mut all = Vec::new();
                for alt in alternatives {
                    all.extend(self.matrix_patterns(alt, ty)?);
                }
                Some(all)
            }
            Pattern::Constructor { path, params } => {
                let args = ty
                    .map(|ty| self.constructor_arg_types(ty, &path.name))
                    .unwrap_or_default();
                let mut combos: Vec<Vec<Pat>> = vec![Vec::new()];
                for (i, param) in params.iter().enumerate() {
                    let options = self.matrix_patterns(param, args.get(i).copied())?;
                    combos = combos
                        .into_iter()
                        .flat_map(|prefix| {
                            options.iter().map(move |option| {
                                let mut next = prefix.clone();
                                next.push(option.clone());
                                next
                            })
                        })
                        .collect();
                }
                Some(
                    combos
                        .into_iter()
                        .map(|args| Pat::Con(path.name.clone(), args))
                        .collect(),
                )
            }
            Pattern::Array(_)
            | Pattern::ArrayRest { .. }
            | Pattern::Object { .. }
            | Pattern::Type { .. }
            | Pattern::Extractor { .. } => None,
        }
    }

    /// The patterns of `types` no row of `rows` matches, each spelled per
    /// column. A column whose constructors are not all present reports the
    /// missing ones and goes no deeper.
    fn missing(&mut self, rows: &[Vec<Pat>], types: &[Option<TypeId>]) -> Option<Vec<Vec<String>>> {
        let Some((&column, rest)) = types.split_first() else {
            return Some(if rows.is_empty() {
                vec![Vec::new()]
            } else {
                Vec::new()
            });
        };
        if rows.is_empty() {
            return Some(vec![vec!["_".to_string(); types.len()]]);
        }
        let default_rows: Vec<Vec<Pat>> = rows
            .iter()
            .filter(|row| matches!(row[0], Pat::Any))
            .map(|row| row[1..].to_vec())
            .collect();
        let signature = column.map_or(Sig::Unknown, |ty| self.signature(ty));
        match signature {
            Sig::Unknown => Some(Vec::new()),
            Sig::Open => {
                let inner = self.missing(&default_rows, rest)?;
                Some(
                    inner
                        .into_iter()
                        .map(|mut tail| {
                            tail.insert(0, "_".to_string());
                            tail
                        })
                        .collect(),
                )
            }
            Sig::Closed(constructors) => {
                let present = |name: &str| {
                    rows.iter()
                        .any(|row| matches!(&row[0], Pat::Con(n, _) if n == name))
                };
                if constructors.iter().any(|c| !present(&c.name)) {
                    let inner = self.missing(&default_rows, rest)?;
                    let mut out = Vec::new();
                    for constructor in constructors.iter().filter(|c| !present(&c.name)) {
                        for tail in &inner {
                            let mut line = vec![constructor.name.clone()];
                            line.extend(tail.iter().cloned());
                            out.push(line);
                        }
                    }
                    return Some(out);
                }
                let mut out = Vec::new();
                for constructor in &constructors {
                    let arity = constructor.args.len();
                    let specialized: Vec<Vec<Pat>> = rows
                        .iter()
                        .filter_map(|row| match &row[0] {
                            Pat::Con(n, args) if *n == constructor.name => {
                                let mut next = args.clone();
                                next.resize(arity, Pat::Any);
                                next.extend(row[1..].iter().cloned());
                                Some(next)
                            }
                            Pat::Any => {
                                let mut next = vec![Pat::Any; arity];
                                next.extend(row[1..].iter().cloned());
                                Some(next)
                            }
                            _ => None,
                        })
                        .collect();
                    let mut next_types: Vec<Option<TypeId>> =
                        constructor.args.iter().map(|t| Some(*t)).collect();
                    next_types.extend(rest.iter().copied());
                    for line in self.missing(&specialized, &next_types)? {
                        let name = if arity == 0 {
                            constructor.name.clone()
                        } else {
                            format!("{}({})", constructor.name, line[..arity].join(", "))
                        };
                        let mut spelled = vec![name];
                        spelled.extend(line[arity..].iter().cloned());
                        out.push(spelled);
                    }
                }
                Some(out)
            }
        }
    }

    /// What values a column of type `ty` can hold.
    fn signature(&mut self, ty: TypeId) -> Sig {
        let mut current = ty;
        let mut nullable = false;
        for _ in 0..8 {
            let kind = self
                .context
                .type_table
                .borrow()
                .get(current)
                .map(|t| t.kind.clone());
            match kind {
                Some(TypeKind::Optional { inner_type }) => {
                    nullable = true;
                    current = inner_type;
                }
                Some(TypeKind::TypeAlias { target_type, .. }) => current = target_type,
                Some(TypeKind::GenericInstance { base_type, .. }) => current = base_type,
                Some(TypeKind::Bool) => {
                    return Sig::Closed(
                        ["true", "false"]
                            .iter()
                            .map(|name| Ctor {
                                name: name.to_string(),
                                args: Vec::new(),
                            })
                            .collect(),
                    );
                }
                Some(TypeKind::Int | TypeKind::Float | TypeKind::String) => return Sig::Open,
                Some(TypeKind::Enum { symbol_id, .. }) => {
                    let mut signature = self.enum_signature(symbol_id, ty);
                    if nullable && let Sig::Closed(constructors) = &mut signature {
                        constructors.push(Ctor {
                            name: "null".to_string(),
                            args: Vec::new(),
                        });
                    }
                    return signature;
                }
                Some(TypeKind::Abstract { symbol_id, .. })
                    if self.enum_abstracts.contains(&symbol_id)
                        || self.context.symbol_table.is_enum_abstract(symbol_id) =>
                {
                    let values = self
                        .class_fields
                        .get(&symbol_id)
                        .map(|fields| {
                            fields
                                .iter()
                                .filter(|(_, symbol, is_static)| {
                                    *is_static
                                        && self
                                            .context
                                            .symbol_table
                                            .get_symbol(*symbol)
                                            .is_some_and(|s| {
                                                !matches!(s.kind, SymbolKind::Function)
                                            })
                                })
                                .filter_map(|(name, _, _)| self.context.string_interner.get(*name))
                                .map(|name| Ctor {
                                    name: name.to_string(),
                                    args: Vec::new(),
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    return if values.is_empty() {
                        Sig::Unknown
                    } else {
                        Sig::Closed(values)
                    };
                }
                _ => return Sig::Unknown,
            }
        }
        Sig::Unknown
    }

    /// An enum's constructors; unknown when any is declared with its own
    /// result type (a GADT), as which of them can occur depends on the type.
    fn enum_signature(&mut self, enum_symbol: crate::tast::SymbolId, ty: TypeId) -> Sig {
        let Some(variants) = self
            .context
            .symbol_table
            .get_enum_variants(enum_symbol)
            .cloned()
        else {
            return Sig::Unknown;
        };
        let mut out = Vec::new();
        for variant in variants {
            let Some(symbol) = self.context.symbol_table.get_symbol(variant) else {
                return Sig::Unknown;
            };
            let (name, declared) = (symbol.name, symbol.type_id);
            let Some(name) = self.context.string_interner.get(name).map(str::to_string) else {
                return Sig::Unknown;
            };
            if self.variant_has_own_result(declared, enum_symbol) {
                return Sig::Unknown;
            }
            let args = self.constructor_arg_types(ty, &name);
            out.push(Ctor { name, args });
        }
        Sig::Closed(out)
    }

    /// Whether a constructor of type `declared` returns something other than
    /// its own enum applied to the enum's parameters.
    fn variant_has_own_result(&self, declared: TypeId, enum_symbol: crate::tast::SymbolId) -> bool {
        let table = self.context.type_table.borrow();
        let Some(TypeKind::Function { return_type, .. }) = table.get(declared).map(|t| &t.kind)
        else {
            return false;
        };
        match table.get(*return_type).map(|t| &t.kind) {
            Some(TypeKind::Enum {
                symbol_id,
                type_args,
            }) => {
                *symbol_id != enum_symbol
                    || type_args.iter().any(|arg| {
                        !matches!(
                            table.get(*arg).map(|t| &t.kind),
                            Some(TypeKind::TypeParameter { .. })
                        )
                    })
            }
            _ => false,
        }
    }

    /// The constructor `name` of the enum or enum abstract `ty` names.
    fn constructor_of(&mut self, ty: TypeId, name: &str) -> Option<crate::tast::SymbolId> {
        let interned = self.context.intern_string(name);
        self.resolve_enum_constructor_of(ty, interned)
    }

    /// The argument types of constructor `name` of `ty`, type arguments applied.
    fn constructor_arg_types(&mut self, ty: TypeId, name: &str) -> Vec<TypeId> {
        let Some(constructor) = self.constructor_of(ty, name) else {
            return Vec::new();
        };
        let Some(declared) = self
            .context
            .symbol_table
            .get_symbol(constructor)
            .map(|s| s.type_id)
        else {
            return Vec::new();
        };
        let applied = self.enum_constructor_for_result(declared, ty);
        match self
            .context
            .type_table
            .borrow()
            .get(applied)
            .map(|t| &t.kind)
        {
            Some(TypeKind::Function { params, .. }) => params.clone(),
            _ => Vec::new(),
        }
    }
}

enum Columns {
    Single(Option<TypeId>),
    Tuple(Vec<TypeId>),
}
