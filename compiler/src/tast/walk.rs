//! Traversal of a typed file that reports every symbol it declares and every
//! symbol an expression refers to, each with the location the TAST holds.

use super::node::{
    StringInterpolationPart, TypedCatchClause, TypedClass, TypedComprehensionFor, TypedExpression,
    TypedExpressionKind, TypedField, TypedFile, TypedFunction, TypedParameter, TypedStatement,
    TypedSwitchCase,
};
use super::{SourceLocation, SymbolId, TypeId};

/// What a walk reports. `at` is the location of the declaring or referring
/// node, which may start before the name itself.
pub trait SymbolSink {
    fn declares(&mut self, symbol: SymbolId, at: SourceLocation);
    /// `member`: the name follows a `.` when written qualified.
    fn refers(&mut self, _symbol: SymbolId, _at: SourceLocation, _member: bool) {}
    fn constructs(&mut self, _ty: TypeId, _at: SourceLocation) {}
    fn refers_typed(&mut self, symbol: SymbolId, at: SourceLocation, member: bool, _ty: TypeId) {
        self.refers(symbol, at, member);
    }
}

pub fn walk_file(file: &TypedFile, sink: &mut impl SymbolSink) {
    for class in &file.classes {
        walk_class(class, sink);
    }
    for interface in &file.interfaces {
        sink.declares(interface.symbol_id, interface.source_location);
        for field in &interface.fields {
            walk_field(field, sink);
        }
    }
    for e in &file.enums {
        sink.declares(e.symbol_id, e.source_location);
        for variant in &e.variants {
            for param in &variant.parameters {
                walk_parameter(param, sink);
            }
        }
    }
    for alias in &file.type_aliases {
        sink.declares(alias.symbol_id, alias.source_location);
    }
    for a in &file.abstracts {
        sink.declares(a.symbol_id, a.source_location);
        for field in &a.fields {
            walk_field(field, sink);
        }
        for f in a.methods.iter().chain(&a.constructors) {
            walk_function(f, sink);
        }
    }
    for f in &file.functions {
        walk_function(f, sink);
    }
}

fn walk_class(class: &TypedClass, sink: &mut impl SymbolSink) {
    sink.declares(class.symbol_id, class.source_location);
    for field in &class.fields {
        walk_field(field, sink);
    }
    for f in class.methods.iter().chain(&class.constructors) {
        walk_function(f, sink);
    }
}

fn walk_field(field: &TypedField, sink: &mut impl SymbolSink) {
    sink.declares(field.symbol_id, field.source_location);
    if let Some(init) = &field.initializer {
        walk_expression(init, sink);
    }
}

pub fn walk_function(f: &TypedFunction, sink: &mut impl SymbolSink) {
    sink.declares(f.symbol_id, f.source_location);
    for param in &f.parameters {
        walk_parameter(param, sink);
    }
    for s in &f.body {
        walk_statement(s, sink);
    }
}

fn walk_parameter(param: &TypedParameter, sink: &mut impl SymbolSink) {
    sink.declares(param.symbol_id, param.source_location);
    if let Some(default) = &param.default_value {
        walk_expression(default, sink);
    }
}

pub fn walk_statement(s: &TypedStatement, sink: &mut impl SymbolSink) {
    match s {
        TypedStatement::Expression { expression, .. } => walk_expression(expression, sink),
        TypedStatement::VarDeclaration {
            symbol_id,
            initializer,
            source_location,
            ..
        } => {
            sink.declares(*symbol_id, *source_location);
            if let Some(init) = initializer {
                walk_expression(init, sink);
            }
        }
        TypedStatement::Assignment { target, value, .. } => {
            walk_expression(target, sink);
            walk_expression(value, sink);
        }
        TypedStatement::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expression(condition, sink);
            walk_statement(then_branch, sink);
            if let Some(e) = else_branch {
                walk_statement(e, sink);
            }
        }
        TypedStatement::While {
            condition, body, ..
        } => {
            walk_expression(condition, sink);
            walk_statement(body, sink);
        }
        TypedStatement::For {
            init,
            condition,
            update,
            body,
            ..
        } => {
            if let Some(init) = init {
                walk_statement(init, sink);
            }
            for e in condition.iter().chain(update) {
                walk_expression(e, sink);
            }
            walk_statement(body, sink);
        }
        TypedStatement::ForIn {
            value_var,
            key_var,
            iterable,
            body,
            source_location,
        } => {
            for var in key_var.iter().chain([value_var]) {
                sink.declares(*var, *source_location);
            }
            walk_expression(iterable, sink);
            walk_statement(body, sink);
        }
        TypedStatement::Return { value, .. } => {
            if let Some(v) = value {
                walk_expression(v, sink);
            }
        }
        TypedStatement::Throw { exception, .. } => walk_expression(exception, sink),
        TypedStatement::Try {
            body,
            catch_clauses,
            finally_block,
            ..
        } => {
            walk_statement(body, sink);
            walk_catches(catch_clauses, sink);
            if let Some(f) = finally_block {
                walk_statement(f, sink);
            }
        }
        TypedStatement::Switch {
            discriminant,
            cases,
            default_case,
            ..
        } => {
            walk_expression(discriminant, sink);
            walk_cases(cases, sink);
            if let Some(d) = default_case {
                walk_statement(d, sink);
            }
        }
        TypedStatement::Block { statements, .. } => {
            for s in statements {
                walk_statement(s, sink);
            }
        }
        TypedStatement::PatternMatch {
            value, patterns, ..
        } => {
            walk_expression(value, sink);
            for case in patterns {
                for var in &case.bound_variables {
                    sink.declares(*var, case.source_location);
                }
                if let Some(g) = &case.guard {
                    walk_expression(g, sink);
                }
                walk_statement(&case.body, sink);
            }
        }
        TypedStatement::MacroExpansion {
            expanded_statements,
            ..
        } => {
            for s in expanded_statements {
                walk_statement(s, sink);
            }
        }
        TypedStatement::Break { .. } | TypedStatement::Continue { .. } => {}
    }
}

fn walk_catches(clauses: &[TypedCatchClause], sink: &mut impl SymbolSink) {
    for c in clauses {
        sink.declares(c.exception_variable, c.source_location);
        if let Some(f) = &c.filter {
            walk_expression(f, sink);
        }
        walk_statement(&c.body, sink);
    }
}

fn walk_cases(cases: &[TypedSwitchCase], sink: &mut impl SymbolSink) {
    for case in cases {
        for v in std::iter::once(&case.case_value).chain(&case.extra_case_values) {
            walk_expression(v, sink);
        }
        if let Some(g) = &case.guard {
            walk_expression(g, sink);
        }
        walk_statement(&case.body, sink);
    }
}

fn walk_comprehension(parts: &[TypedComprehensionFor], sink: &mut impl SymbolSink) {
    for part in parts {
        for var in part.key_var_symbol.iter().chain([&part.var_symbol]) {
            sink.declares(*var, part.source_location);
        }
        walk_expression(&part.iterator, sink);
    }
}

pub fn walk_expression(e: &TypedExpression, sink: &mut impl SymbolSink) {
    let at = e.source_location;
    use TypedExpressionKind as K;
    match &e.kind {
        K::Variable { symbol_id } => sink.refers_typed(*symbol_id, at, false, e.expr_type),
        K::FieldAccess {
            object,
            field_symbol,
            ..
        } => {
            walk_expression(object, sink);
            sink.refers_typed(*field_symbol, at, true, e.expr_type);
        }
        K::StaticFieldAccess {
            class_symbol,
            field_symbol,
        } => {
            sink.refers(*class_symbol, at, false);
            sink.refers_typed(*field_symbol, at, true, e.expr_type);
        }
        K::ArrayAccess { array, index } => {
            walk_expression(array, sink);
            walk_expression(index, sink);
        }
        K::FunctionCall {
            function,
            arguments,
            ..
        } => {
            walk_expression(function, sink);
            for a in arguments {
                walk_expression(a, sink);
            }
        }
        K::MethodCall {
            receiver,
            method_symbol,
            arguments,
            ..
        } => {
            walk_expression(receiver, sink);
            sink.refers(*method_symbol, at, true);
            for a in arguments {
                walk_expression(a, sink);
            }
        }
        K::MethodReference {
            receiver,
            method_symbol,
        } => {
            walk_expression(receiver, sink);
            sink.refers_typed(*method_symbol, at, true, e.expr_type);
        }
        K::StaticMethodCall {
            class_symbol,
            method_symbol,
            arguments,
            ..
        } => {
            sink.refers(*class_symbol, at, false);
            sink.refers(*method_symbol, at, true);
            for a in arguments {
                walk_expression(a, sink);
            }
        }
        K::BinaryOp { left, right, .. } => {
            walk_expression(left, sink);
            walk_expression(right, sink);
        }
        K::UnaryOp { operand, .. } => walk_expression(operand, sink),
        K::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            walk_expression(condition, sink);
            walk_expression(then_expr, sink);
            if let Some(e) = else_expr {
                walk_expression(e, sink);
            }
        }
        K::While {
            condition,
            then_expr,
        } => {
            walk_expression(condition, sink);
            walk_expression(then_expr, sink);
        }
        K::For {
            variable,
            iterable,
            body,
        } => {
            sink.declares(*variable, at);
            walk_expression(iterable, sink);
            walk_expression(body, sink);
        }
        K::ForIn {
            value_var,
            key_var,
            iterable,
            body,
        } => {
            for var in key_var.iter().chain([value_var]) {
                sink.declares(*var, at);
            }
            walk_expression(iterable, sink);
            walk_expression(body, sink);
        }
        K::ArrayLiteral { elements } => {
            for el in elements {
                walk_expression(el, sink);
            }
        }
        K::MapLiteral { entries } => {
            for entry in entries {
                walk_expression(&entry.key, sink);
                walk_expression(&entry.value, sink);
            }
        }
        K::ObjectLiteral { fields } => {
            for f in fields {
                walk_expression(&f.value, sink);
            }
        }
        K::FunctionLiteral {
            parameters, body, ..
        } => {
            for p in parameters {
                walk_parameter(p, sink);
            }
            for s in body {
                walk_statement(s, sink);
            }
        }
        K::Cast { expression, .. }
        | K::Is { expression, .. }
        | K::Throw { expression }
        | K::Meta { expression, .. }
        | K::Await { expression, .. } => walk_expression(expression, sink),
        K::New {
            class_type,
            arguments,
            ..
        } => {
            sink.constructs(*class_type, at);
            for a in arguments {
                walk_expression(a, sink);
            }
        }
        K::Return { value } => {
            if let Some(v) = value {
                walk_expression(v, sink);
            }
        }
        K::VarDeclarationExpr {
            symbol_id,
            initializer,
            ..
        }
        | K::FinalDeclarationExpr {
            symbol_id,
            initializer,
            ..
        } => {
            sink.declares(*symbol_id, at);
            walk_expression(initializer, sink);
        }
        K::StringInterpolation { parts } => {
            for part in parts {
                if let StringInterpolationPart::Expression(e) = part {
                    walk_expression(e, sink);
                }
            }
        }
        K::MacroExpression { arguments, .. } => {
            for a in arguments {
                walk_expression(a, sink);
            }
        }
        K::Block { statements, .. } => {
            for s in statements {
                walk_statement(s, sink);
            }
        }
        K::DollarIdent { arg, .. } => {
            if let Some(a) = arg {
                walk_expression(a, sink);
            }
        }
        K::CompilerSpecific { code, args, .. } => {
            walk_expression(code, sink);
            for a in args {
                walk_expression(a, sink);
            }
        }
        K::Switch {
            discriminant,
            cases,
            default_case,
        } => {
            walk_expression(discriminant, sink);
            walk_cases(cases, sink);
            if let Some(d) = default_case {
                walk_expression(d, sink);
            }
        }
        K::Try {
            try_expr,
            catch_clauses,
            finally_block,
        } => {
            walk_expression(try_expr, sink);
            walk_catches(catch_clauses, sink);
            if let Some(f) = finally_block {
                walk_expression(f, sink);
            }
        }
        K::PatternPlaceholder {
            source_location,
            variable_bindings,
            ..
        } => {
            for (_, var) in variable_bindings {
                sink.declares(*var, *source_location);
            }
        }
        K::ArrayComprehension {
            for_parts,
            expression,
            ..
        } => {
            walk_comprehension(for_parts, sink);
            walk_expression(expression, sink);
        }
        K::MapComprehension {
            for_parts,
            key_expr,
            value_expr,
            ..
        } => {
            walk_comprehension(for_parts, sink);
            walk_expression(key_expr, sink);
            walk_expression(value_expr, sink);
        }
        K::Literal { .. } | K::This { .. } | K::Super { .. } | K::Null | K::Break | K::Continue => {
        }
    }
}
