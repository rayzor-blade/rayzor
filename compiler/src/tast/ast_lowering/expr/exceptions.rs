//! `catch` clauses.

use super::*;
use crate::tast::node::HasSourceLocation;
use crate::tast::{core::*, node::MemoryEffects, node::*, type_resolution, *};
use parser::{
    AbstractDecl, BinaryOp, BlockElement, ClassDecl, ClassField, ClassFieldKind, EnumConstructor,
    EnumDecl, Expr, ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata,
    Modifier, ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

impl<'a> AstLowering<'a> {
    /// Lower a catch clause
    pub(crate) fn lower_catch_clause(
        &mut self,
        catch: &parser::Catch,
    ) -> Result<TypedCatchClause, LoweringError> {
        // Create a new scope for the catch block
        let catch_scope = self.context.enter_scope(ScopeKind::Block);

        // Create symbol for exception variable in the catch scope
        let var_name = self.context.string_interner.intern(&catch.var);
        let var_symbol = self
            .context
            .symbol_table
            .create_variable_in_scope(var_name, self.context.current_scope);

        // Resolve exception type
        let exception_type = if let Some(type_hint) = &catch.type_hint {
            self.lower_type(type_hint)?
        } else {
            // Default to dynamic type if no type specified
            self.context.type_table.borrow().dynamic_type()
        };

        // Set the catch variable's type so field accesses (e.g., e.length) resolve correctly
        self.context
            .symbol_table
            .update_symbol_type(var_symbol, exception_type);

        // Lower filter condition if present (in the catch scope where the exception var is available)
        let filter = if let Some(filter_expr) = &catch.filter {
            Some(self.lower_expression(filter_expr)?)
        } else {
            None
        };

        // Lower catch handler body (in the catch scope where the exception var is available)
        let handler = self.lower_expression(&catch.body)?;

        // Exit the catch scope
        self.context.exit_scope();

        Ok(TypedCatchClause {
            exception_variable: var_symbol,
            exception_type,
            filter,
            body: TypedStatement::Expression {
                expression: handler,
                source_location: self.context.create_location(),
            },
            source_location: self.context.create_location(),
        })
    }
}
