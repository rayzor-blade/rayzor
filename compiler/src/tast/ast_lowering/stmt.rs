//! Statement lowering.

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
    /// Lower a statement (placeholder - not used with new parser)
    fn lower_statement(&mut self, _statement: &str) -> LoweringResult<TypedStatement> {
        let location = self.context.create_location();

        // Placeholder implementation
        Ok(TypedStatement::Expression {
            expression: TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Null,
                usage: VariableUsage::Copy,
                lifetime_id: crate::tast::LifetimeId::first(),
                source_location: location,
                metadata: ExpressionMetadata::default(),
            },
            source_location: location,
        })
    }

    /// Placeholder for old statement lowering - not used with new parser
    fn _old_statement_lowering_placeholder(&mut self) {
        // This was the old statement lowering implementation
        // Not used with the new parser interface
    }
}
