//! Binary and unary operator lowering.

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
    /// Lower a binary operator
    pub(crate) fn lower_binary_operator(
        &mut self,
        operator: &BinaryOp,
    ) -> LoweringResult<BinaryOperator> {
        match operator {
            BinaryOp::Add => Ok(BinaryOperator::Add),
            BinaryOp::Sub => Ok(BinaryOperator::Sub),
            BinaryOp::Mul => Ok(BinaryOperator::Mul),
            BinaryOp::Div => Ok(BinaryOperator::Div),
            BinaryOp::Mod => Ok(BinaryOperator::Mod),
            BinaryOp::Eq => Ok(BinaryOperator::Eq),
            BinaryOp::NotEq => Ok(BinaryOperator::Ne),
            BinaryOp::Lt => Ok(BinaryOperator::Lt),
            BinaryOp::Le => Ok(BinaryOperator::Le),
            BinaryOp::Gt => Ok(BinaryOperator::Gt),
            BinaryOp::Ge => Ok(BinaryOperator::Ge),
            BinaryOp::And => Ok(BinaryOperator::And),
            BinaryOp::Or => Ok(BinaryOperator::Or),
            BinaryOp::BitAnd => Ok(BinaryOperator::BitAnd),
            BinaryOp::BitOr => Ok(BinaryOperator::BitOr),
            BinaryOp::BitXor => Ok(BinaryOperator::BitXor),
            BinaryOp::Shl => Ok(BinaryOperator::Shl),
            BinaryOp::Shr => Ok(BinaryOperator::Shr),
            BinaryOp::Ushr => Ok(BinaryOperator::Ushr),
            BinaryOp::Range => Ok(BinaryOperator::Range),
            BinaryOp::Arrow => Ok(BinaryOperator::Arrow),
            BinaryOp::Is => {
                // 'is' operator needs runtime type checking support
                // For now, lower as a comparison (downstream passes handle it)
                Ok(BinaryOperator::Eq)
            }
            BinaryOp::NullCoal => Ok(BinaryOperator::NullCoal),
        }
    }

    /// Lower a unary operator
    pub(crate) fn lower_unary_operator(
        &mut self,
        operator: &UnaryOp,
    ) -> LoweringResult<UnaryOperator> {
        match operator {
            UnaryOp::Neg => Ok(UnaryOperator::Neg),
            UnaryOp::Not => Ok(UnaryOperator::Not),
            UnaryOp::BitNot => Ok(UnaryOperator::BitNot),
            UnaryOp::PreIncr => Ok(UnaryOperator::PreInc),
            UnaryOp::PostIncr => Ok(UnaryOperator::PostInc),
            UnaryOp::PreDecr => Ok(UnaryOperator::PreDec),
            UnaryOp::PostDecr => Ok(UnaryOperator::PostDec),
        }
    }

    // Property access handling removed for simplicity
}
