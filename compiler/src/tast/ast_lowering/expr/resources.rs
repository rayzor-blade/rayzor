//! `untyped __resources__()`: the table `haxe.Resource` reads, filled with
//! the program's resources once typing is done.

use super::*;
use parser::{Expr, ExprKind, Span};

impl<'a> AstLowering<'a> {
    /// `rayzor.runtime.Resources.table()`. It names no resource, so a module
    /// typed before a macro adds one stays valid.
    pub(crate) fn resource_table_expr(&self, span: Span) -> Expr {
        let node = |kind| Expr { kind, span };
        let mut path = crate::macro_system::assets::RESOURCE_TABLE_CLASS.split('.');
        let root = node(ExprKind::Ident(path.next().unwrap_or_default().to_owned()));
        let class = path.fold(root, |expr, field| {
            node(ExprKind::Field {
                expr: Box::new(expr),
                field: field.to_owned(),
                is_optional: false,
            })
        });
        let table = node(ExprKind::Field {
            expr: Box::new(class),
            field: "table".to_owned(),
            is_optional: false,
        });
        node(ExprKind::Call {
            expr: Box::new(table),
            args: Vec::new(),
        })
    }
}
