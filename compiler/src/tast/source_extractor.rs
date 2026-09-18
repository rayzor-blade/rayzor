use crate::tast::{SourceLocation, node::TypedExpression};

pub fn extract_location_from_expression(expr: &TypedExpression) -> SourceLocation {
    expr.source_location
}
