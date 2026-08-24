//! Derived-trait validation.

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
    /// Validate that derived traits are compatible with field types
    pub(crate) fn validate_derived_traits(
        &self,
        typed_class: &TypedClass,
        derived_traits: &mut Vec<crate::tast::DerivedTrait>,
        class_name: &str,
    ) {
        use crate::tast::DerivedTrait;

        let has_clone = derived_traits.contains(&DerivedTrait::Clone);
        let has_copy = derived_traits.contains(&DerivedTrait::Copy);

        // Validate Clone: all fields must be Clone
        if has_clone {
            let mut non_clone_fields = Vec::new();

            for field in &typed_class.fields {
                if !self.is_type_clone(field.field_type) {
                    let field_name_str = self
                        .context
                        .string_interner
                        .get(field.name)
                        .unwrap_or("?")
                        .to_string();
                    non_clone_fields.push(field_name_str);
                }
            }

            if !non_clone_fields.is_empty() {
                eprintln!(
                    "ERROR: Class '{}' derives Clone but has non-Clone fields:",
                    class_name
                );
                for field_name in &non_clone_fields {
                    eprintln!("  - Field '{}' is not Clone", field_name);
                }
                eprintln!("  All fields must derive Clone or be primitive Copy types");
                eprintln!("  Consider adding @:derive(Clone) to field types or removing Clone from this class");

                // Remove Clone trait to prevent incorrect codegen
                derived_traits.retain(|t| *t != DerivedTrait::Clone);
            }
        }

        // Validate trait dependency chains
        let has_partial_eq = derived_traits.contains(&DerivedTrait::PartialEq);
        let has_eq = derived_traits.contains(&DerivedTrait::Eq);
        let has_partial_ord = derived_traits.contains(&DerivedTrait::PartialOrd);
        let has_ord = derived_traits.contains(&DerivedTrait::Ord);
        let has_hash = derived_traits.contains(&DerivedTrait::Hash);

        // Eq requires PartialEq
        if has_eq && !has_partial_eq {
            eprintln!(
                "ERROR: Class '{}' derives Eq but not PartialEq. Eq requires PartialEq.",
                class_name
            );
            eprintln!("  Use @:derive([PartialEq, Eq]) instead");
            derived_traits.retain(|t| *t != DerivedTrait::Eq);
        }

        // PartialOrd requires PartialEq
        if has_partial_ord && !has_partial_eq {
            eprintln!(
                "ERROR: Class '{}' derives PartialOrd but not PartialEq. PartialOrd requires PartialEq.",
                class_name
            );
            eprintln!("  Use @:derive([PartialEq, PartialOrd]) instead");
            derived_traits.retain(|t| *t != DerivedTrait::PartialOrd);
        }

        // Ord requires PartialOrd + Eq
        if has_ord && (!has_partial_ord || !has_eq) {
            eprintln!(
                "ERROR: Class '{}' derives Ord but is missing required traits.",
                class_name
            );
            eprintln!("  Ord requires PartialEq, Eq, and PartialOrd.");
            eprintln!("  Use @:derive([PartialEq, Eq, PartialOrd, Ord]) instead");
            derived_traits.retain(|t| *t != DerivedTrait::Ord);
        }

        // Validate PartialEq: all fields must support equality
        if has_partial_eq {
            let mut bad_fields = Vec::new();
            for field in &typed_class.fields {
                if !field.is_static && !self.is_type_equatable(field.field_type) {
                    let name = self
                        .context
                        .string_interner
                        .get(field.name)
                        .unwrap_or("?")
                        .to_string();
                    bad_fields.push(name);
                }
            }
            if !bad_fields.is_empty() {
                eprintln!(
                    "ERROR: Class '{}' derives PartialEq but has non-equatable fields:",
                    class_name
                );
                for f in &bad_fields {
                    eprintln!("  - Field '{}' does not support equality", f);
                }
                derived_traits.retain(|t| *t != DerivedTrait::PartialEq);
                derived_traits.retain(|t| *t != DerivedTrait::Eq);
            }
        }

        // Validate PartialOrd: all fields must support ordering
        if derived_traits.contains(&DerivedTrait::PartialOrd) {
            let mut bad_fields = Vec::new();
            for field in &typed_class.fields {
                if !field.is_static && !self.is_type_orderable(field.field_type) {
                    let name = self
                        .context
                        .string_interner
                        .get(field.name)
                        .unwrap_or("?")
                        .to_string();
                    bad_fields.push(name);
                }
            }
            if !bad_fields.is_empty() {
                eprintln!(
                    "ERROR: Class '{}' derives PartialOrd but has non-orderable fields:",
                    class_name
                );
                for f in &bad_fields {
                    eprintln!("  - Field '{}' does not support ordering", f);
                }
                derived_traits.retain(|t| *t != DerivedTrait::PartialOrd);
                derived_traits.retain(|t| *t != DerivedTrait::Ord);
            }
        }

        // Validate Hash: all fields must be hashable
        if derived_traits.contains(&DerivedTrait::Hash) {
            let mut bad_fields = Vec::new();
            for field in &typed_class.fields {
                if !field.is_static && !self.is_type_hashable(field.field_type) {
                    let name = self
                        .context
                        .string_interner
                        .get(field.name)
                        .unwrap_or("?")
                        .to_string();
                    bad_fields.push(name);
                }
            }
            if !bad_fields.is_empty() {
                eprintln!(
                    "ERROR: Class '{}' derives Hash but has non-hashable fields:",
                    class_name
                );
                for f in &bad_fields {
                    eprintln!("  - Field '{}' is not hashable", f);
                }
                derived_traits.retain(|t| *t != DerivedTrait::Hash);
            }
        }

        // Validate Copy: all fields must be Copy
        if has_copy {
            let mut non_copy_fields = Vec::new();

            for field in &typed_class.fields {
                if !self.is_type_copy(field.field_type) {
                    let field_name_str = self
                        .context
                        .string_interner
                        .get(field.name)
                        .unwrap_or("?")
                        .to_string();
                    non_copy_fields.push(field_name_str);
                }
            }

            if !non_copy_fields.is_empty() {
                eprintln!(
                    "ERROR: Class '{}' derives Copy but has non-Copy fields:",
                    class_name
                );
                for field_name in &non_copy_fields {
                    eprintln!("  - Field '{}' is not Copy", field_name);
                }
                eprintln!("  Copy types can only contain primitive Copy types (Int, Float, Bool)");
                eprintln!("  or other classes that derive Copy");
                eprintln!("  Consider using Clone instead of Copy, or remove Copy from this class");

                // Remove Copy trait to prevent incorrect codegen
                derived_traits.retain(|t| *t != DerivedTrait::Copy);
                // Also remove Clone if it was auto-added by Copy
                if !has_clone {
                    derived_traits.retain(|t| *t != DerivedTrait::Clone);
                }
            }
        }

        // Validate Drop: class must have a public drop():Void method
        let has_drop = derived_traits.contains(&DerivedTrait::Drop);
        if has_drop {
            let has_drop_method = typed_class.methods.iter().any(|m| {
                let method_name = self.context.string_interner.get(m.name).unwrap_or("");
                method_name == "drop"
                    && m.parameters.is_empty()
                    && matches!(m.visibility, crate::tast::Visibility::Public)
            });

            if !has_drop_method {
                eprintln!(
                    "ERROR: Class '{}' derives Drop but has no public `drop():Void` method",
                    class_name
                );
                eprintln!("  @:derive(Drop) requires: public function drop():Void {{ ... }}");
                derived_traits.retain(|t| *t != DerivedTrait::Drop);
            }
        }
    }
}
