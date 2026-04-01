//! Type Checker to Diagnostics Integration
//!
//! This module bridges the sophisticated type checking system with the diagnostic
//! reporting infrastructure, converting type checking errors into rich, user-friendly
//! error messages with source locations and suggestions.

use super::node::BinaryOperator;
use super::type_checker::{AccessLevel, TypeCheckError, TypeErrorKind};
use super::{
    InternedString, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeTable,
};
use crate::error_codes::format_error_code;
use diagnostics::{
    Diagnostic, DiagnosticBuilder, DiagnosticSeverity, FileId, SourceMap, SourcePosition,
    SourceSpan,
};
use source_map::SourceFile;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

/// Context for type errors to provide better suggestions
#[derive(Debug, Clone)]
pub enum TypeErrorContext {
    Assignment {
        target_type: TypeId,
    },
    BinaryOperation {
        operator: BinaryOperator,
        other_type: TypeId,
    },
    UnaryOperation {
        operator: super::node::UnaryOperator,
    },
    FunctionCall {
        param_index: usize,
        expected_type: TypeId,
    },
    ReturnStatement {
        expected_type: TypeId,
    },
    ArrayAccess,
    FieldAccess {
        field_name: InternedString,
    },
    Initialization,
    ConditionalExpression,
    LoopCondition,
    SwitchPattern,
    SwitchGuard,
    SwitchExpression,
    GenericConstraint,
    CatchFilter,
    ForInLoop,
}

/// Haxe standard library type knowledge for better suggestions
pub struct HaxeStdTypes {
    /// Types that implement toString() method
    stringable_types: BTreeSet<&'static str>,
    /// Numeric types that can be used in arithmetic
    numeric_types: BTreeSet<&'static str>,
    /// Types that can be iterated in for loops
    iterable_types: BTreeSet<&'static str>,
}

impl HaxeStdTypes {
    pub fn new() -> Self {
        let mut stringable_types = BTreeSet::new();
        stringable_types.insert("Int");
        stringable_types.insert("Float");
        stringable_types.insert("Bool");
        stringable_types.insert("String");

        let mut numeric_types = BTreeSet::new();
        numeric_types.insert("Int");
        numeric_types.insert("Float");

        let mut iterable_types = BTreeSet::new();
        iterable_types.insert("Array");
        iterable_types.insert("String");

        Self {
            stringable_types,
            numeric_types,
            iterable_types,
        }
    }

    pub fn implements_tostring(&self, type_name: &str) -> bool {
        self.stringable_types.contains(type_name)
    }

    pub fn is_numeric(&self, type_name: &str) -> bool {
        self.numeric_types.contains(type_name)
    }

    pub fn is_iterable(&self, type_name: &str) -> bool {
        self.iterable_types.contains(type_name)
    }
}

/// Type-specific suggestion generator for standard Haxe types
pub struct TypeSpecificSuggestions<'a> {
    type_table: &'a Rc<RefCell<TypeTable>>,
    string_interner: &'a StringInterner,
    std_types: HaxeStdTypes,
}

impl<'a> TypeSpecificSuggestions<'a> {
    pub fn new(
        type_table: &'a Rc<RefCell<TypeTable>>,
        string_interner: &'a StringInterner,
    ) -> Self {
        Self {
            type_table,
            string_interner,
            std_types: HaxeStdTypes::new(),
        }
    }

    /// Generate cast suggestions between types
    pub fn suggest_cast(&self, from_type: TypeId, to_type: TypeId) -> Option<String> {
        let from_name = self.get_type_name(from_type);
        let to_name = self.get_type_name(to_type);

        match (from_name.as_str(), to_name.as_str()) {
            ("Int", "String") => Some("Use Std.string(value) or value.toString()".to_string()),
            ("Float", "String") => Some("Use Std.string(value) or value.toString()".to_string()),
            ("Bool", "String") => Some("Use Std.string(value) or value.toString()".to_string()),
            ("String", "Int") => Some("Use Std.parseInt(value) (returns Null<Int>)".to_string()),
            ("String", "Float") => Some("Use Std.parseFloat(value)".to_string()),
            ("Int", "Float") => Some(
                "Automatic conversion available, or use explicit cast (value : Float)".to_string(),
            ),
            ("Float", "Int") => {
                Some("Use Std.int(value) for truncation or Math.round() for rounding".to_string())
            }
            _ if from_name.starts_with("Array<") && to_name == "String" => {
                Some("Use value.join(\",\") or Std.string(value)".to_string())
            }
            _ => None,
        }
    }

    /// Generate operation-specific suggestions
    pub fn suggest_operation(
        &self,
        op: BinaryOperator,
        left_type: TypeId,
        right_type: TypeId,
    ) -> Option<String> {
        let left_name = self.get_type_name(left_type);
        let right_name = self.get_type_name(right_type);

        match op {
            BinaryOperator::Add => {
                if self.std_types.is_numeric(&left_name) && right_name == "String" {
                    Some("Convert number to string: Std.string(leftValue) + rightValue".to_string())
                } else if left_name == "String" && self.std_types.is_numeric(&right_name) {
                    Some("Convert number to string: leftValue + Std.string(rightValue)".to_string())
                } else if !self.std_types.is_numeric(&left_name)
                    && !self.std_types.is_numeric(&right_name)
                {
                    Some("Addition requires numeric types. Use string concatenation or convert to numbers".to_string())
                } else {
                    None
                }
            }
            BinaryOperator::Sub | BinaryOperator::Mul | BinaryOperator::Div => {
                if !self.std_types.is_numeric(&left_name) || !self.std_types.is_numeric(&right_name)
                {
                    Some("Arithmetic operations require numeric types (Int or Float)".to_string())
                } else {
                    None
                }
            }
            BinaryOperator::Eq | BinaryOperator::Ne => {
                if left_name != right_name {
                    Some("Comparing different types. Consider explicit conversion or use strict equality".to_string())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Generate context-aware suggestions
    pub fn suggest_for_context(
        &self,
        from_type: TypeId,
        to_type: TypeId,
        context: &TypeErrorContext,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        let from_name = self.get_type_name(from_type);
        let to_name = self.get_type_name(to_type);

        match context {
            TypeErrorContext::BinaryOperation { operator, .. } => {
                if let Some(suggestion) = self.suggest_operation(*operator, from_type, to_type) {
                    suggestions.push(suggestion);
                }
            }
            TypeErrorContext::Assignment { .. } | TypeErrorContext::Initialization => {
                if let Some(cast_suggestion) = self.suggest_cast(from_type, to_type) {
                    suggestions.push(cast_suggestion);
                }
            }
            TypeErrorContext::FunctionCall { param_index, .. } => {
                if let Some(cast_suggestion) = self.suggest_cast(from_type, to_type) {
                    suggestions.push(format!(
                        "Parameter {} expects {}: {}",
                        param_index + 1,
                        to_name,
                        cast_suggestion
                    ));
                }
            }
            TypeErrorContext::ArrayAccess => {
                if from_name == "String" && to_name == "Int" {
                    suggestions.push(
                        "Array indices must be Int. Use Std.parseInt() to convert String to Int"
                            .to_string(),
                    );
                } else if from_name == "Float" && to_name == "Int" {
                    suggestions.push(
                        "Array indices must be Int. Use Std.int() to convert Float to Int"
                            .to_string(),
                    );
                } else if !self.std_types.is_numeric(&from_name) && to_name == "Int" {
                    suggestions.push(format!(
                        "Array indices must be Int. Convert {} to Int first",
                        from_name
                    ));
                }

                // Suggestions for trying to index non-array types
                if to_name.starts_with("Array<") && !from_name.starts_with("Array<") {
                    if from_name == "String" {
                        suggestions.push(
                            "Strings can be indexed like arrays to get individual characters"
                                .to_string(),
                        );
                    } else {
                        suggestions.push(format!(
                            "Cannot index {}. Use an Array or convert to Array first",
                            from_name
                        ));
                    }
                }
            }
            TypeErrorContext::FieldAccess { field_name } => {
                let field_str = self.string_interner.get(*field_name).unwrap_or("unknown");
                if field_str == "length" && self.std_types.implements_tostring(&from_name) {
                    suggestions.push(format!(
                        "Convert {} to String first: Std.string(value).length",
                        from_name
                    ));
                }
            }
            TypeErrorContext::UnaryOperation { operator } => match operator {
                super::node::UnaryOperator::Not => {
                    if from_name != "Bool" && to_name == "Bool" {
                        suggestions.push(format!(
                            "The ! operator requires a boolean. Convert {} to Bool first",
                            from_name
                        ));
                    }
                }
                super::node::UnaryOperator::Neg => {
                    if !self.std_types.is_numeric(&from_name)
                        && (to_name == "Int" || to_name == "Float")
                    {
                        suggestions.push(format!("The - operator requires a numeric type. Convert {} to Int or Float first", from_name));
                    }
                }
                _ => {}
            },
            TypeErrorContext::ConditionalExpression | TypeErrorContext::LoopCondition => {
                if from_name != "Bool" && to_name == "Bool" {
                    if from_name == "Int" || from_name == "Float" {
                        suggestions.push(format!("Compare {} with 0: value != 0", from_name));
                    } else if from_name == "String" {
                        suggestions.push(
                            "Check string is not empty: str != \"\" or str.length > 0".to_string(),
                        );
                    } else if from_name.starts_with("Array<") {
                        suggestions.push("Check array is not empty: arr.length > 0".to_string());
                    } else {
                        suggestions.push(format!("Convert {} to Bool for condition", from_name));
                    }
                }
            }
            TypeErrorContext::SwitchPattern => {
                suggestions
                    .push("Switch pattern must match the type being switched on".to_string());
                if let Some(cast_suggestion) = self.suggest_cast(from_type, to_type) {
                    suggestions.push(format!("Cast the pattern: {}", cast_suggestion));
                }
            }
            TypeErrorContext::SwitchGuard => {
                if from_name != "Bool" && to_name == "Bool" {
                    suggestions.push("Switch guard must be a boolean expression".to_string());
                }
            }
            _ => {}
        }

        suggestions
    }

    /// Get a readable type name
    fn get_type_name(&self, type_id: TypeId) -> String {
        // Special case for invalid type ID
        if !type_id.is_valid() {
            return "Unknown".to_string();
        }

        match self.type_table.borrow().get(type_id) {
            Some(type_info) => {
                use crate::tast::core::TypeKind;
                match &type_info.kind {
                    TypeKind::Void => "Void".to_string(),
                    TypeKind::Bool => "Bool".to_string(),
                    TypeKind::Int => "Int".to_string(),
                    TypeKind::Float => "Float".to_string(),
                    TypeKind::String => "String".to_string(),
                    TypeKind::Dynamic => "Dynamic".to_string(),
                    TypeKind::Array { element_type } => {
                        format!("Array<{}>", self.get_type_name(*element_type))
                    }
                    TypeKind::Optional { inner_type } => {
                        format!("Null<{}>", self.get_type_name(*inner_type))
                    }
                    _ => format!("Type#{}", type_id.as_raw()),
                }
            }
            None => format!("Type#{}", type_id.as_raw()),
        }
    }
}

/// Type diagnostic emitter that converts type errors to user-friendly diagnostics
pub struct TypeDiagnosticEmitter<'a> {
    type_table: &'a Rc<RefCell<TypeTable>>,
    symbol_table: &'a SymbolTable,
    string_interner: &'a StringInterner,
    source_map: &'a SourceMap,
    suggestion_generator: TypeSpecificSuggestions<'a>,
}

impl<'a> TypeDiagnosticEmitter<'a> {
    /// Create a new type diagnostic emitter
    pub fn new(
        type_table: &'a Rc<RefCell<TypeTable>>,
        symbol_table: &'a SymbolTable,
        string_interner: &'a StringInterner,
        source_map: &'a SourceMap,
    ) -> Self {
        let suggestion_generator = TypeSpecificSuggestions::new(type_table, string_interner);
        Self {
            type_table,
            symbol_table,
            string_interner,
            source_map,
            suggestion_generator,
        }
    }

    /// Get context-aware suggestions for type errors
    pub fn get_suggestions(
        &self,
        from_type: TypeId,
        to_type: TypeId,
        context: &TypeErrorContext,
    ) -> Vec<String> {
        self.suggestion_generator
            .suggest_for_context(from_type, to_type, context)
    }

    /// Convert a type checking error into a rich diagnostic
    pub fn emit_diagnostic(&self, error: TypeCheckError) -> Diagnostic {
        match error.kind {
            TypeErrorKind::TypeMismatch { expected, actual } => self.emit_type_mismatch(
                error.location,
                expected,
                actual,
                &error.context,
                error.suggestion.as_deref(),
            ),
            TypeErrorKind::UndefinedType { name } => {
                self.emit_undefined_type(error.location, name, &error.context)
            }
            TypeErrorKind::UndefinedSymbol { name } => {
                self.emit_undefined_symbol(error.location, name, &error.context)
            }
            TypeErrorKind::InvalidTypeArguments {
                base_type,
                expected_count,
                actual_count,
            } => self.emit_invalid_type_arguments(
                error.location,
                base_type,
                expected_count,
                actual_count,
            ),
            TypeErrorKind::ConstraintViolation {
                type_param,
                constraint,
                violating_type,
            } => self.emit_constraint_violation(
                error.location,
                type_param,
                constraint,
                violating_type,
            ),
            TypeErrorKind::CircularDependency { types } => {
                self.emit_circular_dependency(error.location, types)
            }
            TypeErrorKind::InvalidCast { from_type, to_type } => {
                self.emit_invalid_cast(error.location, from_type, to_type)
            }
            TypeErrorKind::SignatureMismatch {
                expected_params,
                actual_params,
                expected_return,
                actual_return,
            } => self.emit_signature_mismatch(
                error.location,
                expected_params,
                actual_params,
                expected_return,
                actual_return,
            ),
            TypeErrorKind::AccessViolation {
                symbol_id,
                required_access,
            } => self.emit_access_violation(error.location, symbol_id, required_access),
            TypeErrorKind::InferenceFailed { reason } => {
                self.emit_inference_failed(error.location, &reason)
            }
            TypeErrorKind::InterfaceNotImplemented {
                interface_type,
                class_type,
                missing_method,
            } => self.emit_interface_not_implemented(
                error.location,
                interface_type,
                class_type,
                missing_method,
            ),
            TypeErrorKind::MethodSignatureMismatch {
                expected,
                actual,
                method_name,
            } => self.emit_method_signature_mismatch(
                error.location,
                expected,
                actual,
                method_name,
                &error.context,
                error.suggestion.as_deref(),
            ),
            TypeErrorKind::MissingOverride {
                method_name,
                parent_class,
            } => self.emit_missing_override(
                error.location,
                method_name,
                parent_class,
                &error.context,
                error.suggestion.as_deref(),
            ),
            TypeErrorKind::InvalidOverride { method_name } => self.emit_invalid_override(
                error.location,
                method_name,
                &error.context,
                error.suggestion.as_deref(),
            ),
            TypeErrorKind::StaticAccessFromInstance {
                member_name,
                class_name,
            } => self.emit_static_access_from_instance(
                error.location,
                member_name,
                class_name,
                &error.context,
                error.suggestion.as_deref(),
            ),
            TypeErrorKind::InstanceAccessFromStatic {
                member_name,
                class_name,
            } => self.emit_instance_access_from_static(
                error.location,
                member_name,
                class_name,
                &error.context,
                error.suggestion.as_deref(),
            ),
            TypeErrorKind::ImportError { message } => todo!(),
            TypeErrorKind::UnknownSymbol { name } => todo!(),
            TypeErrorKind::SendSyncViolation { type_name, reason } => self
                .emit_send_sync_violation(
                    error.location,
                    &type_name,
                    &reason,
                    error.suggestion.as_deref(),
                ),
            TypeErrorKind::NullAssignmentToNonNull { variable_name } => {
                self.emit_null_assignment_to_not_null(error.location, &variable_name)
            }
            TypeErrorKind::NullableToNonNullParam {
                param_name,
                function_name,
            } => self.emit_nullable_to_non_null_param(error.location, &param_name, &function_name),
            TypeErrorKind::NullableReturn { function_name } => {
                self.emit_nullable_return(error.location, &function_name)
            }
        }
    }

    /// Emit interface not implemented diagnostic
    fn emit_interface_not_implemented(
        &self,
        location: SourceLocation,
        _interface_type: TypeId,
        _class_type: TypeId,
        missing_method: InternedString,
    ) -> Diagnostic {
        let method_name = self
            .string_interner
            .get(missing_method)
            .unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!(
                "Class missing method '{}' required by interface",
                method_name
            ),
            source_span.clone(),
        )
        .code(format_error_code(1008))
        .label(
            source_span,
            format!("missing implementation of '{}'", method_name),
        )
        .help(format!("Add an implementation of method '{}'", method_name))
        .build()
    }

    /// Emit method signature mismatch diagnostic
    fn emit_method_signature_mismatch(
        &self,
        location: SourceLocation,
        expected: TypeId,
        actual: TypeId,
        method_name: InternedString,
        context: &str,
        suggestion: Option<&str>,
    ) -> Diagnostic {
        let method_name_str = self.string_interner.get(method_name).unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        let mut builder = DiagnosticBuilder::error(
            format!("Method '{}' has incorrect signature", method_name_str),
            source_span.clone(),
        )
        .code(format_error_code(1009))
        .label(source_span, "incorrect signature");

        if !context.is_empty() {
            builder = builder.note(context);
        }

        if let Some(suggestion) = suggestion {
            builder = builder.help(suggestion);
        }

        builder.build()
    }

    /// Emit missing override modifier diagnostic
    fn emit_missing_override(
        &self,
        location: SourceLocation,
        method_name: InternedString,
        parent_class: InternedString,
        context: &str,
        suggestion: Option<&str>,
    ) -> Diagnostic {
        let method_name_str = self.string_interner.get(method_name).unwrap_or("<unknown>");
        let parent_class_str = self
            .string_interner
            .get(parent_class)
            .unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        let mut builder = DiagnosticBuilder::error(
            format!(
                "Method '{}' overrides parent method but is missing the 'override' modifier",
                method_name_str
            ),
            source_span.clone(),
        )
        .code(format_error_code(1010))
        .label(
            source_span,
            format!("overrides method from class '{}'", parent_class_str),
        );

        if !context.is_empty() {
            builder = builder.note(context);
        }

        if let Some(suggestion) = suggestion {
            builder = builder.help(suggestion);
        } else {
            builder = builder.help("Add 'override' modifier to the method declaration");
        }

        builder.build()
    }

    /// Emit invalid override diagnostic
    fn emit_invalid_override(
        &self,
        location: SourceLocation,
        method_name: InternedString,
        context: &str,
        suggestion: Option<&str>,
    ) -> Diagnostic {
        let method_name_str = self.string_interner.get(method_name).unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        let mut builder = DiagnosticBuilder::error(
            format!(
                "Method '{}' marked as 'override' but no parent method exists",
                method_name_str
            ),
            source_span.clone(),
        )
        .code(format_error_code(1011))
        .label(source_span, "no parent method to override");

        if !context.is_empty() {
            builder = builder.note(context);
        }

        if let Some(suggestion) = suggestion {
            builder = builder.help(suggestion);
        } else {
            builder = builder.help("Remove the 'override' modifier or check the method name");
        }

        builder.build()
    }

    /// Emit static access from instance diagnostic
    fn emit_static_access_from_instance(
        &self,
        location: SourceLocation,
        member_name: InternedString,
        class_name: InternedString,
        context: &str,
        suggestion: Option<&str>,
    ) -> Diagnostic {
        let member_name_str = self.string_interner.get(member_name).unwrap_or("<unknown>");
        let class_name_str = self.string_interner.get(class_name).unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        let mut builder = DiagnosticBuilder::error(
            format!(
                "Static member '{}' cannot be accessed through instance",
                member_name_str
            ),
            source_span.clone(),
        )
        .code(format_error_code(1012))
        .label(source_span, "static member accessed through instance");

        if !context.is_empty() {
            builder = builder.note(context);
        }

        if let Some(suggestion) = suggestion {
            builder = builder.help(suggestion);
        } else {
            builder = builder.help(format!(
                "Access static members through the class: {}.{}",
                class_name_str, member_name_str
            ));
        }

        builder.build()
    }

    /// Emit instance access from static diagnostic
    fn emit_instance_access_from_static(
        &self,
        location: SourceLocation,
        member_name: InternedString,
        class_name: InternedString,
        context: &str,
        suggestion: Option<&str>,
    ) -> Diagnostic {
        let member_name_str = self.string_interner.get(member_name).unwrap_or("<unknown>");
        let class_name_str = self.string_interner.get(class_name).unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        let mut builder = DiagnosticBuilder::error(
            format!(
                "Instance member '{}' cannot be accessed from static context",
                member_name_str
            ),
            source_span.clone(),
        )
        .code(format_error_code(1013))
        .label(source_span, "instance member accessed from static context");

        if !context.is_empty() {
            builder = builder.note(context);
        }

        if let Some(suggestion) = suggestion {
            builder = builder.help(suggestion);
        } else {
            builder = builder.help(format!(
                "Instance members require an instance of '{}'",
                class_name_str
            ));
        }

        builder.build()
    }

    /// Emit type mismatch diagnostic with enhanced suggestions
    fn emit_type_mismatch(
        &self,
        location: SourceLocation,
        expected: TypeId,
        actual: TypeId,
        context: &str,
        suggestion: Option<&str>,
    ) -> Diagnostic {
        let expected_name = self.format_type_name(expected);
        let actual_name = self.format_type_name(actual);

        let source_span = self.location_to_span(location);

        let mut builder = DiagnosticBuilder::error(
            format!(
                "Type mismatch: expected `{}`, found `{}`",
                expected_name, actual_name
            ),
            source_span.clone(),
        )
        .code(format_error_code(1001)) // E1001: Type mismatch
        .label(
            source_span.clone(),
            format!("expected `{}`, found `{}`", expected_name, actual_name),
        );

        if !context.is_empty() {
            builder = builder.note(context);
        }

        // Use provided suggestion first, then generate smart suggestions
        if let Some(suggestion) = suggestion {
            builder = builder.help(suggestion);
        } else {
            // Generate Haxe-specific suggestions
            if let Some(cast_suggestion) = self.suggestion_generator.suggest_cast(actual, expected)
            {
                builder = builder.help(cast_suggestion);
            }
        }

        // Add helpful notes based on the types involved
        if self.are_similar_types(expected, actual) {
            builder = builder.help(format!(
                "The types `{}` and `{}` are similar but not identical. Consider explicit conversion.",
                expected_name, actual_name
            ));
        }

        builder.build()
    }

    /// Emit undefined type diagnostic
    fn emit_undefined_type(
        &self,
        location: SourceLocation,
        name: InternedString,
        context: &str,
    ) -> Diagnostic {
        let type_name = self.string_interner.get(name).unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!("Undefined type `{}`", type_name),
            source_span.clone(),
        )
        .code(format_error_code(1002)) // E1002: Undefined type
        .label(source_span, format!("type `{}` not found", type_name))
        .note(context)
        .help("Check that the type is imported or defined in the current scope")
        .build()
    }

    /// Emit undefined symbol diagnostic
    fn emit_undefined_symbol(
        &self,
        location: SourceLocation,
        name: InternedString,
        context: &str,
    ) -> Diagnostic {
        let symbol_name = self.string_interner.get(name).unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!("Undefined symbol `{}`", symbol_name),
            source_span.clone(),
        )
        .code(format_error_code(2001)) // E2001: Undefined symbol
        .label(source_span, format!("symbol `{}` not found", symbol_name))
        .note(context)
        .help("Check that the symbol is declared and in scope")
        .build()
    }

    /// Emit access non-object diagnostic
    fn emit_access_non_object(
        &self,
        location: SourceLocation,
        object_type: TypeId,
        field_name: InternedString,
        context: &str,
    ) -> Diagnostic {
        let type_name = self.format_type_name(object_type);
        let field_name_str = self.string_interner.get(field_name).unwrap_or("<unknown>");
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!(
                "Cannot access field `{}` on type `{}`",
                field_name_str, type_name
            ),
            source_span.clone(),
        )
        .code(format_error_code(1202)) // E1202: Field access on non-object
        .label(source_span, format!("field access on `{}`", type_name))
        .note(context)
        .help(format!(
            "Only object types (classes, interfaces) have fields. `{}` is not an object type.",
            type_name
        ))
        .build()
    }

    /// Emit invalid type arguments diagnostic
    fn emit_invalid_type_arguments(
        &self,
        location: SourceLocation,
        base_type: TypeId,
        expected_count: usize,
        actual_count: usize,
    ) -> Diagnostic {
        let type_name = self.format_type_name(base_type);
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!(
                "Wrong number of type arguments for `{}`: expected {}, found {}",
                type_name, expected_count, actual_count
            ),
            source_span.clone(),
        )
        .code(format_error_code(3001)) // E3001: Generic parameter count mismatch
        .label(
            source_span,
            format!("expected {} type arguments", expected_count),
        )
        .help(format!(
            "The type `{}` requires exactly {} type argument{}",
            type_name,
            expected_count,
            if expected_count == 1 { "" } else { "s" }
        ))
        .build()
    }

    /// Emit constraint violation diagnostic
    fn emit_constraint_violation(
        &self,
        location: SourceLocation,
        type_param: TypeId,
        constraint: TypeId,
        violating_type: TypeId,
    ) -> Diagnostic {
        let param_name = self.format_type_name(type_param);
        let constraint_name = self.format_type_name(constraint);
        let violating_name = self.format_type_name(violating_type);
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!(
                "Type parameter `{}` requires constraint `{}`, but `{}` does not satisfy this constraint",
                param_name, constraint_name, violating_name
            ),
            source_span.clone()
        )
        .code(format_error_code(3101))  // E3101: Constraint violation
        .label(source_span, format!("`{}` does not satisfy `{}`", violating_name, constraint_name))
        .help(format!(
            "Ensure that `{}` implements or extends `{}`",
            violating_name, constraint_name
        ))
        .build()
    }

    /// Emit circular dependency diagnostic
    fn emit_circular_dependency(&self, location: SourceLocation, types: Vec<TypeId>) -> Diagnostic {
        let type_names: Vec<String> = types.iter().map(|&t| self.format_type_name(t)).collect();
        let cycle = type_names.join(" -> ");
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error("Circular type dependency detected", source_span.clone())
            .code(format_error_code(1004)) // E1004: Circular type dependency
            .label(source_span, "circular dependency here")
            .note(format!("Dependency cycle: {}", cycle))
            .help("Break the cycle by using interfaces or forward declarations")
            .build()
    }

    /// Emit invalid cast diagnostic
    fn emit_invalid_cast(
        &self,
        location: SourceLocation,
        from_type: TypeId,
        to_type: TypeId,
    ) -> Diagnostic {
        let from_name = self.format_type_name(from_type);
        let to_name = self.format_type_name(to_type);
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!("Cannot cast `{}` to `{}`", from_name, to_name),
            source_span.clone(),
        )
        .code(format_error_code(1003)) // E1003: Invalid type annotation
        .label(
            source_span.clone(),
            format!("invalid cast from `{}` to `{}`", from_name, to_name),
        )
        .help("Consider using explicit conversion methods or checking type compatibility")
        .build()
    }

    /// Emit signature mismatch diagnostic
    fn emit_signature_mismatch(
        &self,
        location: SourceLocation,
        expected_params: Vec<TypeId>,
        actual_params: Vec<TypeId>,
        expected_return: TypeId,
        actual_return: TypeId,
    ) -> Diagnostic {
        let expected_sig = self.format_function_signature(&expected_params, expected_return);
        let actual_sig = self.format_function_signature(&actual_params, actual_return);
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error("Function signature mismatch", source_span.clone())
            .code(format_error_code(1102)) // E1102: Invalid return type
            .label(source_span, "signature mismatch here")
            .note(format!("Expected: {}", expected_sig))
            .note(format!("Found: {}", actual_sig))
            .build()
    }

    /// Emit access violation diagnostic
    fn emit_access_violation(
        &self,
        location: SourceLocation,
        symbol_id: SymbolId,
        required_access: AccessLevel,
    ) -> Diagnostic {
        let symbol_name = self.get_symbol_name(symbol_id);
        let access_str = match required_access {
            AccessLevel::Private => "private",
            AccessLevel::Protected => "protected",
            AccessLevel::Public => "public",
            AccessLevel::Internal => "internal",
        };
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error(
            format!("Cannot access {} member `{}`", access_str, symbol_name),
            source_span.clone(),
        )
        .code(format_error_code(2101)) // E2101: Private symbol access
        .label(source_span, format!("{} member", access_str))
        .help("Make the member public or access it from within the defining class")
        .build()
    }

    /// Emit type inference failure diagnostic
    fn emit_inference_failed(&self, location: SourceLocation, reason: &str) -> Diagnostic {
        let source_span = self.location_to_span(location);

        DiagnosticBuilder::error("Type inference failed", source_span.clone())
            .code(format_error_code(1005)) // E1005: Type inference failed
            .label(source_span, "cannot infer type")
            .note(reason)
            .help("Consider adding explicit type annotations")
            .build()
    }

    /// Emit Send/Sync concurrency violation diagnostic
    fn emit_send_sync_violation(
        &self,
        location: SourceLocation,
        type_name: &str,
        reason: &str,
        suggestion: Option<&str>,
    ) -> Diagnostic {
        let source_span = self.location_to_span_with_length(location, Some(type_name));

        let mut builder = DiagnosticBuilder::error(
            format!("Cannot send value of type `{}` across threads", type_name),
            source_span.clone(),
        )
        .code("E0302")
        .label(source_span, reason.to_string());

        if let Some(hint) = suggestion {
            builder = builder.help(hint);
        } else {
            builder = builder.help(
                "Add @:derive([Send]) to the type declaration, or use a Send-safe alternative",
            );
        }

        builder.build()
    }

    fn emit_null_assignment_to_not_null(
        &self,
        location: SourceLocation,
        variable_name: &str,
    ) -> Diagnostic {
        let source_span = self.location_to_span_with_length(location, Some(variable_name));
        DiagnosticBuilder::error(
            format!(
                "Cannot assign null to @:notNull variable '{}'",
                variable_name
            ),
            source_span,
        )
        .code("E0400")
        .help("Use a non-null value or remove the @:notNull annotation")
        .build()
    }

    fn emit_nullable_to_non_null_param(
        &self,
        location: SourceLocation,
        param_name: &str,
        function_name: &str,
    ) -> Diagnostic {
        let source_span = self.location_to_span_with_length(location, Some(param_name));
        DiagnosticBuilder::error(
            format!(
                "Cannot pass nullable value to @:notNull parameter '{}' of '{}'",
                param_name, function_name
            ),
            source_span,
        )
        .code("E0401")
        .help("Add a null check before the call")
        .build()
    }

    fn emit_nullable_return(&self, location: SourceLocation, function_name: &str) -> Diagnostic {
        let source_span = self.location_to_span_with_length(location, Some(function_name));
        DiagnosticBuilder::error(
            format!(
                "Cannot return nullable value from @:notNull function '{}'",
                function_name
            ),
            source_span,
        )
        .code("E0402")
        .help("Return a non-null value or remove @:notNull from the return type")
        .build()
    }

    /// Helper: Convert SourceLocation to SourceSpan
    /// If token_name is provided, creates a span covering the full token length
    fn location_to_span(&self, location: SourceLocation) -> SourceSpan {
        self.location_to_span_with_length(location, None)
    }

    /// Helper: Convert SourceLocation to SourceSpan with optional token name for proper underlining
    fn location_to_span_with_length(
        &self,
        location: SourceLocation,
        token_name: Option<&str>,
    ) -> SourceSpan {
        let file_id = FileId::new(location.file_id as usize);
        let start_pos = SourcePosition::new(
            location.line as usize,
            location.column as usize,
            location.byte_offset as usize,
        );

        // If we have a token name, calculate the span to cover the whole token
        if let Some(name) = token_name {
            let token_len = name.len().max(1);
            let end_pos = SourcePosition::new(
                location.line as usize,
                location.column as usize + token_len,
                location.byte_offset as usize + token_len,
            );
            SourceSpan::new(start_pos, end_pos, file_id)
        } else {
            // Default to single character span
            SourceSpan::single_position(start_pos, file_id)
        }
    }

    /// Helper: Create a diagnostic with a given span (avoids move issues)
    fn create_diagnostic_with_span(
        &self,
        severity: &str,
        message: String,
        span: &SourceSpan,
        code: &str,
        label_msg: String,
    ) -> DiagnosticBuilder {
        match severity {
            "error" => DiagnosticBuilder::error(message, span.clone())
                .code(code)
                .label(span.clone(), label_msg),
            "warning" => DiagnosticBuilder::warning(message, span.clone())
                .code(code)
                .label(span.clone(), label_msg),
            _ => DiagnosticBuilder::error(message, span.clone())
                .code(code)
                .label(span.clone(), label_msg),
        }
    }

    /// Helper: Format type name for display
    fn format_type_name(&self, type_id: TypeId) -> String {
        // Special case for invalid type ID
        if !type_id.is_valid() {
            return "Unknown".to_string();
        }

        // Look up the type in the TypeTable and format it nicely
        match self.type_table.borrow().get(type_id) {
            Some(type_info) => {
                use crate::tast::core::TypeKind;
                match &type_info.kind {
                    TypeKind::Void => "Void".to_string(),
                    TypeKind::Bool => "Bool".to_string(),
                    TypeKind::Int => "Int".to_string(),
                    TypeKind::Float => "Float".to_string(),
                    TypeKind::String => "String".to_string(),
                    TypeKind::Dynamic => "Dynamic".to_string(),
                    TypeKind::Class { symbol_id, .. } => {
                        // Look up the class name from symbol table
                        if let Some(symbol) = self.symbol_table.get_symbol(*symbol_id) {
                            self.string_interner
                                .get(symbol.name)
                                .unwrap_or("<unknown>")
                                .to_string()
                        } else {
                            format!("Class#{}", symbol_id.as_raw())
                        }
                    }
                    TypeKind::Interface { symbol_id, .. } => {
                        // Look up the interface name from symbol table
                        if let Some(symbol) = self.symbol_table.get_symbol(*symbol_id) {
                            self.string_interner
                                .get(symbol.name)
                                .unwrap_or("<unknown>")
                                .to_string()
                        } else {
                            format!("Interface#{}", symbol_id.as_raw())
                        }
                    }
                    TypeKind::Enum { symbol_id, .. } => {
                        // Look up the enum name from symbol table
                        if let Some(symbol) = self.symbol_table.get_symbol(*symbol_id) {
                            self.string_interner
                                .get(symbol.name)
                                .unwrap_or("<unknown>")
                                .to_string()
                        } else {
                            format!("Enum#{}", symbol_id.as_raw())
                        }
                    }
                    TypeKind::Function {
                        params,
                        return_type,
                        ..
                    } => {
                        let param_types: Vec<String> =
                            params.iter().map(|&p| self.format_type_name(p)).collect();
                        format!(
                            "({}) -> {}",
                            param_types.join(", "),
                            self.format_type_name(*return_type)
                        )
                    }
                    TypeKind::Array { element_type } => {
                        format!("Array<{}>", self.format_type_name(*element_type))
                    }
                    TypeKind::Optional { inner_type } => {
                        format!("Null<{}>", self.format_type_name(*inner_type))
                    }
                    TypeKind::TypeParameter { symbol_id, .. } => {
                        // Look up the type parameter name
                        if let Some(symbol) = self.symbol_table.get_symbol(*symbol_id) {
                            self.string_interner
                                .get(symbol.name)
                                .unwrap_or("<unknown>")
                                .to_string()
                        } else {
                            format!("T#{}", symbol_id.as_raw())
                        }
                    }
                    TypeKind::Placeholder { name } => {
                        // Format placeholder types with their names
                        self.string_interner
                            .get(*name)
                            .unwrap_or("Unknown")
                            .to_string()
                    }
                    TypeKind::Unknown => "Unknown".to_string(),
                    _ => {
                        // For other types, fall back to a descriptive name
                        format!("{:?}", type_info.kind)
                            .split_whitespace()
                            .next()
                            .unwrap_or("Unknown")
                            .to_string()
                    }
                }
            }
            None => {
                // Type not found in table, show the ID
                format!("Type#{}", type_id.as_raw())
            }
        }
    }

    /// Helper: Check if two types are similar (for better error messages)
    fn are_similar_types(&self, _type1: TypeId, _type2: TypeId) -> bool {
        // TODO: Implement similarity checking (e.g., Int vs Float)
        false
    }

    /// Helper: Format function signature for display
    fn format_function_signature(&self, params: &[TypeId], return_type: TypeId) -> String {
        let param_names: Vec<String> = params.iter().map(|&p| self.format_type_name(p)).collect();
        format!(
            "({}) -> {}",
            param_names.join(", "),
            self.format_type_name(return_type)
        )
    }

    /// Helper: Get symbol name for display
    fn get_symbol_name(&self, symbol_id: SymbolId) -> String {
        // Look up symbol name from symbol table and resolve via string interner
        if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
            if let Some(name_str) = self.string_interner.get(symbol.name) {
                return name_str.to_string();
            }
        }
        // Fallback if symbol or name not found
        format!("symbol#{}", symbol_id.as_raw())
    }
}

/// Convenience function to convert TypeCheckError to Diagnostic
pub fn type_error_to_diagnostic(
    error: TypeCheckError,
    type_table: &Rc<RefCell<TypeTable>>,
    symbol_table: &SymbolTable,
    string_interner: &StringInterner,
    source_map: &SourceMap,
) -> Diagnostic {
    let emitter = TypeDiagnosticEmitter::new(type_table, symbol_table, string_interner, source_map);
    emitter.emit_diagnostic(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::core::TypeTable;

    #[test]
    fn test_type_mismatch_diagnostic() {
        // TODO: Add comprehensive tests for diagnostic emission
    }

    #[test]
    fn test_undefined_type_diagnostic() {
        // TODO: Add test for undefined type error conversion
    }
}
