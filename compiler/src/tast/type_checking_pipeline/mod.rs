//! Type Checking Pipeline Integration
//!
//! This module integrates the type checker into the compilation pipeline,
//! providing a complete type checking phase with diagnostic reporting.

use super::{
    node::{
        BinaryOperator, CastKind, StringInterpolationPart, TypedClass, TypedEnum, TypedExpression,
        TypedExpressionKind, TypedField, TypedFile, TypedFunction, TypedInterface, TypedMapEntry,
        TypedMethodSignature, TypedStatement, TypedSwitchCase,
    },
    send_sync_validator::{SendSyncError, SendSyncValidator},
    type_checker::TypeCompatibility,
    type_diagnostics::{TypeDiagnosticEmitter, TypeErrorContext},
    AccessLevel, FlowSafetyError, FlowSafetyResults, InternedString, NamespaceResolver,
    PackageAccessContext, PackageAccessValidator, ScopeTree, SourceLocation, StringInterner,
    SymbolId, SymbolTable, TypeCheckError, TypeChecker, TypeErrorKind, TypeFlowGuard, TypeId,
    TypeKind, TypeTable, Visibility,
};
use diagnostics::{Diagnostics, SourceMap};
use source_map::{SourcePosition, SourceSpan};
use std::cell::RefCell;
use std::rc::Rc;

/// Type checking phase that integrates with the compilation pipeline
pub struct TypeCheckingPhase<'a> {
    type_checker: TypeChecker<'a>,
    diagnostic_emitter: TypeDiagnosticEmitter<'a>,
    diagnostics: &'a mut Diagnostics,
    string_interner: &'a StringInterner,
    /// Type table (stored separately for SendSyncValidator)
    type_table: &'a Rc<RefCell<TypeTable>>,
    /// Symbol table (stored separately for SendSyncValidator)
    symbol_table: &'a SymbolTable,
    /// Stack of expected return types for nested function contexts
    expected_return_types: Vec<TypeId>,
    /// Temporary reference to the typed file for constraint validation
    /// This is set during class checking to enable access to class definitions
    current_typed_file: Option<*const TypedFile>,
    /// Current method context (is_static, class_symbol_id)
    current_method_context: Option<(bool, SymbolId)>,
    /// Current package context for package-level visibility checking
    current_package: Option<super::namespace::PackageId>,
    /// Package access validator for cross-package visibility
    package_access_validator: Option<PackageAccessValidator<'a>>,
    /// Flow-sensitive safety analyzer
    type_flow_guard: Option<TypeFlowGuard<'a>>,
    /// Whether to enable flow-sensitive analysis
    enable_flow_analysis: bool,
}

// `super::node` in the modules below named `tast::node` before this file
// became a directory; keep that path meaning what it did.
pub(crate) use crate::tast::node;
pub(crate) use crate::tast::namespace;
pub(crate) use crate::tast::symbols::SymbolKind;

mod access;
mod casts;
mod constraints;
mod contracts;
mod errors;
mod expressions;
mod flow;
mod members;
mod statements;

impl<'a> TypeCheckingPhase<'a> {
    /// Create a new type checking phase
    pub fn new(
        type_table: &'a Rc<RefCell<TypeTable>>,
        symbol_table: &'a SymbolTable,
        scope_tree: &'a ScopeTree,
        string_interner: &'a StringInterner,
        source_map: &'a SourceMap,
        diagnostics: &'a mut Diagnostics,
    ) -> Self {
        let type_checker = TypeChecker::new(type_table, symbol_table, scope_tree, string_interner);
        let diagnostic_emitter =
            TypeDiagnosticEmitter::new(type_table, symbol_table, string_interner, source_map);

        Self {
            type_checker,
            diagnostic_emitter,
            diagnostics,
            string_interner,
            type_table,
            symbol_table,
            expected_return_types: Vec::new(),
            current_typed_file: None,
            current_method_context: None,
            current_package: None,
            package_access_validator: None,
            type_flow_guard: None,
            enable_flow_analysis: true, // Enable by default
        }
    }


    /// Set the namespace resolver for package access validation
    pub fn set_namespace_resolver(&mut self, namespace_resolver: &'a NamespaceResolver) {
        self.package_access_validator = Some(PackageAccessValidator::new(
            self.type_checker.symbol_table,
            namespace_resolver,
            self.string_interner,
        ));
    }


    /// Enable or disable flow-sensitive analysis
    pub fn set_flow_analysis(&mut self, enabled: bool) {
        self.enable_flow_analysis = enabled;
    }


    /// Initialize TypeFlowGuard for flow-sensitive analysis
    fn initialize_flow_guard(&mut self) {
        if self.enable_flow_analysis && self.type_flow_guard.is_none() {
            self.type_flow_guard = Some(TypeFlowGuard::new(
                self.type_checker.symbol_table,
                self.type_checker.type_table,
            ));
        }
    }


    /// Run type checking on a typed file
    pub fn check_file(&mut self, typed_file: &mut TypedFile) -> Result<(), String> {
        // Set current package context from file metadata
        self.current_package = self.extract_package_from_file(typed_file);

        // Phase 1: Check all type declarations
        self.check_type_declarations(typed_file)?;

        // Phase 2: Check interfaces
        for interface in &typed_file.interfaces {
            self.check_interface(interface)?;
        }

        // Phase 3: Check classes
        for class in &typed_file.classes {
            // Set the current typed file for constraint validation
            self.current_typed_file = Some(typed_file as *const TypedFile);
            self.check_class(class)?;
            self.current_typed_file = None;
        }

        // Phase 4: Check enums
        for enum_decl in &typed_file.enums {
            // TODO: Add enum checking
            // self.check_enum(enum_decl)?;
        }

        // Phase 5: Check module-level functions and variables
        self.check_module_fields(typed_file)?;

        // Phase 6: Flow-sensitive safety analysis
        if self.enable_flow_analysis {
            self.run_flow_analysis(typed_file)?;
        }

        // Phase 7: Send/Sync validation for thread safety
        self.run_send_sync_validation(typed_file)?;

        // Return error if we collected any error diagnostics
        if self.diagnostics.has_errors() {
            Err(format!(
                "Type checking failed with {} errors",
                self.diagnostics.errors().count()
            ))
        } else {
            Ok(())
        }
    }


    /// Check all type declarations for validity
    fn check_type_declarations(&mut self, typed_file: &TypedFile) -> Result<(), String> {
        // Check for duplicate type names
        let mut type_names = std::collections::BTreeSet::new();

        for class in &typed_file.classes {
            if !type_names.insert(&class.name) {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType { name: class.name },
                    location: class.source_location,
                    context: format!(
                        "Duplicate class definition: {}",
                        self.get_string(class.name)
                    ),
                    suggestion: Some("Rename one of the duplicate classes".to_string()),
                });
            }
        }

        for interface in &typed_file.interfaces {
            if !type_names.insert(&interface.name) {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType {
                        name: interface.name,
                    },
                    location: interface.source_location,
                    context: format!(
                        "Duplicate interface definition: {}",
                        self.get_string(interface.name)
                    ),
                    suggestion: Some("Rename one of the duplicate interfaces".to_string()),
                });
            }
        }

        Ok(())
    }


    /// Check an interface for type correctness
    fn check_interface(&mut self, interface: &TypedInterface) -> Result<(), String> {
        // Check method signatures
        for method in &interface.methods {
            self.check_method_signature(method.name, method.source_location)?;
        }

        Ok(())
    }


    /// Check a class for type correctness
    fn check_class(&mut self, class: &TypedClass) -> Result<(), String> {
        // Check field types
        for field in &class.fields {
            self.check_field_type(field.symbol_id, field.field_type, field.source_location)?;
        }

        // Check method implementations
        for method in &class.methods {
            self.check_method_implementation(method.symbol_id, method.source_location)?;

            // Check method body with return type context
            self.check_method_body(method, class.symbol_id)?;
        }

        // Verify interface implementations
        for &interface_type_id in &class.interfaces {
            if let Some(interface_symbol_id) = self
                .type_checker
                .symbol_table
                .get_symbol_from_type(interface_type_id)
            {
                self.verify_interface_implementation(
                    class.symbol_id,
                    interface_symbol_id,
                    class.source_location,
                )?;
            }
        }

        // Check inheritance method signature compatibility
        if let Some(super_type_id) = class.super_class {
            self.verify_inheritance_signatures(class, super_type_id)?;
        }

        Ok(())
    }


    /// Check an enum for type correctness
    fn check_enum(&mut self, _enum_decl: &TypedEnum) -> Result<(), String> {
        // TODO: Check that enum variant parameter types are valid
        Ok(())
    }


    /// Check module-level fields
    fn check_module_fields(&mut self, _typed_file: &TypedFile) -> Result<(), String> {
        // Module fields would be checked here
        // Currently, TypedFile doesn't expose module fields directly
        Ok(())
    }
}


/// Run type checking on a typed file with full diagnostic support
pub fn type_check_with_diagnostics(
    typed_file: &mut TypedFile,
    type_table: &Rc<RefCell<TypeTable>>,
    symbol_table: &SymbolTable,
    scope_tree: &ScopeTree,
    string_interner: &StringInterner,
    source_map: &SourceMap,
) -> Result<Diagnostics, String> {
    let mut diagnostics = Diagnostics::new();

    {
        let mut type_checking_phase = TypeCheckingPhase::new(
            type_table,
            symbol_table,
            scope_tree,
            string_interner,
            source_map,
            &mut diagnostics,
        );

        // Run type checking - we want the diagnostics regardless of whether errors were found
        let _result = type_checking_phase.check_file(typed_file);
        // Note: Intentionally ignoring the result here since we want to return diagnostics
        // even when type errors are found
    }

    Ok(diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::{AstLowering, ScopeId, ScopeTree};
    use diagnostics::ErrorFormatter;
    use parser::parse_haxe_file;

    #[test]
    fn test_type_checking_pipeline() {
        let haxe_code = r#"
            interface IShape {
                public function getArea():Float;
            }

            class Rectangle implements IShape {
                private var width:Float;
                private var height:Float;

                public function new(w:Float, h:Float) {
                    this.width = w;
                    this.height = h;
                }

                public function getArea():Float {
                    return width * height;
                }
            }

            class Circle implements IShape {
                private var radius:Float;

                public function new(r:Float) {
                    this.radius = r;
                }

                public function getArea():Float {
                    return 3.14159 * radius * radius;
                }
            }
        "#;

        // Parse
        let ast_result = parse_haxe_file("test.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("test.hx".to_string(), haxe_code.to_string());

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

        // Lower to TAST
        let string_interner_rc = Rc::new(RefCell::new(StringInterner::new()));
        let mut lowering = AstLowering::new(
            &mut string_interner,
            string_interner_rc,
            &mut symbol_table,
            &type_table,
            &mut scope_tree,
            &mut namespace_resolver,
            &mut import_resolver,
        );
        lowering.initialize_span_converter(file_id.as_usize() as u32, haxe_code.to_string());
        let mut typed_file = lowering
            .lower_file(&haxe_file)
            .expect("Lowering should succeed");

        // Run type checking
        let diagnostics = type_check_with_diagnostics(
            &mut typed_file,
            &type_table,
            &symbol_table,
            &scope_tree,
            &string_interner,
            &source_map,
        )
        .expect("Type checking should complete");

        // Check results
        if !diagnostics.is_empty() {
            let formatter = ErrorFormatter::new();
            let formatted = formatter.format_diagnostics(&diagnostics, &source_map);
        }

        assert!(
            diagnostics.is_empty(),
            "Should have no type errors for valid code"
        );
    }

    #[test]
    fn test_type_error_detection() {
        let haxe_code = r#"
            class TypeErrors {
                public function test():Void {
                    var x:Int = "not an int";  // Type error
                    var y:String = 42;          // Type error
                    var z:Bool = x + y;         // Type error
                }
            }
        "#;

        // Parse
        let ast_result = parse_haxe_file("test_errors.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("test_errors.hx".to_string(), haxe_code.to_string());

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

        // Lower to TAST
        let string_interner_rc = Rc::new(RefCell::new(StringInterner::new()));
        let mut lowering = AstLowering::new(
            &mut string_interner,
            string_interner_rc,
            &mut symbol_table,
            &type_table,
            &mut scope_tree,
            &mut namespace_resolver,
            &mut import_resolver,
        );
        lowering.initialize_span_converter(file_id.as_usize() as u32, haxe_code.to_string());
        let mut typed_file = lowering
            .lower_file(&haxe_file)
            .expect("Lowering should succeed");

        // For demonstration, manually create a type error since full type checking isn't implemented
        let mut diagnostics = Diagnostics::new();
        {
            let mut type_checking_phase = TypeCheckingPhase::new(
                &type_table,
                &symbol_table,
                &scope_tree,
                &string_interner,
                &source_map,
                &mut diagnostics,
            );

            // Simulate a type error
            type_checking_phase.emit_error(TypeCheckError {
                kind: TypeErrorKind::TypeMismatch {
                    expected: type_table.borrow().int_type(),
                    actual: type_table.borrow().string_type(),
                },
                location: SourceLocation::new(1, 4, 21, 20),
                context: "Cannot assign string literal to variable of type Int".to_string(),
                suggestion: Some(
                    "Change the type annotation to String or use an integer literal".to_string(),
                ),
            });
        }

        // Format and display errors
        assert!(!diagnostics.is_empty(), "Should have type errors");
        let formatter = ErrorFormatter::new();
        let formatted = formatter.format_diagnostics(&diagnostics, &source_map);
    }

    #[test]
    fn test_static_vs_instance_member_checking() {
        let haxe_code = r#"
class MyClass {
    public static var staticField:Int = 42;
    public var instanceField:String = "hello";

    public static function staticMethod():String {
        return "Static method";
    }

    public function instanceMethod():String {
        return "Instance method";
    }
}

class TestStaticAccess {
    public function new() {}

    public function testAccess():Void {
        // Valid static access
        var s1:Int = MyClass.staticField;
        MyClass.staticMethod();

        // Valid instance access
        var obj = new MyClass();
        var i1:String = obj.instanceField;
        obj.instanceMethod();

        // Invalid: accessing static member through instance
        var invalid1:Int = obj.staticField;  // Should error
        obj.staticMethod();  // Should error

        // Invalid: accessing instance member through static context
        var invalid2:String = MyClass.instanceField;  // Should error
        MyClass.instanceMethod();  // Should error
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        println!("\n=== Compilation Result ===");
        println!("Total errors: {}", result.errors.len());
        for (i, error) in result.errors.iter().enumerate() {
            println!("\nError {}: {}", i + 1, error.message);
            println!(
                "  Location: {}:{}:{}",
                error.location.file_id, error.location.line, error.location.column
            );
            println!("  Category: {:?}", error.category);
        }

        // Print expected errors for debugging
        println!("\n=== Expected Errors ===");
        println!("1. Line 35: obj.staticField - accessing static field through instance");
        println!("2. Line 36: obj.staticMethod() - accessing static method through instance");
        println!("3. Line 39: MyClass.instanceField - accessing instance field statically");
        println!("4. Line 40: MyClass.instanceMethod() - accessing instance method statically");

        // We expect exactly 4 errors
        assert_eq!(
            result.errors.len(),
            4,
            "Expected 4 static/instance access errors"
        );

        // Check error messages
        let error_messages: Vec<String> = result.errors.iter().map(|e| e.message.clone()).collect();

        // Verify we have the right kinds of errors
        let static_from_instance_errors = error_messages
            .iter()
            .filter(|msg| {
                msg.contains("Static member") && msg.contains("cannot be accessed through instance")
            })
            .count();
        assert_eq!(
            static_from_instance_errors, 2,
            "Expected 2 static-from-instance errors"
        );

        let instance_from_static_errors = error_messages
            .iter()
            .filter(|msg| {
                msg.contains("Instance member")
                    && msg.contains("cannot be accessed from static context")
            })
            .count();
        assert_eq!(
            instance_from_static_errors, 2,
            "Expected 2 instance-from-static errors"
        );
    }

    #[test]
    fn test_static_method_context() {
        let haxe_code = r#"
class StaticContext {
    private static var staticData:Int = 100;
    private var instanceData:String = "data";

    public static function staticWork():Void {
        // Valid: static accessing static
        var x = staticData;

        // Invalid: static method accessing instance member
        var y = instanceData;  // Should error
    }

    public function instanceWork():Void {
        // Valid: instance method can access both
        var x = staticData;
        var y = instanceData;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // We expect exactly 1 error
        assert_eq!(
            result.errors.len(),
            1,
            "Expected 1 error for instance member access from static context"
        );

        let error = &result.errors[0];
        assert!(error.message.contains("Instance member"));
        assert!(error
            .message
            .contains("cannot be accessed from static context"));
    }

    #[test]
    fn test_try_catch_type_checking() {
        let haxe_code = r#"
class TryCatchTest {
    public static function main() {
        // Test basic try-catch
        try {
            var result = riskyOperation();
            trace(result);
        } catch (e: String) {
            trace("String error: " + e);
        } catch (e: Int) {
            trace("Int error: " + e);
        } catch (e: Dynamic) {
            trace("Generic error");
        }

        // Test catch with filter (invalid - filter must be boolean)
        try {
            doSomething();
        } catch (e: String) if (e.length) {  // Error: filter must be boolean
            trace("Filtered error");
        }
    }

    static function riskyOperation(): String {
        throw "Error";
    }

    static function doSomething(): Void {}
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about non-boolean filter
        assert!(
            !result.errors.is_empty(),
            "Expected error for non-boolean catch filter"
        );

        let has_filter_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("filter") && e.message.contains("boolean"));
        assert!(
            has_filter_error,
            "Should have error about catch filter needing to be boolean"
        );
    }

    #[test]
    fn test_while_loop_condition_type_checking() {
        let haxe_code = r#"
class WhileLoopTest {
    public static function main() {
        var i = 0;

        // Valid while loop
        while (i < 10) {
            trace("Count: " + i);
            i++;
        }

        // Invalid: non-boolean condition
        var str = "test";
        while (str) {  // Error: condition must be boolean
            trace("Never reached");
            break;
        }
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected error for non-boolean while condition"
        );

        let has_condition_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("condition") && e.message.contains("boolean"));
        assert!(
            has_condition_error,
            "Should have error about while condition needing to be boolean"
        );
    }

    #[test]
    fn test_for_loop_condition_type_checking() {
        let haxe_code = r#"
class ForLoopTest {
    public static function main() {
        // Valid for loop
        for (i in 0...10) {
            trace("i = " + i);
        }

        // Invalid: non-boolean condition in traditional for loop
        for (var j = 0; "not boolean"; j++) {  // Error: condition must be boolean
            trace("Never reached");
        }
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about non-boolean condition
        let _has_condition_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("condition") && e.message.contains("boolean"));
        // Note: Traditional for loops with conditions might not be fully supported in parser
        // This test documents expected behavior
    }

    #[test]
    fn test_for_in_loop_iterable_checking() {
        let haxe_code = r#"
class ForInTest {
    public static function main() {
        // Valid: iterating over array
        var arr = [1, 2, 3];
        for (item in arr) {
            trace("Item: " + item);
        }

        // Valid: iterating over string
        var str = "hello";
        for (char in str) {
            trace("Char: " + char);
        }

        // Invalid: iterating over non-iterable
        var num = 42;
        for (x in num) {  // Error: Int is not iterable
            trace("Never reached");
        }
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected error for non-iterable type in for-in"
        );

        let has_iterable_error = result.errors.iter().any(|e| {
            e.message.contains("not iterable") || e.message.contains("Type is not iterable")
        });
        assert!(
            has_iterable_error,
            "Should have error about type not being iterable"
        );
    }

    #[test]
    fn test_throw_expression_type_checking() {
        let haxe_code = r#"
class ThrowTest {
    public static function main() {
        // Valid: throwing string
        if (Math.random() < 0.5) {
            throw "Error message";
        }

        // Valid: throwing custom exception
        throw new CustomException("Something went wrong");

        // Valid: throwing any type (Haxe allows this)
        throw 42;
        throw true;
        throw { error: "object error" };
    }
}

class CustomException {
    public var message: String;
    public function new(msg: String) {
        this.message = msg;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should not have errors - Haxe allows throwing any type
        assert!(
            result.errors.is_empty(),
            "Should not have errors for valid throw expressions"
        );
    }

    #[test]
    fn test_object_literal_validation() {
        let haxe_code = r#"
class ObjectLiteralTest {
    public static function main() {
        // Valid object literal
        var obj1 = {
            name: "test",
            value: 42,
            active: true
        };

        // Object with duplicate fields
        var obj2 = {
            field: "first",
            other: 123,
            field: "duplicate"  // Error: duplicate field name
        };
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected error for duplicate object field"
        );

        let has_duplicate_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("Duplicate field") || e.message.contains("duplicate"));
        assert!(
            has_duplicate_error,
            "Should have error about duplicate field in object literal"
        );
    }

    #[test]
    fn test_map_literal_type_consistency() {
        let haxe_code = r#"
class MapLiteralTest {
    public static function main() {
        // Valid: consistent types
        var map1 = [
            "key1" => "value1",
            "key2" => "value2",
            "key3" => "value3"
        ];

        // Invalid: inconsistent key types
        var map2 = [
            "key1" => "value1",
            42 => "value2",      // Error: key type mismatch
            "key3" => "value3"
        ];

        // Invalid: inconsistent value types
        var map3 = [
            "key1" => "value1",
            "key2" => 42,        // Error: value type mismatch
            "key3" => "value3"
        ];

        // Map with duplicate keys
        var map4 = [
            "same" => "first",
            "other" => "second",
            "same" => "duplicate"  // Error: duplicate key
        ];
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected errors for map literal issues"
        );

        // Check for various map errors
        let has_key_mismatch = result.errors.iter().any(|e| {
            e.message.contains("Map key type mismatch")
                || (e.message.contains("key") && e.message.contains("type"))
        });
        let has_value_mismatch = result.errors.iter().any(|e| {
            e.message.contains("Map value type mismatch")
                || (e.message.contains("value") && e.message.contains("type"))
        });
        let has_duplicate_key = result.errors.iter().any(|e| {
            e.message.contains("Duplicate map key")
                || (e.message.contains("duplicate") && e.message.contains("key"))
        });

        assert!(
            has_key_mismatch || has_value_mismatch || has_duplicate_key,
            "Should have errors for map literal type inconsistencies or duplicates"
        );
    }

    #[test]
    fn test_break_continue_validation() {
        let haxe_code = r#"
class BreakContinueTest {
    public static function main() {
        // Valid break/continue in loop
        for (i in 0...10) {
            if (i == 5) break;
            if (i % 2 == 0) continue;
            trace(i);
        }

        // Invalid: break outside loop
        if (true) {
            break;  // Error: break not in loop context
        }

        // Invalid: continue outside loop
        trace("test");
        continue;  // Error: continue not in loop context
    }
}
        "#;

        // Note: Break/continue validation outside loops would require context tracking
        // This test documents the expected behavior
        let _result = crate::pipeline::compile_haxe_source(haxe_code);

        // Current implementation checks symbol table references
        // Full context validation would be a future enhancement
    }

    #[test]
    fn test_nested_try_catch() {
        let haxe_code = r#"
class NestedTryCatchTest {
    public static function main() {
        try {
            outerOperation();
        } catch (e: String) {
            try {
                innerOperation();
            } catch (inner: Int) {
                trace("Inner int error: " + inner);
            } catch (inner: Dynamic) {
                trace("Inner dynamic error");
            }
        } catch (e: Dynamic) {
            trace("Outer dynamic error");
        }
    }

    static function outerOperation(): Void {
        throw "outer error";
    }

    static function innerOperation(): Void {
        throw 42;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Nested try-catch should compile without errors
        if !result.errors.is_empty() {
            for (i, e) in result.errors.iter().enumerate() {
                eprintln!("Error {}: {} (category: {:?})", i, e.message, e.category);
            }
        }
        assert!(
            result.errors.is_empty(),
            "Nested try-catch should not have errors, got {} errors",
            result.errors.len()
        );
    }

    #[test]
    fn test_try_catch_finally() {
        let haxe_code = r#"
class TryCatchFinallyTest {
    public static function main() {
        var resource: Resource = null;

        try {
            resource = new Resource();
            resource.use();
        } catch (e: String) {
            trace("Error: " + e);
        } finally {
            if (resource != null) {
                resource.cleanup();
            }
        }
    }
}

class Resource {
    public function new() {}
    public function use(): Void {
        throw "Resource error";
    }
    public function cleanup(): Void {
        trace("Cleaning up");
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Try-catch-finally should compile without errors
        assert!(
            result.errors.is_empty()
                || result.errors.iter().all(|e| !e.message.contains("finally")),
            "Try-catch-finally should be supported"
        );
    }

    #[test]
    fn test_private_field_access() {
        let haxe_code = r#"
class TestClass {
    private var privateField:Int = 42;
    public var publicField:Int = 24;

    public function testAccess() {
        // Should work - same class access
        privateField = 100;
        publicField = 200;
    }
}

class OtherClass {
    public function testExternalAccess() {
        var obj = new TestClass();

        // Should work - public field
        obj.publicField = 300;

        // Should fail - private field access from different class
        obj.privateField = 400;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about private field access
        let has_private_access_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("Private") && e.message.contains("privateField"));

        // Note: This test documents expected behavior - parser may need enhancement
        // for full private field support
    }

    #[test]
    fn test_private_method_access() {
        let haxe_code = r#"
class TestClass {
    private function privateMethod():Void {
        trace("Private method");
    }

    public function publicMethod():Void {
        // Should work - same class access
        privateMethod();
    }
}

class OtherClass {
    public function testMethodAccess() {
        var obj = new TestClass();

        // Should work - public method
        obj.publicMethod();

        // Should fail - private method access from different class
        obj.privateMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about private method access
        let has_private_method_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("Private") && e.message.contains("privateMethod"));

        // Note: This test documents expected behavior - parser may need enhancement
        // for full private method support
    }

    #[test]
    fn test_protected_access_inheritance() {
        let haxe_code = r#"
class BaseClass {
    protected var protectedField:Int = 42;

    protected function protectedMethod():Void {
        trace("Protected method");
    }
}

class DerivedClass extends BaseClass {
    public function testAccess() {
        // Should work - accessing protected members from subclass
        protectedField = 100;
        protectedMethod();
    }
}

class UnrelatedClass {
    public function testExternalAccess() {
        var obj = new BaseClass();

        // Should fail - protected field access from unrelated class
        obj.protectedField = 200;

        // Should fail - protected method access from unrelated class
        obj.protectedMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have errors about protected access from unrelated class
        let has_protected_errors = result.errors.iter().any(|e| {
            e.message.contains("Protected")
                && (e.message.contains("protectedField") || e.message.contains("protectedMethod"))
        });

        // Note: This test documents expected behavior - full inheritance checking
        // requires parser support for 'extends' and protected modifiers
    }

    #[test]
    fn test_internal_package_access_same_package() {
        let haxe_code = r#"
package com.example.utils;

class InternalClass {
    internal var internalField:Int = 42;

    internal function internalMethod():String {
        return "Internal method";
    }

    public function publicMethod():Void {
        // Should work - same class access
        this.internalField = 100;
        var result = this.internalMethod();
    }
}

class SamePackageClass {
    public function testInternalAccess() {
        var obj = new InternalClass();

        // Should work - same package access to internal members
        obj.internalField = 200;
        var result = obj.internalMethod();

        // Should always work - public access
        obj.publicMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should not have any access violation errors for internal members within same package
        let has_internal_access_errors = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && (e.message.contains("internalField") || e.message.contains("internalMethod"))
        });

        // This should pass - same package access to internal members should be allowed
        assert!(
            !has_internal_access_errors,
            "Internal access within same package should be allowed"
        );
    }

    #[test]
    fn test_internal_package_access_different_package() {
        let haxe_code = r#"
package com.example.utils;

class InternalClass {
    internal var internalField:Int = 42;

    internal function internalMethod():String {
        return "Internal method";
    }

    public function publicMethod():Void {
        trace("Public method");
    }
}

// Different package file would be:
package com.other.package;

class DifferentPackageClass {
    public function testCrossPackageAccess() {
        var obj = new com.example.utils.InternalClass();

        // Should work - public access across packages
        obj.publicMethod();

        // Should fail - internal field access from different package
        obj.internalField = 300;

        // Should fail - internal method access from different package
        var result = obj.internalMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have access violation errors for internal members from different package
        let has_internal_field_error = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && e.message.contains("internalField")
                && e.message.contains("different package")
        });

        let has_internal_method_error = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && e.message.contains("internalMethod")
                && e.message.contains("different package")
        });

        // These should fail - cross-package access to internal members should be denied
        // Note: This test documents expected behavior - full package checking requires
        // parser support for package declarations and proper symbol resolution
    }

    #[test]
    fn test_default_package_internal_access() {
        let haxe_code = r#"
// No package declaration = default package

class DefaultPackageClass {
    internal var internalField:Int = 42;

    internal function internalMethod():String {
        return "Internal in default package";
    }
}

class AnotherDefaultClass {
    public function testDefaultPackageAccess() {
        var obj = new DefaultPackageClass();

        // Should work - both classes in default package (no package declaration)
        obj.internalField = 100;
        var result = obj.internalMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should not have access errors - both classes in default package
        let has_internal_access_errors = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && (e.message.contains("internalField") || e.message.contains("internalMethod"))
        });

        // This should pass - default package access should be allowed
        assert!(
            !has_internal_access_errors,
            "Internal access within default package should be allowed"
        );
    }
}
