//! Pipeline validation utilities
//!
//! This module provides comprehensive validation functions to test the compilation pipeline
//! and ensure that AST to TAST lowering preserves all critical information.

use crate::{
    pipeline::{compile_haxe_file, CompilationResult},
    tast::{node::*, StringInterner},
};

use std::{cell::RefCell, rc::Rc};

/// Validate the pipeline with a simple Haxe program
pub fn validate_simple_pipeline() -> CompilationResult {
    let simple_haxe = r#"
        class Main {
            static function main() {
                trace("Hello, World!");
            }
        }
    "#;

    compile_haxe_file("test.hx", simple_haxe)
}

/// Validate the pipeline with a more complex Haxe program
pub fn validate_complex_pipeline() -> CompilationResult {
    let complex_haxe = r#"
        package examples;

        enum Color {
            Red;
            Green;
            Blue;
            RGB(r:Int, g:Int, b:Int);
        }

        class ColorProcessor {
            public static function toString(color:Color):String {
                return switch (color) {
                    case Red: "red";
                    case Green: "green";
                    case Blue: "blue";
                    case RGB(r, g, b): 'rgb($r, $g, $b)';
                }
            }
        }

        class Main {
            static function main() {
                var colors = [Red, Green, Blue, RGB(255, 128, 0)];
                for (color in colors) {
                    trace(ColorProcessor.toString(color));
                }
            }
        }
    "#;

    compile_haxe_file("ColorProcessor.hx", complex_haxe)
}

/// Comprehensive validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub compilation_result: CompilationResult,
    pub validation_errors: Vec<ValidationError>,
    pub validation_warnings: Vec<ValidationWarning>,
    pub info_preservation_score: u8, // 0-100 score for how well information is preserved
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
    pub category: ValidationErrorCategory,
    pub severity: ValidationSeverity,
}

#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
    pub category: ValidationWarningCategory,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationErrorCategory {
    /// Information was lost during AST to TAST lowering
    InformationLoss,
    /// Type information is incorrect or incomplete
    TypeInformation,
    /// Symbol resolution failed
    SymbolResolution,
    /// Source location information is missing
    SourceLocation,
    /// Metadata preservation failed
    MetadataPreservation,
    /// Structural integrity issues
    StructuralIntegrity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationWarningCategory {
    /// Type defaulted to Dynamic when it could be inferred
    TypeDefaulting,
    /// Metadata was simplified or partially lost
    MetadataSimplification,
    /// Performance concerns in the lowering process
    Performance,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValidationSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Validate a comprehensive set of Haxe language features
pub fn validate_comprehensive_pipeline() -> ValidationResult {
    let comprehensive_haxe = r#"
        package validation.test;

        import sys.io.File;
        using StringTools;

        // Test enum with complex constructors
        enum Result<T, E> {
            Ok(value: T);
            Err(error: E);
        }

        // Test interface with generic type parameters
        interface Processor<T> {
            function process(input: T): Result<T, String>;
            function validate(input: T): Bool;
        }

        // Test abstract type with from/to conversions
        abstract Money(Int) from Int to Int {
            public function new(amount: Int) {
                this = amount;
            }

            public function add(other: Money): Money {
                return new Money(this + other);
            }

            @:from
            public static function fromFloat(value: Float): Money {
                return new Money(Math.floor(value));
            }

            @:to
            public function toString(): String {
                return '$this cents';
            }
        }

        // Test class with complex inheritance and generics
        class DataProcessor<T> implements Processor<T> {
            private var cache: Map<String, T>;
            public static var instanceCount: Int = 0;

            public function new() {
                this.cache = new Map();
                DataProcessor.instanceCount++;
            }

            public function process(input: T): Result<T, String> {
                if (validate(input)) {
                    return Ok(input);
                } else {
                    return Err("Invalid input");
                }
            }

            public function validate(input: T): Bool {
                return input != null;
            }

            private function getCacheKey(input: T): String {
                return Std.string(input);
            }
        }

        // Test typedef with complex type
        typedef ProcessorConfig = {
            var maxRetries: Int;
            var timeout: Float;
            var ?debug: Bool;
        }

        // Test class with complex methods and properties
        class ValidationTest {
            public var result(get, set): String;
            private var _result: String = "";

            public function new() {}

            function get_result(): String {
                return _result;
            }

            function set_result(value: String): String {
                return _result = value;
            }

            public static function main() {
                var processor = new DataProcessor<Int>();
                var money = new Money(100);
                var config: ProcessorConfig = {
                    maxRetries: 3,
                    timeout: 5.0,
                    debug: true
                };

                switch (processor.process(42)) {
                    case Ok(value): trace('Success: $value');
                    case Err(error): trace('Error: $error');
                }
            }
        }
    "#;

    let compilation_result = compile_haxe_file("comprehensive_validation.hx", comprehensive_haxe);

    // Perform detailed validation
    let mut validation_errors = Vec::new();
    let mut validation_warnings = Vec::new();
    let mut info_score = 100u8;

    // Validate the compilation result
    validate_compilation_result(
        &compilation_result,
        &mut validation_errors,
        &mut validation_warnings,
        &mut info_score,
    );

    // Validate TAST structure if compilation succeeded
    if !compilation_result.typed_files.is_empty() {
        validate_tast_structure(
            &compilation_result.typed_files[0],
            &mut validation_errors,
            &mut validation_warnings,
            &mut info_score,
        );
    }

    ValidationResult {
        compilation_result,
        validation_errors,
        validation_warnings,
        info_preservation_score: info_score,
    }
}

/// Validate that all critical information is preserved during lowering
fn validate_compilation_result(
    result: &CompilationResult,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    // Check if compilation succeeded
    if !result.errors.is_empty() {
        errors.push(ValidationError {
            message: format!("Compilation failed with {} errors", result.errors.len()),
            category: ValidationErrorCategory::StructuralIntegrity,
            severity: ValidationSeverity::Critical,
        });
        *score = (*score).saturating_sub(50);
    }

    // Check if any files were processed
    if result.stats.files_processed == 0 {
        errors.push(ValidationError {
            message: "No files were processed".to_string(),
            category: ValidationErrorCategory::StructuralIntegrity,
            severity: ValidationSeverity::Critical,
        });
        *score = 0;
    }

    // Check if TAST files were generated
    if result.typed_files.is_empty() && result.errors.is_empty() {
        errors.push(ValidationError {
            message: "No TAST files generated despite successful compilation".to_string(),
            category: ValidationErrorCategory::InformationLoss,
            severity: ValidationSeverity::High,
        });
        *score = (*score).saturating_sub(30);
    }

    // Check for high error rates
    if result.stats.error_count > result.stats.files_processed * 3 {
        warnings.push(ValidationWarning {
            message: format!(
                "High error rate: {} errors for {} files",
                result.stats.error_count, result.stats.files_processed
            ),
            category: ValidationWarningCategory::Performance,
        });
        *score = (*score).saturating_sub(10);
    }
}

/// Validate the structure and completeness of the TAST
fn validate_tast_structure(
    tast_file: &TypedFile,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    let string_interner = tast_file.string_interner();

    // Check if classes were properly lowered
    validate_classes(
        &tast_file.classes,
        string_interner.clone(),
        errors,
        warnings,
        score,
    );

    // Check if interfaces were properly lowered
    validate_interfaces(
        &tast_file.interfaces,
        string_interner.clone(),
        errors,
        warnings,
        score,
    );

    // Check if enums were properly lowered
    validate_enums(
        &tast_file.enums,
        string_interner.clone(),
        errors,
        warnings,
        score,
    );

    // Check if functions were properly lowered
    validate_functions(
        &tast_file.functions,
        string_interner.clone(),
        errors,
        warnings,
        score,
    );

    // Check metadata preservation
    validate_metadata(&tast_file.metadata, errors, warnings, score);
}

/// Validate that classes preserve all AST information
fn validate_classes(
    classes: &[TypedClass],
    string_interner: Rc<RefCell<StringInterner>>,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    for class in classes {
        // Check if class has proper source location
        if !class.source_location.is_valid() {
            let class_name = string_interner
                .borrow()
                .get(class.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            errors.push(ValidationError {
                message: format!("Class '{}' has unknown source location", class_name),
                category: ValidationErrorCategory::SourceLocation,
                severity: ValidationSeverity::Medium,
            });
            *score = (*score).saturating_sub(5);
        }

        // Check if class has proper symbol ID
        if !class.symbol_id.is_valid() {
            let class_name = string_interner
                .borrow()
                .get(class.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            errors.push(ValidationError {
                message: format!("Class '{}' has invalid symbol ID", class_name),
                category: ValidationErrorCategory::SymbolResolution,
                severity: ValidationSeverity::High,
            });
            *score = (*score).saturating_sub(15);
        }

        // Check if methods preserve information
        for method in &class.methods {
            let class_name = string_interner
                .borrow()
                .get(class.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            validate_method_information(
                method,
                &format!("class '{}'", class_name),
                string_interner.clone(),
                errors,
                warnings,
                score,
            );
        }

        // Check if fields preserve information
        for field in &class.fields {
            if !field.source_location.is_valid() {
                let class_name = string_interner
                    .borrow()
                    .get(class.name)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                let field_name = string_interner
                    .borrow()
                    .get(field.name)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                warnings.push(ValidationWarning {
                    message: format!(
                        "Field '{}' in class '{}' has unknown source location",
                        field_name, class_name
                    ),
                    category: ValidationWarningCategory::MetadataSimplification,
                });
                *score = (*score).saturating_sub(2);
            }
        }
    }
}

/// Validate that interfaces preserve all AST information
fn validate_interfaces(
    interfaces: &[TypedInterface],
    string_interner: Rc<RefCell<StringInterner>>,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    for interface in interfaces {
        let interface_name = &interface.name;

        // Check if interface has proper source location
        if !interface.source_location.is_valid() {
            errors.push(ValidationError {
                message: format!(
                    "Interface '{:?}' has unknown source location",
                    interface_name
                ),
                category: ValidationErrorCategory::SourceLocation,
                severity: ValidationSeverity::Medium,
            });
            *score = (*score).saturating_sub(5);
        }

        // Check if interface has proper symbol ID
        if !interface.symbol_id.is_valid() {
            errors.push(ValidationError {
                message: format!("Interface '{:?}' has invalid symbol ID", interface_name),
                category: ValidationErrorCategory::SymbolResolution,
                severity: ValidationSeverity::High,
            });
            *score = (*score).saturating_sub(15);
        }

        // Check if method signatures preserve information
        for method in &interface.methods {
            if !method.source_location.is_valid() {
                let method_name = &method.name;
                warnings.push(ValidationWarning {
                    message: format!(
                        "Method '{:?}' in interface '{:?}' has unknown source location",
                        method_name, interface_name
                    ),
                    category: ValidationWarningCategory::MetadataSimplification,
                });
                *score = (*score).saturating_sub(2);
            }
        }
    }
}

/// Validate that enums preserve all AST information
fn validate_enums(
    enums: &[TypedEnum],
    string_interner: Rc<RefCell<StringInterner>>,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    for enum_def in enums {
        let enum_name = &enum_def.name;

        // Check if enum has proper source location
        if !enum_def.source_location.is_valid() {
            errors.push(ValidationError {
                message: format!("Enum '{:?}' has unknown source location", enum_name),
                category: ValidationErrorCategory::SourceLocation,
                severity: ValidationSeverity::Medium,
            });
            *score = (*score).saturating_sub(5);
        }

        // Check if enum has proper symbol ID
        if !enum_def.symbol_id.is_valid() {
            errors.push(ValidationError {
                message: format!("Enum '{:?}' has invalid symbol ID", enum_name),
                category: ValidationErrorCategory::SymbolResolution,
                severity: ValidationSeverity::High,
            });
            *score = (*score).saturating_sub(15);
        }

        // Check if variants preserve information
        for variant in &enum_def.variants {
            if !variant.source_location.is_valid() {
                warnings.push(ValidationWarning {
                    message: format!(
                        "Variant '{:?}' in enum '{:?}' has unknown source location",
                        variant.name, enum_name
                    ),
                    category: ValidationWarningCategory::MetadataSimplification,
                });
                *score = (*score).saturating_sub(2);
            }
        }
    }
}

/// Validate that functions preserve all AST information
fn validate_functions(
    functions: &[TypedFunction],
    string_interner: Rc<RefCell<StringInterner>>,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    for function in functions {
        validate_method_information(
            function,
            "global",
            string_interner.clone(),
            errors,
            warnings,
            score,
        );
    }
}

/// Validate that method/function information is preserved
fn validate_method_information(
    method: &TypedFunction,
    context: &str,
    string_interner: Rc<RefCell<StringInterner>>,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    let method_name = string_interner
        .borrow()
        .get(method.name)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "<unknown>".to_string());

    // Check if method has proper source location
    if !method.source_location.is_valid() {
        errors.push(ValidationError {
            message: format!(
                "Method '{}' in '{}' has unknown source location",
                method_name, context
            ),
            category: ValidationErrorCategory::SourceLocation,
            severity: ValidationSeverity::Medium,
        });
        *score = (*score).saturating_sub(3);
    }

    // Check if method has proper symbol ID
    if !method.symbol_id.is_valid() {
        errors.push(ValidationError {
            message: format!(
                "Method '{}' in '{}' has invalid symbol ID",
                method_name, context
            ),
            category: ValidationErrorCategory::SymbolResolution,
            severity: ValidationSeverity::High,
        });
        *score = (*score).saturating_sub(10);
    }

    // Check if return type is properly resolved
    if !method.return_type.is_valid() {
        errors.push(ValidationError {
            message: format!(
                "Method '{}' in '{}' has invalid return type",
                method_name, context
            ),
            category: ValidationErrorCategory::TypeInformation,
            severity: ValidationSeverity::High,
        });
        *score = (*score).saturating_sub(10);
    }

    // Check if parameters preserve information
    for param in &method.parameters {
        let param_name = string_interner
            .borrow()
            .get(param.name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "<unknown>".to_string());

        if !param.param_type.is_valid() {
            errors.push(ValidationError {
                message: format!(
                    "Parameter '{}' in method '{}' has invalid type",
                    param_name, method_name
                ),
                category: ValidationErrorCategory::TypeInformation,
                severity: ValidationSeverity::High,
            });
            *score = (*score).saturating_sub(5);
        }

        if !param.source_location.is_valid() {
            warnings.push(ValidationWarning {
                message: format!(
                    "Parameter '{}' in method '{}' has unknown source location",
                    param_name, method_name
                ),
                category: ValidationWarningCategory::MetadataSimplification,
            });
            *score = (*score).saturating_sub(1);
        }
    }
}

/// Validate that metadata is preserved
fn validate_metadata(
    metadata: &crate::tast::node::FileMetadata,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<ValidationWarning>,
    score: &mut u8,
) {
    // Check if package information is preserved
    if metadata.package_name.is_none() {
        warnings.push(ValidationWarning {
            message: "Package information was not preserved".to_string(),
            category: ValidationWarningCategory::MetadataSimplification,
        });
        *score = (*score).saturating_sub(5);
    }

    // Check if file path is preserved
    if metadata.file_path.is_empty() {
        errors.push(ValidationError {
            message: "File path information was lost".to_string(),
            category: ValidationErrorCategory::MetadataPreservation,
            severity: ValidationSeverity::Medium,
        });
        *score = (*score).saturating_sub(8);
    }

    // Check if timestamp is reasonable
    if metadata.timestamp == 0 {
        warnings.push(ValidationWarning {
            message: "File timestamp was not preserved".to_string(),
            category: ValidationWarningCategory::MetadataSimplification,
        });
        *score = (*score).saturating_sub(2);
    }
}

/// Print a comprehensive summary of validation results
pub fn print_validation_summary(name: &str, result: &ValidationResult) {
    println!("=== {} ===", name);
    println!(
        "Information Preservation Score: {}/100",
        result.info_preservation_score
    );
    println!(
        "Files processed: {}",
        result.compilation_result.stats.files_processed
    );
    println!(
        "Parse time: {}μs",
        result.compilation_result.stats.parse_time_us
    );
    println!(
        "Lowering time: {}μs",
        result.compilation_result.stats.lowering_time_us
    );
    println!(
        "Total time: {}μs",
        result.compilation_result.stats.total_time_us
    );
    println!(
        "TAST files: {}",
        result.compilation_result.typed_files.len()
    );
    println!(
        "Compilation errors: {}",
        result.compilation_result.errors.len()
    );
    println!(
        "Compilation warnings: {}",
        result.compilation_result.warnings.len()
    );
    println!("Validation errors: {}", result.validation_errors.len());
    println!("Validation warnings: {}", result.validation_warnings.len());

    // Print critical validation errors
    let critical_errors: Vec<_> = result
        .validation_errors
        .iter()
        .filter(|e| e.severity == ValidationSeverity::Critical)
        .collect();

    if !critical_errors.is_empty() {
        println!("\nCRITICAL VALIDATION ERRORS:");
        for error in critical_errors {
            println!("  - {}", error.message);
        }
    }

    // Print high priority validation errors
    let high_errors: Vec<_> = result
        .validation_errors
        .iter()
        .filter(|e| e.severity == ValidationSeverity::High)
        .collect();

    if !high_errors.is_empty() {
        println!("\nHIGH PRIORITY VALIDATION ERRORS:");
        for error in high_errors {
            println!("  - {}", error.message);
        }
    }

    // Print validation warnings
    if !result.validation_warnings.is_empty() {
        println!("\nVALIDATION WARNINGS:");
        for warning in &result.validation_warnings {
            println!("  - {}", warning.message);
        }
    }

    // Print TAST structure summary
    if !result.compilation_result.typed_files.is_empty() {
        let file = &result.compilation_result.typed_files[0];
        println!("\nTAST STRUCTURE:");
        println!("  - Functions: {}", file.functions.len());
        println!("  - Classes: {}", file.classes.len());
        println!("  - Interfaces: {}", file.interfaces.len());
        println!("  - Enums: {}", file.enums.len());
        println!("  - Abstracts: {}", file.abstracts.len());
        println!("  - Type aliases: {}", file.type_aliases.len());
        if let Some(pkg) = &file.metadata.package_name {
            println!("  - Package: {}", pkg);
        }
    }

    // Print overall assessment
    println!("\nOVERALL ASSESSMENT:");
    if result.info_preservation_score >= 90 {
        println!("  EXCELLENT - Information preservation is excellent");
    } else if result.info_preservation_score >= 80 {
        println!("  GOOD - Information preservation is good with minor issues");
    } else if result.info_preservation_score >= 70 {
        println!("  ACCEPTABLE - Information preservation is acceptable but needs improvement");
    } else if result.info_preservation_score >= 50 {
        println!("  POOR - Significant information loss detected");
    } else {
        println!("  CRITICAL - Major information loss or structural issues");
    }

    println!();
}

/// Print a summary of compilation results (legacy function)
pub fn print_result_summary(name: &str, result: &CompilationResult) {
    println!("=== {} ===", name);
    println!("Files processed: {}", result.stats.files_processed);
    println!("Parse time: {}μs", result.stats.parse_time_us);
    println!("Total time: {}μs", result.stats.total_time_us);
    println!("TAST files: {}", result.typed_files.len());
    println!("Errors: {}", result.errors.len());
    println!("Warnings: {}", result.warnings.len());

    if !result.errors.is_empty() {
        println!("First error: {}", result.errors[0].message);
    }

    if !result.typed_files.is_empty() {
        let file = &result.typed_files[0];
        println!("TAST summary:");
        println!("  - Functions: {}", file.functions.len());
        println!("  - Classes: {}", file.classes.len());
        println!("  - Enums: {}", file.enums.len());
        if let Some(pkg) = &file.metadata.package_name {
            println!("  - Package: {}", pkg);
        }
    }

    println!();
}

/// Validate specific language constructs in isolation
pub fn validate_type_parameters() -> ValidationResult {
    let type_param_haxe = r#"
        // Test various type parameter scenarios
        class GenericClass<T, U extends Comparable<U>> {
            public function new() {}
            public function process<V>(input: V): T {
                return null;
            }
        }

        interface GenericInterface<T> {
            function work<U>(input: T): U;
        }

        enum GenericEnum<T> {
            Some(value: T);
            None;
        }

        abstract GenericAbstract<T>(Array<T>) {
            public function new(arr: Array<T>) {
                this = arr;
            }
        }
    "#;

    let compilation_result = compile_haxe_file("type_param_test.hx", type_param_haxe);
    let mut validation_errors = Vec::new();
    let mut validation_warnings = Vec::new();
    let mut info_score = 100u8;

    validate_compilation_result(
        &compilation_result,
        &mut validation_errors,
        &mut validation_warnings,
        &mut info_score,
    );

    if !compilation_result.typed_files.is_empty() {
        validate_tast_structure(
            &compilation_result.typed_files[0],
            &mut validation_errors,
            &mut validation_warnings,
            &mut info_score,
        );

        // Additional type parameter specific validations
        let file = &compilation_result.typed_files[0];
        for class in &file.classes {
            if !class.type_parameters.is_empty() {
                for tp in &class.type_parameters {
                    if !tp.symbol_id.is_valid() {
                        validation_errors.push(ValidationError {
                            message: format!(
                                "Type parameter '{:?}' has invalid symbol ID",
                                tp.name
                            ),
                            category: ValidationErrorCategory::TypeInformation,
                            severity: ValidationSeverity::High,
                        });
                        info_score = info_score.saturating_sub(10);
                    }
                }
            }
        }
    }

    ValidationResult {
        compilation_result,
        validation_errors,
        validation_warnings,
        info_preservation_score: info_score,
    }
}

/// Validate property handling (getters/setters)
pub fn validate_properties() -> ValidationResult {
    let property_haxe = r#"
        class PropertyTest {
            public var readOnly(default, null): String;
            public var writeOnly(null, default): String;
            public var readWrite(default, default): String;
            public var computed(get, set): String;
            private var _computed: String = "";

            public function new() {
                readOnly = "read only";
                writeOnly = "write only";
                readWrite = "read write";
            }

            function get_computed(): String {
                return _computed;
            }

            function set_computed(value: String): String {
                return _computed = value;
            }
        }
    "#;

    let compilation_result = compile_haxe_file("property_test.hx", property_haxe);
    let mut validation_errors = Vec::new();
    let mut validation_warnings = Vec::new();
    let mut info_score = 100u8;

    validate_compilation_result(
        &compilation_result,
        &mut validation_errors,
        &mut validation_warnings,
        &mut info_score,
    );

    if !compilation_result.typed_files.is_empty() {
        validate_tast_structure(
            &compilation_result.typed_files[0],
            &mut validation_errors,
            &mut validation_warnings,
            &mut info_score,
        );
    }

    ValidationResult {
        compilation_result,
        validation_errors,
        validation_warnings,
        info_preservation_score: info_score,
    }
}

/// Validate metadata preservation
pub fn validate_metadata_preservation() -> ValidationResult {
    let metadata_haxe = r#"
        @:native("CustomName")
        @:final
        class MetadataTest {
            @:isVar
            public var field: String;

            @:overload(function(x: Int): Void {})
            @:overload(function(x: String): Void {})
            public function method(x: Dynamic): Void {}

            @:getter(field)
            public function getField(): String {
                return field;
            }
        }
    "#;

    let compilation_result = compile_haxe_file("metadata_test.hx", metadata_haxe);
    let mut validation_errors = Vec::new();
    let mut validation_warnings = Vec::new();
    let mut info_score = 100u8;

    validate_compilation_result(
        &compilation_result,
        &mut validation_errors,
        &mut validation_warnings,
        &mut info_score,
    );

    if !compilation_result.typed_files.is_empty() {
        validate_tast_structure(
            &compilation_result.typed_files[0],
            &mut validation_errors,
            &mut validation_warnings,
            &mut info_score,
        );
    }

    ValidationResult {
        compilation_result,
        validation_errors,
        validation_warnings,
        info_preservation_score: info_score,
    }
}

/// Run all validation tests and return overall assessment
pub fn run_comprehensive_validation() -> ValidationResult {
    let mut overall_result = validate_comprehensive_pipeline();

    // Run additional specific tests
    let type_param_result = validate_type_parameters();
    let property_result = validate_properties();
    let metadata_result = validate_metadata_preservation();

    // Combine results
    overall_result
        .validation_errors
        .extend(type_param_result.validation_errors);
    overall_result
        .validation_errors
        .extend(property_result.validation_errors);
    overall_result
        .validation_errors
        .extend(metadata_result.validation_errors);

    overall_result
        .validation_warnings
        .extend(type_param_result.validation_warnings);
    overall_result
        .validation_warnings
        .extend(property_result.validation_warnings);
    overall_result
        .validation_warnings
        .extend(metadata_result.validation_warnings);

    // Calculate combined score
    let scores = [
        overall_result.info_preservation_score,
        type_param_result.info_preservation_score,
        property_result.info_preservation_score,
        metadata_result.info_preservation_score,
    ];

    overall_result.info_preservation_score =
        (scores.iter().map(|&s| s as u32).sum::<u32>() / scores.len() as u32).min(100) as u8;

    overall_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_validation() {
        let result = validate_simple_pipeline();
        assert!(result.stats.files_processed > 0);
        assert!(result.stats.parse_time_us > 0);
    }

    #[test]
    fn test_complex_validation() {
        let result = validate_complex_pipeline();
        assert!(result.stats.files_processed > 0);
        assert!(result.stats.parse_time_us > 0);
    }

    #[test]
    fn test_comprehensive_validation() {
        let result = validate_comprehensive_pipeline();

        // Should have processed at least one file
        assert!(result.compilation_result.stats.files_processed > 0);
        assert!(result.compilation_result.stats.parse_time_us > 0);

        // Should have validation results
        assert!(result.info_preservation_score <= 100);

        // Print results for debugging
        print_validation_summary("Comprehensive Test", &result);
    }

    #[test]
    fn test_type_parameter_validation() {
        let result = validate_type_parameters();

        // Should have processed the type parameter test
        assert!(result.compilation_result.stats.files_processed > 0);

        // Print results for debugging
        print_validation_summary("Type Parameter Test", &result);
    }

    #[test]
    fn test_property_validation() {
        let result = validate_properties();

        // Should have processed the property test
        assert!(result.compilation_result.stats.files_processed > 0);

        // Print results for debugging
        print_validation_summary("Property Test", &result);
    }

    #[test]
    fn test_metadata_validation() {
        let result = validate_metadata_preservation();

        // Should have processed the metadata test
        assert!(result.compilation_result.stats.files_processed > 0);

        // Print results for debugging
        print_validation_summary("Metadata Test", &result);
    }

    #[test]
    fn test_validation_error_categories() {
        let result = validate_comprehensive_pipeline();

        // Test that different error categories are properly detected
        let _has_critical = result
            .validation_errors
            .iter()
            .any(|e| e.severity == ValidationSeverity::Critical);
        let _has_high = result
            .validation_errors
            .iter()
            .any(|e| e.severity == ValidationSeverity::High);

        // Should have some form of validation feedback
        assert!(
            result.validation_errors.len() > 0
                || result.validation_warnings.len() > 0
                || result.info_preservation_score < 100
        );
    }

    #[test]
    fn test_information_preservation_scoring() {
        let result = validate_comprehensive_pipeline();

        // Score should be between 0 and 100
        assert!(result.info_preservation_score <= 100);

        // If there are critical errors, score should be low
        let has_critical = result
            .validation_errors
            .iter()
            .any(|e| e.severity == ValidationSeverity::Critical);
        if has_critical {
            assert!(result.info_preservation_score < 90);
        }

        // If there are high priority errors, score should be affected
        let has_high = result
            .validation_errors
            .iter()
            .any(|e| e.severity == ValidationSeverity::High);
        if has_high {
            assert!(result.info_preservation_score < 95);
        }
    }

    #[test]
    fn test_overall_validation_suite() {
        let result = run_comprehensive_validation();

        // Should have processed multiple tests
        assert!(result.compilation_result.stats.files_processed > 0);

        // Should have combined validation results
        assert!(result.info_preservation_score <= 100);

        // Print comprehensive results
        print_validation_summary("Overall Validation Suite", &result);

        // Assert that the pipeline is working at some level
        assert!(
            result.info_preservation_score > 0,
            "Pipeline should preserve at least some information"
        );
    }
}
