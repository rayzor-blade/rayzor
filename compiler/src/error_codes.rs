//! Global Error Code Registry for the Haxe Compiler
//!
//! This module defines a unified error numbering system across all compilation phases.
//! Error codes are organized by range to avoid conflicts between different modules.
//!
//! # Error Code Ranges
//!
//! - E0001-E0999: Parser and syntax errors
//! - E1000-E1999: Type system and type checking errors
//! - E2000-E2999: Symbol resolution and scope errors
//! - E3000-E3999: Generic and constraint errors
//! - E4000-E4999: Import and module system errors
//! - E5000-E5999: Code generation and optimization errors
//! - E6000-E6999: Metadata and annotation errors
//! - E7000-E7999: Macro and compile-time errors
//! - E8000-E8999: Platform and target-specific errors
//! - E9000-E9999: Internal compiler errors and assertions
//!
//! # Subcategory Organization
//!
//! Within each range, the first digit after the thousands indicates subcategory:
//! - 0: General/basic errors
//! - 1: Syntax and structure errors
//! - 2: Declaration and definition errors
//! - 3: Expression and statement errors
//! - 4: Control flow errors
//! - 5: Access and visibility errors
//! - 6: Literal and constant errors
//! - 7: Pattern matching errors
//! - 8: Conversion and casting errors
//! - 9: Advanced/complex errors

use std::collections::BTreeMap;
use std::fmt;

/// Error code struct containing the numeric code and human-readable description
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorCode {
    /// The numeric error code (e.g., 1001)
    pub code: u16,
    /// Human-readable error category
    pub category: &'static str,
    /// Brief description of what this error means
    pub description: &'static str,
    /// Optional help text with suggestions for fixing the error
    pub help: Option<&'static str>,
}

impl ErrorCode {
    /// Create a new error code
    pub const fn new(
        code: u16,
        category: &'static str,
        description: &'static str,
        help: Option<&'static str>,
    ) -> Self {
        Self {
            code,
            category,
            description,
            help,
        }
    }

    /// Format the error code as "E{code:04}" (e.g., "E1001")
    pub fn format_code(&self) -> String {
        format!("E{:04}", self.code)
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}]: {}",
            self.format_code(),
            self.category,
            self.description
        )
    }
}

/// Registry containing all defined error codes
pub struct ErrorCodeRegistry {
    codes: BTreeMap<u16, ErrorCode>,
}

impl ErrorCodeRegistry {
    /// Create a new registry with all predefined error codes
    pub fn new() -> Self {
        let mut registry = Self {
            codes: BTreeMap::new(),
        };
        registry.register_all_codes();
        registry
    }

    /// Get an error code by its numeric value
    pub fn get(&self, code: u16) -> Option<&ErrorCode> {
        self.codes.get(&code)
    }

    /// Get an error code by its formatted string (e.g., "E1001")
    pub fn get_by_string(&self, code_str: &str) -> Option<&ErrorCode> {
        if let Some(stripped) = code_str.strip_prefix('E') {
            if let Ok(code_num) = stripped.parse::<u16>() {
                return self.get(code_num);
            }
        }
        None
    }

    /// Register a new error code
    fn register(&mut self, error_code: ErrorCode) {
        self.codes.insert(error_code.code, error_code);
    }

    /// Register all predefined error codes
    fn register_all_codes(&mut self) {
        // ===== PARSER ERRORS (E0001-E0999) =====

        // General parser errors (E0001-E0099)
        self.register(ErrorCode::new(
            1,
            "Parser",
            "Unexpected token",
            Some("Check for missing punctuation or keywords"),
        ));
        self.register(ErrorCode::new(
            2,
            "Parser",
            "Missing closing delimiter",
            Some("Ensure all brackets, braces, and parentheses are properly closed"),
        ));
        self.register(ErrorCode::new(
            3,
            "Parser",
            "Expected token",
            Some("Review the syntax requirements for this language construct"),
        ));
        self.register(ErrorCode::new(
            4,
            "Parser",
            "Invalid syntax",
            Some("Check the language documentation for correct syntax"),
        ));
        self.register(ErrorCode::new(
            5,
            "Parser",
            "Unexpected end of file",
            Some("The file ended unexpectedly; check for incomplete statements"),
        ));

        // Declaration errors (E0100-E0199)
        self.register(ErrorCode::new(
            101,
            "Parser",
            "Invalid class declaration",
            Some("Class declarations must follow 'class ClassName' syntax"),
        ));
        self.register(ErrorCode::new(
            102,
            "Parser",
            "Invalid interface declaration",
            Some("Interface declarations must follow 'interface InterfaceName' syntax"),
        ));
        self.register(ErrorCode::new(
            103,
            "Parser",
            "Invalid function declaration",
            Some("Function declarations must include 'function' keyword and name"),
        ));
        self.register(ErrorCode::new(
            104,
            "Parser",
            "Invalid variable declaration",
            Some("Variable declarations must specify a name and optionally a type"),
        ));
        self.register(ErrorCode::new(
            105,
            "Parser",
            "Invalid package declaration",
            Some("Package declarations must be at the top of the file"),
        ));
        self.register(ErrorCode::new(
            106,
            "Parser",
            "Invalid import declaration",
            Some("Import statements must specify a valid module path"),
        ));

        // Expression errors (E0200-E0299)
        self.register(ErrorCode::new(
            201,
            "Parser",
            "Invalid expression",
            Some("Check expression syntax and operator precedence"),
        ));
        self.register(ErrorCode::new(
            202,
            "Parser",
            "Malformed function call",
            Some("Function calls require parentheses around arguments"),
        ));
        self.register(ErrorCode::new(
            203,
            "Parser",
            "Invalid array literal",
            Some("Array literals must be enclosed in brackets [...]"),
        ));
        self.register(ErrorCode::new(
            204,
            "Parser",
            "Invalid object literal",
            Some("Object literals must be enclosed in braces {...}"),
        ));

        // ===== TYPE SYSTEM ERRORS (E1000-E1999) =====

        // Basic type errors (E1000-E1099)
        self.register(ErrorCode::new(
            1001,
            "Type",
            "Type mismatch",
            Some("Ensure the assigned value matches the expected type"),
        ));
        self.register(ErrorCode::new(
            1002,
            "Type",
            "Undefined type",
            Some("Check that the type is imported or defined in scope"),
        ));
        self.register(ErrorCode::new(
            1003,
            "Type",
            "Invalid type annotation",
            Some("Type annotations must use valid type syntax"),
        ));
        self.register(ErrorCode::new(
            1004,
            "Type",
            "Circular type dependency",
            Some("Avoid creating circular references between types"),
        ));
        self.register(ErrorCode::new(
            1005,
            "Type",
            "Type inference failed",
            Some("Provide explicit type annotations where inference is ambiguous"),
        ));

        // Function type errors (E1100-E1199)
        self.register(ErrorCode::new(
            1101,
            "Type",
            "Function arity mismatch",
            Some("Ensure the correct number of arguments are provided"),
        ));
        self.register(ErrorCode::new(
            1102,
            "Type",
            "Invalid return type",
            Some("Function return value must match declared return type"),
        ));
        self.register(ErrorCode::new(
            1103,
            "Type",
            "Parameter type mismatch",
            Some("Function arguments must match parameter types"),
        ));

        // Object and field type errors (E1200-E1299)
        self.register(ErrorCode::new(
            1201,
            "Type",
            "Undefined field",
            Some("Check that the field exists on the object type"),
        ));
        self.register(ErrorCode::new(
            1202,
            "Type",
            "Field access on non-object",
            Some("Field access is only valid on object types"),
        ));
        self.register(ErrorCode::new(
            1203,
            "Type",
            "Field type mismatch",
            Some("Assigned value must match the field's declared type"),
        ));
        self.register(ErrorCode::new(
            1204,
            "Type",
            "Private field access",
            Some("Private fields can only be accessed within the defining class"),
        ));

        // Array and indexing errors (E1300-E1399)
        self.register(ErrorCode::new(
            1301,
            "Type",
            "Invalid array index",
            Some("Array indices must be integers"),
        ));
        self.register(ErrorCode::new(
            1302,
            "Type",
            "Index on non-indexable type",
            Some("Only arrays, strings, and maps support indexing"),
        ));
        self.register(ErrorCode::new(
            1303,
            "Type",
            "Array element type mismatch",
            Some("All array elements must be compatible with the array type"),
        ));

        // ===== SYMBOL RESOLUTION ERRORS (E2000-E2999) =====

        // Basic symbol errors (E2000-E2099)
        self.register(ErrorCode::new(
            2001,
            "Symbol",
            "Undefined symbol",
            Some("Check that the identifier is declared and in scope"),
        ));
        self.register(ErrorCode::new(
            2002,
            "Symbol",
            "Symbol already defined",
            Some("Choose a different name or check for duplicate declarations"),
        ));
        self.register(ErrorCode::new(
            2003,
            "Symbol",
            "Symbol not in scope",
            Some("Ensure the symbol is accessible from the current context"),
        ));
        self.register(ErrorCode::new(
            2004,
            "Symbol",
            "Ambiguous symbol reference",
            Some("Use fully qualified names to resolve ambiguity"),
        ));

        // Scope and visibility errors (E2100-E2199)
        self.register(ErrorCode::new(
            2101,
            "Symbol",
            "Private symbol access",
            Some("Private symbols can only be accessed within their defining scope"),
        ));
        self.register(ErrorCode::new(
            2102,
            "Symbol",
            "Protected symbol access",
            Some("Protected symbols are only accessible to subclasses"),
        ));

        // ===== GENERIC AND CONSTRAINT ERRORS (E3000-E3999) =====

        // Generic parameter errors (E3000-E3099)
        self.register(ErrorCode::new(
            3001,
            "Generic",
            "Generic parameter count mismatch",
            Some("Provide the correct number of type parameters"),
        ));
        self.register(ErrorCode::new(
            3002,
            "Generic",
            "Invalid generic instantiation",
            Some("Check that type arguments satisfy the generic constraints"),
        ));
        self.register(ErrorCode::new(
            3003,
            "Generic",
            "Unconstrained generic parameter",
            Some("Consider adding constraints to limit the generic parameter"),
        ));

        // Constraint errors (E3100-E3199)
        self.register(ErrorCode::new(
            3101,
            "Generic",
            "Constraint violation",
            Some("The type does not satisfy the required constraints"),
        ));
        self.register(ErrorCode::new(
            3102,
            "Generic",
            "Recursive constraint",
            Some("Avoid creating circular constraint dependencies"),
        ));
        self.register(ErrorCode::new(
            3103,
            "Generic",
            "Constraint resolution failed",
            Some("Unable to determine if constraints are satisfied"),
        ));

        // ===== IMPORT AND MODULE ERRORS (E4000-E4999) =====

        // Import errors (E4000-E4099)
        self.register(ErrorCode::new(
            4001,
            "Import",
            "Module not found",
            Some("Check that the module path is correct and the file exists"),
        ));
        self.register(ErrorCode::new(
            4002,
            "Import",
            "Circular import dependency",
            Some("Restructure code to avoid circular module dependencies"),
        ));
        self.register(ErrorCode::new(
            4003,
            "Import",
            "Invalid import path",
            Some("Import paths must use valid module naming conventions"),
        ));

        // ===== INTERNAL COMPILER ERRORS (E9000-E9999) =====

        // Internal errors (E9000-E9099)
        self.register(ErrorCode::new(
            9001,
            "Internal",
            "Compiler assertion failed",
            Some("This is an internal compiler error; please report it"),
        ));
        self.register(ErrorCode::new(
            9002,
            "Internal",
            "Unexpected compiler state",
            Some("This is an internal compiler error; please report it"),
        ));
        self.register(ErrorCode::new(
            9999,
            "Internal",
            "Unknown error",
            Some("An unexpected error occurred; please report it with context"),
        ));
    }

    /// Get all error codes in a specific range
    pub fn get_range(&self, start: u16, end: u16) -> Vec<&ErrorCode> {
        let mut codes: Vec<&ErrorCode> = self
            .codes
            .values()
            .filter(|code| code.code >= start && code.code <= end)
            .collect();
        codes.sort_by_key(|code| code.code);
        codes
    }

    /// Get all parser error codes (E0001-E0999)
    pub fn get_parser_errors(&self) -> Vec<&ErrorCode> {
        self.get_range(1, 999)
    }

    /// Get all type system error codes (E1000-E1999)
    pub fn get_type_errors(&self) -> Vec<&ErrorCode> {
        self.get_range(1000, 1999)
    }

    /// Get all symbol resolution error codes (E2000-E2999)
    pub fn get_symbol_errors(&self) -> Vec<&ErrorCode> {
        self.get_range(2000, 2999)
    }

    /// Get all generic/constraint error codes (E3000-E3999)
    pub fn get_generic_errors(&self) -> Vec<&ErrorCode> {
        self.get_range(3000, 3999)
    }

    /// Validate that an error code is registered
    pub fn is_valid_code(&self, code: u16) -> bool {
        self.codes.contains_key(&code)
    }
}

/// Global error code registry instance
static REGISTRY: std::sync::OnceLock<ErrorCodeRegistry> = std::sync::OnceLock::new();

/// Get the global error code registry
pub fn error_registry() -> &'static ErrorCodeRegistry {
    REGISTRY.get_or_init(|| ErrorCodeRegistry::new())
}

/// Helper function to get error code by number
pub fn get_error_code(code: u16) -> Option<&'static ErrorCode> {
    error_registry().get(code)
}

/// Helper function to format error code string (e.g., 1001 -> "E1001")
pub fn format_error_code(code: u16) -> String {
    format!("E{:04}", code)
}

/// Helper function to parse error code from string (e.g., "E1001" -> Some(1001))
pub fn parse_error_code(code_str: &str) -> Option<u16> {
    if let Some(stripped) = code_str.strip_prefix('E') {
        stripped.parse::<u16>().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_creation() {
        let code = ErrorCode::new(1001, "Type", "Type mismatch", Some("Check types"));
        assert_eq!(code.code, 1001);
        assert_eq!(code.category, "Type");
        assert_eq!(code.description, "Type mismatch");
        assert_eq!(code.help, Some("Check types"));
        assert_eq!(code.format_code(), "E1001");
    }

    #[test]
    fn test_registry_functionality() {
        let registry = ErrorCodeRegistry::new();

        // Test getting a known error code
        let type_mismatch = registry.get(1001).unwrap();
        assert_eq!(type_mismatch.description, "Type mismatch");

        // Test getting by string
        let by_string = registry.get_by_string("E1001").unwrap();
        assert_eq!(by_string.code, 1001);

        // Test invalid codes
        assert!(registry.get(65535).is_none());
        assert!(registry.get_by_string("INVALID").is_none());
    }

    #[test]
    fn test_error_code_ranges() {
        let registry = ErrorCodeRegistry::new();

        // Test parser errors range
        let parser_errors = registry.get_parser_errors();
        assert!(!parser_errors.is_empty());
        assert!(parser_errors.iter().all(|e| e.code >= 1 && e.code <= 999));

        // Test type errors range
        let type_errors = registry.get_type_errors();
        assert!(!type_errors.is_empty());
        assert!(type_errors.iter().all(|e| e.code >= 1000 && e.code <= 1999));

        // Test symbol errors range
        let symbol_errors = registry.get_symbol_errors();
        assert!(!symbol_errors.is_empty());
        assert!(symbol_errors
            .iter()
            .all(|e| e.code >= 2000 && e.code <= 2999));
    }

    #[test]
    fn test_global_registry() {
        let reg1 = error_registry();
        let reg2 = error_registry();

        // Should be the same instance
        assert!(std::ptr::eq(reg1, reg2));

        // Should contain expected codes
        assert!(reg1.is_valid_code(1001));
        assert!(reg1.is_valid_code(2001));
        assert!(!reg1.is_valid_code(65000));
    }

    #[test]
    fn test_helper_functions() {
        assert_eq!(format_error_code(1001), "E1001");
        assert_eq!(format_error_code(42), "E0042");

        assert_eq!(parse_error_code("E1001"), Some(1001));
        assert_eq!(parse_error_code("E0042"), Some(42));
        assert_eq!(parse_error_code("1001"), None);
        assert_eq!(parse_error_code("INVALID"), None);

        let code = get_error_code(1001).unwrap();
        assert_eq!(code.description, "Type mismatch");
    }
}
