use super::bytecode::{BytecodeCompiler, Chunk, CompiledClassInfo};
use super::errors::{MacroDiagnostic, MacroError};
use super::value::MacroParam;
use crate::tast::SourceLocation;
use parser::{ClassDecl, ClassField, ClassFieldKind, Expr, HaxeFile, Modifier, TypeDeclaration};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Default maximum macro expansion depth
const DEFAULT_MAX_DEPTH: usize = 256;

/// A registered macro function definition
#[derive(Debug, Clone)]
pub struct MacroDefinition {
    /// Simple function name
    pub name: String,
    /// Fully qualified name (package.Class.function)
    pub qualified_name: String,
    /// Parameters
    pub params: Vec<MacroParam>,
    /// The function body AST (Arc for O(1) clone on each macro invocation)
    pub body: Arc<Expr>,
    /// Whether this is a @:build macro
    pub is_build_macro: bool,
    /// Source file where defined
    pub source_file: String,
    /// Source location of the definition
    pub location: SourceLocation,
}

/// Entry for a pending @:build macro application
#[derive(Debug, Clone)]
pub struct BuildMacroEntry {
    /// Qualified name of the macro to call
    pub macro_name: String,
    /// Arguments from the @:build metadata
    pub args: Vec<Expr>,
    /// The class/interface this macro applies to
    pub target_class: String,
    /// Source location of the @:build metadata
    pub location: SourceLocation,
}

/// Registry for macro definitions, tracking what macros are available
/// and managing expansion state.
#[derive(Debug, Clone)]
pub struct MacroRegistry {
    /// Macro function definitions indexed by qualified name
    macros: BTreeMap<String, MacroDefinition>,
    /// Build macros pending application
    build_macros: Vec<BuildMacroEntry>,
    /// Current expansion depth counter
    expansion_depth: usize,
    /// Maximum allowed expansion depth
    max_depth: usize,
    /// Set of macros currently being expanded (for circular dependency detection)
    expanding: Vec<String>,
    /// Diagnostics accumulated during scanning
    diagnostics: Vec<MacroDiagnostic>,
    /// Compiled bytecode chunks for macro functions (keyed by qualified name).
    /// Only populated when RAYZOR_MACRO_VM=1 is set.
    compiled: BTreeMap<String, Arc<Chunk>>,
    /// Compiled class data for VM class dispatch (keyed by class name).
    compiled_classes: BTreeMap<String, CompiledClassInfo>,
}

impl MacroRegistry {
    pub fn new() -> Self {
        Self {
            macros: BTreeMap::new(),
            build_macros: Vec::new(),
            expansion_depth: 0,
            max_depth: DEFAULT_MAX_DEPTH,
            expanding: Vec::new(),
            diagnostics: Vec::new(),
            compiled: BTreeMap::new(),
            compiled_classes: BTreeMap::new(),
        }
    }

    /// Set the maximum expansion depth
    pub fn set_max_depth(&mut self, max_depth: usize) {
        self.max_depth = max_depth;
    }

    /// Scan a parsed file and register all macro definitions found
    pub fn scan_and_register(
        &mut self,
        file: &HaxeFile,
        source_file: &str,
    ) -> Result<(), MacroError> {
        let package_prefix = file
            .package
            .as_ref()
            .map(|p| p.path.join("."))
            .unwrap_or_default();

        for decl in &file.declarations {
            match decl {
                TypeDeclaration::Class(class) => {
                    self.scan_class(class, &package_prefix, source_file)?;
                }
                TypeDeclaration::Interface(_) => {
                    // Interfaces can't have macro functions
                }
                TypeDeclaration::Enum(_) => {
                    // Enums can't have macro functions
                }
                TypeDeclaration::Typedef(_) => {
                    // Typedefs can't have macro functions
                }
                TypeDeclaration::Abstract(_) => {
                    // Abstracts could have macro functions; scan similar to class
                }
                TypeDeclaration::Conditional(_) => {
                    // Skip conditional compilation blocks for now
                }
            }
        }

        Ok(())
    }

    /// Scan a class declaration for macro functions and @:build metadata
    fn scan_class(
        &mut self,
        class: &ClassDecl,
        package_prefix: &str,
        source_file: &str,
    ) -> Result<(), MacroError> {
        let class_qualified = if package_prefix.is_empty() {
            class.name.clone()
        } else {
            format!("{}.{}", package_prefix, class.name)
        };

        // Check for @:build metadata on the class
        for meta in &class.meta {
            if meta.name == "build" || meta.name == ":build" {
                self.build_macros.push(BuildMacroEntry {
                    macro_name: self.extract_build_macro_name(&meta.params),
                    args: meta.params.clone(),
                    target_class: class_qualified.clone(),
                    location: SourceLocation::new(0, 0, 0, meta.span.start as u32),
                });
            }
        }

        // Scan fields for macro functions
        for field in &class.fields {
            if field.modifiers.contains(&Modifier::Macro) {
                self.register_macro_field(field, &class_qualified, source_file)?;
            }
        }

        Ok(())
    }

    /// Register a macro function from a class field
    fn register_macro_field(
        &mut self,
        field: &ClassField,
        class_qualified: &str,
        source_file: &str,
    ) -> Result<(), MacroError> {
        if let ClassFieldKind::Function(func) = &field.kind {
            let qualified_name = format!("{}.{}", class_qualified, func.name);

            let params: Vec<MacroParam> = func
                .params
                .iter()
                .map(|p| MacroParam::from_function_param(p))
                .collect();

            let body = match &func.body {
                Some(body) => Arc::new(body.as_ref().clone()),
                None => {
                    return Err(MacroError::InvalidDefinition {
                        message: format!("macro function '{}' must have a body", qualified_name),
                        location: SourceLocation::new(0, 0, 0, field.span.start as u32),
                    });
                }
            };

            let definition = MacroDefinition {
                name: func.name.clone(),
                qualified_name: qualified_name.clone(),
                params,
                body,
                is_build_macro: false,
                source_file: source_file.to_string(),
                location: SourceLocation::new(0, 0, 0, field.span.start as u32),
            };

            // Bytecode compilation is deferred — the tiering scheduler in the
            // interpreter will compile hot macros on-demand after they cross the
            // call-count threshold (morsel-parallelism-inspired scheduling).

            self.macros.insert(qualified_name, definition);
        }

        Ok(())
    }

    /// Extract the macro name from @:build metadata arguments
    fn extract_build_macro_name(&self, args: &[Expr]) -> String {
        if let Some(first) = args.first() {
            // Try to extract a qualified name from the expression
            self.expr_to_qualified_name(first)
        } else {
            String::new()
        }
    }

    /// Convert an expression to a qualified name string (for macro references)
    fn expr_to_qualified_name(&self, expr: &Expr) -> String {
        use parser::ExprKind;
        match &expr.kind {
            ExprKind::Ident(name) => name.clone(),
            ExprKind::Field {
                expr: base, field, ..
            } => {
                let base_name = self.expr_to_qualified_name(base);
                format!("{}.{}", base_name, field)
            }
            ExprKind::Call { expr: callee, .. } => {
                // @:build(MacroTools.buildFields()) — extract the function path
                self.expr_to_qualified_name(callee)
            }
            _ => format!("{:?}", expr.kind),
        }
    }

    /// Look up a macro by its qualified name
    pub fn get_macro(&self, qualified_name: &str) -> Option<&MacroDefinition> {
        self.macros.get(qualified_name)
    }

    /// Look up a compiled bytecode chunk by qualified name.
    pub fn get_compiled(&self, qualified_name: &str) -> Option<Arc<Chunk>> {
        self.compiled.get(qualified_name).cloned()
    }

    /// Insert a compiled bytecode chunk (used by the tiering scheduler).
    pub fn insert_compiled(&mut self, qualified_name: String, chunk: Arc<Chunk>) {
        self.compiled.insert(qualified_name, chunk);
    }

    /// Look up a macro by simple name (searches all registered macros)
    pub fn find_macro_by_name(&self, name: &str) -> Option<&MacroDefinition> {
        // First try exact match
        if let Some(def) = self.macros.get(name) {
            return Some(def);
        }
        // Then try matching by simple name suffix
        self.macros.values().find(|def| def.name == name)
    }

    /// Get all pending build macros
    pub fn build_macros(&self) -> &[BuildMacroEntry] {
        &self.build_macros
    }

    /// Get all registered macro definitions
    pub fn all_macros(&self) -> impl Iterator<Item = &MacroDefinition> {
        self.macros.values()
    }

    /// Number of registered macros
    pub fn macro_count(&self) -> usize {
        self.macros.len()
    }

    /// Enter a macro expansion (tracks depth and circular dependencies)
    pub fn enter_expansion(&mut self, macro_name: &str) -> Result<(), MacroError> {
        if self.expansion_depth >= self.max_depth {
            return Err(MacroError::RecursionLimitExceeded {
                macro_name: macro_name.to_string(),
                depth: self.expansion_depth + 1,
                max_depth: self.max_depth,
                location: SourceLocation::unknown(),
            });
        }

        if self.expanding.contains(&macro_name.to_string()) {
            let mut chain = self.expanding.clone();
            chain.push(macro_name.to_string());
            return Err(MacroError::CircularDependency {
                chain,
                location: SourceLocation::unknown(),
            });
        }

        self.expansion_depth += 1;
        self.expanding.push(macro_name.to_string());
        Ok(())
    }

    /// Exit a macro expansion
    pub fn exit_expansion(&mut self, macro_name: &str) {
        if self.expansion_depth > 0 {
            self.expansion_depth -= 1;
        }
        self.expanding.retain(|n| n != macro_name);
    }

    /// Get the current expansion depth
    pub fn expansion_depth(&self) -> usize {
        self.expansion_depth
    }

    /// Take accumulated diagnostics
    pub fn take_diagnostics(&mut self) -> Vec<MacroDiagnostic> {
        std::mem::take(&mut self.diagnostics)
    }

    /// Check if a name refers to a registered macro
    pub fn is_macro(&self, name: &str) -> bool {
        self.macros.contains_key(name) || self.macros.values().any(|def| def.name == name)
    }

    /// Compile class constructors/methods from a ClassRegistry for VM dispatch.
    /// Called when RAYZOR_MACRO_VM=1 is set.
    pub fn compile_classes(&mut self, class_registry: &super::class_registry::ClassRegistry) {
        use super::class_registry::ClassInfo;
        use super::value::MacroValue;

        // Iterate all classes in the registry
        for class_name in class_registry.iter_class_names() {
            if let Some(class_info) = class_registry.find_class(class_name) {
                let mut compiled = CompiledClassInfo {
                    constructor: None,
                    instance_methods: std::collections::BTreeMap::new(),
                    static_methods: std::collections::BTreeMap::new(),
                    instance_vars: Vec::new(),
                };

                // Compile instance var defaults
                for var in &class_info.instance_vars {
                    let default_val = if let Some(ref init) = var.init_expr {
                        // Try to evaluate simple literal defaults
                        Self::eval_simple_literal(init)
                    } else {
                        MacroValue::Null
                    };
                    compiled.instance_vars.push((var.name.clone(), default_val));
                }

                // Compile constructor
                if let Some(ref ctor) = class_info.constructor {
                    if let Ok(chunk) = BytecodeCompiler::compile_method(
                        &format!("{}.new", class_info.name),
                        &ctor.params,
                        &ctor.body,
                        true, // is_constructor
                    ) {
                        compiled.constructor = Some(Arc::new(chunk));
                    }
                }

                // Compile instance methods
                for (method_name, method_info) in &class_info.instance_methods {
                    if let Ok(chunk) = BytecodeCompiler::compile_method(
                        &format!("{}.{}", class_info.name, method_name),
                        &method_info.params,
                        &method_info.body,
                        false,
                    ) {
                        compiled
                            .instance_methods
                            .insert(method_name.clone(), Arc::new(chunk));
                    }
                }

                // Compile static methods
                for (method_name, method_info) in &class_info.static_methods {
                    if let Ok(chunk) = BytecodeCompiler::compile_method(
                        &format!("{}.{}", class_info.name, method_name),
                        &method_info.params,
                        &method_info.body,
                        false,
                    ) {
                        compiled
                            .static_methods
                            .insert(method_name.clone(), Arc::new(chunk));
                    }
                }

                // Store under both simple and qualified names
                self.compiled_classes
                    .insert(class_info.name.clone(), compiled);
            }
        }
    }

    /// Evaluate a simple literal expression to a MacroValue.
    fn eval_simple_literal(expr: &Expr) -> super::value::MacroValue {
        use super::value::MacroValue;
        match &expr.kind {
            parser::ExprKind::Int(i) => MacroValue::Int(*i),
            parser::ExprKind::Float(f) => MacroValue::Float(*f),
            parser::ExprKind::String(s) => MacroValue::from_str(s),
            parser::ExprKind::Bool(b) => MacroValue::Bool(*b),
            parser::ExprKind::Null => MacroValue::Null,
            // Negative literal: -N
            parser::ExprKind::Unary {
                op: parser::UnaryOp::Neg,
                expr,
            } => match &expr.kind {
                parser::ExprKind::Int(i) => MacroValue::Int(-i),
                parser::ExprKind::Float(f) => MacroValue::Float(-f),
                _ => MacroValue::Null,
            },
            _ => MacroValue::Null,
        }
    }

    /// Get compiled class data for VM dispatch.
    pub fn get_compiled_classes(&self) -> &BTreeMap<String, CompiledClassInfo> {
        &self.compiled_classes
    }
}

impl Default for MacroRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_file(source: &str) -> HaxeFile {
        parser::parse_haxe_file("test.hx", source, false).expect("parse should succeed")
    }

    #[test]
    fn test_scan_empty_file() {
        let mut registry = MacroRegistry::new();
        let file = make_test_file("class Empty {}");
        registry
            .scan_and_register(&file, "test.hx")
            .expect("scan should succeed");
        assert_eq!(registry.macro_count(), 0);
    }

    #[test]
    fn test_scan_macro_function() {
        let mut registry = MacroRegistry::new();
        let source = r#"
package test;

class MacroTools {
    public macro function buildFields():Array<Dynamic> {
        return [];
    }

    public function normalFunc():Void {}
}
"#;
        let file = make_test_file(source);
        registry
            .scan_and_register(&file, "test.hx")
            .expect("scan should succeed");
        assert_eq!(registry.macro_count(), 1);

        let macro_def = registry
            .get_macro("test.MacroTools.buildFields")
            .expect("macro should be registered");
        assert_eq!(macro_def.name, "buildFields");
        assert_eq!(macro_def.qualified_name, "test.MacroTools.buildFields");
    }

    #[test]
    fn test_find_macro_by_name() {
        let mut registry = MacroRegistry::new();
        let source = r#"
class Tools {
    macro function myMacro():Void {
        trace("hello");
    }
}
"#;
        let file = make_test_file(source);
        registry
            .scan_and_register(&file, "test.hx")
            .expect("scan should succeed");

        assert!(registry.find_macro_by_name("myMacro").is_some());
        assert!(registry.find_macro_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_expansion_depth_tracking() {
        let mut registry = MacroRegistry::new();
        registry.set_max_depth(3);

        assert!(registry.enter_expansion("a").is_ok());
        assert_eq!(registry.expansion_depth(), 1);

        assert!(registry.enter_expansion("b").is_ok());
        assert_eq!(registry.expansion_depth(), 2);

        assert!(registry.enter_expansion("c").is_ok());
        assert_eq!(registry.expansion_depth(), 3);

        // Should exceed limit
        assert!(registry.enter_expansion("d").is_err());

        registry.exit_expansion("c");
        registry.exit_expansion("b");
        registry.exit_expansion("a");
        assert_eq!(registry.expansion_depth(), 0);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut registry = MacroRegistry::new();

        assert!(registry.enter_expansion("a").is_ok());
        assert!(registry.enter_expansion("b").is_ok());

        // "a" is already being expanded
        let err = registry.enter_expansion("a").unwrap_err();
        match err {
            MacroError::CircularDependency { chain, .. } => {
                assert_eq!(chain, vec!["a", "b", "a"]);
            }
            _ => panic!("expected CircularDependency error"),
        }
    }

    #[test]
    fn test_is_macro() {
        let mut registry = MacroRegistry::new();
        let source = r#"
package pkg;

class M {
    macro function doStuff():Void {
        return;
    }
}
"#;
        let file = make_test_file(source);
        registry.scan_and_register(&file, "test.hx").unwrap();

        assert!(registry.is_macro("pkg.M.doStuff"));
        assert!(registry.is_macro("doStuff"));
        assert!(!registry.is_macro("nonexistent"));
    }

    // ===== Edge case tests (Phase 7) =====

    #[test]
    fn test_depth_limit_error_message() {
        let mut registry = MacroRegistry::new();
        registry.set_max_depth(2);

        registry.enter_expansion("a").unwrap();
        registry.enter_expansion("b").unwrap();
        let err = registry.enter_expansion("c").unwrap_err();
        match err {
            MacroError::RecursionLimitExceeded {
                macro_name,
                depth,
                max_depth,
                ..
            } => {
                assert_eq!(macro_name, "c");
                assert_eq!(depth, 3);
                assert_eq!(max_depth, 2);
            }
            _ => panic!("expected RecursionLimitExceeded"),
        }
    }

    #[test]
    fn test_enter_exit_balanced() {
        let mut registry = MacroRegistry::new();
        registry.enter_expansion("a").unwrap();
        registry.enter_expansion("b").unwrap();
        registry.exit_expansion("b");
        assert_eq!(registry.expansion_depth(), 1);
        registry.exit_expansion("a");
        assert_eq!(registry.expansion_depth(), 0);

        // Re-enter should work after full exit
        assert!(registry.enter_expansion("a").is_ok());
        assert_eq!(registry.expansion_depth(), 1);
    }

    #[test]
    fn test_scan_multiple_macros() {
        let mut registry = MacroRegistry::new();
        let source = r#"
class MacroUtils {
    macro static function first() { return 1; }
    macro static function second() { return 2; }
    static function notMacro() { return 3; }
}
"#;
        let file = make_test_file(source);
        registry.scan_and_register(&file, "test.hx").unwrap();

        assert!(registry.is_macro("first") || registry.is_macro("MacroUtils.first"));
        assert!(registry.is_macro("second") || registry.is_macro("MacroUtils.second"));
        assert!(!registry.is_macro("notMacro"));
    }

    #[test]
    fn test_scan_empty_class() {
        let mut registry = MacroRegistry::new();
        let source = "class Empty {}";
        let file = make_test_file(source);
        registry.scan_and_register(&file, "test.hx").unwrap();
        assert_eq!(registry.expansion_depth(), 0);
    }
}
