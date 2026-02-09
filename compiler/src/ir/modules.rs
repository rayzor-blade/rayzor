//! HIR Modules
//!
//! This module defines the top-level compilation unit representation in HIR,
//! including modules, global variables, type definitions, and function declarations.

use super::{IrFunction, IrFunctionId, IrId, IrSourceLocation, IrType, IrValue, Linkage};
use crate::tast::{SymbolId, TypeId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// HIR module - represents a compilation unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrModule {
    /// Module name
    pub name: String,

    /// Source file path
    pub source_file: String,

    /// Functions defined in this module (BTreeMap for deterministic iteration)
    pub functions: BTreeMap<IrFunctionId, IrFunction>,

    /// Global variables
    pub globals: HashMap<IrGlobalId, IrGlobal>,

    /// Type definitions
    pub types: HashMap<IrTypeDefId, IrTypeDef>,

    /// String constants pool
    pub string_pool: StringPool,

    /// External function declarations (BTreeMap for deterministic iteration)
    pub extern_functions: BTreeMap<IrFunctionId, IrExternFunction>,

    /// Module metadata
    pub metadata: ModuleMetadata,

    /// Next available IDs (pub for MIR builder)
    pub next_function_id: u32,
    pub next_global_id: u32,
    pub next_typedef_id: u32,

    /// Symbol-to-register mapping for memory safety validation
    /// Maps TAST SymbolId to MIR IrId for ownership tracking
    pub symbol_to_register: HashMap<SymbolId, IrId>,

    /// Register-to-symbol reverse mapping
    pub register_to_symbol: HashMap<IrId, SymbolId>,
}

/// Global variable identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IrGlobalId(pub u32);

impl std::fmt::Display for IrGlobalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// Type definition identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IrTypeDefId(pub u32);

impl std::fmt::Display for IrTypeDefId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ty{}", self.0)
    }
}

/// Global variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrGlobal {
    /// Global identifier
    pub id: IrGlobalId,

    /// Variable name
    pub name: String,

    /// Original TAST symbol
    pub symbol_id: SymbolId,

    /// Variable type
    pub ty: IrType,

    /// Initial value (if any)
    pub initializer: Option<IrValue>,

    /// Whether this is mutable
    pub mutable: bool,

    /// Linkage type
    pub linkage: Linkage,

    /// Alignment requirement
    pub alignment: Option<u32>,

    /// Source location
    pub source_location: IrSourceLocation,
}

/// Type definition (for structs, enums, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTypeDef {
    /// Type identifier
    pub id: IrTypeDefId,

    /// Type name
    pub name: String,

    /// Original TAST type
    pub type_id: TypeId,

    /// Type definition
    pub definition: IrTypeDefinition,

    /// Source location
    pub source_location: IrSourceLocation,

    /// Super class type id (for class types with inheritance)
    #[serde(default)]
    pub super_type_id: Option<TypeId>,
}

/// Type definition variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrTypeDefinition {
    /// Struct type
    Struct { fields: Vec<IrField>, packed: bool },

    /// Enum type
    Enum {
        variants: Vec<IrEnumVariant>,
        discriminant_type: IrType,
    },

    /// Type alias
    Alias { aliased_type: IrType },

    /// Opaque type (forward declaration)
    Opaque,
}

/// Struct field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrField {
    /// Field name
    pub name: String,

    /// Field type
    pub ty: IrType,

    /// Field offset (computed during layout)
    pub offset: Option<u32>,
}

/// Enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrEnumVariant {
    /// Variant name
    pub name: String,

    /// Discriminant value
    pub discriminant: i64,

    /// Associated data (if any)
    pub fields: Vec<IrField>,
}

/// External function declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrExternFunction {
    /// Function ID
    pub id: IrFunctionId,

    /// Function name (may be mangled)
    pub name: String,

    /// Original symbol
    pub symbol_id: SymbolId,

    /// Function signature
    pub signature: super::IrFunctionSignature,

    /// Which library/module this comes from
    pub source: String,
}

/// String constant pool for efficient string storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringPool {
    /// String constants indexed by ID
    strings: HashMap<u32, String>,

    /// Reverse mapping for deduplication
    string_to_id: HashMap<String, u32>,

    /// Next available ID
    next_id: u32,
}

impl StringPool {
    pub fn new() -> Self {
        Self {
            strings: HashMap::new(),
            string_to_id: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a string to the pool, returning its ID
    pub fn add(&mut self, s: String) -> u32 {
        if let Some(&id) = self.string_to_id.get(&s) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.string_to_id.insert(s.clone(), id);
            self.strings.insert(id, s);
            id
        }
    }

    /// Get a string by ID
    pub fn get(&self, id: u32) -> Option<&str> {
        self.strings.get(&id).map(|s| s.as_str())
    }

    /// Merge strings from another pool into this one (deduplicating by value)
    pub fn merge_from(&mut self, other: &StringPool) {
        for (_id, s) in &other.strings {
            self.add(s.clone());
        }
    }
}

/// Module metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    pub target_triple: Option<String>,

    /// Source language version
    pub language_version: String,

    /// Optimization level
    pub optimization_level: OptimizationLevel,

    /// Debug info level
    pub debug_info: DebugInfoLevel,

    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    O1,
    /// Standard optimization
    O2,
    /// Aggressive optimization
    O3,
    /// Optimize for size
    Os,
}

/// Debug info level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugInfoLevel {
    /// No debug info
    None,
    /// Line numbers only
    LineOnly,
    /// Full debug info
    Full,
}

impl Default for ModuleMetadata {
    fn default() -> Self {
        Self {
            target_triple: None,
            language_version: "1.0".to_string(),
            optimization_level: OptimizationLevel::None,
            debug_info: DebugInfoLevel::Full,
            attributes: HashMap::new(),
        }
    }
}

impl IrModule {
    /// Create a new HIR module
    pub fn new(name: String, source_file: String) -> Self {
        Self {
            name,
            source_file,
            functions: BTreeMap::new(),
            globals: HashMap::new(),
            types: HashMap::new(),
            string_pool: StringPool::new(),
            extern_functions: BTreeMap::new(),
            metadata: ModuleMetadata::default(),
            next_function_id: 0,
            next_global_id: 0,
            next_typedef_id: 0,
            symbol_to_register: HashMap::new(),
            register_to_symbol: HashMap::new(),
        }
    }

    /// Add a function to the module
    pub fn add_function(&mut self, function: IrFunction) -> IrFunctionId {
        let id = function.id;
        self.functions.insert(id, function);
        self.next_function_id = self.next_function_id.max(id.0 + 1);
        id
    }

    /// Allocate a new function ID
    pub fn alloc_function_id(&mut self) -> IrFunctionId {
        let id = IrFunctionId(self.next_function_id);
        self.next_function_id += 1;
        id
    }

    /// Add a global variable
    pub fn add_global(&mut self, global: IrGlobal) -> IrGlobalId {
        let id = global.id;
        self.globals.insert(id, global);
        self.next_global_id = self.next_global_id.max(id.0 + 1);
        id
    }

    /// Allocate a new global ID
    pub fn alloc_global_id(&mut self) -> IrGlobalId {
        let id = IrGlobalId(self.next_global_id);
        self.next_global_id += 1;
        id
    }

    /// Add a type definition
    pub fn add_type(&mut self, typedef: IrTypeDef) -> IrTypeDefId {
        let id = typedef.id;
        self.types.insert(id, typedef);
        self.next_typedef_id = self.next_typedef_id.max(id.0 + 1);
        id
    }

    /// Allocate a new type definition ID
    pub fn alloc_typedef_id(&mut self) -> IrTypeDefId {
        let id = IrTypeDefId(self.next_typedef_id);
        self.next_typedef_id += 1;
        id
    }

    /// Add an external function declaration
    pub fn add_extern_function(&mut self, extern_fn: IrExternFunction) {
        self.extern_functions.insert(extern_fn.id, extern_fn);
    }

    /// Get all functions (internal and external)
    pub fn all_functions(&self) -> impl Iterator<Item = (IrFunctionId, &str)> {
        self.functions
            .iter()
            .map(|(id, f)| (*id, f.name.as_str()))
            .chain(
                self.extern_functions
                    .iter()
                    .map(|(id, f)| (*id, f.name.as_str())),
            )
    }

    /// Verify module integrity
    pub fn verify(&self) -> Result<(), String> {
        // Verify all functions
        for (id, function) in &self.functions {
            function
                .verify()
                .map_err(|e| format!("Function {} error: {}", id, e))?;
        }

        // TODO: Verify globals, types, etc.

        Ok(())
    }

    /// Get module statistics
    pub fn stats(&self) -> ModuleStats {
        ModuleStats {
            function_count: self.functions.len(),
            global_count: self.globals.len(),
            type_count: self.types.len(),
            string_count: self.string_pool.strings.len(),
            extern_function_count: self.extern_functions.len(),
        }
    }
}

/// Module statistics
#[derive(Debug)]
pub struct ModuleStats {
    pub function_count: usize,
    pub global_count: usize,
    pub type_count: usize,
    pub string_count: usize,
    pub extern_function_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        let module = IrModule::new("test".to_string(), "test.hx".to_string());
        assert_eq!(module.name, "test");
        assert_eq!(module.source_file, "test.hx");
        assert!(module.functions.is_empty());
    }

    #[test]
    fn test_string_pool() {
        let mut pool = StringPool::new();
        let id1 = pool.add("hello".to_string());
        let id2 = pool.add("world".to_string());
        let id3 = pool.add("hello".to_string()); // Duplicate

        assert_eq!(id1, id3); // Deduplication
        assert_ne!(id1, id2);
        assert_eq!(pool.get(id1), Some("hello"));
        assert_eq!(pool.get(id2), Some("world"));
    }
}
