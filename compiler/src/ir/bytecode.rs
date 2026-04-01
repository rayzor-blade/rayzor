//! MIR Bytecode Format
//!
//! This module provides serialization and deserialization of MIR (Mid-level IR)
//! to a compact bytecode format for:
//! - Incremental compilation (avoid recompiling unchanged modules)
//! - Module caching (fast startup)
//! - Build artifacts (distribute pre-compiled libraries)
//!
//! # Bytecode Format
//!
//! The bytecode format is designed to be:
//! - **Compact**: Minimal size for fast I/O
//! - **Versionable**: Can evolve without breaking compatibility
//! - **Portable**: Architecture-independent (uses portable type sizes)
//! - **Fast to deserialize**: Direct mapping to MIR structures
//!
//! ## File Structure
//!
//! ```text
//! +------------------+
//! | Magic (4 bytes)  |  "RZBC" (Rayzor ByteCode)
//! +------------------+
//! | Version (4)      |  Format version number
//! +------------------+
//! | Checksum (8)     |  xxHash64 of content
//! +------------------+
//! | Metadata Section |  Module name, dependencies, timestamps
//! +------------------+
//! | Type Table       |  Serialized types
//! +------------------+
//! | Function Table   |  Function signatures and metadata
//! +------------------+
//! | CFG Data         |  Control flow graphs with phi nodes
//! +------------------+
//! | Constant Pool    |  String and numeric constants
//! +------------------+
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use compiler::ir::bytecode::{BytecodeWriter, BytecodeReader};
//!
//! // Serialize MIR to bytecode
//! let bytecode = BytecodeWriter::new()
//!     .write_module(&mir_module)?
//!     .to_bytes();
//!
//! std::fs::write("output.rzbc", &bytecode)?;
//!
//! // Deserialize bytecode to MIR
//! let bytecode = std::fs::read("output.rzbc")?;
//! let mir_module = BytecodeReader::new(&bytecode)
//!     .read_module()?;
//! ```

use std::collections::BTreeMap;
use std::io::{self, Write, Read};
use crate::ir::{IrModule, IrFunction, IrFunctionId, IrInstruction, IrTerminator, IrBlockId, IrId, IrType};
use crate::ir::cfg::ControlFlowGraph;

/// Magic number identifying Rayzor bytecode files
const MAGIC: &[u8; 4] = b"RZBC";

/// Current bytecode format version
const VERSION: u32 = 1;

/// Metadata about a compiled module
#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    /// Module name (e.g., "haxe.ds.StringMap")
    pub name: String,

    /// Source file path
    pub source_path: String,

    /// Source file modification timestamp (Unix time)
    pub source_timestamp: u64,

    /// Compilation timestamp
    pub compile_timestamp: u64,

    /// Dependencies (other modules this module imports)
    pub dependencies: Vec<String>,

    /// Compiler version used
    pub compiler_version: String,
}

/// Writer for serializing MIR to bytecode
pub struct BytecodeWriter {
    buffer: Vec<u8>,
    metadata: Option<ModuleMetadata>,
}

impl BytecodeWriter {
    /// Create a new bytecode writer
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            metadata: None,
        }
    }

    /// Set module metadata
    pub fn with_metadata(mut self, metadata: ModuleMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Write a MIR module to bytecode
    pub fn write_module(mut self, module: &IrModule) -> Result<Self, BytecodeError> {
        // Write header
        self.write_header()?;

        // Write metadata section
        self.write_metadata()?;

        // Write type table
        self.write_type_table(module)?;

        // Write function table
        self.write_function_table(module)?;

        // Write CFG data for each function
        self.write_cfg_data(module)?;

        // Write constant pool
        self.write_constant_pool(module)?;

        // Compute and write checksum
        self.finalize_checksum()?;

        Ok(self)
    }

    /// Get the serialized bytecode
    pub fn to_bytes(self) -> Vec<u8> {
        self.buffer
    }

    fn write_header(&mut self) -> Result<(), BytecodeError> {
        // Magic number
        self.buffer.extend_from_slice(MAGIC);

        // Version
        self.write_u32(VERSION)?;

        // Placeholder for checksum (will be filled later)
        self.write_u64(0)?;

        Ok(())
    }

    fn write_metadata(&mut self) -> Result<(), BytecodeError> {
        let metadata = self.metadata.as_ref().ok_or(BytecodeError::MissingMetadata)?;

        self.write_string(&metadata.name)?;
        self.write_string(&metadata.source_path)?;
        self.write_u64(metadata.source_timestamp)?;
        self.write_u64(metadata.compile_timestamp)?;

        // Write dependencies
        self.write_u32(metadata.dependencies.len() as u32)?;
        for dep in &metadata.dependencies {
            self.write_string(dep)?;
        }

        self.write_string(&metadata.compiler_version)?;

        Ok(())
    }

    fn write_type_table(&mut self, _module: &IrModule) -> Result<(), BytecodeError> {
        // TODO: Serialize type information
        // For now, write empty type table
        self.write_u32(0)?;
        Ok(())
    }

    fn write_function_table(&mut self, module: &IrModule) -> Result<(), BytecodeError> {
        // Write number of functions
        self.write_u32(module.functions.len() as u32)?;

        for (func_id, function) in &module.functions {
            self.write_function_id(*func_id)?;
            self.write_string(&function.name)?;

            // Write signature
            self.write_u32(function.signature.parameters.len() as u32)?;
            for param in &function.signature.parameters {
                self.write_ir_id(param.reg)?;
                self.write_type(&param.ty)?;
            }

            self.write_type(&function.signature.return_type)?;

            // Write locals count
            self.write_u32(function.locals.len() as u32)?;
        }

        Ok(())
    }

    fn write_cfg_data(&mut self, module: &IrModule) -> Result<(), BytecodeError> {
        for (func_id, function) in &module.functions {
            self.write_function_id(*func_id)?;
            self.write_cfg(&function.cfg)?;
        }

        Ok(())
    }

    fn write_cfg(&mut self, cfg: &ControlFlowGraph) -> Result<(), BytecodeError> {
        // Write number of blocks
        self.write_u32(cfg.blocks.len() as u32)?;

        for (block_id, block) in &cfg.blocks {
            self.write_block_id(*block_id)?;

            // Write phi nodes
            self.write_u32(block.phi_nodes.len() as u32)?;
            for phi in &block.phi_nodes {
                self.write_ir_id(phi.dest)?;
                self.write_type(&phi.ty)?;

                // Write incoming values
                self.write_u32(phi.incoming.len() as u32)?;
                for (block_id, value_id) in &phi.incoming {
                    self.write_block_id(*block_id)?;
                    self.write_ir_id(*value_id)?;
                }
            }

            // Write instructions
            self.write_u32(block.instructions.len() as u32)?;
            for inst in &block.instructions {
                self.write_instruction(inst)?;
            }

            // Write terminator
            self.write_terminator(&block.terminator)?;
        }

        // Write entry block
        self.write_block_id(cfg.entry_block)?;

        Ok(())
    }

    fn write_instruction(&mut self, inst: &IrInstruction) -> Result<(), BytecodeError> {
        // Write instruction opcode
        let opcode = instruction_opcode(inst);
        self.write_u8(opcode)?;

        // Write instruction-specific data
        match inst {
            IrInstruction::Add { dest, left, right } |
            IrInstruction::Sub { dest, left, right } |
            IrInstruction::Mul { dest, left, right } |
            IrInstruction::Div { dest, left, right } |
            IrInstruction::Mod { dest, left, right } => {
                self.write_ir_id(*dest)?;
                self.write_ir_id(*left)?;
                self.write_ir_id(*right)?;
            }
            IrInstruction::LoadInt { dest, value } => {
                self.write_ir_id(*dest)?;
                self.write_i64(*value)?;
            }
            IrInstruction::LoadFloat { dest, value } => {
                self.write_ir_id(*dest)?;
                self.write_f64(*value)?;
            }
            IrInstruction::LoadBool { dest, value } => {
                self.write_ir_id(*dest)?;
                self.write_u8(if *value { 1 } else { 0 })?;
            }
            IrInstruction::LoadNull { dest } => {
                self.write_ir_id(*dest)?;
            }
            IrInstruction::Copy { dest, src } => {
                self.write_ir_id(*dest)?;
                self.write_ir_id(*src)?;
            }
            IrInstruction::CallDirect { dest, func_id, args, arg_ownership: _, type_args, is_tail_call } => {
                self.write_ir_id(*dest)?;
                self.write_function_id(*func_id)?;
                self.write_u32(args.len() as u32)?;
                for arg in args {
                    self.write_ir_id(*arg)?;
                }
                // Write type args count and types for generic instantiation
                self.write_u32(type_args.len() as u32)?;
                for ty in type_args {
                    self.write_type(ty)?;
                }
                // Write tail call flag
                self.write_u8(if *is_tail_call { 1 } else { 0 })?;
            }
            IrInstruction::Cmp { dest, op, left, right } => {
                self.write_ir_id(*dest)?;
                self.write_u8(*op as u8)?;
                self.write_ir_id(*left)?;
                self.write_ir_id(*right)?;
            }
            // Add other instruction types...
            _ => {
                // For now, skip unsupported instructions
                // TODO: Implement all instruction types
            }
        }

        Ok(())
    }

    fn write_terminator(&mut self, term: &IrTerminator) -> Result<(), BytecodeError> {
        match term {
            IrTerminator::Return { value } => {
                self.write_u8(0)?; // Return opcode
                if let Some(val) = value {
                    self.write_u8(1)?; // Has value
                    self.write_ir_id(*val)?;
                } else {
                    self.write_u8(0)?; // No value
                }
            }
            IrTerminator::Branch { target } => {
                self.write_u8(1)?; // Branch opcode
                self.write_block_id(*target)?;
            }
            IrTerminator::CondBranch { condition, then_block, else_block } => {
                self.write_u8(2)?; // CondBranch opcode
                self.write_ir_id(*condition)?;
                self.write_block_id(*then_block)?;
                self.write_block_id(*else_block)?;
            }
            IrTerminator::Unreachable => {
                self.write_u8(3)?; // Unreachable opcode
            }
        }

        Ok(())
    }

    fn write_constant_pool(&mut self, _module: &IrModule) -> Result<(), BytecodeError> {
        // TODO: Write string and constant data
        // For now, write empty constant pool
        self.write_u32(0)?;
        Ok(())
    }

    fn finalize_checksum(&mut self) -> Result<(), BytecodeError> {
        // Compute checksum of all data after the checksum field
        let checksum = compute_checksum(&self.buffer[16..]);

        // Write checksum at offset 8
        self.buffer[8..16].copy_from_slice(&checksum.to_le_bytes());

        Ok(())
    }

    // Helper write methods
    fn write_u8(&mut self, value: u8) -> Result<(), BytecodeError> {
        self.buffer.write_all(&[value]).map_err(BytecodeError::IoError)
    }

    fn write_u32(&mut self, value: u32) -> Result<(), BytecodeError> {
        self.buffer.write_all(&value.to_le_bytes()).map_err(BytecodeError::IoError)
    }

    fn write_u64(&mut self, value: u64) -> Result<(), BytecodeError> {
        self.buffer.write_all(&value.to_le_bytes()).map_err(BytecodeError::IoError)
    }

    fn write_i64(&mut self, value: i64) -> Result<(), BytecodeError> {
        self.buffer.write_all(&value.to_le_bytes()).map_err(BytecodeError::IoError)
    }

    fn write_f64(&mut self, value: f64) -> Result<(), BytecodeError> {
        self.buffer.write_all(&value.to_le_bytes()).map_err(BytecodeError::IoError)
    }

    fn write_string(&mut self, s: &str) -> Result<(), BytecodeError> {
        self.write_u32(s.len() as u32)?;
        self.buffer.write_all(s.as_bytes()).map_err(BytecodeError::IoError)
    }

    fn write_ir_id(&mut self, id: IrId) -> Result<(), BytecodeError> {
        self.write_u32(id.0)
    }

    fn write_function_id(&mut self, id: IrFunctionId) -> Result<(), BytecodeError> {
        self.write_u32(id.0)
    }

    fn write_block_id(&mut self, id: IrBlockId) -> Result<(), BytecodeError> {
        self.write_u32(id.0)
    }

    fn write_type(&mut self, ty: &IrType) -> Result<(), BytecodeError> {
        // Write type discriminant
        match ty {
            IrType::Void => self.write_u8(0)?,
            IrType::Bool => self.write_u8(1)?,
            IrType::Int => self.write_u8(2)?,
            IrType::Float => self.write_u8(3)?,
            IrType::String => self.write_u8(4)?,
            IrType::Dynamic => self.write_u8(5)?,
            IrType::Class(_) => {
                self.write_u8(6)?;
                // TODO: Write class type ID
                self.write_u32(0)?;
            }
            IrType::Function { params, return_type } => {
                self.write_u8(7)?;
                self.write_u32(params.len() as u32)?;
                for param in params {
                    self.write_type(param)?;
                }
                self.write_type(return_type)?;
            }
            IrType::Pointer(_) => {
                self.write_u8(8)?;
                // TODO: Write pointee type
                self.write_u32(0)?;
            }
        }
        Ok(())
    }
}

/// Reader for deserializing bytecode to MIR
pub struct BytecodeReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> BytecodeReader<'a> {
    /// Create a new bytecode reader
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    /// Read and validate a MIR module from bytecode
    pub fn read_module(&mut self) -> Result<(IrModule, ModuleMetadata), BytecodeError> {
        // Read and validate header
        self.read_header()?;

        // Read metadata
        let metadata = self.read_metadata()?;

        // Read type table
        self.read_type_table()?;

        // Read function table
        let functions = self.read_function_table()?;

        // Read CFG data
        self.read_cfg_data(&functions)?;

        // Read constant pool
        self.read_constant_pool()?;

        let module = IrModule {
            functions,
            globals: BTreeMap::new(), // TODO: Serialize globals
            structs: BTreeMap::new(),  // TODO: Serialize structs
        };

        Ok((module, metadata))
    }

    fn read_header(&mut self) -> Result<(), BytecodeError> {
        // Check magic number
        let magic = self.read_bytes(4)?;
        if magic != MAGIC {
            return Err(BytecodeError::InvalidMagic);
        }

        // Check version
        let version = self.read_u32()?;
        if version != VERSION {
            return Err(BytecodeError::UnsupportedVersion(version));
        }

        // Read and validate checksum
        let stored_checksum = self.read_u64()?;
        let computed_checksum = compute_checksum(&self.data[16..]);

        if stored_checksum != computed_checksum {
            return Err(BytecodeError::ChecksumMismatch);
        }

        Ok(())
    }

    fn read_metadata(&mut self) -> Result<ModuleMetadata, BytecodeError> {
        Ok(ModuleMetadata {
            name: self.read_string()?,
            source_path: self.read_string()?,
            source_timestamp: self.read_u64()?,
            compile_timestamp: self.read_u64()?,
            dependencies: {
                let count = self.read_u32()? as usize;
                (0..count).map(|_| self.read_string()).collect::<Result<Vec<_>, _>>()?
            },
            compiler_version: self.read_string()?,
        })
    }

    fn read_type_table(&mut self) -> Result<(), BytecodeError> {
        let _count = self.read_u32()?;
        // TODO: Read type table
        Ok(())
    }

    fn read_function_table(&mut self) -> Result<BTreeMap<IrFunctionId, IrFunction>, BytecodeError> {
        let count = self.read_u32()? as usize;
        let mut functions = BTreeMap::new();

        for _ in 0..count {
            let func_id = self.read_function_id()?;
            let name = self.read_string()?;

            // Read signature
            let param_count = self.read_u32()? as usize;
            let mut parameters = Vec::new();
            for _ in 0..param_count {
                let reg = self.read_ir_id()?;
                let ty = self.read_type()?;
                parameters.push(crate::ir::IrParameter { reg, ty });
            }

            let return_type = self.read_type()?;

            let _locals_count = self.read_u32()?;

            let function = IrFunction {
                name,
                signature: crate::ir::IrSignature {
                    parameters,
                    return_type,
                },
                cfg: ControlFlowGraph::new(), // Will be filled in read_cfg_data
                locals: BTreeMap::new(), // TODO: Read locals
            };

            functions.insert(func_id, function);
        }

        Ok(functions)
    }

    fn read_cfg_data(&mut self, _functions: &BTreeMap<IrFunctionId, IrFunction>) -> Result<(), BytecodeError> {
        // TODO: Read CFG data for each function
        Ok(())
    }

    fn read_constant_pool(&mut self) -> Result<(), BytecodeError> {
        let _count = self.read_u32()?;
        // TODO: Read constant pool
        Ok(())
    }

    // Helper read methods
    fn read_bytes(&mut self, count: usize) -> Result<&'a [u8], BytecodeError> {
        if self.offset + count > self.data.len() {
            return Err(BytecodeError::UnexpectedEof);
        }
        let bytes = &self.data[self.offset..self.offset + count];
        self.offset += count;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, BytecodeError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, BytecodeError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, BytecodeError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, BytecodeError> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| BytecodeError::InvalidUtf8)
    }

    fn read_ir_id(&mut self) -> Result<IrId, BytecodeError> {
        Ok(IrId(self.read_u32()?))
    }

    fn read_function_id(&mut self) -> Result<IrFunctionId, BytecodeError> {
        Ok(IrFunctionId(self.read_u32()?))
    }

    fn read_type(&mut self) -> Result<IrType, BytecodeError> {
        let discriminant = self.read_u8()?;
        match discriminant {
            0 => Ok(IrType::Void),
            1 => Ok(IrType::Bool),
            2 => Ok(IrType::Int),
            3 => Ok(IrType::Float),
            4 => Ok(IrType::String),
            5 => Ok(IrType::Dynamic),
            6 => {
                let _class_id = self.read_u32()?;
                // TODO: Look up actual class type
                Ok(IrType::Dynamic) // Placeholder
            }
            7 => {
                let param_count = self.read_u32()? as usize;
                let mut params = Vec::new();
                for _ in 0..param_count {
                    params.push(self.read_type()?);
                }
                let return_type = Box::new(self.read_type()?);
                Ok(IrType::Function { params, return_type })
            }
            8 => {
                let _pointee_id = self.read_u32()?;
                // TODO: Read actual pointee type
                Ok(IrType::Pointer(Box::new(IrType::Dynamic))) // Placeholder
            }
            _ => Err(BytecodeError::InvalidTypeDiscriminant(discriminant)),
        }
    }
}

/// Bytecode errors
#[derive(Debug)]
pub enum BytecodeError {
    InvalidMagic,
    UnsupportedVersion(u32),
    ChecksumMismatch,
    UnexpectedEof,
    InvalidUtf8,
    InvalidTypeDiscriminant(u8),
    MissingMetadata,
    IoError(io::Error),
}

impl std::fmt::Display for BytecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BytecodeError::InvalidMagic => write!(f, "Invalid magic number in bytecode file"),
            BytecodeError::UnsupportedVersion(v) => write!(f, "Unsupported bytecode version: {}", v),
            BytecodeError::ChecksumMismatch => write!(f, "Bytecode checksum mismatch"),
            BytecodeError::UnexpectedEof => write!(f, "Unexpected end of bytecode file"),
            BytecodeError::InvalidUtf8 => write!(f, "Invalid UTF-8 in bytecode"),
            BytecodeError::InvalidTypeDiscriminant(d) => write!(f, "Invalid type discriminant: {}", d),
            BytecodeError::MissingMetadata => write!(f, "Missing module metadata"),
            BytecodeError::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for BytecodeError {}

/// Get opcode for an instruction (for serialization)
fn instruction_opcode(inst: &IrInstruction) -> u8 {
    match inst {
        IrInstruction::Add { .. } => 0,
        IrInstruction::Sub { .. } => 1,
        IrInstruction::Mul { .. } => 2,
        IrInstruction::Div { .. } => 3,
        IrInstruction::Mod { .. } => 4,
        IrInstruction::LoadInt { .. } => 5,
        IrInstruction::LoadFloat { .. } => 6,
        IrInstruction::LoadBool { .. } => 7,
        IrInstruction::LoadNull { .. } => 8,
        IrInstruction::Copy { .. } => 9,
        IrInstruction::CallDirect { .. } => 10,
        IrInstruction::Cmp { .. } => 11,
        // Add more instruction opcodes...
        _ => 255, // Unknown/unsupported
    }
}

/// Compute xxHash64 checksum
fn compute_checksum(data: &[u8]) -> u64 {
    // Simple FNV-1a hash for now (replace with xxHash later for better performance)
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_header() {
        let writer = BytecodeWriter::new()
            .with_metadata(ModuleMetadata {
                name: "test.Module".to_string(),
                source_path: "test.hx".to_string(),
                source_timestamp: 1234567890,
                compile_timestamp: 1234567900,
                dependencies: vec!["haxe.String".to_string()],
                compiler_version: "0.1.0".to_string(),
            });

        let module = IrModule {
            functions: BTreeMap::new(),
            globals: BTreeMap::new(),
            structs: BTreeMap::new(),
        };

        let bytecode = writer.write_module(&module).unwrap().to_bytes();

        // Verify magic number
        assert_eq!(&bytecode[0..4], b"RZBC");

        // Verify version
        let version = u32::from_le_bytes([bytecode[4], bytecode[5], bytecode[6], bytecode[7]]);
        assert_eq!(version, VERSION);
    }

    #[test]
    fn test_round_trip_empty_module() {
        let metadata = ModuleMetadata {
            name: "test.Empty".to_string(),
            source_path: "empty.hx".to_string(),
            source_timestamp: 1000,
            compile_timestamp: 2000,
            dependencies: vec![],
            compiler_version: "0.1.0".to_string(),
        };

        let module = IrModule {
            functions: BTreeMap::new(),
            globals: BTreeMap::new(),
            structs: BTreeMap::new(),
        };

        // Serialize
        let bytecode = BytecodeWriter::new()
            .with_metadata(metadata.clone())
            .write_module(&module)
            .unwrap()
            .to_bytes();

        // Deserialize
        let mut reader = BytecodeReader::new(&bytecode);
        let (decoded_module, decoded_metadata) = reader.read_module().unwrap();

        // Verify
        assert_eq!(decoded_metadata.name, metadata.name);
        assert_eq!(decoded_module.functions.len(), 0);
    }
}
