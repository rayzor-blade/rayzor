use tracing::debug;

use crate::ir::hir::HirCapture;
use crate::ir::{BinaryOp, IrBuilder, IrId, IrType};
use crate::tast::{SymbolId, TypeId};

/// Represents a single captured variable in the environment
#[derive(Debug, Clone)]
pub struct EnvironmentField {
    /// Index in the captures array
    pub index: usize,
    /// Symbol of the captured variable
    pub symbol: SymbolId,
    /// Final type after conversion (what the lambda code expects)
    pub ty: IrType,
    /// Storage type in environment (how it's actually stored)
    pub storage_ty: IrType,
    /// Byte offset in environment struct
    pub offset: usize,
    /// Whether a cast is needed from storage_ty to ty
    pub needs_cast: bool,
}

/// Describes the layout of a closure environment
#[derive(Debug, Clone)]
pub struct EnvironmentLayout {
    /// Fields in the environment, in order
    pub fields: Vec<EnvironmentField>,
    /// Total size in bytes
    pub total_size: usize,
    /// Alignment requirement
    pub alignment: usize,
}

impl EnvironmentLayout {
    /// Create a new environment layout from captures
    pub fn new<F>(captures: &[HirCapture], type_converter: F) -> Self
    where
        F: Fn(TypeId) -> IrType,
    {
        let mut offset = 0;
        let mut fields = Vec::with_capacity(captures.len());

        for (index, capture) in captures.iter().enumerate() {
            let final_ty = type_converter(capture.ty);
            let storage_ty = IrType::I64; // Always store as I64 (pointer-sized)

            debug!(
                "DEBUG EnvironmentLayout: Capture {} (sym {:?}) - final_ty={:?}",
                index, capture.symbol, final_ty
            );

            // Determine if cast is needed
            // IMPORTANT: Only cast for scalar integer types (I8, I16, I32).
            // Pointer types (Ptr, Ref, Struct, etc.) should NEVER be cast from I64
            // to a smaller type, as this would truncate the pointer value!
            let needs_cast = match &final_ty {
                IrType::I8 => true,                 // I64 → I8 cast needed
                IrType::I16 => true,                // I64 → I16 cast needed
                IrType::I32 => true,                // I64 → I32 cast needed
                IrType::U8 => true,                 // I64 → U8 cast needed
                IrType::U16 => true,                // I64 → U16 cast needed
                IrType::U32 => true,                // I64 → U32 cast needed
                IrType::I64 | IrType::U64 => false, // Already 64-bit
                // CRITICAL: Pointer and reference types are stored as I64 but
                // should NOT be cast down - they must remain as the full 64-bit value
                IrType::Ptr(_) => false,
                IrType::Ref(_) => false,
                IrType::Struct { .. } => false,
                IrType::Function { .. } => false,
                IrType::String => false,
                IrType::Array(..) => false,
                IrType::Slice(_) => false,
                IrType::Any => false,
                IrType::Opaque { .. } => false,
                IrType::TypeVar(_) => false,
                IrType::Generic { .. } => false, // Generic types treated as pointers
                // Bool and floating point
                IrType::Bool => true, // I64 → I8 for bool
                // Bit-preserving, not a numeric conversion. The value went in
                // by reinterpreting its bits into the i64 slot, so it has to
                // come back out the same way — `build_cast` between I64 and a
                // float is a bitcast for exactly this reason. Leaving it uncast
                // handed the lambda body an integer holding the float's bit
                // pattern, and every arithmetic on it was integer arithmetic:
                // a captured 2.5 read as 4612811918334230528, and `* 2.0`
                // saturated to i64::MAX. Silently, on both backends.
                IrType::F32 | IrType::F64 => true,
                IrType::Void => false,
                IrType::Union { .. } => false,
                // SIMD vector types are not captured directly
                IrType::Vector { .. } => false,
            };

            fields.push(EnvironmentField {
                index,
                symbol: capture.symbol,
                ty: final_ty,
                storage_ty,
                offset,
                needs_cast,
            });

            // Always use 8-byte alignment for simplicity
            offset += 8;
        }

        EnvironmentLayout {
            fields,
            total_size: offset,
            alignment: 8,
        }
    }

    /// Find field by symbol
    pub fn find_field(&self, symbol: SymbolId) -> Option<&EnvironmentField> {
        self.fields.iter().find(|f| f.symbol == symbol)
    }

    /// Generate code to load a captured variable from the environment
    ///
    /// Returns the register containing the final value (after cast if needed)
    pub fn load_field(
        &self,
        builder: &mut IrBuilder,
        env_ptr: IrId,
        symbol: SymbolId,
    ) -> Option<IrId> {
        let field = self.find_field(symbol)?;

        // Calculate field address: env_ptr + offset
        let offset_const = builder.build_int(field.offset as i64, IrType::I64)?;
        let field_ptr = builder.build_binop(BinaryOp::Add, env_ptr, offset_const)?;
        // Register the pointer's type
        builder.register_local(field_ptr, IrType::Ptr(Box::new(IrType::Void)))?;

        // Load the value (always as I64 from storage)
        let loaded = builder.build_load(field_ptr, field.storage_ty.clone())?;
        // Register the loaded value's type - CRITICAL: For pointer types, register with
        // the SEMANTIC type (field.ty = Ptr), not storage type (I64). This ensures that
        // downstream type checks recognize this as a pointer and don't truncate it.
        let register_ty = if matches!(
            field.ty,
            IrType::Ptr(_) | IrType::Ref(_) | IrType::String | IrType::Any
        ) {
            debug!("DEBUG: EnvironmentLayout registering loaded {:?} as pointer type {:?} (storage was {:?})", loaded, field.ty, field.storage_ty);
            field.ty.clone()
        } else {
            debug!(
                "DEBUG: EnvironmentLayout registering loaded {:?} with storage type {:?}",
                loaded, field.storage_ty
            );
            field.storage_ty.clone()
        };
        builder.register_local(loaded, register_ty)?;
        debug!(
            "DEBUG: EnvironmentLayout successfully registered loaded {:?}",
            loaded
        );

        // Cast if needed
        let final_reg = if field.needs_cast {
            // A float goes in and comes out by REINTERPRETING its bits, never by
            // converting: the i64 slot holds the bit pattern, so `build_cast`
            // here would sitofp the pattern into a nonsense magnitude. F32 goes
            // through i32 first — a 4-byte float and an 8-byte slot are not a
            // legal bitcast pair.
            let casted = match field.ty {
                IrType::F64 => builder.build_bitcast(loaded, IrType::F64)?,
                IrType::F32 => {
                    let narrowed = builder.build_cast(loaded, IrType::I64, IrType::I32)?;
                    builder.register_local(narrowed, IrType::I32)?;
                    builder.build_bitcast(narrowed, IrType::F32)?
                }
                _ => builder.build_cast(loaded, field.storage_ty.clone(), field.ty.clone())?,
            };
            // Register the casted value's type
            builder.register_local(casted, field.ty.clone())?;
            casted
        } else {
            loaded
        };

        Some(final_reg)
    }

    /// Generate code to store a value back to a captured variable in the environment
    ///
    /// This handles the reverse of load_field: cast from final type to storage type if needed,
    /// then store to the environment.
    pub fn store_field(
        &self,
        builder: &mut IrBuilder,
        env_ptr: IrId,
        symbol: SymbolId,
        value: IrId,
    ) -> Option<()> {
        let field = self.find_field(symbol)?;

        // Cast if needed (I32 → I64 for storage). Floats are reinterpreted
        // rather than converted, mirroring `load_field`.
        let store_value = if field.needs_cast {
            match field.ty {
                IrType::F64 => builder.build_bitcast(value, IrType::I64)?,
                IrType::F32 => {
                    let bits = builder.build_bitcast(value, IrType::I32)?;
                    builder.register_local(bits, IrType::I32)?;
                    builder.build_cast(bits, IrType::I32, IrType::I64)?
                }
                _ => builder.build_cast(value, field.ty.clone(), field.storage_ty.clone())?,
            }
        } else {
            value
        };

        // Calculate field address: env_ptr + offset
        let offset_const = builder.build_int(field.offset as i64, IrType::I64)?;
        let field_ptr = builder.build_binop(BinaryOp::Add, env_ptr, offset_const)?;
        // Register the pointer's type
        builder.register_local(field_ptr, IrType::Ptr(Box::new(IrType::Void)))?;

        // Store the value
        builder.build_store(field_ptr, store_value)?;

        Some(())
    }
}
