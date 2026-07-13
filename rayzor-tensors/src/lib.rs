//! rayzor-tensors: the shared tensor layer (Tensor/QTensor + the ML kernels and
//! their private worker/tensor pools), extracted from rayzor-runtime so the core
//! runtime stays general. Portable compute comes from `rayzor-runtime-core`; the
//! one general-runtime service it needs (`perf_core_count`) resolves from the
//! host binary at load (RTLD_GLOBAL).

// The kernels are C-ABI FFI entry points: raw-pointer args and callers that own
// the safety contract. Same crate-level allows the runtime carried for them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::missing_safety_doc)]

pub mod haxe_sys;
pub mod heap_check;
pub mod kernel_timing;
pub mod quant;
pub mod tensor;
pub mod tensor_pool;
pub mod tensor_simd;
pub mod worker_pool;

/// Read an env var by its current `RZT_`/`NUE_` name, falling back to the
/// legacy `RAYZOR_` name. Kept Result-typed so call sites keep their existing
/// `.ok()`/`.map`/`.map_or`/`match` shape. The legacy alias is transitional.
pub(crate) fn env_var(primary: &str, legacy: &str) -> Result<String, std::env::VarError> {
    std::env::var(primary).or_else(|_| std::env::var(legacy))
}

/// Host-runtime services resolved from the loading binary.
pub mod topology {
    extern "C" {
        #[link_name = "rayzor_topology_perf_core_count"]
        fn perf_core_count_ffi() -> i32;
    }
    #[inline]
    pub fn rayzor_topology_perf_core_count() -> i32 {
        unsafe { perf_core_count_ffi() }
    }
}

// ============================================================================
// Plugin ABI surface (native host loader).
//
// The host dlopens this dylib via `[build] native-libs`, checks the ABI
// handshake, then reads the runtime symbol table below and registers each
// (name -> fn-ptr) with the JIT so the `@:native("rayzor_tensor_*")` /
// `@:native("qtensor_*")` externs resolve to these kernels. Mirrors the
// nue-plugins loader contract (`plugin_init` -> `SymbolEntry` triples).
// ============================================================================

rayzor_plugin::export_abi_version!();

#[repr(C)]
pub struct SymbolEntry {
    pub name_ptr: *const u8,
    pub name_len: usize,
    pub fn_ptr: *const core::ffi::c_void,
}

macro_rules! entry {
    ($name:expr, $fn:path) => {
        SymbolEntry {
            name_ptr: ($name as &[u8]).as_ptr(),
            name_len: ($name as &[u8]).len(),
            fn_ptr: $fn as *const core::ffi::c_void,
        }
    };
}

/// JIT linkage entry point: hands the host the tensor/quant kernel symbol
/// table. The host reads `count` entries of `(name_ptr, name_len, fn_ptr)`.
#[no_mangle]
pub unsafe extern "C" fn plugin_init(out_count: *mut usize) -> *const SymbolEntry {
    let entries = Box::new([
        entry!(
            b"rayzor_plugin_tensor_data",
            crate::tensor::rayzor_plugin_tensor_data
        ),
        entry!(
            b"rayzor_plugin_tensor_dtype",
            crate::tensor::rayzor_plugin_tensor_dtype
        ),
        entry!(
            b"rayzor_plugin_tensor_ndim",
            crate::tensor::rayzor_plugin_tensor_ndim
        ),
        entry!(
            b"rayzor_plugin_tensor_shape",
            crate::tensor::rayzor_plugin_tensor_shape
        ),
        entry!(
            b"rayzor_plugin_tensor_is_contiguous",
            crate::tensor::rayzor_plugin_tensor_is_contiguous
        ),
        entry!(
            b"rayzor_plugin_tensor_alloc_zeros",
            crate::tensor::rayzor_plugin_tensor_alloc_zeros
        ),
        entry!(b"rayzor_tensor_zeros", crate::tensor::rayzor_tensor_zeros),
        entry!(b"rayzor_tensor_uninit", crate::tensor::rayzor_tensor_uninit),
        entry!(b"rayzor_tensor_ones", crate::tensor::rayzor_tensor_ones),
        entry!(b"rayzor_tensor_full", crate::tensor::rayzor_tensor_full),
        entry!(
            b"rayzor_tensor_from_array",
            crate::tensor::rayzor_tensor_from_array
        ),
        entry!(
            b"rayzor_tensor_from_bytes_f16",
            crate::tensor::rayzor_tensor_from_bytes_f16
        ),
        entry!(
            b"rayzor_tensor_from_bytes_f32",
            crate::tensor::rayzor_tensor_from_bytes_f32
        ),
        entry!(
            b"rayzor_tensor_from_bytes_q8_0",
            crate::tensor::rayzor_tensor_from_bytes_q8_0
        ),
        entry!(b"rayzor_tensor_rand", crate::tensor::rayzor_tensor_rand),
        entry!(b"rayzor_tensor_shape", crate::tensor::rayzor_tensor_shape),
        entry!(b"rayzor_tensor_ndim", crate::tensor::rayzor_tensor_ndim),
        entry!(b"rayzor_tensor_numel", crate::tensor::rayzor_tensor_numel),
        entry!(b"rayzor_tensor_dtype", crate::tensor::rayzor_tensor_dtype),
        entry!(b"rayzor_tensor_device", crate::tensor::rayzor_tensor_device),
        entry!(
            b"rayzor_tensor_numa_node",
            crate::tensor::rayzor_tensor_numa_node
        ),
        entry!(
            b"rayzor_tensor_shape_ptr",
            crate::tensor::rayzor_tensor_shape_ptr
        ),
        entry!(
            b"rayzor_tensor_shape_ndim",
            crate::tensor::rayzor_tensor_shape_ndim
        ),
        entry!(b"rayzor_tensor_get", crate::tensor::rayzor_tensor_get),
        entry!(
            b"rayzor_tensor_get_flat",
            crate::tensor::rayzor_tensor_get_flat
        ),
        entry!(
            b"rayzor_tensor_set_flat",
            crate::tensor::rayzor_tensor_set_flat
        ),
        entry!(
            b"rayzor_tensor_topk_scan",
            crate::tensor::rayzor_tensor_topk_scan
        ),
        entry!(
            b"rayzor_tensor_flash_attn_decode",
            crate::tensor::rayzor_tensor_flash_attn_decode
        ),
        entry!(b"rayzor_tensor_set", crate::tensor::rayzor_tensor_set),
        entry!(
            b"rayzor_tensor_append_along_0_f32",
            crate::tensor::rayzor_tensor_append_along_0_f32
        ),
        entry!(
            b"rayzor_tensor_broadcast_repeat_0_f32",
            crate::tensor::rayzor_tensor_broadcast_repeat_0_f32
        ),
        entry!(
            b"rayzor_tensor_reshape",
            crate::tensor::rayzor_tensor_reshape
        ),
        entry!(
            b"rayzor_tensor_transpose",
            crate::tensor::rayzor_tensor_transpose
        ),
        entry!(
            b"rayzor_tensor_permute",
            crate::tensor::rayzor_tensor_permute
        ),
        entry!(b"rayzor_tensor_slice", crate::tensor::rayzor_tensor_slice),
        entry!(b"rayzor_tensor_add", crate::tensor::rayzor_tensor_add),
        entry!(
            b"rayzor_tensor_add_into",
            crate::tensor::rayzor_tensor_add_into
        ),
        entry!(b"rayzor_tensor_sub", crate::tensor::rayzor_tensor_sub),
        entry!(b"rayzor_tensor_mul", crate::tensor::rayzor_tensor_mul),
        entry!(
            b"rayzor_tensor_silu_mul",
            crate::tensor::rayzor_tensor_silu_mul
        ),
        entry!(b"rayzor_tensor_div", crate::tensor::rayzor_tensor_div),
        entry!(b"rayzor_tensor_sqrt", crate::tensor::rayzor_tensor_sqrt),
        entry!(b"rayzor_tensor_exp", crate::tensor::rayzor_tensor_exp),
        entry!(b"rayzor_tensor_log", crate::tensor::rayzor_tensor_log),
        entry!(b"rayzor_tensor_relu", crate::tensor::rayzor_tensor_relu),
        entry!(b"rayzor_tensor_gelu", crate::tensor::rayzor_tensor_gelu),
        entry!(b"rayzor_tensor_silu", crate::tensor::rayzor_tensor_silu),
        entry!(
            b"rayzor_tensor_softmax",
            crate::tensor::rayzor_tensor_softmax
        ),
        entry!(
            b"rayzor_tensor_layer_norm",
            crate::tensor::rayzor_tensor_layer_norm
        ),
        entry!(
            b"rayzor_tensor_rms_norm",
            crate::tensor::rayzor_tensor_rms_norm
        ),
        entry!(
            b"rayzor_tensor_rms_norm_weight",
            crate::tensor::rayzor_tensor_rms_norm_weight
        ),
        entry!(b"rayzor_tensor_sum", crate::tensor::rayzor_tensor_sum),
        entry!(b"rayzor_tensor_mean", crate::tensor::rayzor_tensor_mean),
        entry!(b"rayzor_tensor_max", crate::tensor::rayzor_tensor_max),
        entry!(b"rayzor_tensor_min", crate::tensor::rayzor_tensor_min),
        entry!(b"rayzor_tensor_dot", crate::tensor::rayzor_tensor_dot),
        entry!(b"rayzor_tensor_matmul", crate::tensor::rayzor_tensor_matmul),
        entry!(
            b"rayzor_tensor_matmul_t",
            crate::tensor::rayzor_tensor_matmul_t
        ),
        entry!(
            b"rayzor_tensor_matmul_t_threaded",
            crate::tensor::rayzor_tensor_matmul_t_threaded
        ),
        entry!(b"rayzor_tensor_rope", crate::tensor::rayzor_tensor_rope),
        entry!(
            b"rayzor_tensor_rope_cos_table",
            crate::tensor::rayzor_tensor_rope_cos_table
        ),
        entry!(
            b"rayzor_tensor_rope_sin_table",
            crate::tensor::rayzor_tensor_rope_sin_table
        ),
        entry!(
            b"rayzor_tensor_rope_cos_table_f16",
            crate::tensor::rayzor_tensor_rope_cos_table_f16
        ),
        entry!(
            b"rayzor_tensor_rope_sin_table_f16",
            crate::tensor::rayzor_tensor_rope_sin_table_f16
        ),
        entry!(
            b"rayzor_tensor_gather_rows",
            crate::tensor::rayzor_tensor_gather_rows
        ),
        entry!(b"rayzor_tensor_bmm", crate::tensor::rayzor_tensor_bmm),
        entry!(
            b"rayzor_tensor_bmm_threaded",
            crate::tensor::rayzor_tensor_bmm_threaded
        ),
        entry!(
            b"rayzor_tensor_expand_kv_heads_axis1_f32",
            crate::tensor::rayzor_tensor_expand_kv_heads_axis1_f32
        ),
        entry!(
            b"rayzor_tensor_causal_mask_",
            crate::tensor::rayzor_tensor_causal_mask_
        ),
        entry!(b"rayzor_tensor_scale", crate::tensor::rayzor_tensor_scale),
        entry!(
            b"rayzor_tensor_transpose_last2",
            crate::tensor::rayzor_tensor_transpose_last2
        ),
        entry!(b"rayzor_tensor_data", crate::tensor::rayzor_tensor_data),
        entry!(b"rayzor_tensor_free", crate::tensor::rayzor_tensor_free),
        entry!(b"rayzor_tensor_clone", crate::tensor::rayzor_tensor_clone),
        entry!(
            b"rayzor_tensor_arc_clone",
            crate::tensor::rayzor_tensor_arc_clone
        ),
        entry!(
            b"rayzor_tensor_deep_clone",
            crate::tensor::rayzor_tensor_deep_clone
        ),
        entry!(
            b"rayzor_qtensor_from_f32_int8",
            crate::quant::rayzor_qtensor_from_f32_int8
        ),
        entry!(
            b"rayzor_qtensor_wrap_q4_k_m",
            crate::quant::rayzor_qtensor_wrap_q4_k_m
        ),
        entry!(
            b"rayzor_qtensor_from_bytes_q4_k_m",
            crate::quant::rayzor_qtensor_from_bytes_q4_k_m
        ),
        entry!(
            b"rayzor_qtensor_from_bytes_q6_k",
            crate::quant::rayzor_qtensor_from_bytes_q6_k
        ),
        entry!(
            b"rayzor_qtensor_requant_q6k_to_q4km",
            crate::quant::rayzor_qtensor_requant_q6k_to_q4km
        ),
        entry!(b"rayzor_qtensor_rows", crate::quant::rayzor_qtensor_rows),
        entry!(b"rayzor_qtensor_cols", crate::quant::rayzor_qtensor_cols),
        entry!(
            b"rayzor_qtensor_data_ptr",
            crate::quant::rayzor_qtensor_data_ptr
        ),
        entry!(b"rayzor_qtensor_numel", crate::quant::rayzor_qtensor_numel),
        entry!(
            b"rayzor_qtensor_scheme",
            crate::quant::rayzor_qtensor_scheme
        ),
        entry!(
            b"rayzor_qtensor_dequant",
            crate::quant::rayzor_qtensor_dequant
        ),
        entry!(
            b"rayzor_qtensor_matmul_f32",
            crate::quant::rayzor_qtensor_matmul_f32
        ),
        entry!(
            b"rayzor_tensor_matmul_qt_t_f32",
            crate::quant::rayzor_tensor_matmul_qt_t_f32
        ),
        entry!(
            b"rayzor_tensor_matmul_qt_t_f32_chunk",
            crate::quant::rayzor_tensor_matmul_qt_t_f32_chunk
        ),
        entry!(
            b"rayzor_tensor_matmul_qt_t_f32_threaded",
            crate::quant::rayzor_tensor_matmul_qt_t_f32_threaded
        ),
        entry!(
            b"rayzor_tensor_matmul_qkv_qt_t_f32_threaded",
            crate::quant::rayzor_tensor_matmul_qkv_qt_t_f32_threaded
        ),
        entry!(
            b"rayzor_tensor_gather_rows_q6_k",
            crate::quant::rayzor_tensor_gather_rows_q6_k
        ),
        entry!(b"rayzor_qtensor_free", crate::quant::rayzor_qtensor_free),
        entry!(b"rayzor_qtensor_clone", crate::quant::rayzor_qtensor_clone),
        entry!(
            b"rayzor_qtensor_arc_clone",
            crate::quant::rayzor_qtensor_arc_clone
        ),
        entry!(
            b"rayzor_qtensor_deep_clone",
            crate::quant::rayzor_qtensor_deep_clone
        ),
    ]);
    if !out_count.is_null() {
        *out_count = entries.len();
    }
    Box::leak(entries).as_ptr()
}
