//! Runtime plugin implementation
//!
//! This registers all runtime functions as a plugin

/// Thread-safe function pointer wrapper
pub struct FunctionPtr(*const u8);

unsafe impl Send for FunctionPtr {}
unsafe impl Sync for FunctionPtr {}

impl FunctionPtr {
    pub const fn new(ptr: *const u8) -> Self {
        FunctionPtr(ptr)
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.0
    }
}

/// Runtime symbol for inventory-based registration
pub struct RuntimeSymbol {
    pub name: &'static str,
    pub ptr: FunctionPtr,
}

inventory::collect!(RuntimeSymbol);

/// Register a runtime symbol
macro_rules! register_symbol {
    ($name:expr, $func:path) => {
        inventory::submit! {
            RuntimeSymbol {
                name: $name,
                ptr: FunctionPtr::new($func as *const u8),
            }
        }
    };
}

// ============================================================================
// Vec Functions (Simple pointer-based API)
// ============================================================================
register_symbol!("haxe_vec_new_ptr", crate::vec_plugin::haxe_vec_new_ptr);
register_symbol!("haxe_vec_push_ptr", crate::vec_plugin::haxe_vec_push_ptr);
register_symbol!("haxe_vec_get_ptr", crate::vec_plugin::haxe_vec_get_ptr);
register_symbol!("haxe_vec_len_ptr", crate::vec_plugin::haxe_vec_len_ptr);
register_symbol!("haxe_vec_free_ptr", crate::vec_plugin::haxe_vec_free_ptr);

// ============================================================================
// String Functions (Comprehensive Haxe String API)
// ============================================================================

// Creation
register_symbol!("haxe_string_new", crate::haxe_string::haxe_string_new);
register_symbol!(
    "haxe_string_from_cstr",
    crate::haxe_string::haxe_string_from_cstr
);
register_symbol!(
    "haxe_string_from_bytes",
    crate::haxe_string::haxe_string_from_bytes
);

// Properties
register_symbol!("haxe_string_length", crate::haxe_string::haxe_string_length);
register_symbol!(
    "haxe_string_char_at",
    crate::haxe_string::haxe_string_char_at
);
register_symbol!(
    "haxe_string_char_code_at",
    crate::haxe_string::haxe_string_char_code_at
);

// Operations
// Use the pointer-returning version from string.rs to avoid struct return ABI issues
register_symbol!("haxe_string_concat", crate::string::haxe_string_concat_ptr);
register_symbol!(
    "haxe_string_substring",
    crate::haxe_string::haxe_string_substring
);
register_symbol!("haxe_string_substr", crate::haxe_string::haxe_string_substr);
register_symbol!(
    "haxe_string_to_upper_case",
    crate::haxe_string::haxe_string_to_upper_case
);
register_symbol!(
    "haxe_string_to_lower_case",
    crate::haxe_string::haxe_string_to_lower_case
);
register_symbol!(
    "haxe_string_index_of",
    crate::haxe_string::haxe_string_index_of
);
// Commented out old out-param version to avoid conflicts
// register_symbol!("haxe_string_split", crate::haxe_string::haxe_string_split);
register_symbol!(
    "haxe_string_split_array",
    crate::haxe_string::haxe_string_split_array
);
register_symbol!(
    "haxe_string_replace",
    crate::haxe_string::haxe_string_replace
);
register_symbol!(
    "haxe_string_starts_with",
    crate::string::haxe_string_starts_with
);
register_symbol!(
    "haxe_string_ends_with",
    crate::string::haxe_string_ends_with
);
register_symbol!("haxe_string_contains", crate::string::haxe_string_contains);

// Memory
register_symbol!("haxe_string_free", crate::haxe_string::haxe_string_free);

// I/O
register_symbol!("haxe_string_print", crate::haxe_string::haxe_string_print);
register_symbol!(
    "haxe_string_println",
    crate::haxe_string::haxe_string_println
);
register_symbol!(
    "haxe_string_to_cstr",
    crate::haxe_string::haxe_string_to_cstr
);

// ============================================================================
// Array Functions (Generic Dynamic Array)
// ============================================================================

// Creation
register_symbol!("haxe_array_new", crate::haxe_array::haxe_array_new);
register_symbol!(
    "haxe_array_from_elements",
    crate::haxe_array::haxe_array_from_elements
);

// Properties
register_symbol!("haxe_array_length", crate::haxe_array::haxe_array_length);

// Access
register_symbol!("haxe_array_get", crate::haxe_array::haxe_array_get);
register_symbol!("haxe_array_set", crate::haxe_array::haxe_array_set);
register_symbol!("haxe_array_set_i64", crate::haxe_array::haxe_array_set_i64);
register_symbol!("haxe_array_set_f64", crate::haxe_array::haxe_array_set_f64);
register_symbol!(
    "haxe_array_set_null",
    crate::haxe_array::haxe_array_set_null
);
register_symbol!("haxe_array_get_ptr", crate::haxe_array::haxe_array_get_ptr);

// Modification
register_symbol!("haxe_array_push", crate::haxe_array::haxe_array_push);
register_symbol!("haxe_array_pop", crate::haxe_array::haxe_array_pop);
register_symbol!("haxe_array_pop_i64", crate::haxe_array::haxe_array_pop_i64);
register_symbol!("haxe_array_pop_ptr", crate::haxe_array::haxe_array_pop_ptr);
register_symbol!("haxe_array_insert", crate::haxe_array::haxe_array_insert);
register_symbol!("haxe_array_remove", crate::haxe_array::haxe_array_remove);
register_symbol!("haxe_array_reverse", crate::haxe_array::haxe_array_reverse);

// Operations
register_symbol!("haxe_array_copy", crate::haxe_array::haxe_array_copy);
register_symbol!("haxe_array_slice", crate::haxe_array::haxe_array_slice);
register_symbol!("haxe_array_join", crate::haxe_array::haxe_array_join);

// Higher-order methods
register_symbol!("haxe_array_map", crate::haxe_array::haxe_array_map);
register_symbol!("haxe_array_filter", crate::haxe_array::haxe_array_filter);
register_symbol!("haxe_array_sort", crate::haxe_array::haxe_array_sort);

// Memory
register_symbol!("haxe_array_free", crate::haxe_array::haxe_array_free);

// Specialized integer operations
register_symbol!(
    "haxe_array_push_i32",
    crate::haxe_array::haxe_array_push_i32
);
register_symbol!("haxe_array_get_i32", crate::haxe_array::haxe_array_get_i32);
register_symbol!(
    "haxe_array_push_i64",
    crate::haxe_array::haxe_array_push_i64
);
register_symbol!("haxe_array_get_i64", crate::haxe_array::haxe_array_get_i64);
register_symbol!(
    "haxe_array_push_f64",
    crate::haxe_array::haxe_array_push_f64
);
register_symbol!("haxe_array_get_f64", crate::haxe_array::haxe_array_get_f64);

// ============================================================================
// Math Functions
// ============================================================================

// Constants
register_symbol!("haxe_math_pi", crate::haxe_math::haxe_math_pi);
register_symbol!("haxe_math_e", crate::haxe_math::haxe_math_e);

// Basic operations
register_symbol!("haxe_math_abs", crate::haxe_math::haxe_math_abs);
register_symbol!("haxe_math_min", crate::haxe_math::haxe_math_min);
register_symbol!("haxe_math_max", crate::haxe_math::haxe_math_max);
register_symbol!("haxe_math_floor", crate::haxe_math::haxe_math_floor);
register_symbol!("haxe_math_ceil", crate::haxe_math::haxe_math_ceil);
register_symbol!("haxe_math_round", crate::haxe_math::haxe_math_round);

// Trigonometric
register_symbol!("haxe_math_sin", crate::haxe_math::haxe_math_sin);
register_symbol!("haxe_math_cos", crate::haxe_math::haxe_math_cos);
register_symbol!("haxe_math_tan", crate::haxe_math::haxe_math_tan);
register_symbol!("haxe_math_asin", crate::haxe_math::haxe_math_asin);
register_symbol!("haxe_math_acos", crate::haxe_math::haxe_math_acos);
register_symbol!("haxe_math_atan", crate::haxe_math::haxe_math_atan);
register_symbol!("haxe_math_atan2", crate::haxe_math::haxe_math_atan2);

// Exponential and logarithmic
register_symbol!("haxe_math_exp", crate::haxe_math::haxe_math_exp);
register_symbol!("haxe_math_log", crate::haxe_math::haxe_math_log);
register_symbol!("haxe_math_pow", crate::haxe_math::haxe_math_pow);
register_symbol!("haxe_math_sqrt", crate::haxe_math::haxe_math_sqrt);

// Special
register_symbol!("haxe_math_is_nan", crate::haxe_math::haxe_math_is_nan);
register_symbol!("haxe_math_is_finite", crate::haxe_math::haxe_math_is_finite);
register_symbol!("haxe_math_random", crate::haxe_math::haxe_math_random);

// ============================================================================
// Sys Functions (System and I/O)
// ============================================================================

// Console I/O
register_symbol!("haxe_sys_print_int", crate::haxe_sys::haxe_sys_print_int);
register_symbol!(
    "haxe_sys_print_float",
    crate::haxe_sys::haxe_sys_print_float
);
register_symbol!("haxe_sys_print_bool", crate::haxe_sys::haxe_sys_print_bool);
register_symbol!("haxe_sys_println", crate::haxe_sys::haxe_sys_println);

// Trace (Runtime logging)
register_symbol!("haxe_trace_int", crate::haxe_sys::haxe_trace_int);
register_symbol!("haxe_trace_float", crate::haxe_sys::haxe_trace_float);
register_symbol!("haxe_trace_bool", crate::haxe_sys::haxe_trace_bool);
register_symbol!("haxe_trace_string", crate::haxe_sys::haxe_trace_string);
register_symbol!(
    "haxe_trace_string_struct",
    crate::haxe_sys::haxe_trace_string_struct
);
register_symbol!("haxe_trace_any", crate::haxe_sys::haxe_trace_any);

// Enum RTTI
register_symbol!("haxe_register_enum", crate::type_system::haxe_register_enum);
register_symbol!(
    "haxe_register_enum_start",
    crate::type_system::haxe_register_enum_start
);
register_symbol!(
    "haxe_register_enum_variant",
    crate::type_system::haxe_register_enum_variant
);
register_symbol!(
    "haxe_register_enum_finish",
    crate::type_system::haxe_register_enum_finish
);
register_symbol!(
    "haxe_enum_variant_name",
    crate::type_system::haxe_enum_variant_name
);
register_symbol!("haxe_trace_enum", crate::type_system::haxe_trace_enum);
register_symbol!(
    "haxe_trace_enum_boxed",
    crate::type_system::haxe_trace_enum_boxed
);
register_symbol!(
    "haxe_trace_enum_boxed_typed",
    crate::type_system::haxe_trace_enum_boxed_typed
);

// Std.string() - Type-specific conversions
register_symbol!(
    "haxe_string_from_int",
    crate::haxe_sys::haxe_string_from_int
);
register_symbol!(
    "haxe_string_from_float",
    crate::haxe_sys::haxe_string_from_float
);
register_symbol!(
    "haxe_string_from_bool",
    crate::haxe_sys::haxe_string_from_bool
);
register_symbol!(
    "haxe_string_from_string",
    crate::haxe_sys::haxe_string_from_string
);
register_symbol!(
    "haxe_string_from_null",
    crate::haxe_sys::haxe_string_from_null
);
register_symbol!("haxe_string_literal", crate::haxe_sys::haxe_string_literal);
register_symbol!("haxe_string_upper", crate::haxe_sys::haxe_string_upper);
register_symbol!("haxe_string_lower", crate::haxe_sys::haxe_string_lower);

// String class methods (working with *const HaxeString from haxe_sys)
// These use `_ptr` suffix to avoid conflicts with haxe_string.rs module
register_symbol!("haxe_string_len", crate::haxe_sys::haxe_string_len);
register_symbol!(
    "haxe_string_char_at_ptr",
    crate::haxe_sys::haxe_string_char_at_ptr
);
register_symbol!(
    "haxe_string_char_code_at_ptr",
    crate::haxe_sys::haxe_string_char_code_at_ptr
);
register_symbol!(
    "haxe_string_index_of_ptr",
    crate::haxe_sys::haxe_string_index_of_ptr
);
register_symbol!(
    "haxe_string_last_index_of_ptr",
    crate::haxe_sys::haxe_string_last_index_of_ptr
);
register_symbol!(
    "haxe_string_substr_ptr",
    crate::haxe_sys::haxe_string_substr_ptr
);
register_symbol!(
    "haxe_string_substring_ptr",
    crate::haxe_sys::haxe_string_substring_ptr
);
register_symbol!(
    "haxe_string_from_char_code",
    crate::haxe_sys::haxe_string_from_char_code
);
register_symbol!("haxe_string_copy", crate::haxe_sys::haxe_string_copy);
register_symbol!(
    "haxe_string_split_ptr",
    crate::haxe_sys::haxe_string_split_ptr
);

// Program control
register_symbol!("haxe_sys_exit", crate::haxe_sys::haxe_sys_exit);
register_symbol!("haxe_sys_time", crate::haxe_sys::haxe_sys_time);
register_symbol!("haxe_sys_args_count", crate::haxe_sys::haxe_sys_args_count);

// Environment
register_symbol!("haxe_sys_get_env", crate::haxe_sys::haxe_sys_get_env);
register_symbol!("haxe_sys_put_env", crate::haxe_sys::haxe_sys_put_env);

// Working directory
register_symbol!("haxe_sys_get_cwd", crate::haxe_sys::haxe_sys_get_cwd);
register_symbol!("haxe_sys_set_cwd", crate::haxe_sys::haxe_sys_set_cwd);

// Sleep
register_symbol!("haxe_sys_sleep", crate::haxe_sys::haxe_sys_sleep);

// System info
register_symbol!(
    "haxe_sys_system_name",
    crate::haxe_sys::haxe_sys_system_name
);
register_symbol!("haxe_sys_cpu_time", crate::haxe_sys::haxe_sys_cpu_time);
register_symbol!(
    "haxe_sys_program_path",
    crate::haxe_sys::haxe_sys_program_path
);

// Command execution
register_symbol!("haxe_sys_command", crate::haxe_sys::haxe_sys_command);
register_symbol!("haxe_sys_get_char", crate::haxe_sys::haxe_sys_get_char);

// ============================================================================
// File I/O (sys.io.File)
// ============================================================================
register_symbol!(
    "haxe_file_get_content",
    crate::haxe_sys::haxe_file_get_content
);
register_symbol!(
    "haxe_file_save_content",
    crate::haxe_sys::haxe_file_save_content
);
register_symbol!("haxe_file_copy", crate::haxe_sys::haxe_file_copy);
register_symbol!("haxe_file_read", crate::haxe_sys::haxe_file_read);
register_symbol!("haxe_file_write", crate::haxe_sys::haxe_file_write);
register_symbol!("haxe_file_append", crate::haxe_sys::haxe_file_append);
register_symbol!("haxe_file_update", crate::haxe_sys::haxe_file_update);
register_symbol!("haxe_file_get_bytes", crate::haxe_sys::haxe_file_get_bytes);
register_symbol!(
    "haxe_file_save_bytes",
    crate::haxe_sys::haxe_file_save_bytes
);

// ============================================================================
// FileInput (sys.io.FileInput)
// ============================================================================
register_symbol!(
    "haxe_fileinput_read_byte",
    crate::haxe_sys::haxe_fileinput_read_byte
);
register_symbol!(
    "haxe_fileinput_read_bytes",
    crate::haxe_sys::haxe_fileinput_read_bytes
);
register_symbol!("haxe_fileinput_seek", crate::haxe_sys::haxe_fileinput_seek);
register_symbol!("haxe_fileinput_tell", crate::haxe_sys::haxe_fileinput_tell);
register_symbol!("haxe_fileinput_eof", crate::haxe_sys::haxe_fileinput_eof);
register_symbol!(
    "haxe_fileinput_close",
    crate::haxe_sys::haxe_fileinput_close
);

// ============================================================================
// FileOutput (sys.io.FileOutput)
// ============================================================================
register_symbol!(
    "haxe_fileoutput_write_byte",
    crate::haxe_sys::haxe_fileoutput_write_byte
);
register_symbol!(
    "haxe_fileoutput_write_bytes",
    crate::haxe_sys::haxe_fileoutput_write_bytes
);
register_symbol!(
    "haxe_fileoutput_seek",
    crate::haxe_sys::haxe_fileoutput_seek
);
register_symbol!(
    "haxe_fileoutput_tell",
    crate::haxe_sys::haxe_fileoutput_tell
);
register_symbol!(
    "haxe_fileoutput_flush",
    crate::haxe_sys::haxe_fileoutput_flush
);
register_symbol!(
    "haxe_fileoutput_close",
    crate::haxe_sys::haxe_fileoutput_close
);

// ============================================================================
// FileSystem (sys.FileSystem)
// ============================================================================
register_symbol!(
    "haxe_filesystem_exists",
    crate::haxe_sys::haxe_filesystem_exists
);
register_symbol!(
    "haxe_filesystem_is_directory",
    crate::haxe_sys::haxe_filesystem_is_directory
);
register_symbol!(
    "haxe_filesystem_is_file",
    crate::haxe_sys::haxe_filesystem_is_file
);
register_symbol!(
    "haxe_filesystem_create_directory",
    crate::haxe_sys::haxe_filesystem_create_directory
);
register_symbol!(
    "haxe_filesystem_delete_file",
    crate::haxe_sys::haxe_filesystem_delete_file
);
register_symbol!(
    "haxe_filesystem_delete_directory",
    crate::haxe_sys::haxe_filesystem_delete_directory
);
register_symbol!(
    "haxe_filesystem_rename",
    crate::haxe_sys::haxe_filesystem_rename
);
register_symbol!(
    "haxe_filesystem_full_path",
    crate::haxe_sys::haxe_filesystem_full_path
);
register_symbol!(
    "haxe_filesystem_absolute_path",
    crate::haxe_sys::haxe_filesystem_absolute_path
);
register_symbol!(
    "haxe_filesystem_stat",
    crate::haxe_sys::haxe_filesystem_stat
);
register_symbol!(
    "haxe_filesystem_read_directory",
    crate::haxe_sys::haxe_filesystem_read_directory
);

// ============================================================================
// Date Class
// ============================================================================
register_symbol!("haxe_date_new", crate::haxe_sys::haxe_date_new);
register_symbol!("haxe_date_now", crate::haxe_sys::haxe_date_now);
register_symbol!("haxe_date_from_time", crate::haxe_sys::haxe_date_from_time);
register_symbol!(
    "haxe_date_from_string",
    crate::haxe_sys::haxe_date_from_string
);
register_symbol!("haxe_date_get_time", crate::haxe_sys::haxe_date_get_time);
register_symbol!("haxe_date_get_hours", crate::haxe_sys::haxe_date_get_hours);
register_symbol!(
    "haxe_date_get_minutes",
    crate::haxe_sys::haxe_date_get_minutes
);
register_symbol!(
    "haxe_date_get_seconds",
    crate::haxe_sys::haxe_date_get_seconds
);
register_symbol!(
    "haxe_date_get_full_year",
    crate::haxe_sys::haxe_date_get_full_year
);
register_symbol!("haxe_date_get_month", crate::haxe_sys::haxe_date_get_month);
register_symbol!("haxe_date_get_date", crate::haxe_sys::haxe_date_get_date);
register_symbol!("haxe_date_get_day", crate::haxe_sys::haxe_date_get_day);
register_symbol!(
    "haxe_date_get_utc_hours",
    crate::haxe_sys::haxe_date_get_utc_hours
);
register_symbol!(
    "haxe_date_get_utc_minutes",
    crate::haxe_sys::haxe_date_get_utc_minutes
);
register_symbol!(
    "haxe_date_get_utc_seconds",
    crate::haxe_sys::haxe_date_get_utc_seconds
);
register_symbol!(
    "haxe_date_get_utc_full_year",
    crate::haxe_sys::haxe_date_get_utc_full_year
);
register_symbol!(
    "haxe_date_get_utc_month",
    crate::haxe_sys::haxe_date_get_utc_month
);
register_symbol!(
    "haxe_date_get_utc_date",
    crate::haxe_sys::haxe_date_get_utc_date
);
register_symbol!(
    "haxe_date_get_utc_day",
    crate::haxe_sys::haxe_date_get_utc_day
);
register_symbol!(
    "haxe_date_get_timezone_offset",
    crate::haxe_sys::haxe_date_get_timezone_offset
);
register_symbol!("haxe_date_to_string", crate::haxe_sys::haxe_date_to_string);

// ============================================================================
// Bytes (rayzor.Bytes / haxe.io.Bytes)
// ============================================================================
register_symbol!("haxe_bytes_alloc", crate::haxe_sys::haxe_bytes_alloc);
register_symbol!(
    "haxe_bytes_of_string",
    crate::haxe_sys::haxe_bytes_of_string
);
register_symbol!("haxe_bytes_length", crate::haxe_sys::haxe_bytes_length);
register_symbol!("haxe_bytes_get", crate::haxe_sys::haxe_bytes_get);
register_symbol!("haxe_bytes_set", crate::haxe_sys::haxe_bytes_set);
register_symbol!("haxe_bytes_sub", crate::haxe_sys::haxe_bytes_sub);
register_symbol!("haxe_bytes_blit", crate::haxe_sys::haxe_bytes_blit);
register_symbol!("haxe_bytes_fill", crate::haxe_sys::haxe_bytes_fill);
register_symbol!("haxe_bytes_compare", crate::haxe_sys::haxe_bytes_compare);
register_symbol!(
    "haxe_bytes_to_string",
    crate::haxe_sys::haxe_bytes_to_string
);
register_symbol!(
    "haxe_bytes_get_int16",
    crate::haxe_sys::haxe_bytes_get_int16
);
register_symbol!(
    "haxe_bytes_get_int32",
    crate::haxe_sys::haxe_bytes_get_int32
);
register_symbol!(
    "haxe_bytes_get_int64",
    crate::haxe_sys::haxe_bytes_get_int64
);
register_symbol!(
    "haxe_bytes_get_float",
    crate::haxe_sys::haxe_bytes_get_float
);
register_symbol!(
    "haxe_bytes_get_double",
    crate::haxe_sys::haxe_bytes_get_double
);
register_symbol!(
    "haxe_bytes_set_int16",
    crate::haxe_sys::haxe_bytes_set_int16
);
register_symbol!(
    "haxe_bytes_set_int32",
    crate::haxe_sys::haxe_bytes_set_int32
);
register_symbol!(
    "haxe_bytes_set_int64",
    crate::haxe_sys::haxe_bytes_set_int64
);
register_symbol!(
    "haxe_bytes_set_float",
    crate::haxe_sys::haxe_bytes_set_float
);
register_symbol!(
    "haxe_bytes_set_double",
    crate::haxe_sys::haxe_bytes_set_double
);
register_symbol!("haxe_bytes_free", crate::haxe_sys::haxe_bytes_free);

// ============================================================================
// StringMap<T> (haxe.ds.StringMap)
// ============================================================================
register_symbol!("haxe_stringmap_new", crate::haxe_sys::haxe_stringmap_new);
register_symbol!("haxe_stringmap_set", crate::haxe_sys::haxe_stringmap_set);
register_symbol!("haxe_stringmap_get", crate::haxe_sys::haxe_stringmap_get);
register_symbol!(
    "haxe_stringmap_exists",
    crate::haxe_sys::haxe_stringmap_exists
);
register_symbol!(
    "haxe_stringmap_remove",
    crate::haxe_sys::haxe_stringmap_remove
);
register_symbol!(
    "haxe_stringmap_clear",
    crate::haxe_sys::haxe_stringmap_clear
);
register_symbol!(
    "haxe_stringmap_count",
    crate::haxe_sys::haxe_stringmap_count
);
register_symbol!("haxe_stringmap_keys", crate::haxe_sys::haxe_stringmap_keys);
register_symbol!(
    "haxe_stringmap_to_string",
    crate::haxe_sys::haxe_stringmap_to_string
);

// ============================================================================
// IntMap<T> (haxe.ds.IntMap)
// ============================================================================
register_symbol!("haxe_intmap_new", crate::haxe_sys::haxe_intmap_new);
register_symbol!("haxe_intmap_set", crate::haxe_sys::haxe_intmap_set);
register_symbol!("haxe_intmap_get", crate::haxe_sys::haxe_intmap_get);
register_symbol!("haxe_intmap_exists", crate::haxe_sys::haxe_intmap_exists);
register_symbol!("haxe_intmap_remove", crate::haxe_sys::haxe_intmap_remove);
register_symbol!("haxe_intmap_clear", crate::haxe_sys::haxe_intmap_clear);
register_symbol!("haxe_intmap_count", crate::haxe_sys::haxe_intmap_count);
register_symbol!("haxe_intmap_keys", crate::haxe_sys::haxe_intmap_keys);
register_symbol!(
    "haxe_intmap_to_string",
    crate::haxe_sys::haxe_intmap_to_string
);

// ============================================================================
// Type System (Dynamic values and Std.string)
// ============================================================================

// Boxing functions: Convert concrete values to Dynamic
register_symbol!("haxe_box_int", crate::type_system::haxe_box_int);
register_symbol!("haxe_box_float", crate::type_system::haxe_box_float);
register_symbol!("haxe_box_bool", crate::type_system::haxe_box_bool);
register_symbol!("haxe_box_string", crate::type_system::haxe_box_string);
register_symbol!("haxe_box_null", crate::type_system::haxe_box_null);

// Unboxing functions: Extract concrete values from Dynamic
register_symbol!("haxe_unbox_int", crate::type_system::haxe_unbox_int);
register_symbol!("haxe_unbox_float", crate::type_system::haxe_unbox_float);
register_symbol!("haxe_unbox_bool", crate::type_system::haxe_unbox_bool);
register_symbol!("haxe_unbox_string", crate::type_system::haxe_unbox_string);

// Pointer-based boxing/unboxing for MIR (simpler ABI)
register_symbol!("haxe_box_int_ptr", crate::type_system::haxe_box_int_ptr);
register_symbol!("haxe_box_float_ptr", crate::type_system::haxe_box_float_ptr);
register_symbol!("haxe_box_bool_ptr", crate::type_system::haxe_box_bool_ptr);
register_symbol!("haxe_unbox_int_ptr", crate::type_system::haxe_unbox_int_ptr);
register_symbol!(
    "haxe_unbox_float_ptr",
    crate::type_system::haxe_unbox_float_ptr
);
register_symbol!(
    "haxe_unbox_bool_ptr",
    crate::type_system::haxe_unbox_bool_ptr
);

// Reference type boxing/unboxing (Classes, Enums, Anonymous, Arrays, etc.)
register_symbol!(
    "haxe_box_reference_ptr",
    crate::type_system::haxe_box_reference_ptr
);
register_symbol!(
    "haxe_unbox_reference_ptr",
    crate::type_system::haxe_unbox_reference_ptr
);

// Std class functions
register_symbol!("haxe_std_string", crate::type_system::haxe_std_string);
register_symbol!(
    "haxe_std_string_ptr",
    crate::type_system::haxe_std_string_ptr
);
register_symbol!("haxe_std_int", crate::type_system::haxe_std_int);
register_symbol!("haxe_std_parse_int", crate::type_system::haxe_std_parse_int);
register_symbol!(
    "haxe_std_parse_float",
    crate::type_system::haxe_std_parse_float
);
register_symbol!("haxe_std_random", crate::type_system::haxe_std_random);

// Memory management for Dynamic values
register_symbol!("haxe_free_dynamic", crate::type_system::haxe_free_dynamic);

// ============================================================================
// Concurrency Functions (Thread, Arc, Mutex, Channel)
// ============================================================================

// Thread functions
register_symbol!(
    "rayzor_thread_spawn",
    crate::concurrency::rayzor_thread_spawn
);
register_symbol!("rayzor_thread_join", crate::concurrency::rayzor_thread_join);
register_symbol!(
    "rayzor_thread_is_finished",
    crate::concurrency::rayzor_thread_is_finished
);
register_symbol!(
    "rayzor_thread_yield_now",
    crate::concurrency::rayzor_thread_yield_now
);
register_symbol!(
    "rayzor_thread_sleep",
    crate::concurrency::rayzor_thread_sleep
);
register_symbol!(
    "rayzor_thread_current_id",
    crate::concurrency::rayzor_thread_current_id
);
// Thread tracking for JIT safety
register_symbol!(
    "rayzor_wait_all_threads",
    crate::concurrency::rayzor_wait_all_threads
);
register_symbol!(
    "rayzor_active_thread_count",
    crate::concurrency::rayzor_active_thread_count
);

// Arc functions
register_symbol!("rayzor_arc_init", crate::concurrency::rayzor_arc_init);
register_symbol!("rayzor_arc_clone", crate::concurrency::rayzor_arc_clone);
register_symbol!("rayzor_arc_get", crate::concurrency::rayzor_arc_get);
register_symbol!(
    "rayzor_arc_strong_count",
    crate::concurrency::rayzor_arc_strong_count
);
register_symbol!(
    "rayzor_arc_try_unwrap",
    crate::concurrency::rayzor_arc_try_unwrap
);
register_symbol!("rayzor_arc_as_ptr", crate::concurrency::rayzor_arc_as_ptr);

// Box<T> functions
register_symbol!("rayzor_box_init", crate::box_runtime::rayzor_box_init);
register_symbol!("rayzor_box_unbox", crate::box_runtime::rayzor_box_unbox);
register_symbol!("rayzor_box_raw", crate::box_runtime::rayzor_box_raw);
register_symbol!("rayzor_box_free", crate::box_runtime::rayzor_box_free);

// TinyCC runtime compiler functions
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_create",
    crate::tinycc_runtime::rayzor_tcc_create
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_compile",
    crate::tinycc_runtime::rayzor_tcc_compile
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_add_symbol",
    crate::tinycc_runtime::rayzor_tcc_add_symbol
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_add_value_symbol",
    crate::tinycc_runtime::rayzor_tcc_add_value_symbol
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_free_value",
    crate::tinycc_runtime::rayzor_tcc_free_value
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_relocate",
    crate::tinycc_runtime::rayzor_tcc_relocate
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_get_symbol",
    crate::tinycc_runtime::rayzor_tcc_get_symbol
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_delete",
    crate::tinycc_runtime::rayzor_tcc_delete
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_add_framework",
    crate::tinycc_runtime::rayzor_tcc_add_framework
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_add_include_path",
    crate::tinycc_runtime::rayzor_tcc_add_include_path
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_add_file",
    crate::tinycc_runtime::rayzor_tcc_add_file
);
#[cfg(feature = "tcc-runtime")]
register_symbol!(
    "rayzor_tcc_add_clib",
    crate::tinycc_runtime::rayzor_tcc_add_clib
);
#[cfg(feature = "tcc-runtime")]
register_symbol!("rayzor_tcc_call0", crate::tinycc_runtime::rayzor_tcc_call0);
#[cfg(feature = "tcc-runtime")]
register_symbol!("rayzor_tcc_call1", crate::tinycc_runtime::rayzor_tcc_call1);
#[cfg(feature = "tcc-runtime")]
register_symbol!("rayzor_tcc_call2", crate::tinycc_runtime::rayzor_tcc_call2);
#[cfg(feature = "tcc-runtime")]
register_symbol!("rayzor_tcc_call3", crate::tinycc_runtime::rayzor_tcc_call3);

// Mutex functions
register_symbol!("rayzor_mutex_init", crate::concurrency::rayzor_mutex_init);
register_symbol!("rayzor_mutex_lock", crate::concurrency::rayzor_mutex_lock);
register_symbol!(
    "rayzor_mutex_try_lock",
    crate::concurrency::rayzor_mutex_try_lock
);
register_symbol!(
    "rayzor_mutex_is_locked",
    crate::concurrency::rayzor_mutex_is_locked
);
register_symbol!(
    "rayzor_mutex_guard_get",
    crate::concurrency::rayzor_mutex_guard_get
);
register_symbol!(
    "rayzor_mutex_unlock",
    crate::concurrency::rayzor_mutex_unlock
);

// Channel functions
register_symbol!(
    "rayzor_channel_init",
    crate::concurrency::rayzor_channel_init
);
register_symbol!(
    "rayzor_channel_send",
    crate::concurrency::rayzor_channel_send
);
register_symbol!(
    "rayzor_channel_try_send",
    crate::concurrency::rayzor_channel_try_send
);
register_symbol!(
    "rayzor_channel_receive",
    crate::concurrency::rayzor_channel_receive
);
register_symbol!(
    "rayzor_channel_try_receive",
    crate::concurrency::rayzor_channel_try_receive
);
register_symbol!(
    "rayzor_channel_close",
    crate::concurrency::rayzor_channel_close
);
register_symbol!(
    "rayzor_channel_is_closed",
    crate::concurrency::rayzor_channel_is_closed
);
register_symbol!("rayzor_channel_len", crate::concurrency::rayzor_channel_len);
register_symbol!(
    "rayzor_channel_capacity",
    crate::concurrency::rayzor_channel_capacity
);
register_symbol!(
    "rayzor_channel_is_empty",
    crate::concurrency::rayzor_channel_is_empty
);
register_symbol!(
    "rayzor_channel_is_full",
    crate::concurrency::rayzor_channel_is_full
);

// Semaphore functions (for sys.thread.Lock and sys.thread.Semaphore)
register_symbol!(
    "rayzor_semaphore_init",
    crate::concurrency::rayzor_semaphore_init
);
register_symbol!(
    "rayzor_semaphore_acquire",
    crate::concurrency::rayzor_semaphore_acquire
);
register_symbol!(
    "rayzor_semaphore_try_acquire",
    crate::concurrency::rayzor_semaphore_try_acquire
);
register_symbol!(
    "rayzor_semaphore_release",
    crate::concurrency::rayzor_semaphore_release
);
register_symbol!(
    "rayzor_semaphore_count",
    crate::concurrency::rayzor_semaphore_count
);

// sys.thread.Lock wrapper functions
register_symbol!("sys_lock_wait", crate::concurrency::sys_lock_wait);

// sys.thread.Semaphore wrapper functions
register_symbol!(
    "sys_semaphore_try_acquire_nowait",
    crate::concurrency::sys_semaphore_try_acquire_nowait
);

// sys.thread.Thread wrapper functions
register_symbol!("sys_thread_create", crate::concurrency::sys_thread_create);
register_symbol!("sys_thread_join", crate::concurrency::sys_thread_join);
register_symbol!(
    "sys_thread_is_finished",
    crate::concurrency::sys_thread_is_finished
);
register_symbol!("sys_thread_yield", crate::concurrency::sys_thread_yield);
register_symbol!("sys_thread_sleep", crate::concurrency::sys_thread_sleep);
register_symbol!("sys_thread_current", crate::concurrency::sys_thread_current);

// sys.thread.Mutex wrapper functions
register_symbol!("sys_mutex_alloc", crate::concurrency::sys_mutex_alloc);
register_symbol!("sys_mutex_acquire", crate::concurrency::sys_mutex_acquire);
register_symbol!(
    "sys_mutex_try_acquire",
    crate::concurrency::sys_mutex_try_acquire
);
register_symbol!("sys_mutex_release", crate::concurrency::sys_mutex_release);

// sys.thread.Deque<T>
register_symbol!("sys_deque_alloc", crate::concurrency::sys_deque_alloc);
register_symbol!("sys_deque_add", crate::concurrency::sys_deque_add);
register_symbol!("sys_deque_push", crate::concurrency::sys_deque_push);
register_symbol!("sys_deque_pop", crate::concurrency::sys_deque_pop);

// sys.thread.Condition
register_symbol!(
    "sys_condition_alloc",
    crate::concurrency::sys_condition_alloc
);
register_symbol!(
    "sys_condition_acquire",
    crate::concurrency::sys_condition_acquire
);
register_symbol!(
    "sys_condition_try_acquire",
    crate::concurrency::sys_condition_try_acquire
);
register_symbol!(
    "sys_condition_release",
    crate::concurrency::sys_condition_release
);
register_symbol!("sys_condition_wait", crate::concurrency::sys_condition_wait);
register_symbol!(
    "sys_condition_signal",
    crate::concurrency::sys_condition_signal
);
register_symbol!(
    "sys_condition_broadcast",
    crate::concurrency::sys_condition_broadcast
);

// ============================================================================
// Generic Vec<T> Functions
// ============================================================================

// Vec<Int> -> VecI32
register_symbol!("rayzor_vec_i32_new", crate::generic_vec::rayzor_vec_i32_new);
register_symbol!(
    "rayzor_vec_i32_with_capacity",
    crate::generic_vec::rayzor_vec_i32_with_capacity
);
register_symbol!(
    "rayzor_vec_i32_push",
    crate::generic_vec::rayzor_vec_i32_push
);
register_symbol!("rayzor_vec_i32_pop", crate::generic_vec::rayzor_vec_i32_pop);
register_symbol!("rayzor_vec_i32_get", crate::generic_vec::rayzor_vec_i32_get);
register_symbol!("rayzor_vec_i32_set", crate::generic_vec::rayzor_vec_i32_set);
register_symbol!("rayzor_vec_i32_len", crate::generic_vec::rayzor_vec_i32_len);
register_symbol!(
    "rayzor_vec_i32_capacity",
    crate::generic_vec::rayzor_vec_i32_capacity
);
register_symbol!(
    "rayzor_vec_i32_is_empty",
    crate::generic_vec::rayzor_vec_i32_is_empty
);
register_symbol!(
    "rayzor_vec_i32_clear",
    crate::generic_vec::rayzor_vec_i32_clear
);
register_symbol!(
    "rayzor_vec_i32_first",
    crate::generic_vec::rayzor_vec_i32_first
);
register_symbol!(
    "rayzor_vec_i32_last",
    crate::generic_vec::rayzor_vec_i32_last
);
register_symbol!(
    "rayzor_vec_i32_sort",
    crate::generic_vec::rayzor_vec_i32_sort
);
register_symbol!(
    "rayzor_vec_i32_sort_by",
    crate::generic_vec::rayzor_vec_i32_sort_by
);
register_symbol!(
    "rayzor_vec_i32_free",
    crate::generic_vec::rayzor_vec_i32_free
);

// Vec<Int64> -> VecI64
register_symbol!("rayzor_vec_i64_new", crate::generic_vec::rayzor_vec_i64_new);
register_symbol!(
    "rayzor_vec_i64_push",
    crate::generic_vec::rayzor_vec_i64_push
);
register_symbol!("rayzor_vec_i64_pop", crate::generic_vec::rayzor_vec_i64_pop);
register_symbol!("rayzor_vec_i64_get", crate::generic_vec::rayzor_vec_i64_get);
register_symbol!("rayzor_vec_i64_set", crate::generic_vec::rayzor_vec_i64_set);
register_symbol!("rayzor_vec_i64_len", crate::generic_vec::rayzor_vec_i64_len);
register_symbol!(
    "rayzor_vec_i64_is_empty",
    crate::generic_vec::rayzor_vec_i64_is_empty
);
register_symbol!(
    "rayzor_vec_i64_clear",
    crate::generic_vec::rayzor_vec_i64_clear
);
register_symbol!(
    "rayzor_vec_i64_first",
    crate::generic_vec::rayzor_vec_i64_first
);
register_symbol!(
    "rayzor_vec_i64_last",
    crate::generic_vec::rayzor_vec_i64_last
);
register_symbol!(
    "rayzor_vec_i64_free",
    crate::generic_vec::rayzor_vec_i64_free
);

// Vec<Float> -> VecF64
register_symbol!("rayzor_vec_f64_new", crate::generic_vec::rayzor_vec_f64_new);
register_symbol!(
    "rayzor_vec_f64_push",
    crate::generic_vec::rayzor_vec_f64_push
);
register_symbol!("rayzor_vec_f64_pop", crate::generic_vec::rayzor_vec_f64_pop);
register_symbol!("rayzor_vec_f64_get", crate::generic_vec::rayzor_vec_f64_get);
register_symbol!("rayzor_vec_f64_set", crate::generic_vec::rayzor_vec_f64_set);
register_symbol!("rayzor_vec_f64_len", crate::generic_vec::rayzor_vec_f64_len);
register_symbol!(
    "rayzor_vec_f64_is_empty",
    crate::generic_vec::rayzor_vec_f64_is_empty
);
register_symbol!(
    "rayzor_vec_f64_clear",
    crate::generic_vec::rayzor_vec_f64_clear
);
register_symbol!(
    "rayzor_vec_f64_first",
    crate::generic_vec::rayzor_vec_f64_first
);
register_symbol!(
    "rayzor_vec_f64_last",
    crate::generic_vec::rayzor_vec_f64_last
);
register_symbol!(
    "rayzor_vec_f64_sort",
    crate::generic_vec::rayzor_vec_f64_sort
);
register_symbol!(
    "rayzor_vec_f64_sort_by",
    crate::generic_vec::rayzor_vec_f64_sort_by
);
register_symbol!(
    "rayzor_vec_f64_free",
    crate::generic_vec::rayzor_vec_f64_free
);

// Vec<T> (reference types) -> VecPtr
register_symbol!("rayzor_vec_ptr_new", crate::generic_vec::rayzor_vec_ptr_new);
register_symbol!(
    "rayzor_vec_ptr_push",
    crate::generic_vec::rayzor_vec_ptr_push
);
register_symbol!("rayzor_vec_ptr_pop", crate::generic_vec::rayzor_vec_ptr_pop);
register_symbol!("rayzor_vec_ptr_get", crate::generic_vec::rayzor_vec_ptr_get);
register_symbol!("rayzor_vec_ptr_set", crate::generic_vec::rayzor_vec_ptr_set);
register_symbol!("rayzor_vec_ptr_len", crate::generic_vec::rayzor_vec_ptr_len);
register_symbol!(
    "rayzor_vec_ptr_is_empty",
    crate::generic_vec::rayzor_vec_ptr_is_empty
);
register_symbol!(
    "rayzor_vec_ptr_clear",
    crate::generic_vec::rayzor_vec_ptr_clear
);
register_symbol!(
    "rayzor_vec_ptr_first",
    crate::generic_vec::rayzor_vec_ptr_first
);
register_symbol!(
    "rayzor_vec_ptr_last",
    crate::generic_vec::rayzor_vec_ptr_last
);
register_symbol!(
    "rayzor_vec_ptr_sort_by",
    crate::generic_vec::rayzor_vec_ptr_sort_by
);
register_symbol!(
    "rayzor_vec_ptr_free",
    crate::generic_vec::rayzor_vec_ptr_free
);

// Vec<Bool> -> VecBool
register_symbol!(
    "rayzor_vec_bool_new",
    crate::generic_vec::rayzor_vec_bool_new
);
register_symbol!(
    "rayzor_vec_bool_push",
    crate::generic_vec::rayzor_vec_bool_push
);
register_symbol!(
    "rayzor_vec_bool_pop",
    crate::generic_vec::rayzor_vec_bool_pop
);
register_symbol!(
    "rayzor_vec_bool_get",
    crate::generic_vec::rayzor_vec_bool_get
);
register_symbol!(
    "rayzor_vec_bool_set",
    crate::generic_vec::rayzor_vec_bool_set
);
register_symbol!(
    "rayzor_vec_bool_len",
    crate::generic_vec::rayzor_vec_bool_len
);
register_symbol!(
    "rayzor_vec_bool_is_empty",
    crate::generic_vec::rayzor_vec_bool_is_empty
);
register_symbol!(
    "rayzor_vec_bool_clear",
    crate::generic_vec::rayzor_vec_bool_clear
);
register_symbol!(
    "rayzor_vec_bool_free",
    crate::generic_vec::rayzor_vec_bool_free
);

// ============================================================================
// Anonymous Object Functions (Arc-based, COW)
// ============================================================================
register_symbol!(
    "rayzor_ensure_shape",
    crate::anon_object::rayzor_ensure_shape
);
register_symbol!("rayzor_anon_new", crate::anon_object::rayzor_anon_new);
register_symbol!("rayzor_anon_clone", crate::anon_object::rayzor_anon_clone);
register_symbol!("rayzor_anon_drop", crate::anon_object::rayzor_anon_drop);
register_symbol!(
    "rayzor_anon_get_field_by_index",
    crate::anon_object::rayzor_anon_get_field_by_index
);
register_symbol!(
    "rayzor_anon_set_field_by_index",
    crate::anon_object::rayzor_anon_set_field_by_index
);
register_symbol!(
    "rayzor_anon_has_field",
    crate::anon_object::rayzor_anon_has_field
);
register_symbol!(
    "rayzor_anon_get_field",
    crate::anon_object::rayzor_anon_get_field
);
register_symbol!(
    "rayzor_anon_set_field",
    crate::anon_object::rayzor_anon_set_field
);
register_symbol!(
    "rayzor_anon_delete_field",
    crate::anon_object::rayzor_anon_delete_field
);
register_symbol!("rayzor_anon_fields", crate::anon_object::rayzor_anon_fields);
register_symbol!("rayzor_anon_copy", crate::anon_object::rayzor_anon_copy);
register_symbol!(
    "rayzor_register_shape",
    crate::anon_object::rayzor_register_shape
);

// ============================================================================
// Reflect + Type API
// ============================================================================
register_symbol!(
    "haxe_reflect_has_field",
    crate::reflect::haxe_reflect_has_field
);
register_symbol!("haxe_reflect_field", crate::reflect::haxe_reflect_field);
register_symbol!(
    "haxe_reflect_set_field",
    crate::reflect::haxe_reflect_set_field
);
register_symbol!(
    "haxe_reflect_delete_field",
    crate::reflect::haxe_reflect_delete_field
);
register_symbol!("haxe_reflect_fields", crate::reflect::haxe_reflect_fields);
register_symbol!(
    "haxe_reflect_is_object",
    crate::reflect::haxe_reflect_is_object
);
register_symbol!(
    "haxe_reflect_is_function",
    crate::reflect::haxe_reflect_is_function
);
register_symbol!("haxe_reflect_copy", crate::reflect::haxe_reflect_copy);
register_symbol!("haxe_type_typeof", crate::reflect::haxe_type_typeof);

// ============================================================================
// Memory Allocation (libc malloc/free for heap allocations)
// ============================================================================
register_symbol!("malloc", libc::malloc);
register_symbol!("free", libc::free);

// ============================================================================
// Global Variable Storage (for static class fields)
// ============================================================================
register_symbol!("rayzor_global_store", crate::rayzor_global_store);
register_symbol!("rayzor_global_load", crate::rayzor_global_load);

// ============================================================================
// Tracked Heap Allocator (Rust allocator with double-free protection)
// ============================================================================
register_symbol!("rayzor_tracked_alloc", crate::rayzor_tracked_alloc);
register_symbol!("rayzor_tracked_realloc", crate::rayzor_tracked_realloc);
register_symbol!("rayzor_tracked_free", crate::rayzor_tracked_free);

// ============================================================================
// CString Runtime (rayzor.CString — null-terminated C string interop)
// ============================================================================
register_symbol!(
    "rayzor_cstring_from",
    crate::cstring_runtime::rayzor_cstring_from
);
register_symbol!(
    "rayzor_cstring_to_string",
    crate::cstring_runtime::rayzor_cstring_to_string
);
register_symbol!(
    "rayzor_cstring_free",
    crate::cstring_runtime::rayzor_cstring_free
);

// ============================================================================
// Tensor Runtime (rayzor.ds.Tensor)
// ============================================================================

// Construction
register_symbol!("rayzor_tensor_zeros", crate::tensor::rayzor_tensor_zeros);
register_symbol!("rayzor_tensor_ones", crate::tensor::rayzor_tensor_ones);
register_symbol!("rayzor_tensor_full", crate::tensor::rayzor_tensor_full);
register_symbol!(
    "rayzor_tensor_from_array",
    crate::tensor::rayzor_tensor_from_array
);
register_symbol!("rayzor_tensor_rand", crate::tensor::rayzor_tensor_rand);

// Properties
register_symbol!("rayzor_tensor_shape", crate::tensor::rayzor_tensor_shape);
register_symbol!("rayzor_tensor_ndim", crate::tensor::rayzor_tensor_ndim);
register_symbol!("rayzor_tensor_numel", crate::tensor::rayzor_tensor_numel);
register_symbol!("rayzor_tensor_dtype", crate::tensor::rayzor_tensor_dtype);
register_symbol!(
    "rayzor_tensor_shape_ptr",
    crate::tensor::rayzor_tensor_shape_ptr
);
register_symbol!(
    "rayzor_tensor_shape_ndim",
    crate::tensor::rayzor_tensor_shape_ndim
);

// Element access
register_symbol!("rayzor_tensor_get", crate::tensor::rayzor_tensor_get);
register_symbol!("rayzor_tensor_set", crate::tensor::rayzor_tensor_set);

// Reshape / Transpose
register_symbol!(
    "rayzor_tensor_reshape",
    crate::tensor::rayzor_tensor_reshape
);
register_symbol!(
    "rayzor_tensor_transpose",
    crate::tensor::rayzor_tensor_transpose
);

// Arithmetic
register_symbol!("rayzor_tensor_add", crate::tensor::rayzor_tensor_add);
register_symbol!("rayzor_tensor_sub", crate::tensor::rayzor_tensor_sub);
register_symbol!("rayzor_tensor_mul", crate::tensor::rayzor_tensor_mul);
register_symbol!("rayzor_tensor_div", crate::tensor::rayzor_tensor_div);

// Math
register_symbol!("rayzor_tensor_sqrt", crate::tensor::rayzor_tensor_sqrt);
register_symbol!("rayzor_tensor_exp", crate::tensor::rayzor_tensor_exp);
register_symbol!("rayzor_tensor_log", crate::tensor::rayzor_tensor_log);
register_symbol!("rayzor_tensor_relu", crate::tensor::rayzor_tensor_relu);

// Reductions
register_symbol!("rayzor_tensor_sum", crate::tensor::rayzor_tensor_sum);
register_symbol!("rayzor_tensor_mean", crate::tensor::rayzor_tensor_mean);
register_symbol!("rayzor_tensor_dot", crate::tensor::rayzor_tensor_dot);

// Linear algebra
register_symbol!("rayzor_tensor_matmul", crate::tensor::rayzor_tensor_matmul);

// Interop
register_symbol!("rayzor_tensor_data", crate::tensor::rayzor_tensor_data);
register_symbol!("rayzor_tensor_free", crate::tensor::rayzor_tensor_free);

// ============================================================================
// Exception Handling (setjmp/longjmp)
// ============================================================================
register_symbol!(
    "rayzor_exception_push_handler",
    crate::exception::rayzor_exception_push_handler
);
register_symbol!(
    "rayzor_exception_pop_handler",
    crate::exception::rayzor_exception_pop_handler
);
register_symbol!("rayzor_throw", crate::exception::rayzor_throw);
register_symbol!(
    "rayzor_get_exception",
    crate::exception::rayzor_get_exception
);

// ============================================================================
// EReg Functions (Regular Expressions)
// ============================================================================
register_symbol!("haxe_ereg_new", crate::ereg::haxe_ereg_new);
register_symbol!("haxe_ereg_match", crate::ereg::haxe_ereg_match);
register_symbol!("haxe_ereg_matched", crate::ereg::haxe_ereg_matched);
register_symbol!(
    "haxe_ereg_matched_left",
    crate::ereg::haxe_ereg_matched_left
);
register_symbol!(
    "haxe_ereg_matched_right",
    crate::ereg::haxe_ereg_matched_right
);
register_symbol!("haxe_ereg_matched_pos", crate::ereg::haxe_ereg_matched_pos);
register_symbol!("haxe_ereg_match_sub", crate::ereg::haxe_ereg_match_sub);
register_symbol!("haxe_ereg_split", crate::ereg::haxe_ereg_split);
register_symbol!("haxe_ereg_replace", crate::ereg::haxe_ereg_replace);
register_symbol!("haxe_ereg_escape", crate::ereg::haxe_ereg_escape);

// ============================================================================
// TinyCC Runtime API (rayzor.runtime.CC) — registered above

/// Rayzor Runtime Plugin
pub struct RayzorRuntimePlugin;

impl RayzorRuntimePlugin {
    pub fn new() -> Self {
        RayzorRuntimePlugin
    }
}

impl Default for RayzorRuntimePlugin {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the runtime plugin instance
pub fn get_plugin() -> Box<dyn rayzor_plugin::RuntimePlugin> {
    Box::new(RayzorRuntimePlugin)
}

// Note: We manually implement RuntimePlugin instead of using proc macros
// to avoid the complexity of creating a separate proc-macro crate for now

impl rayzor_plugin::RuntimePlugin for RayzorRuntimePlugin {
    fn name(&self) -> &str {
        "rayzor_runtime"
    }

    fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
        inventory::iter::<RuntimeSymbol>
            .into_iter()
            .map(|sym| (sym.name, sym.ptr.as_ptr()))
            .collect()
    }
}
