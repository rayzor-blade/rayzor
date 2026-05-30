// Crate-level lint allows for clippy compliance
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_parens)]
#![allow(unused_doc_comments)]
#![allow(dead_code)]
#![allow(unreachable_patterns)]
#![allow(unused_comparisons)]
#![allow(duplicate_macro_attributes)]
#![allow(mismatched_lifetime_syntaxes)]
// Clippy lints
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(clippy::approx_constant)]
#![allow(clippy::bind_instead_of_map)]
#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::collapsible_str_replace)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::double_ended_iterator_last)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::explicit_auto_deref)]
#![allow(clippy::extra_unused_lifetimes)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::get_first)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::inherent_to_string)]
#![allow(clippy::items_after_test_module)]
#![allow(clippy::iter_overeager_cloned)]
#![allow(clippy::just_underscores_and_digits)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::len_zero)]
#![allow(clippy::let_and_return)]
#![allow(clippy::let_unit_value)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_find)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_map)]
#![allow(clippy::manual_range_patterns)]
#![allow(clippy::manual_strip)]
#![allow(clippy::map_entry)]
#![allow(clippy::map_flatten)]
#![allow(clippy::map_identity)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::match_single_binding)]
#![allow(clippy::missing_const_for_thread_local)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_return)]
#![allow(clippy::new_without_default)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::option_map_unit_fn)]
#![allow(clippy::println_empty_string)]
#![allow(clippy::question_mark)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::redundant_field_names)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::result_large_err)]
#![allow(clippy::search_is_some)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::single_component_path_imports)]
#![allow(clippy::single_match)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::unnecessary_mut_passed)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::unneeded_struct_pattern)]
#![allow(clippy::unused_enumerate_index)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::useless_format)]
#![allow(clippy::useless_vec)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::while_let_on_iterator)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::manual_unwrap_or_default)]
#![allow(clippy::manual_range_contains)]
#![allow(elided_lifetimes_in_paths)]

pub mod codegen;
pub mod compilation;
pub mod compiler_plugin; // Compiler-level plugin system for stdlib method mappings
pub mod dependency_graph;
pub mod error_codes;
pub mod hxml;
pub mod ir;
pub mod logging;
pub mod macro_system;
pub mod pipeline;
pub mod rpkg; // RPKG package format (native package distribution)
pub mod semantic_graph;
pub mod stdlib; // MIR-based standard library
pub mod tast;
pub mod tools;
pub mod workspace;

// Re-export plugin system from separate crate (avoids cyclic dependency)
pub use rayzor_plugin as plugin;

/// Build ID of the compiler binary, set from `RAYZOR_BUILD_ID` env var at
/// build time (see `compiler/build.rs` — UNIX-epoch second tick). Exposed so
/// downstream binaries (e.g. `rayzor` top-level CLI) can fold it into their
/// own cache keys without needing to thread `env!` through a separate
/// `build.rs`. Matches the value used by `.blade` per-module caches in
/// `compilation.rs`.
pub const BUILD_ID: &str = env!("RAYZOR_BUILD_ID");

// #[cfg(test)]
// mod pipeline_test;

pub mod pipeline_validation;
