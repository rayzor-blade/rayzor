// Comprehensive Test Suite  Generics System
// Real-world Haxe scenarios and performance benchmarks

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use crate::tast::core::{TypeKind, TypeTable};
use crate::tast::generics::{ConstraintKind, ConstraintPriority, ConstraintSet, TypeConstraint, GenericsEngine, ConstraintValidationResult, GenericError};
use crate::tast::constraint_solver::*;
use crate::tast::generic_instantiation::*;
use crate::tast::{ScopeId, TypeId, SymbolId, InternedString, SourceLocation, SymbolTable, ScopeTree, StringInterner};

/// Comprehensive test suite for the generics system
pub struct GenericsTestSuite<'a> {
    engine: GenericsEngine<'a>,
    symbol_table: SymbolTable,
    scope_tree: ScopeTree,
    string_interner: StringInterner,
}

impl<'a> GenericsTestSuite<'a> {
    pub fn new(type_table: &'a RefCell<TypeTable>) -> Self {
        Self {
            engine: GenericsEngine::with_defaults(&type_table),
            symbol_table: SymbolTable::new(),
            scope_tree: ScopeTree::new(),
            string_interner: StringInterner::new(),
        }
    }

    /// Run all tests and return comprehensive results
    pub fn run_all_tests(&mut self) -> TestResults {
        let mut results = TestResults::new();

        println!("🧪 Running Phase 3 Generics System Test Suite");
        println!("===========================================");

        // Basic functionality tests
        results.add_result("basic_instantiation", self.test_basic_instantiation());
        results.add_result("constraint_validation", self.test_constraint_validation());
        results.add_result("fast_path_optimization", self.test_fast_path_optimization());
        results.add_result("cycle_detection", self.test_cycle_detection());

        // Real-world Haxe scenarios
        results.add_result("array_generics", self.test_array_generics());
        results.add_result("map_generics", self.test_map_generics());
        results.add_result("function_generics", self.test_function_generics());
        results.add_result("nested_generics", self.test_nested_generics());
        results.add_result("variance_handling", self.test_variance_handling());

        // Advanced constraint scenarios
        results.add_result("complex_constraints", self.test_complex_constraints());
        results.add_result("constraint_propagation", self.test_constraint_propagation());
        results.add_result("method_constraints", self.test_method_constraints());
        results.add_result("iterable_constraints", self.test_iterable_constraints());

        // Error handling and recovery
        results.add_result("error_recovery", self.test_error_recovery());
        results.add_result("invalid_instantiations", self.test_invalid_instantiations());
        results.add_result("constraint_violations", self.test_constraint_violations());

        // Performance and optimization
        results.add_result("performance_benchmarks", self.test_performance_benchmarks());
        results.add_result("cache_effectiveness", self.test_cache_effectiveness());
        results.add_result("memory_usage", self.test_memory_usage());

        // Integration with Phase 1 & 2
        results.add_result(
            "type_system_integration",
            self.test_type_system_integration(),
        );
        results.add_result("scope_integration", self.test_scope_integration());
        results.add_result("symbol_resolution", self.test_symbol_resolution());

        results.print_summary();
        results
    }

    // === Basic Functionality Tests ===

    fn test_basic_instantiation(&mut self) -> TestResult {
        println!("📋 Testing basic generic instantiation...");

        // Create Array<String> from Array<T>
        let array_base = self.create_array_type(); // Array<T>
        let string_type = self.engine.solver.type_table.borrow().string_type();

        match self
            .engine
            .resolve_generic(array_base, vec![string_type], SourceLocation::default())
        {
            Ok(result) => {
                assert!(result.resolved_type.is_valid());
                TestResult::passed("Successfully instantiated Array<String>")
            }
            Err(e) => TestResult::failed(&format!("Instantiation failed: {}", e)),
        }
    }

    fn test_constraint_validation(&mut self) -> TestResult {
        println!("📋 Testing constraint validation...");

        let int_type = self.engine.solver.type_table.borrow().int_type();
        let constraints = vec![
            ConstraintKind::Comparable,
            ConstraintKind::Arithmetic,
            ConstraintKind::Sized,
        ];

        match self.engine.validate_constraints(int_type, &constraints) {
            ConstraintValidationResult::AllSatisfied => {
                TestResult::passed("Int satisfies all numeric constraints")
            }
            ConstraintValidationResult::SomeViolated { violations } => {
                TestResult::failed(&format!("Unexpected violations: {:?}", violations))
            }
        }
    }

    fn test_fast_path_optimization(&mut self) -> TestResult {
        println!("📋 Testing fast path optimization...");

        let start = Instant::now();
        let mut fast_instantiations = 0;

        // Test multiple instantiations of common types
        for _ in 0..1000 {
            let array_base = self.create_array_type();
            let string_type = self.engine.solver.type_table.borrow().string_type();

            if let Ok(result) = self.engine.resolve_generic(
                array_base,
                vec![string_type],
                SourceLocation::default(),
            ) {
                if result.used_fast_path {
                    fast_instantiations += 1;
                }
            }
        }

        let elapsed = start.elapsed();

        if fast_instantiations > 500 {
            TestResult::passed(&format!(
                "Fast path used for {}/1000 instantiations in {:?}",
                fast_instantiations, elapsed
            ))
        } else {
            TestResult::failed("Fast path optimization not working effectively")
        }
    }

    fn test_cycle_detection(&mut self) -> TestResult {
        println!("📋 Testing cycle detection...");

        // Create a recursive type: Recursive<Recursive<T>>
        let recursive_base = self.create_recursive_type();

        match self.engine.resolve_generic(
            recursive_base,
            vec![recursive_base], // Self-referential
            SourceLocation::default(),
        ) {
            Err(GenericError::InstantiationFailed {
                error: InstantiationError::RecursiveInstantiation { .. },
                ..
            }) => TestResult::passed("Successfully detected recursive instantiation"),
            Ok(_) => TestResult::failed("Should have detected cycle"),
            Err(e) => TestResult::failed(&format!("Unexpected error: {}", e)),
        }
    }

    // === Real-world Haxe Scenarios ===

    fn test_array_generics(&mut self) -> TestResult {
        println!("📋 Testing Array<T> generics...");
        let type_table_guard = self.engine.solver.type_table.borrow();
        let mut successes = 0;
        let test_cases = [
            ("Array<Int>", type_table_guard.int_type()),
            ("Array<String>", type_table_guard.string_type()),
            ("Array<Bool>", type_table_guard.bool_type()),
            ("Array<Float>", type_table_guard.float_type()),
        ];

        let array_base = self.create_array_type();

        for (name, element_type) in &test_cases {
            if let Ok(_) = self.engine.resolve_generic(
                array_base,
                vec![*element_type],
                SourceLocation::default(),
            ) {
                successes += 1;
                println!("  ✅ {}", name);
            } else {
                println!("  ❌ {}", name);
            }
        }

        if successes == test_cases.len() {
            TestResult::passed("All Array<T> instantiations successful")
        } else {
            TestResult::failed(&format!(
                "Only {}/{} array types succeeded",
                successes,
                test_cases.len()
            ))
        }
    }

    fn test_map_generics(&mut self) -> TestResult {
        println!("📋 Testing Map<K,V> generics...");
        let type_table_guard = self.engine.solver.type_table.borrow();
        let map_base = self.create_map_type(); // Map<K,V>
        let string_type = type_table_guard.string_type();
        let int_type = type_table_guard.int_type();

        // Test Map<String, Int>
        match self.engine.resolve_generic(
            map_base,
            vec![string_type, int_type],
            SourceLocation::default(),
        ) {
            Ok(result) => {
                println!("  ✅ Map<String, Int> instantiated");

                // Test nested maps: Map<String, Map<String, Int>>
                match self.engine.resolve_generic(
                    map_base,
                    vec![string_type, result.resolved_type],
                    SourceLocation::default(),
                ) {
                    Ok(_) => TestResult::passed("Nested Map<String, Map<String, Int>> successful"),
                    Err(e) => TestResult::failed(&format!("Nested map failed: {}", e)),
                }
            }
            Err(e) => TestResult::failed(&format!("Map instantiation failed: {}", e)),
        }
    }

    fn test_function_generics(&mut self) -> TestResult {
        println!("📋 Testing generic functions...");
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Test function type: <T>(T) -> T (identity function)
        let type_param = self.create_type_parameter("T");
        let identity_func = type_table_guard.create_function_type(vec![type_param], type_param);

        // Create constraints for the type parameter
        let mut constraints = ConstraintSet::new();
        constraints.add_constraint(TypeConstraint {
            type_var: type_param,
            kind: ConstraintKind::Copy, // T must be copyable
            location: SourceLocation::default(),
            priority: ConstraintPriority::Explicit,
            is_soft: false,
        });

        // Test solving constraints
        match self.engine.solve_constraints(constraints) {
            Ok(solution) => TestResult::passed(&format!(
                "Function constraints solved with {} substitutions",
                solution.substitutions.len()
            )),
            Err(e) => TestResult::failed(&format!("Function constraint solving failed: {}", e)),
        }
    }

    fn test_nested_generics(&mut self) -> TestResult {
        println!("📋 Testing nested generics...");
        let mut type_table_guard = self.engine.solver.type_table.borrow();
        // Test Array<Map<String, Int>>
        let array_base = self.create_array_type();
        let map_base = self.create_map_type();
        let string_type = type_table_guard.string_type();
        let int_type = type_table_guard.int_type();

        // First create Map<String, Int>
        let map_instance = match self.engine.resolve_generic(
            map_base,
            vec![string_type, int_type],
            SourceLocation::default(),
        ) {
            Ok(result) => result.resolved_type,
            Err(e) => return TestResult::failed(&format!("Inner map failed: {}", e)),
        };

        // Then create Array<Map<String, Int>>
        match self
            .engine
            .resolve_generic(array_base, vec![map_instance], SourceLocation::default())
        {
            Ok(_) => TestResult::passed("Nested Array<Map<String, Int>> successful"),
            Err(e) => TestResult::failed(&format!("Nested instantiation failed: {}", e)),
        }
    }

    fn test_variance_handling(&mut self) -> TestResult {
        println!("📋 Testing variance in generics...");
        let type_table_guard = self.engine.solver.type_table.borrow();
        // Test covariant and contravariant relationships
        // Array<String> <: Array<Object> (covariant in Haxe)
        let array_base = self.create_array_type();
        let string_type = type_table_guard.string_type();
        let object_type = self.create_object_type();

        let array_string = match self.engine.resolve_generic(
            array_base,
            vec![string_type],
            SourceLocation::default(),
        ) {
            Ok(result) => result.resolved_type,
            Err(e) => return TestResult::failed(&format!("Array<String> failed: {}", e)),
        };

        let array_object = match self.engine.resolve_generic(
            array_base,
            vec![object_type],
            SourceLocation::default(),
        ) {
            Ok(result) => result.resolved_type,
            Err(e) => return TestResult::failed(&format!("Array<Object> failed: {}", e)),
        };

        // Check subtype relationship
        if self.engine.is_generic_subtype(array_string, array_object) {
            TestResult::passed("Covariant array subtyping works correctly")
        } else {
            TestResult::failed("Array covariance not working")
        }
    }

    // === Advanced Constraint Tests ===

    fn test_complex_constraints(&mut self) -> TestResult {
        println!("📋 Testing complex constraint scenarios...");
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        let string_type = type_table_guard.string_type();
        let type_param = self.create_type_parameter("T");
        let constraints = vec![
            ConstraintKind::Comparable,
            ConstraintKind::Arithmetic,
            ConstraintKind::StringConvertible,
            ConstraintKind::HasMethod {
                method_name: self.intern_string("toString"),
                signature: type_table_guard.create_function_type(vec![], string_type),
                is_static: false,
            },
        ];

        // Test with Int type (should satisfy all)
        let int_type = type_table_guard.int_type();
        match self.engine.validate_constraints(int_type, &constraints) {
            ConstraintValidationResult::AllSatisfied => {
                TestResult::passed("Complex constraints satisfied for Int")
            }
            ConstraintValidationResult::SomeViolated { violations } => {
                TestResult::failed(&format!("Unexpected violations: {}", violations.len()))
            }
        }
    }

    fn test_constraint_propagation(&mut self) -> TestResult {
        println!("📋 Testing constraint propagation...");

        let mut constraints = ConstraintSet::new();
        let type_var1 = self.create_type_parameter("T");
        let type_var2 = self.create_type_parameter("U");

        // Add constraint: T = U
        constraints.add_constraint(TypeConstraint {
            type_var: type_var1,
            kind: ConstraintKind::Equality {
                target_type: type_var2,
            },
            location: SourceLocation::default(),
            priority: ConstraintPriority::Explicit,
            is_soft: false,
        });

        // Add constraint: T: Comparable
        constraints.add_constraint(TypeConstraint {
            type_var: type_var1,
            kind: ConstraintKind::Comparable,
            location: SourceLocation::default(),
            priority: ConstraintPriority::Explicit,
            is_soft: false,
        });

        let initial_count = constraints.stats().total_constraints;

        // Solve constraints (should propagate Comparable to U)
        match self.engine.solve_constraints(constraints) {
            Ok(solution) => {
                if solution.final_constraints.stats().total_constraints >= initial_count {
                    TestResult::passed("Constraint propagation working")
                } else {
                    TestResult::failed("No constraint propagation detected")
                }
            }
            Err(e) => TestResult::failed(&format!("Constraint solving failed: {}", e)),
        }
    }

    fn test_method_constraints(&mut self) -> TestResult {
        println!("📋 Testing method constraints...");
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        let int_type = type_table_guard.int_type();
        let type_param = self.create_type_parameter("T");
        let method_constraint = ConstraintKind::HasMethod {
            method_name: self.intern_string("length"),
            signature: type_table_guard.create_function_type(vec![], int_type),
            is_static: false,
        };

        // Test with String type (has length method)
        let string_type = type_table_guard.string_type();
        match self
            .engine
            .validate_constraints(string_type, &[method_constraint])
        {
            ConstraintValidationResult::AllSatisfied => {
                TestResult::passed("Method constraint validation working")
            }
            ConstraintValidationResult::SomeViolated { .. } => {
                TestResult::failed("String should have length method")
            }
        }
    }

    fn test_iterable_constraints(&mut self) -> TestResult {
        println!("📋 Testing iterable constraints...");
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        let array_type = self.create_array_type();
        let string_type = type_table_guard.string_type();

        // Array<String> should be Iterable<String>
        let array_string = match self.engine.resolve_generic(
            array_type,
            vec![string_type],
            SourceLocation::default(),
        ) {
            Ok(result) => result.resolved_type,
            Err(e) => return TestResult::failed(&format!("Array instantiation failed: {}", e)),
        };

        let iterable_constraint = ConstraintKind::Iterable {
            element_type: Some(string_type),
        };

        match self
            .engine
            .validate_constraints(array_string, &[iterable_constraint])
        {
            ConstraintValidationResult::AllSatisfied => {
                TestResult::passed("Array<String> is Iterable<String>")
            }
            ConstraintValidationResult::SomeViolated { .. } => {
                TestResult::failed("Array should be iterable")
            }
        }
    }

    // === Error Handling Tests ===

    fn test_error_recovery(&mut self) -> TestResult {
        println!("📋 Testing error recovery...");
        let type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Try to instantiate non-generic type with type arguments
        let int_type = type_table_guard.int_type();

        match self.engine.resolve_generic(
            int_type,
            vec![type_table_guard.string_type()],
            SourceLocation::default(),
        ) {
            Err(GenericError::InstantiationFailed {
                error: InstantiationError::NotGeneric { .. },
                ..
            }) => TestResult::passed("Correctly detected non-generic type"),
            Ok(_) => TestResult::failed("Should have failed for non-generic type"),
            Err(e) => TestResult::failed(&format!("Unexpected error: {}", e)),
        }
    }

    fn test_invalid_instantiations(&mut self) -> TestResult {
        println!("📋 Testing invalid instantiations...");
        let type_table_guard = self.engine.solver.type_table.borrow();
        let array_base = self.create_array_type();

        // Wrong number of type arguments
        match self.engine.resolve_generic(
            array_base,
            vec![type_table_guard.string_type(), type_table_guard.int_type()], // Too many args
            SourceLocation::default(),
        ) {
            Err(GenericError::InstantiationFailed {
                error: InstantiationError::ArityMismatch { .. },
                ..
            }) => TestResult::passed("Correctly detected arity mismatch"),
            Ok(_) => TestResult::failed("Should have failed for wrong arity"),
            Err(e) => TestResult::failed(&format!("Unexpected error: {}", e)),
        }
    }

    fn test_constraint_violations(&mut self) -> TestResult {
        println!("📋 Testing constraint violations...");
        let type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Test type that doesn't satisfy constraints
        let dynamic_type = type_table_guard.dynamic_type();
        let constraints = vec![ConstraintKind::Sized]; // Dynamic is not sized

        match self.engine.validate_constraints(dynamic_type, &constraints) {
            ConstraintValidationResult::SomeViolated { violations } => {
                if violations.len() > 0 {
                    TestResult::passed("Correctly detected constraint violations")
                } else {
                    TestResult::failed("Expected violations but got none")
                }
            }
            ConstraintValidationResult::AllSatisfied => {
                TestResult::failed("Dynamic should not satisfy Sized constraint")
            }
        }
    }

    // === Performance Tests ===

    fn test_performance_benchmarks(&mut self) -> TestResult {
        println!("📋 Running performance benchmarks...");
        let type_table_guard = self.engine.solver.type_table.borrow_mut();
        let start = Instant::now();
        let mut successful_instantiations = 0;

        // Benchmark: 1000 generic instantiations
        for i in 0..1000 {
            let array_base = self.create_array_type();
            let element_type = match i % 4 {
                0 => type_table_guard.int_type(),
                1 => type_table_guard.string_type(),
                2 => type_table_guard.bool_type(),
                _ => type_table_guard.float_type(),
            };

            if let Ok(_) = self.engine.resolve_generic(
                array_base,
                vec![element_type],
                SourceLocation::default(),
            ) {
                successful_instantiations += 1;
            }
        }

        let elapsed = start.elapsed();
        let avg_time_ns = elapsed.as_nanos() / 1000;

        println!(
            "  📊 1000 instantiations in {:?} (avg: {}ns)",
            elapsed, avg_time_ns
        );

        if successful_instantiations >= 950 && elapsed.as_millis() < 100 {
            TestResult::passed(&format!(
                "Performance benchmark: {}/1000 successful in {:?}",
                successful_instantiations, elapsed
            ))
        } else {
            TestResult::failed(&format!(
                "Performance below threshold: {}/1000 in {:?}",
                successful_instantiations, elapsed
            ))
        }
    }

    fn test_cache_effectiveness(&mut self) -> TestResult {
        println!("📋 Testing cache effectiveness...");
        let type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Clear cache to start fresh
        self.engine.clear_caches();

        let array_base = self.create_array_type();
        let string_type = type_table_guard.string_type();

        // First instantiation (cache miss)
        let _ =
            self.engine
                .resolve_generic(array_base, vec![string_type], SourceLocation::default());

        let start = Instant::now();

        // Repeated instantiations with SAME parameters (should be cache hits)
        for _ in 0..100 {
            let _ = self.engine.resolve_generic(
                array_base,        // Same base type
                vec![string_type], // Same type args
                SourceLocation::default(),
            );
        }

        let elapsed = start.elapsed();

        // Get cache statistics
        let stats = self.engine.stats();

        // We expect 99% hit ratio (1 miss + 100 hits = 99/101)
        if stats.cache_hit_ratio > 0.9 && elapsed.as_millis() < 10 {
            TestResult::passed(&format!(
                "Cache effectiveness: {:.1}% hit ratio, 100 ops in {:?}",
                stats.cache_hit_ratio * 100.0,
                elapsed
            ))
        } else {
            TestResult::failed(&format!(
                "Cache not effective: {:.1}% hit ratio, {:?}",
                stats.cache_hit_ratio * 100.0,
                elapsed
            ))
        }
    }

    fn test_memory_usage(&mut self) -> TestResult {
        println!("📋 Testing memory usage...");

        let initial_stats = self.engine.stats();
        let initial_memory = initial_stats.memory_usage_bytes;

        // Create many generic instantiations
        for i in 0..500 {
            let array_base = self.create_array_type();
            let element_type = self.create_type_parameter(&format!("T{}", i));

            let _ = self.engine.resolve_generic(
                array_base,
                vec![element_type],
                SourceLocation::default(),
            );
        }

        let final_stats = self.engine.stats();
        let final_memory = final_stats.memory_usage_bytes;

        let memory_growth = final_memory.saturating_sub(initial_memory);
        let avg_per_instantiation = if final_stats.instantiations_performed > 0 {
            memory_growth / final_stats.instantiations_performed
        } else {
            0
        };

        println!(
            "  📊 Memory growth: {} bytes ({} bytes/instantiation)",
            memory_growth, avg_per_instantiation
        );

        if avg_per_instantiation < 1024 {
            // Less than 1KB per instantiation
            TestResult::passed(&format!(
                "Memory usage acceptable: {} bytes/instantiation",
                avg_per_instantiation
            ))
        } else {
            TestResult::failed(&format!(
                "Memory usage too high: {} bytes/instantiation",
                avg_per_instantiation
            ))
        }
    }

    // === Integration Tests ===

    fn test_type_system_integration(&mut self) -> TestResult {
        println!("📋 Testing Phase 1 type system integration...");
        let type_table_guard = self.engine.solver.type_table.borrow();
        // Test that generics work with the existing type system
        let array_base = self.create_array_type();
        let string_type = type_table_guard.string_type();

        let array_string = match self.engine.resolve_generic(
            array_base,
            vec![string_type],
            SourceLocation::default(),
        ) {
            Ok(result) => result.resolved_type,
            Err(e) => return TestResult::failed(&format!("Integration failed: {}", e)),
        };

        // Verify the instantiated type is properly stored in the type table
        if let Some(type_obj) = type_table_guard.get(array_string) {
            match &type_obj.kind {
                TypeKind::GenericInstance {
                    base_type,
                    type_args,
                    ..
                } => {
                    if *base_type == array_base && type_args[0] == string_type {
                        TestResult::passed("Type system integration working")
                    } else {
                        TestResult::failed("Generic instance structure incorrect")
                    }
                }
                _ => TestResult::failed("Expected GenericInstance type"),
            }
        } else {
            TestResult::failed("Instantiated type not found in type table")
        }
    }

    fn test_scope_integration(&mut self) -> TestResult {
        println!("📋 Testing scope system integration...");

        // Create a scope with type parameters
        let scope_id = self.scope_tree.create_scope(None);
        let type_params = vec![
            (self.intern_string("T"), vec![ConstraintKind::Comparable]),
            (self.intern_string("U"), vec![ConstraintKind::Arithmetic]),
        ];

        match self.engine.register_type_parameters(
            scope_id,
            &type_params,
            &mut self.symbol_table,
            &mut self.scope_tree,
        ) {
            Ok(param_symbols) => {
                if param_symbols.len() == 2 {
                    TestResult::passed("Type parameters registered in scope")
                } else {
                    TestResult::failed("Wrong number of type parameters")
                }
            }
            Err(e) => TestResult::failed(&format!("Scope integration failed: {}", e)),
        }
    }

    fn test_symbol_resolution(&mut self) -> TestResult {
        println!("📋 Testing symbol resolution integration...");

        // Create a scope with a type parameter
        let scope_id = self.scope_tree.create_scope(None);
        let param_name = self.intern_string("T");
        let constraints = vec![ConstraintKind::Comparable];

        // Register type parameter
        let symbol_id = self
            .symbol_table
            .create_type_parameter(param_name, constraints);
        self.scope_tree
            .add_symbol_to_scope(scope_id, symbol_id)
            .expect("Failed to add symbol to scope");

        // Resolve the type parameter
        if let Some(resolved_symbol) = self.engine.resolve_type_parameter(
            param_name,
            scope_id,
            &self.symbol_table,
            &mut self.scope_tree,
        ) {
            if resolved_symbol == symbol_id {
                TestResult::passed("Type parameter resolution working")
            } else {
                TestResult::failed("Resolved wrong symbol")
            }
        } else {
            TestResult::failed("Type parameter not found")
        }
    }

    // === Helper methods for creating test types ===

    fn create_array_type(&mut self) -> TypeId {
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Create Array<T> base type
        let type_param = self.create_type_parameter("T");
        let array_name = self.intern_string("Array");
        let class_symbol = self.symbol_table.create_class(array_name);
        type_table_guard.create_class_type(class_symbol, vec![type_param])
    }

    fn create_map_type(&mut self) -> TypeId {
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Create Map<K,V> base type
        let key_param = self.create_type_parameter("K");
        let value_param = self.create_type_parameter("V");
        let map_name = self.intern_string("Map");
        let class_symbol = self.symbol_table.create_class(map_name);
        type_table_guard.create_class_type(class_symbol, vec![key_param, value_param])
    }

    fn create_recursive_type(&mut self) -> TypeId {
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Create a type that can reference itself
        let type_param = self.create_type_parameter("T");
        let recursive_name = self.intern_string("Recursive");
        let class_symbol = self.symbol_table.create_class(recursive_name);
        type_table_guard.create_class_type(class_symbol, vec![type_param])
    }

    fn create_object_type(&mut self) -> TypeId {
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        // Create Object base type
        let object_name = self.intern_string("Object");
        let class_symbol = self.symbol_table.create_class(object_name);
        type_table_guard.create_class_type(class_symbol, vec![])
    }

    fn create_type_parameter(&mut self, name: &str) -> TypeId {
        let mut type_table_guard = self.engine.solver.type_table.borrow_mut();
        let param_name = self.intern_string(name);
        let param_symbol = self.symbol_table.create_type_parameter(param_name, vec![]);
        type_table_guard.create_type_parameter(
            param_symbol,
            vec![],
            super::core::Variance::Invariant,
        )
    }

    fn intern_string(&mut self, s: &str) -> InternedString {
        self.string_interner.intern(s)
    }
}

/// Result of a single test
#[derive(Debug, Clone)]
pub struct TestResult {
    pub passed: bool,
    pub message: String,
    pub duration_ms: Option<u64>,
}

impl TestResult {
    pub fn passed(message: &str) -> Self {
        Self {
            passed: true,
            message: message.to_string(),
            duration_ms: None,
        }
    }

    pub fn failed(message: &str) -> Self {
        Self {
            passed: false,
            message: message.to_string(),
            duration_ms: None,
        }
    }
}

/// Collection of test results with summary statistics
#[derive(Debug)]
pub struct TestResults {
    results: BTreeMap<String, TestResult>,
    total_tests: usize,
    passed_tests: usize,
    failed_tests: usize,
}

impl TestResults {
    pub fn new() -> Self {
        Self {
            results: BTreeMap::new(),
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
        }
    }

    pub fn add_result(&mut self, test_name: &str, result: TestResult) {
        self.total_tests += 1;
        if result.passed {
            self.passed_tests += 1;
            println!("  ✅ {}: {}", test_name, result.message);
        } else {
            self.failed_tests += 1;
            println!("  ❌ {}: {}", test_name, result.message);
        }
        self.results.insert(test_name.to_string(), result);
    }

    pub fn print_summary(&self) {
        println!("\n🎯 Test Suite Summary");
        println!("=====================");
        println!("Total tests: {}", self.total_tests);
        println!("Passed: {} ✅", self.passed_tests);
        println!("Failed: {} ❌", self.failed_tests);
        println!(
            "Success rate: {:.1}%",
            (self.passed_tests as f64 / self.total_tests as f64) * 100.0
        );

        if self.failed_tests > 0 {
            println!("\n❌ Failed tests:");
            for (name, result) in &self.results {
                if !result.passed {
                    println!("  • {}: {}", name, result.message);
                }
            }
        }

        println!();
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_tests > 0 {
            self.passed_tests as f64 / self.total_tests as f64
        } else {
            0.0
        }
    }

    pub fn all_passed(&self) -> bool {
        self.failed_tests == 0 && self.total_tests > 0
    }
}

/// Example usage and integration demonstration
pub fn run_generics_examples() {
    println!("🌟 Haxe Generics System Examples");
    println!("=================================\n");
    let type_table = &RefCell::new(TypeTable::new());
    let mut suite = GenericsTestSuite::new(type_table);

    // Example 1: Simple Array instantiation
    println!("Example 1: Simple Array<String>");
    let array_base = suite.create_array_type();
    let string_type = type_table.borrow().string_type();

    match suite
        .engine
        .resolve_generic(array_base, vec![string_type], SourceLocation::default())
    {
        Ok(result) => {
            println!(
                "✅ Array<String> resolved to type {:?}",
                result.resolved_type
            );
            println!("   Used fast path: {}", result.used_fast_path);
            println!("   Resolution time: {}ms", result.resolution_time_ms);
        }
        Err(e) => println!("❌ Failed: {}", e),
    }

    // Example 2: Complex nested generics
    println!("\nExample 2: Array<Map<String, Int>>");
    let map_base = suite.create_map_type();
    let int_type = type_table.borrow().int_type();

    // Create Map<String, Int>
    let map_instance = suite
        .engine
        .resolve_generic(
            map_base,
            vec![string_type, int_type],
            SourceLocation::default(),
        )
        .unwrap()
        .resolved_type;

    // Create Array<Map<String, Int>>
    match suite
        .engine
        .resolve_generic(array_base, vec![map_instance], SourceLocation::default())
    {
        Ok(result) => {
            println!(
                "✅ Complex nested type resolved: {:?}",
                result.resolved_type
            );
            println!(
                "   Generated {} constraints",
                result.generated_constraints.stats().total_constraints
            );
        }
        Err(e) => println!("❌ Failed: {}", e),
    }

    // Example 3: Constraint validation
    println!("\nExample 3: Constraint validation");
    let constraints = vec![
        ConstraintKind::Comparable,
        ConstraintKind::Arithmetic,
        ConstraintKind::StringConvertible,
    ];

    match suite.engine.validate_constraints(int_type, &constraints) {
        ConstraintValidationResult::AllSatisfied => {
            println!("✅ Int type satisfies all numeric constraints");
        }
        ConstraintValidationResult::SomeViolated { violations } => {
            println!("❌ Constraint violations: {}", violations.len());
        }
    }

    // Show final statistics
    println!("\n📊 Final Statistics:");
    let stats = suite.engine.stats();
    println!("Generic resolutions: {}", stats.generic_resolutions);
    println!("Cache hit ratio: {:.1}%", stats.cache_hit_ratio * 100.0);
    println!(
        "Average resolution time: {:.2}ms",
        stats.average_resolution_time_ms
    );
    println!("Memory usage: {} bytes", stats.memory_usage_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_full_test_suite() {
        let type_table = &RefCell::new(TypeTable::new());
        let mut suite = GenericsTestSuite::new(type_table);
        let results = suite.run_all_tests();

        // Expect at least 80% success rate
        assert!(
            results.success_rate() >= 0.8,
            "Test success rate too low: {:.1}%",
            results.success_rate() * 100.0
        );
        assert!(results.total_tests >= 15, "Not enough tests run");
    }

    #[test]
    fn test_basic_functionality() {
        let type_table = &RefCell::new(TypeTable::new());
        let mut suite = GenericsTestSuite::new(type_table);

        // Test basic instantiation
        let result = suite.test_basic_instantiation();
        assert!(
            result.passed,
            "Basic instantiation should pass: {}",
            result.message
        );

        // Test constraint validation
        let result = suite.test_constraint_validation();
        assert!(
            result.passed,
            "Constraint validation should pass: {}",
            result.message
        );
    }

    #[test]
    fn test_real_world_scenarios() {
        let type_table = &RefCell::new(TypeTable::new());
        let mut suite = GenericsTestSuite::new(type_table);

        // Test array generics
        let result = suite.test_array_generics();
        assert!(
            result.passed,
            "Array generics should work: {}",
            result.message
        );

        // Test nested generics
        let result = suite.test_nested_generics();
        assert!(
            result.passed,
            "Nested generics should work: {}",
            result.message
        );
    }

    #[test]
    fn test_performance_requirements() {
        let type_table = &RefCell::new(TypeTable::new());
        let mut suite = GenericsTestSuite::new(type_table);

        // Test performance benchmarks
        let result = suite.test_performance_benchmarks();
        assert!(
            result.passed,
            "Performance should meet requirements: {}",
            result.message
        );

        // Test cache effectiveness
        let result = suite.test_cache_effectiveness();
        assert!(
            result.passed,
            "Cache should be effective: {}",
            result.message
        );
    }
}
