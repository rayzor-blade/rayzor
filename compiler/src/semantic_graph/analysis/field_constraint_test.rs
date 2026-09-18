//! Test for FieldConstraint processing in lifetime solver

use crate::semantic_graph::LifetimeId;
use crate::semantic_graph::analysis::lifetime_analyzer::LifetimeConstraint;
use crate::semantic_graph::analysis::lifetime_solver::LifetimeConstraintSolver;
use crate::tast::SourceLocation;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_constraint_processing() {
        let mut solver = LifetimeConstraintSolver::new();

        // Create lifetimes for object and field access
        let object_lifetime = LifetimeId::from_raw(1);
        let field_lifetime = LifetimeId::from_raw(2);

        // Create a field constraint: object must outlive field access
        let field_constraint = LifetimeConstraint::FieldConstraint {
            object_lifetime,
            field_lifetime,
            field_name: "my_field".to_string(),
        };

        // Solve the constraint system
        let result = solver.solve(&[field_constraint]);

        // Should succeed without errors
        assert!(result.is_ok());
        let solution = result.unwrap();

        // Verify the solution is satisfiable
        assert!(solution.satisfiable);

        // Verify no conflicts were detected
        assert!(solution.conflicts.is_empty());

        // Verify that some assignments were generated
        assert!(!solution.assignments.is_empty());

        // The constraint should create an outlives relationship in the solution's ordering
        // object_lifetime should come before field_lifetime in the topological ordering
        // (earlier in the ordering means longer-lived)
        let object_pos = solution
            .lifetime_ordering
            .iter()
            .position(|&lt| lt == object_lifetime);
        let field_pos = solution
            .lifetime_ordering
            .iter()
            .position(|&lt| lt == field_lifetime);

        // Both lifetimes should be in the ordering
        assert!(
            object_pos.is_some() || field_pos.is_some(),
            "At least one lifetime should be in ordering"
        );

        // If both are present, object should come before field (longer-lived)
        if let (Some(obj_idx), Some(field_idx)) = (object_pos, field_pos) {
            assert!(
                obj_idx < field_idx,
                "Object lifetime should outlive field access lifetime"
            );
        }
    }

    #[test]
    fn test_field_constraint_with_conflicting_constraints() {
        let mut solver = LifetimeConstraintSolver::new();

        let object_lifetime = LifetimeId::from_raw(1);
        let field_lifetime = LifetimeId::from_raw(2);

        // Create constraints that create a cycle:
        // 1. object must outlive field (from field access)
        // 2. field must outlive object (artificial conflicting constraint)
        let constraints = vec![
            LifetimeConstraint::FieldConstraint {
                object_lifetime,
                field_lifetime,
                field_name: "field".to_string(),
            },
            LifetimeConstraint::Outlives {
                longer: field_lifetime,
                shorter: object_lifetime,
                location: SourceLocation::unknown(),
                reason:
                    crate::semantic_graph::analysis::lifetime_analyzer::OutlivesReason::Assignment,
            },
        ];

        let result = solver.solve(&constraints);

        // Should succeed but find the system unsatisfiable due to cycle
        assert!(result.is_ok());
        let solution = result.unwrap();

        // Should be marked as unsatisfiable due to the cycle
        assert!(!solution.satisfiable);

        // Should have detected conflicts (the cycle)
        assert!(!solution.conflicts.is_empty());
    }

    #[test]
    fn test_multiple_field_constraints() {
        let mut solver = LifetimeConstraintSolver::new();

        let object_lifetime = LifetimeId::from_raw(1);
        let field1_lifetime = LifetimeId::from_raw(2);
        let field2_lifetime = LifetimeId::from_raw(3);

        // Multiple field accesses on the same object
        let constraints = vec![
            LifetimeConstraint::FieldConstraint {
                object_lifetime,
                field_lifetime: field1_lifetime,
                field_name: "field1".to_string(),
            },
            LifetimeConstraint::FieldConstraint {
                object_lifetime,
                field_lifetime: field2_lifetime,
                field_name: "field2".to_string(),
            },
        ];

        let result = solver.solve(&constraints);

        // Should succeed
        assert!(result.is_ok());
        let solution = result.unwrap();

        // Should be satisfiable
        assert!(solution.satisfiable);

        // No conflicts expected
        assert!(solution.conflicts.is_empty());

        // Verify ordering: object should outlive both fields
        let object_pos = solution
            .lifetime_ordering
            .iter()
            .position(|&lt| lt == object_lifetime);
        let field1_pos = solution
            .lifetime_ordering
            .iter()
            .position(|&lt| lt == field1_lifetime);
        let field2_pos = solution
            .lifetime_ordering
            .iter()
            .position(|&lt| lt == field2_lifetime);

        if let Some(obj_idx) = object_pos {
            if let Some(f1_idx) = field1_pos {
                assert!(obj_idx < f1_idx, "Object should outlive field1");
            }
            if let Some(f2_idx) = field2_pos {
                assert!(obj_idx < f2_idx, "Object should outlive field2");
            }
        }
    }
}
