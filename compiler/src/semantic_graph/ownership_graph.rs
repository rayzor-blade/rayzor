//! Ownership Graph implementation for memory safety analysis
//!
//! The ownership graph tracks ownership relationships, borrowing patterns,
//! and lifetime constraints to enable Rust-style memory safety checking
//! in Haxe code. This enables static detection of use-after-free,
//! double-free, and data race conditions.
//!
//! Key features:
//! - Variable ownership tracking (owned, borrowed, moved)
//! - Lifetime constraint solving
//! - Borrowing relationship analysis
//! - Move semantics verification
//! - Memory safety violation detection

use super::{SourceLocation, SourceLocationTracker, SourceLocationTracking, SymbolId};
use crate::semantic_graph::analysis::lifetime_analyzer::LifetimeConstraint;
use crate::tast::collections::{new_id_map, new_id_set, IdMap, IdSet};
use crate::tast::{BlockId, BorrowEdgeId, DataFlowNodeId, LifetimeId, MoveEdgeId, ScopeId, TypeId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// Complete ownership and lifetime tracking graph
#[derive(Debug, Clone)]
pub struct OwnershipGraph {
    /// All variables and their ownership information
    pub variables: IdMap<SymbolId, OwnershipNode>,

    /// All lifetimes in the program
    pub lifetimes: IdMap<LifetimeId, Lifetime>,

    /// All borrowing relationships
    pub borrow_edges: IdMap<BorrowEdgeId, BorrowEdge>,

    /// All ownership transfer (move) relationships
    pub move_edges: IdMap<MoveEdgeId, MoveEdge>,

    /// Lifetime constraints that must be satisfied
    pub lifetime_constraints: Vec<LifetimeConstraint>,

    /// Ownership analysis statistics
    pub statistics: OwnershipStatistics,

    /// Source location tracking for error reporting
    pub location_tracker: SourceLocationTracker,

    /// Use sites for each variable (for use-after-move detection)
    pub use_sites: BTreeMap<SymbolId, Vec<SourceLocation>>,

    /// Next available IDs for allocation
    next_lifetime_id: u32,
    next_borrow_edge_id: u32,
    next_move_edge_id: u32,
}

/// Information about a variable's ownership and lifetime
#[derive(Debug, Clone)]
pub struct OwnershipNode {
    /// The variable this node represents
    pub variable: SymbolId,

    /// Lifetime of this variable
    pub lifetime: LifetimeId,

    /// Current ownership kind
    pub ownership_kind: OwnershipKind,

    /// Variables that borrow from this variable
    pub borrowed_by: Vec<BorrowEdgeId>,

    /// Variables this variable borrows from
    pub borrows_from: Vec<BorrowEdgeId>,

    /// Location where this variable was allocated
    pub allocation_site: Option<DataFlowNodeId>,

    /// Location where this variable was last moved
    pub move_site: Option<MoveEdgeId>,

    /// Whether this variable has been moved
    pub is_moved: bool,

    /// Type of the variable (for ownership analysis)
    pub variable_type: TypeId,

    /// Scope where this variable is defined
    pub scope: ScopeId,
}

/// Different kinds of ownership for variables
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipKind {
    /// Full ownership - can move and mutate
    Owned,

    /// Immutable borrow - can read but not modify
    Borrowed,

    /// Mutable borrow - exclusive access for modification
    BorrowedMut,

    /// Shared ownership (reference counted) - for Haxe interop
    Shared,

    /// Unknown ownership - analysis couldn't determine
    Unknown,

    /// Moved - no longer accessible
    Moved,
}

/// Lifetime information and constraints
#[derive(Debug, Clone)]
pub struct Lifetime {
    /// Unique identifier
    pub id: LifetimeId,

    /// Scope this lifetime is bound to
    pub scope: ScopeId,

    /// Variables that have this lifetime
    pub variables: Vec<SymbolId>,

    /// Constraints on this lifetime
    pub constraints: Vec<LifetimeConstraint>,

    /// Whether this lifetime is inferred or explicit
    pub is_inferred: bool,

    /// Source location where this lifetime originates
    pub source_location: SourceLocation,
}

/// Borrowing relationship between two variables
#[derive(Debug, Clone)]
pub struct BorrowEdge {
    /// Unique identifier
    pub id: BorrowEdgeId,

    /// Variable doing the borrowing
    pub borrower: SymbolId,

    /// Variable being borrowed
    pub borrowed: SymbolId,

    /// Type of borrow (immutable or mutable)
    pub borrow_type: BorrowType,

    /// Scope where the borrow is active
    pub borrow_scope: ScopeId,

    /// Location where the borrow occurs
    pub borrow_location: SourceLocation,

    /// Lifetime of the borrow
    pub borrow_lifetime: LifetimeId,
}

/// Ownership transfer (move) relationship
#[derive(Debug, Clone)]
pub struct MoveEdge {
    /// Unique identifier
    pub id: MoveEdgeId,

    /// Variable being moved from
    pub source: SymbolId,

    /// Variable being moved to (or expression consuming the value)
    pub destination: Option<SymbolId>,

    /// Location where the move occurs
    pub move_location: SourceLocation,

    /// Type of move operation
    pub move_type: MoveType,

    /// Whether this move invalidates the source
    pub invalidates_source: bool,
}

/// Type of borrowing relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowType {
    /// Immutable borrow (&T)
    Immutable,

    /// Mutable borrow (&mut T)
    Mutable,

    /// Weak reference (for breaking cycles)
    Weak,
}

/// Type of move operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveType {
    /// Explicit move (assignment or parameter passing)
    Explicit,

    /// Implicit move (return value, function call)
    Implicit,

    /// Move into function call
    FunctionCall,

    /// Move for destruction
    Destruction,
}

// /// Lifetime constraint that must be satisfied
// #[derive(Debug, Clone)]
// pub enum LifetimeConstraint {
//     /// One lifetime must outlive another
//     Outlives {
//         longer: LifetimeId,
//         shorter: LifetimeId,
//         reason: String,
//     },

//     /// Two lifetimes must be equal
//     Equal {
//         left: LifetimeId,
//         right: LifetimeId,
//         reason: String,
//     },

//     /// Constraint from function call parameter lifetimes
//     CallConstraint {
//         call_site: DataFlowNodeId,
//         param_lifetimes: Vec<LifetimeId>,
//         return_lifetime: Option<LifetimeId>,
//     },

//     /// Constraint from field access
//     FieldConstraint {
//         object_lifetime: LifetimeId,
//         field_lifetime: LifetimeId,
//         field_name: String,
//     },
// }

/// Statistics about ownership analysis
#[derive(Debug, Clone, Default)]
pub struct OwnershipStatistics {
    /// Total number of variables tracked
    pub variable_count: usize,

    /// Number of lifetime constraints
    pub constraint_count: usize,

    /// Number of borrowing relationships
    pub borrow_count: usize,

    /// Number of move operations
    pub move_count: usize,

    /// Number of ownership violations found
    pub violation_count: usize,

    /// Number of stack allocation opportunities
    pub stack_allocation_opportunities: usize,

    /// Analysis time in microseconds
    pub analysis_time_us: u64,
}

impl OwnershipGraph {
    /// Create a new empty ownership graph
    pub fn new() -> Self {
        let mut graph = Self {
            variables: new_id_map(),
            lifetimes: new_id_map(),
            borrow_edges: new_id_map(),
            move_edges: new_id_map(),
            lifetime_constraints: Vec::new(),
            statistics: OwnershipStatistics::default(),
            location_tracker: SourceLocationTracker::new(),
            use_sites: BTreeMap::new(),
            next_lifetime_id: 2, // Start after global and static
            next_borrow_edge_id: 1,
            next_move_edge_id: 1,
        };

        // Add global and static lifetimes
        graph.add_lifetime(Lifetime {
            id: LifetimeId::global(),
            scope: ScopeId::from_raw(0), // Global scope
            variables: Vec::new(),
            constraints: Vec::new(),
            is_inferred: false,
            source_location: SourceLocation::unknown(),
        });

        graph.add_lifetime(Lifetime {
            id: LifetimeId::static_lifetime(),
            scope: ScopeId::from_raw(0), // Global scope
            variables: Vec::new(),
            constraints: Vec::new(),
            is_inferred: false,
            source_location: SourceLocation::unknown(),
        });

        graph
    }

    /// Add a variable to the ownership graph
    pub fn add_variable(
        &mut self,
        variable: SymbolId,
        variable_type: TypeId,
        scope: ScopeId,
    ) -> &mut OwnershipNode {
        let lifetime = self.allocate_lifetime(scope);

        let node = OwnershipNode {
            variable,
            lifetime,
            ownership_kind: OwnershipKind::Owned, // Default to owned
            borrowed_by: Vec::new(),
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type,
            scope,
        };

        self.variables.insert(variable, node);
        self.statistics.variable_count += 1;

        self.variables.get_mut(&variable).unwrap()
    }

    /// Add a lifetime to the graph
    pub fn add_lifetime(&mut self, lifetime: Lifetime) -> LifetimeId {
        let id = lifetime.id;
        self.lifetimes.insert(id, lifetime);
        id
    }

    /// Allocate a new lifetime for a scope
    pub fn allocate_lifetime(&mut self, scope: ScopeId) -> LifetimeId {
        let id = LifetimeId::from_raw(self.next_lifetime_id);
        self.next_lifetime_id += 1;

        let lifetime = Lifetime {
            id,
            scope,
            variables: Vec::new(),
            constraints: Vec::new(),
            is_inferred: true,
            source_location: SourceLocation::unknown(),
        };

        self.add_lifetime(lifetime);
        id
    }

    /// Record a borrowing relationship
    pub fn add_borrow(
        &mut self,
        borrower: SymbolId,
        borrowed: SymbolId,
        borrow_type: BorrowType,
        borrow_scope: ScopeId,
        location: SourceLocation,
    ) -> BorrowEdgeId {
        let id = BorrowEdgeId::from_raw(self.next_borrow_edge_id);
        self.next_borrow_edge_id += 1;

        let borrow_lifetime = self.allocate_lifetime(borrow_scope);

        let edge = BorrowEdge {
            id,
            borrower,
            borrowed,
            borrow_type,
            borrow_scope,
            borrow_location: location,
            borrow_lifetime,
        };

        // Update ownership nodes
        if let Some(borrower_node) = self.variables.get_mut(&borrower) {
            borrower_node.borrows_from.push(id);
            // Only update ownership kind if not already shared
            if borrower_node.ownership_kind != OwnershipKind::Shared {
                borrower_node.ownership_kind = match borrow_type {
                    BorrowType::Immutable => OwnershipKind::Borrowed,
                    BorrowType::Mutable => OwnershipKind::BorrowedMut,
                    BorrowType::Weak => OwnershipKind::Borrowed,
                };
            }
        }

        if let Some(borrowed_node) = self.variables.get_mut(&borrowed) {
            borrowed_node.borrowed_by.push(id);
        }

        self.borrow_edges.insert(id, edge);
        self.statistics.borrow_count += 1;

        // Add lifetime constraint: borrowed data must outlive borrow
        self.add_lifetime_constraint(LifetimeConstraint::Outlives {
            longer: self
                .get_variable_lifetime(borrowed)
                .unwrap_or(LifetimeId::global()),
            shorter: borrow_lifetime,
            reason: super::analysis::lifetime_analyzer::OutlivesReason::Borrow,
            location: location,
        });

        id
    }

    /// Record a move operation
    pub fn add_move(
        &mut self,
        source: SymbolId,
        destination: Option<SymbolId>,
        location: SourceLocation,
        move_type: MoveType,
    ) -> MoveEdgeId {
        let id = MoveEdgeId::from_raw(self.next_move_edge_id);
        self.next_move_edge_id += 1;

        let edge = MoveEdge {
            id,
            source,
            destination,
            move_location: location,
            move_type,
            invalidates_source: true, // Most moves invalidate source
        };

        // Update source variable as moved
        if let Some(source_node) = self.variables.get_mut(&source) {
            source_node.is_moved = true;
            source_node.move_site = Some(id);
            source_node.ownership_kind = OwnershipKind::Moved;
        }

        // Update destination variable if it exists
        if let Some(dest_var) = destination {
            if let Some(dest_node) = self.variables.get_mut(&dest_var) {
                dest_node.ownership_kind = OwnershipKind::Owned;
            }
        }

        self.move_edges.insert(id, edge);
        self.statistics.move_count += 1;

        id
    }

    /// Record a use site for a variable (for use-after-move detection)
    pub fn record_use(&mut self, variable: SymbolId, location: SourceLocation) {
        self.use_sites.entry(variable).or_default().push(location);
    }

    /// Add a lifetime constraint
    pub fn add_lifetime_constraint(&mut self, constraint: LifetimeConstraint) {
        self.lifetime_constraints.push(constraint);
        self.statistics.constraint_count += 1;
    }

    /// Get the lifetime of a variable
    pub fn get_variable_lifetime(&self, variable: SymbolId) -> Option<LifetimeId> {
        self.variables.get(&variable).map(|node| node.lifetime)
    }

    /// Get the ownership kind of a variable
    pub fn get_ownership_kind(&self, variable: SymbolId) -> Option<OwnershipKind> {
        self.variables
            .get(&variable)
            .map(|node| node.ownership_kind)
    }

    /// Check if a variable has been moved
    pub fn is_moved(&self, variable: SymbolId) -> bool {
        self.variables
            .get(&variable)
            .map(|node| node.is_moved)
            .unwrap_or(false)
    }

    /// Get all variables borrowing from a given variable
    pub fn get_borrowers(&self, variable: SymbolId) -> Vec<SymbolId> {
        if let Some(node) = self.variables.get(&variable) {
            node.borrowed_by
                .iter()
                .filter_map(|&edge_id| self.borrow_edges.get(&edge_id).map(|edge| edge.borrower))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all variables a given variable borrows from
    pub fn get_borrowed_from(&self, variable: SymbolId) -> Vec<SymbolId> {
        if let Some(node) = self.variables.get(&variable) {
            node.borrows_from
                .iter()
                .filter_map(|&edge_id| self.borrow_edges.get(&edge_id).map(|edge| edge.borrowed))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Check if there are any mutable and immutable borrows of the same variable
    pub fn has_aliasing_violations(&self) -> Vec<OwnershipViolation> {
        let mut violations = Vec::new();

        for (variable, node) in &self.variables {
            let mut has_mutable_borrow = false;
            let mut has_immutable_borrow = false;
            let mut mutable_locations = Vec::new();
            let mut immutable_locations = Vec::new();

            for &edge_id in &node.borrowed_by {
                if let Some(edge) = self.borrow_edges.get(&edge_id) {
                    match edge.borrow_type {
                        BorrowType::Mutable => {
                            has_mutable_borrow = true;
                            mutable_locations.push(edge.borrow_location.clone());
                        }
                        BorrowType::Immutable => {
                            has_immutable_borrow = true;
                            immutable_locations.push(edge.borrow_location.clone());
                        }
                        BorrowType::Weak => {
                            // Weak references don't count for aliasing
                        }
                    }
                }
            }

            if has_mutable_borrow && has_immutable_borrow {
                violations.push(OwnershipViolation::AliasingViolation {
                    variable: *variable,
                    mutable_borrow_locations: mutable_locations,
                    immutable_borrow_locations: immutable_locations,
                });
            }
        }

        violations
    }

    /// Check for use-after-move violations
    pub fn check_use_after_move(&self) -> Vec<OwnershipViolation> {
        let mut violations = Vec::new();

        for (variable, node) in &self.variables {
            if node.is_moved {
                if let Some(move_edge_id) = node.move_site {
                    if let Some(move_edge) = self.move_edges.get(&move_edge_id) {
                        // Only report if there's a use AFTER the move
                        if let Some(uses) = self.use_sites.get(variable) {
                            for use_loc in uses {
                                if use_loc.line > move_edge.move_location.line
                                    || (use_loc.line == move_edge.move_location.line
                                        && use_loc.column > move_edge.move_location.column)
                                {
                                    violations.push(OwnershipViolation::UseAfterMove {
                                        variable: *variable,
                                        use_location: use_loc.clone(),
                                        move_location: move_edge.move_location.clone(),
                                        move_type: move_edge.move_type,
                                    });
                                    break; // One violation per variable is enough
                                }
                            }
                        }
                    }
                }
            }
        }

        violations
    }

    /// Update statistics for the ownership graph
    pub fn update_statistics(&mut self) {
        self.statistics.variable_count = self.variables.len();
        self.statistics.constraint_count = self.lifetime_constraints.len();
        self.statistics.borrow_count = self.borrow_edges.len();
        self.statistics.move_count = self.move_edges.len();
    }

    /// Validate the ownership graph for consistency
    pub fn validate(&self) -> Result<(), OwnershipValidationError> {
        // Check that all referenced lifetimes exist
        for (_, node) in &self.variables {
            if !self.lifetimes.contains_key(&node.lifetime) {
                return Err(OwnershipValidationError::InvalidLifetime {
                    variable: node.variable,
                    lifetime: node.lifetime,
                });
            }
        }

        // Check that all borrow edges reference valid variables
        for (_, edge) in &self.borrow_edges {
            if !self.variables.contains_key(&edge.borrower) {
                return Err(OwnershipValidationError::InvalidBorrow {
                    edge_id: edge.id,
                    variable: edge.borrower,
                });
            }
            if !self.variables.contains_key(&edge.borrowed) {
                return Err(OwnershipValidationError::InvalidBorrow {
                    edge_id: edge.id,
                    variable: edge.borrowed,
                });
            }
        }

        // Check that all move edges reference valid variables
        for (_, edge) in &self.move_edges {
            if !self.variables.contains_key(&edge.source) {
                return Err(OwnershipValidationError::InvalidMove {
                    edge_id: edge.id,
                    variable: edge.source,
                });
            }
        }

        Ok(())
    }
}

/// Ownership violation detected by analysis
#[derive(Debug, Clone)]
pub enum OwnershipViolation {
    /// Variable used after being moved
    UseAfterMove {
        variable: SymbolId,
        use_location: SourceLocation,
        move_location: SourceLocation,
        move_type: MoveType,
    },

    /// Aliasing violation (mutable and immutable borrows)
    AliasingViolation {
        variable: SymbolId,
        mutable_borrow_locations: Vec<SourceLocation>,
        immutable_borrow_locations: Vec<SourceLocation>,
    },

    /// Dangling pointer (use after lifetime expires)
    DanglingPointer {
        variable: SymbolId,
        use_location: SourceLocation,
        expired_lifetime: LifetimeId,
    },

    /// Double free (attempting to free already freed memory)
    DoubleFree {
        variable: SymbolId,
        first_free: SourceLocation,
        second_free: SourceLocation,
    },
}

/// Validation errors in ownership graph
#[derive(Debug)]
pub enum OwnershipValidationError {
    InvalidLifetime {
        variable: SymbolId,
        lifetime: LifetimeId,
    },
    InvalidBorrow {
        edge_id: BorrowEdgeId,
        variable: SymbolId,
    },
    InvalidMove {
        edge_id: MoveEdgeId,
        variable: SymbolId,
    },
}

impl fmt::Display for OwnershipKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OwnershipKind::Owned => write!(f, "owned"),
            OwnershipKind::Borrowed => write!(f, "borrowed"),
            OwnershipKind::BorrowedMut => write!(f, "borrowed_mut"),
            OwnershipKind::Shared => write!(f, "shared"),
            OwnershipKind::Unknown => write!(f, "unknown"),
            OwnershipKind::Moved => write!(f, "moved"),
        }
    }
}

impl fmt::Display for BorrowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BorrowType::Immutable => write!(f, "&"),
            BorrowType::Mutable => write!(f, "&mut"),
            BorrowType::Weak => write!(f, "&weak"),
        }
    }
}

impl fmt::Display for MoveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoveType::Explicit => write!(f, "explicit"),
            MoveType::Implicit => write!(f, "implicit"),
            MoveType::FunctionCall => write!(f, "function_call"),
            MoveType::Destruction => write!(f, "destruction"),
        }
    }
}

impl fmt::Display for OwnershipValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OwnershipValidationError::InvalidLifetime { variable, lifetime } => {
                write!(
                    f,
                    "Invalid lifetime {:?} for variable {:?}",
                    lifetime, variable
                )
            }
            OwnershipValidationError::InvalidBorrow { edge_id, variable } => {
                write!(
                    f,
                    "Invalid borrow edge {:?} references non-existent variable {:?}",
                    edge_id, variable
                )
            }
            OwnershipValidationError::InvalidMove { edge_id, variable } => {
                write!(
                    f,
                    "Invalid move edge {:?} references non-existent variable {:?}",
                    edge_id, variable
                )
            }
        }
    }
}

impl std::error::Error for OwnershipValidationError {}

impl Default for OwnershipGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceLocationTracking for OwnershipGraph {
    fn location_tracker(&self) -> &SourceLocationTracker {
        &self.location_tracker
    }

    fn location_tracker_mut(&mut self) -> &mut SourceLocationTracker {
        &mut self.location_tracker
    }
}
