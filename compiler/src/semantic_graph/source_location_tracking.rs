//! Source Location Tracking System for Semantic Graph
//!
//! This module provides utilities for tracking source locations throughout
//! the semantic graph construction and analysis process. It helps maintain
//! accurate location information for error reporting and debugging.

use crate::tast::{BlockId, CallSiteId, DataFlowNodeId, SourceLocation, SymbolId};
use std::collections::BTreeMap;

/// Source location tracker for semantic graph construction
#[derive(Debug, Clone)]
pub struct SourceLocationTracker {
    /// Symbol to source location mapping
    symbol_locations: BTreeMap<SymbolId, SourceLocation>,

    /// Data flow node to source location mapping
    node_locations: BTreeMap<DataFlowNodeId, SourceLocation>,

    /// Basic block to source location mapping
    block_locations: BTreeMap<BlockId, SourceLocation>,

    /// Call site to source location mapping
    call_site_locations: BTreeMap<CallSiteId, SourceLocation>,

    /// Default location for missing mappings
    default_location: SourceLocation,
}

impl SourceLocationTracker {
    /// Create a new source location tracker
    pub fn new() -> Self {
        Self {
            symbol_locations: BTreeMap::new(),
            node_locations: BTreeMap::new(),
            block_locations: BTreeMap::new(),
            call_site_locations: BTreeMap::new(),
            default_location: SourceLocation::unknown(),
        }
    }

    /// Set the default location for missing mappings
    pub fn set_default_location(&mut self, location: SourceLocation) {
        self.default_location = location;
    }

    /// Add a symbol location mapping
    pub fn add_symbol_location(&mut self, symbol: SymbolId, location: SourceLocation) {
        self.symbol_locations.insert(symbol, location);
    }

    /// Add a node location mapping
    pub fn add_node_location(&mut self, node: DataFlowNodeId, location: SourceLocation) {
        self.node_locations.insert(node, location);
    }

    /// Add a block location mapping
    pub fn add_block_location(&mut self, block: BlockId, location: SourceLocation) {
        self.block_locations.insert(block, location);
    }

    /// Add a call site location mapping
    pub fn add_call_site_location(&mut self, call_site: CallSiteId, location: SourceLocation) {
        self.call_site_locations.insert(call_site, location);
    }

    /// Get the source location for a symbol
    pub fn get_symbol_location(&self, symbol: SymbolId) -> SourceLocation {
        self.symbol_locations
            .get(&symbol)
            .cloned()
            .unwrap_or(self.default_location)
    }

    /// Get the source location for a node
    pub fn get_node_location(&self, node: DataFlowNodeId) -> SourceLocation {
        self.node_locations
            .get(&node)
            .cloned()
            .unwrap_or(self.default_location)
    }

    /// Get the source location for a block
    pub fn get_block_location(&self, block: BlockId) -> SourceLocation {
        self.block_locations
            .get(&block)
            .cloned()
            .unwrap_or(self.default_location)
    }

    /// Get the source location for a call site
    pub fn get_call_site_location(&self, call_site: CallSiteId) -> SourceLocation {
        self.call_site_locations
            .get(&call_site)
            .cloned()
            .unwrap_or(self.default_location)
    }

    /// Get the best available location for a constraint
    pub fn get_constraint_location(
        &self,
        symbol: Option<SymbolId>,
        node: Option<DataFlowNodeId>,
        block: Option<BlockId>,
    ) -> SourceLocation {
        // Try to find the most specific location available
        if let Some(node_id) = node {
            if let Some(location) = self.node_locations.get(&node_id) {
                return *location;
            }
        }

        if let Some(symbol_id) = symbol {
            if let Some(location) = self.symbol_locations.get(&symbol_id) {
                return *location;
            }
        }

        if let Some(block_id) = block {
            if let Some(location) = self.block_locations.get(&block_id) {
                return *location;
            }
        }

        self.default_location
    }

    /// Clear all location mappings
    pub fn clear(&mut self) {
        self.symbol_locations.clear();
        self.node_locations.clear();
        self.block_locations.clear();
        self.call_site_locations.clear();
    }
}

impl Default for SourceLocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for adding source location tracking to analysis components
pub trait SourceLocationTracking {
    /// Get the source location tracker
    fn location_tracker(&self) -> &SourceLocationTracker;

    /// Get the mutable source location tracker
    fn location_tracker_mut(&mut self) -> &mut SourceLocationTracker;

    /// Get location for a symbol with fallback
    fn get_symbol_location_or_default(&self, symbol: SymbolId) -> SourceLocation {
        self.location_tracker().get_symbol_location(symbol)
    }

    /// Get location for a node with fallback
    fn get_node_location_or_default(&self, node: DataFlowNodeId) -> SourceLocation {
        self.location_tracker().get_node_location(node)
    }

    /// Get location for a block with fallback
    fn get_block_location_or_default(&self, block: BlockId) -> SourceLocation {
        self.location_tracker().get_block_location(block)
    }
}

/// Helper function to create a source location from line and column
pub fn create_source_location(
    line: u32,
    column: u32,
    file_id: u32,
    byte_offset: u32,
) -> SourceLocation {
    SourceLocation {
        line,
        column,
        file_id,
        byte_offset,
    }
}

/// Helper function to merge two source locations, preferring the first valid one
pub fn merge_source_locations(primary: SourceLocation, fallback: SourceLocation) -> SourceLocation {
    if primary.is_valid() {
        primary
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_tracker() {
        let mut tracker = SourceLocationTracker::new();

        let symbol = SymbolId::from_raw(1);
        let location = create_source_location(10, 5, 1, 0);

        tracker.add_symbol_location(symbol, location);

        let retrieved = tracker.get_symbol_location(symbol);
        assert_eq!(retrieved.line, 10);
        assert_eq!(retrieved.column, 5);
        assert_eq!(retrieved.file_id, 1);
    }

    #[test]
    fn test_location_fallback() {
        let tracker = SourceLocationTracker::new();

        // Should return default location for unknown symbol
        let symbol = SymbolId::from_raw(999);
        let location = tracker.get_symbol_location(symbol);
        assert_eq!(location, SourceLocation::unknown());
    }

    #[test]
    fn test_constraint_location_priority() {
        let mut tracker = SourceLocationTracker::new();

        let symbol = SymbolId::from_raw(1);
        let node = DataFlowNodeId::from_raw(1);
        let block = BlockId::from_raw(1);

        let symbol_loc = create_source_location(10, 5, 0, 0);
        let node_loc = create_source_location(15, 10, 0, 1);
        let block_loc = create_source_location(20, 15, 0, 2);

        tracker.add_symbol_location(symbol, symbol_loc);
        tracker.add_node_location(node, node_loc);
        tracker.add_block_location(block, block_loc);

        // Should prefer node location over symbol location
        let location = tracker.get_constraint_location(Some(symbol), Some(node), Some(block));
        assert_eq!(location.line, 15);

        // Should prefer symbol location when node is not available
        let location = tracker.get_constraint_location(Some(symbol), None, Some(block));
        assert_eq!(location.line, 10);

        // Should use block location when others are not available
        let location = tracker.get_constraint_location(None, None, Some(block));
        assert_eq!(location.line, 20);
    }
}
