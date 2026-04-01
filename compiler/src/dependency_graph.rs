//! Dependency Graph for Multi-File Compilation
//!
//! This module provides dependency analysis and topological sorting for Haxe files.
//! It detects circular dependencies and determines the correct compilation order.

use parser::HaxeFile;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Represents a file in the dependency graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileNode {
    /// Fully qualified package name (e.g., "com.example.model.User")
    pub qualified_name: String,

    /// File path (for error reporting)
    pub file_path: String,

    /// Index in the original file list
    pub file_index: usize,
}

/// Dependency graph for file compilation ordering
pub struct DependencyGraph {
    /// Nodes in the graph (package name -> node info)
    nodes: BTreeMap<String, FileNode>,

    /// Edges in the graph (from -> [to])
    /// If A imports B, there's an edge from A to B (A depends on B)
    edges: BTreeMap<String, BTreeSet<String>>,

    /// Reverse edges (to -> [from])
    reverse_edges: BTreeMap<String, BTreeSet<String>>,
}

/// Result of dependency analysis
#[derive(Debug)]
pub struct DependencyAnalysis {
    /// Files in topological order (dependencies first)
    pub compilation_order: Vec<usize>,

    /// Detected circular dependencies (if any)
    pub circular_dependencies: Vec<CircularDependency>,
}

/// Represents a circular dependency cycle
#[derive(Debug, Clone)]
pub struct CircularDependency {
    /// The cycle of package names (e.g., ["A", "B", "C", "A"])
    pub cycle: Vec<String>,

    /// File paths involved in the cycle
    pub file_paths: Vec<String>,
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            reverse_edges: BTreeMap::new(),
        }
    }

    /// Build a dependency graph from a list of parsed Haxe files
    ///
    /// # Arguments
    /// * `files` - List of parsed HaxeFile structures
    ///
    /// # Returns
    /// A dependency graph with all files and their import relationships
    pub fn from_files(files: &[HaxeFile]) -> Self {
        let mut graph = Self::new();

        // First pass: Register all files as nodes
        for (index, file) in files.iter().enumerate() {
            let package_name = Self::extract_package_name(file);
            let file_path = file.filename.clone();

            let node = FileNode {
                qualified_name: package_name.clone(),
                file_path,
                file_index: index,
            };

            graph.nodes.insert(package_name, node);
        }

        // Second pass: Add edges based on imports
        // Edge direction: if A imports B, edge is B → A (B must compile before A)
        for (_index, file) in files.iter().enumerate() {
            let from_package = Self::extract_package_name(file);

            // Get all imported packages
            let imported_packages = Self::extract_imports(file);

            for imported_package in imported_packages {
                // Only add edge if the imported package is in our file set
                if graph.nodes.contains_key(&imported_package) {
                    // Edge from dependency TO dependent (dependency must come first)
                    graph.add_edge(&imported_package, &from_package);
                }
                // If not in our file set, it's either stdlib or external - ignore
            }
        }

        graph
    }

    /// Extract the package name from a file
    /// Returns the full qualified name including the main type
    fn extract_package_name(file: &HaxeFile) -> String {
        let package_prefix = if let Some(ref pkg) = file.package {
            pkg.path.join(".")
        } else {
            String::new()
        };

        // Get the main type name from the file
        // In Haxe, the file name should match the main type name
        let type_name = if let Some(decl) = file.declarations.first() {
            match decl {
                parser::TypeDeclaration::Class(c) => &c.name,
                parser::TypeDeclaration::Interface(i) => &i.name,
                parser::TypeDeclaration::Enum(e) => &e.name,
                parser::TypeDeclaration::Typedef(t) => &t.name,
                parser::TypeDeclaration::Abstract(a) => &a.name,
                parser::TypeDeclaration::Conditional(_) => "Conditional",
            }
        } else {
            // Fallback: extract from filename
            file.filename
                .split('/')
                .last()
                .and_then(|name| name.strip_suffix(".hx"))
                .unwrap_or("Unknown")
        };

        if package_prefix.is_empty() {
            type_name.to_string()
        } else {
            format!("{}.{}", package_prefix, type_name)
        }
    }

    /// Extract all imported package names from a file
    fn extract_imports(file: &HaxeFile) -> Vec<String> {
        file.imports
            .iter()
            .map(|import| import.path.join("."))
            .collect()
    }

    /// Add an edge from one package to another
    fn add_edge(&mut self, from: &str, to: &str) {
        // Add forward edge
        self.edges
            .entry(from.to_string())
            .or_insert_with(BTreeSet::new)
            .insert(to.to_string());

        // Add reverse edge
        self.reverse_edges
            .entry(to.to_string())
            .or_insert_with(BTreeSet::new)
            .insert(from.to_string());
    }

    /// Perform dependency analysis
    ///
    /// This detects circular dependencies and computes a valid compilation order.
    /// If circular dependencies exist, they are reported but a best-effort order
    /// is still provided.
    pub fn analyze(&self) -> DependencyAnalysis {
        let mut circular_dependencies = Vec::new();

        // Detect cycles using DFS
        let mut visited = BTreeSet::new();
        let mut rec_stack = BTreeSet::new();
        let mut path = Vec::new();

        for node_name in self.nodes.keys() {
            if !visited.contains(node_name) {
                self.detect_cycles(
                    node_name,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut circular_dependencies,
                );
            }
        }

        // Compute topological order using Kahn's algorithm
        let compilation_order = self.topological_sort();

        DependencyAnalysis {
            compilation_order,
            circular_dependencies,
        }
    }

    /// Detect cycles using DFS
    fn detect_cycles(
        &self,
        node: &str,
        visited: &mut BTreeSet<String>,
        rec_stack: &mut BTreeSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<CircularDependency>,
    ) {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = self.edges.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    self.detect_cycles(neighbor, visited, rec_stack, path, cycles);
                } else if rec_stack.contains(neighbor) {
                    // Found a cycle!
                    let cycle_start = path.iter().position(|n| n == neighbor).unwrap();
                    let cycle_path: Vec<String> = path[cycle_start..].to_vec();
                    let mut cycle_path_closed = cycle_path.clone();
                    cycle_path_closed.push(neighbor.to_string()); // Close the cycle

                    let file_paths: Vec<String> = cycle_path
                        .iter()
                        .filter_map(|name| self.nodes.get(name))
                        .map(|node| node.file_path.clone())
                        .collect();

                    cycles.push(CircularDependency {
                        cycle: cycle_path_closed,
                        file_paths,
                    });
                }
            }
        }

        path.pop();
        rec_stack.remove(node);
    }

    /// Compute topological sort using Kahn's algorithm
    ///
    /// Returns file indices in compilation order (dependencies first)
    fn topological_sort(&self) -> Vec<usize> {
        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        let mut result = Vec::new();

        // Calculate in-degree for each node
        for node_name in self.nodes.keys() {
            in_degree.insert(node_name.clone(), 0);
        }

        for neighbors in self.edges.values() {
            for neighbor in neighbors {
                *in_degree.entry(neighbor.clone()).or_insert(0) += 1;
            }
        }

        // Start with nodes that have no dependencies
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(name, _)| name.clone())
            .collect();

        // Process nodes in topological order
        while let Some(node_name) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&node_name) {
                result.push(node.file_index);
            }

            // Reduce in-degree for neighbors
            if let Some(neighbors) = self.edges.get(&node_name) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        // If we didn't process all nodes, there's a cycle
        // In this case, add remaining nodes in arbitrary order
        if result.len() < self.nodes.len() {
            for (node_name, node) in &self.nodes {
                if !result.contains(&node.file_index) {
                    result.push(node.file_index);
                }
            }
        }

        result
    }

    /// Get all dependencies of a given package (transitive)
    pub fn get_all_dependencies(&self, package: &str) -> BTreeSet<String> {
        let mut deps = BTreeSet::new();
        let mut to_visit = VecDeque::new();
        to_visit.push_back(package.to_string());

        while let Some(current) = to_visit.pop_front() {
            if let Some(neighbors) = self.edges.get(&current) {
                for neighbor in neighbors {
                    if deps.insert(neighbor.clone()) {
                        to_visit.push_back(neighbor.clone());
                    }
                }
            }
        }

        deps
    }

    /// Get all dependents of a given package (transitive)
    pub fn get_all_dependents(&self, package: &str) -> BTreeSet<String> {
        let mut dependents = BTreeSet::new();
        let mut to_visit = VecDeque::new();
        to_visit.push_back(package.to_string());

        while let Some(current) = to_visit.pop_front() {
            if let Some(neighbors) = self.reverse_edges.get(&current) {
                for neighbor in neighbors {
                    if dependents.insert(neighbor.clone()) {
                        to_visit.push_back(neighbor.clone());
                    }
                }
            }
        }

        dependents
    }

    /// Get direct dependencies of a package
    pub fn get_direct_dependencies(&self, package: &str) -> Vec<String> {
        self.edges
            .get(package)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get direct dependents of a package
    pub fn get_direct_dependents(&self, package: &str) -> Vec<String> {
        self.reverse_edges
            .get(package)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl CircularDependency {
    /// Format the circular dependency as a readable error message
    pub fn format_error(&self) -> String {
        let cycle_str = self.cycle.join(" -> ");
        let files_str = self.file_paths.join("\n  ");

        format!(
            "Circular dependency detected:\n  {}\n\nFiles involved:\n  {}",
            cycle_str, files_str
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parser::{HaxeFile, Import, ImportMode, Package, Span};

    fn create_test_file(
        name: &str,
        package: Option<Vec<&str>>,
        imports: Vec<Vec<&str>>,
    ) -> HaxeFile {
        HaxeFile {
            filename: format!("{}.hx", name),
            input: None,
            package: package.map(|p| Package {
                path: p.iter().map(|s| s.to_string()).collect(),
                span: Span::new(0, 0),
            }),
            imports: imports
                .into_iter()
                .map(|i| Import {
                    path: i.iter().map(|s| s.to_string()).collect(),
                    mode: ImportMode::Normal,
                    span: Span::new(0, 0),
                })
                .collect(),
            using: Vec::new(),
            module_fields: Vec::new(),
            declarations: Vec::new(),
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn test_simple_dependency_order() {
        // A depends on B, B depends on C
        let files = vec![
            create_test_file("A", Some(vec!["com"]), vec![vec!["com", "B"]]),
            create_test_file("B", Some(vec!["com"]), vec![vec!["com", "C"]]),
            create_test_file("C", Some(vec!["com"]), vec![]),
        ];

        let graph = DependencyGraph::from_files(&files);
        let analysis = graph.analyze();

        assert_eq!(analysis.circular_dependencies.len(), 0);

        // C should come before B, B before A
        let order = analysis.compilation_order;
        assert_eq!(order.len(), 3);

        let c_idx = order.iter().position(|&i| i == 2).unwrap();
        let b_idx = order.iter().position(|&i| i == 1).unwrap();
        let a_idx = order.iter().position(|&i| i == 0).unwrap();

        assert!(c_idx < b_idx);
        assert!(b_idx < a_idx);
    }

    #[test]
    fn test_circular_dependency_detection() {
        // A -> B -> C -> A (circular)
        let files = vec![
            create_test_file("A", Some(vec!["com"]), vec![vec!["com", "B"]]),
            create_test_file("B", Some(vec!["com"]), vec![vec!["com", "C"]]),
            create_test_file("C", Some(vec!["com"]), vec![vec!["com", "A"]]),
        ];

        let graph = DependencyGraph::from_files(&files);
        let analysis = graph.analyze();

        assert!(analysis.circular_dependencies.len() > 0);

        let cycle = &analysis.circular_dependencies[0];
        assert!(cycle.cycle.len() >= 3);
    }

    #[test]
    fn test_independent_files() {
        // A, B, C have no dependencies
        let files = vec![
            create_test_file("A", Some(vec!["com"]), vec![]),
            create_test_file("B", Some(vec!["com"]), vec![]),
            create_test_file("C", Some(vec!["com"]), vec![]),
        ];

        let graph = DependencyGraph::from_files(&files);
        let analysis = graph.analyze();

        assert_eq!(analysis.circular_dependencies.len(), 0);
        assert_eq!(analysis.compilation_order.len(), 3);
    }
}
