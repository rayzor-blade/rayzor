use super::value::MacroValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Scoped variable environment for the macro interpreter.
///
/// Implements lexical scoping with nested scope frames. Variables
/// are looked up from innermost to outermost scope.
#[derive(Debug, Clone)]
pub struct Environment {
    /// Stack of variable scopes (innermost last)
    scopes: Vec<HashMap<String, MacroValue>>,
}

impl Environment {
    /// Create a new environment with a single global scope
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    /// Push a new scope onto the stack (e.g., entering a block or function)
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the innermost scope (e.g., leaving a block or function)
    ///
    /// Returns the popped scope's variables, or None if only the global scope remains.
    pub fn pop_scope(&mut self) -> Option<HashMap<String, MacroValue>> {
        if self.scopes.len() > 1 {
            self.scopes.pop()
        } else {
            None
        }
    }

    /// Look up a variable by name, searching from innermost to outermost scope
    pub fn get(&self, name: &str) -> Option<&MacroValue> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        None
    }

    /// Set an existing variable's value.
    ///
    /// Searches from innermost to outermost scope, updating the first
    /// occurrence found. Returns true if the variable was found and updated.
    pub fn set(&mut self, name: &str, value: MacroValue) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(name) {
                *slot = value;
                return true;
            }
        }
        false
    }

    /// Mutate a single field on an Object variable in-place.
    ///
    /// Avoids the clone→COW→reassign cycle for `this.field = value` patterns.
    /// Finds the variable by name, checks it is an Object, and inserts the field
    /// directly. Returns true if the variable was found and is an Object.
    pub fn mutate_object_field(&mut self, var_name: &str, field: &str, value: MacroValue) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(mac_val) = scope.get_mut(var_name) {
                if let MacroValue::Object(ref mut arc_map) = mac_val {
                    Arc::make_mut(arc_map).insert(field.to_string(), value);
                    return true;
                }
                return false;
            }
        }
        false
    }

    /// Define a new variable in the current (innermost) scope.
    ///
    /// If a variable with the same name already exists in the current scope,
    /// it will be overwritten (shadowing).
    pub fn define(&mut self, name: &str, value: MacroValue) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    /// Define a new variable using an already-owned String key.
    ///
    /// Avoids the `&str` → `to_string()` allocation in `define()` when the caller
    /// already has an owned String (e.g., cloned param names in function calls).
    pub fn define_owned(&mut self, name: String, value: MacroValue) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value);
        }
    }

    /// Check if a variable exists in any scope
    pub fn contains(&self, name: &str) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|scope| scope.contains_key(name))
    }

    /// Get the current scope depth (0 = global only)
    pub fn depth(&self) -> usize {
        self.scopes.len() - 1
    }

    /// Get all variable names visible in the current scope (for debugging/error messages)
    pub fn visible_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for scope in self.scopes.iter().rev() {
            for name in scope.keys() {
                if seen.insert(name.as_str()) {
                    names.push(name.as_str());
                }
            }
        }
        names.sort();
        names
    }

    /// Create a snapshot of all variables (for closures)
    pub fn capture_all(&self) -> HashMap<String, MacroValue> {
        let mut captured = HashMap::new();
        // From outermost to innermost so inner scopes shadow outer
        for scope in &self.scopes {
            for (name, value) in scope {
                captured.insert(name.clone(), value.clone());
            }
        }
        captured
    }

    /// Selectively capture only the specified variables (for closures)
    ///
    /// Only clones values for variables that are actually referenced in the closure body.
    /// This is much cheaper than `capture_all()` when the closure only uses a few variables
    /// from a large enclosing scope.
    pub fn capture_used(
        &self,
        used_names: &std::collections::HashSet<String>,
    ) -> HashMap<String, MacroValue> {
        let mut captured = HashMap::new();
        // From outermost to innermost so inner scopes shadow outer
        for scope in &self.scopes {
            for (name, value) in scope {
                if used_names.contains(name) {
                    captured.insert(name.clone(), value.clone());
                }
            }
        }
        captured
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_and_get() {
        let mut env = Environment::new();
        env.define("x", MacroValue::Int(42));
        assert_eq!(env.get("x"), Some(&MacroValue::Int(42)));
        assert_eq!(env.get("y"), None);
    }

    #[test]
    fn test_scoping() {
        let mut env = Environment::new();
        env.define("x", MacroValue::Int(1));

        env.push_scope();
        env.define("x", MacroValue::Int(2)); // shadow
        env.define("y", MacroValue::Int(3));
        assert_eq!(env.get("x"), Some(&MacroValue::Int(2)));
        assert_eq!(env.get("y"), Some(&MacroValue::Int(3)));

        env.pop_scope();
        assert_eq!(env.get("x"), Some(&MacroValue::Int(1)));
        assert_eq!(env.get("y"), None);
    }

    #[test]
    fn test_set_updates_existing() {
        let mut env = Environment::new();
        env.define("x", MacroValue::Int(1));

        env.push_scope();
        // set should update the outer scope's variable, not create a new one
        assert!(env.set("x", MacroValue::Int(99)));
        assert_eq!(env.get("x"), Some(&MacroValue::Int(99)));

        env.pop_scope();
        // The outer scope's value should be updated
        assert_eq!(env.get("x"), Some(&MacroValue::Int(99)));
    }

    #[test]
    fn test_set_returns_false_for_undefined() {
        let mut env = Environment::new();
        assert!(!env.set("nonexistent", MacroValue::Int(1)));
    }

    #[test]
    fn test_contains() {
        let mut env = Environment::new();
        env.define("x", MacroValue::Int(1));
        assert!(env.contains("x"));
        assert!(!env.contains("y"));
    }

    #[test]
    fn test_depth() {
        let mut env = Environment::new();
        assert_eq!(env.depth(), 0);
        env.push_scope();
        assert_eq!(env.depth(), 1);
        env.push_scope();
        assert_eq!(env.depth(), 2);
        env.pop_scope();
        assert_eq!(env.depth(), 1);
    }

    #[test]
    fn test_visible_names() {
        let mut env = Environment::new();
        env.define("b", MacroValue::Int(1));
        env.define("a", MacroValue::Int(2));
        env.push_scope();
        env.define("c", MacroValue::Int(3));

        let names = env.visible_names();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_capture_all() {
        let mut env = Environment::new();
        env.define("x", MacroValue::Int(1));
        env.push_scope();
        env.define("x", MacroValue::Int(2)); // shadows
        env.define("y", MacroValue::Int(3));

        let captured = env.capture_all();
        // Inner scope's x should win
        assert_eq!(captured.get("x"), Some(&MacroValue::Int(2)));
        assert_eq!(captured.get("y"), Some(&MacroValue::Int(3)));
    }

    #[test]
    fn test_cannot_pop_global_scope() {
        let mut env = Environment::new();
        assert!(env.pop_scope().is_none());
        // Should still work after trying to pop
        env.define("x", MacroValue::Int(1));
        assert_eq!(env.get("x"), Some(&MacroValue::Int(1)));
    }
}
