//! HXML Build File Parser
//!
//! Haxe uses .hxml files to specify compilation parameters.
//! This module provides compatibility with existing Haxe projects.
//!
//! Rayzor simplifies HXML by:
//! - Ignoring traditional Haxe targets (--js, --cpp, etc.)
//! - Defaulting to Rayzor's JIT mode
//! - Supporting --rayzor-jit (default) or --rayzor-compile for AOT
//!
//! # Example HXML File
//! ```hxml
//! -cp src
//! -main Main
//! -lib lime
//! -D analyzer-optimize
//! --rayzor-jit
//! ```

use std::path::PathBuf;

/// HXML compilation configuration
#[derive(Debug, Clone)]
pub struct HxmlConfig {
    /// Class paths (-cp)
    pub class_paths: Vec<PathBuf>,

    /// Main class (-main)
    pub main_class: Option<String>,

    /// Output file (optional for JIT, required for compile)
    pub output: Option<PathBuf>,

    /// Rayzor compilation mode
    pub mode: RayzorMode,

    /// Libraries to include (-lib)
    pub libraries: Vec<String>,

    /// Defines (-D)
    pub defines: Vec<(String, Option<String>)>,

    /// Resources to embed (-resource)
    pub resources: Vec<(PathBuf, Option<String>)>,

    /// Debug mode (-debug)
    pub debug: bool,

    /// Verbose output (-v)
    pub verbose: bool,

    /// Additional compiler flags
    pub compiler_flags: Vec<String>,

    /// Source files
    pub source_files: Vec<PathBuf>,
}

impl Default for HxmlConfig {
    fn default() -> Self {
        Self {
            class_paths: Vec::new(),
            main_class: None,
            output: None,
            mode: RayzorMode::Jit, // Default to JIT
            libraries: Vec::new(),
            defines: Vec::new(),
            resources: Vec::new(),
            debug: false,
            verbose: false,
            compiler_flags: Vec::new(),
            source_files: Vec::new(),
        }
    }
}

/// Rayzor compilation mode
#[derive(Debug, Clone, PartialEq)]
pub enum RayzorMode {
    /// JIT compile and execute (default)
    Jit,
    /// AOT compile to native binary
    Compile,
}

impl HxmlConfig {
    /// Parse an HXML file
    pub fn from_file(path: &PathBuf) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read HXML file: {}", e))?;

        Self::from_string(&content)
    }

    /// Parse HXML content from a string
    pub fn from_string(content: &str) -> Result<Self, String> {
        let mut config = HxmlConfig::default();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse the line
            if let Some(rest) = line.strip_prefix("-cp ") {
                config.class_paths.push(PathBuf::from(rest.trim()));
            } else if let Some(rest) = line.strip_prefix("--class-path ") {
                config.class_paths.push(PathBuf::from(rest.trim()));
            } else if let Some(rest) = line.strip_prefix("-main ") {
                config.main_class = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("-lib ") {
                config.libraries.push(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("-D ") {
                let parts: Vec<&str> = rest.trim().splitn(2, '=').collect();
                if parts.len() == 2 {
                    config
                        .defines
                        .push((parts[0].to_string(), Some(parts[1].to_string())));
                } else {
                    config.defines.push((parts[0].to_string(), None));
                }
            } else if line == "--rayzor-jit" {
                config.mode = RayzorMode::Jit;
            } else if line == "--rayzor-compile" {
                config.mode = RayzorMode::Compile;
            } else if let Some(rest) = line.strip_prefix("--rayzor-jit ") {
                config.mode = RayzorMode::Jit;
                config.output = Some(PathBuf::from(rest.trim()));
            } else if let Some(rest) = line.strip_prefix("--rayzor-compile ") {
                config.mode = RayzorMode::Compile;
                config.output = Some(PathBuf::from(rest.trim()));
            } else if let Some(rest) = line.strip_prefix("-output ") {
                config.output = Some(PathBuf::from(rest.trim()));
            } else if let Some(rest) = line.strip_prefix("--output ") {
                config.output = Some(PathBuf::from(rest.trim()));
            } else if line.starts_with("--js ")
                || line.starts_with("--cpp ")
                || line.starts_with("--cs ")
                || line.starts_with("--java ")
                || line.starts_with("--python ")
                || line.starts_with("--lua ")
                || line.starts_with("--php ")
            {
                // Ignore traditional Haxe targets - Rayzor uses --rayzor-jit or --rayzor-compile
                if config.verbose {
                    eprintln!(
                        "Note: Ignoring traditional Haxe target: {}. Use --rayzor-jit or --rayzor-compile",
                        line
                    );
                }
            } else if let Some(rest) = line.strip_prefix("-resource ") {
                let parts: Vec<&str> = rest.trim().splitn(2, '@').collect();
                if parts.len() == 2 {
                    config
                        .resources
                        .push((PathBuf::from(parts[0]), Some(parts[1].to_string())));
                } else {
                    config.resources.push((PathBuf::from(parts[0]), None));
                }
            } else if line == "-debug" {
                config.debug = true;
            } else if line == "-v" || line == "--verbose" {
                config.verbose = true;
            } else if line.starts_with('-') || line.starts_with("--") {
                // Unknown compiler flag - store for potential future use
                config.compiler_flags.push(line.to_string());
            } else {
                // Assume it's a source file
                config.source_files.push(PathBuf::from(line));
            }
        }

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.main_class.is_none() && self.source_files.is_empty() {
            return Err("No main class or source files specified".to_string());
        }

        // Compile mode requires output file
        if self.mode == RayzorMode::Compile && self.output.is_none() {
            return Err("Compile mode requires an output file. Use --rayzor-compile <output> or --output <file>".to_string());
        }

        Ok(())
    }

    /// Get a summary of the configuration
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Class paths: {}\n", self.class_paths.len()));
        s.push_str(&format!("Main class: {:?}\n", self.main_class));
        s.push_str(&format!("Mode: {:?}\n", self.mode));
        s.push_str(&format!("Output: {:?}\n", self.output));
        s.push_str(&format!("Libraries: {}\n", self.libraries.len()));
        s.push_str(&format!("Defines: {}\n", self.defines.len()));
        s.push_str(&format!("Debug: {}\n", self.debug));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hxml() {
        let hxml = r#"
# Build configuration
-cp src
-cp lib
-main Main
-lib lime
-D analyzer-optimize
-D MY_FLAG=value
--rayzor-jit
-debug
-v
        "#;

        let config = HxmlConfig::from_string(hxml).unwrap();

        assert_eq!(config.class_paths.len(), 2);
        assert_eq!(config.main_class, Some("Main".to_string()));
        assert_eq!(config.mode, RayzorMode::Jit);
        assert_eq!(config.libraries.len(), 1);
        assert_eq!(config.defines.len(), 2);
        assert_eq!(config.debug, true);
        assert_eq!(config.verbose, true);
    }

    #[test]
    fn test_jit_mode() {
        let hxml = "-main Test\n--rayzor-jit";
        let config = HxmlConfig::from_string(hxml).unwrap();
        assert_eq!(config.mode, RayzorMode::Jit);
    }

    #[test]
    fn test_compile_mode() {
        let hxml = "-main Test\n--rayzor-compile output.bin";
        let config = HxmlConfig::from_string(hxml).unwrap();
        assert_eq!(config.mode, RayzorMode::Compile);
        assert_eq!(config.output, Some(PathBuf::from("output.bin")));
    }

    #[test]
    fn test_ignore_traditional_targets() {
        let hxml = "-main Test\n--js output.js\n--rayzor-jit";
        let config = HxmlConfig::from_string(hxml).unwrap();
        // Should use Rayzor mode, ignore --js
        assert_eq!(config.mode, RayzorMode::Jit);
    }

    #[test]
    fn test_default_to_jit() {
        let hxml = "-main Test\n-cp src";
        let config = HxmlConfig::from_string(hxml).unwrap();
        // Should default to JIT mode
        assert_eq!(config.mode, RayzorMode::Jit);
    }
}
