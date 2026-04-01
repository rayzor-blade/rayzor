//! Standard Library Loader
//!
//! This module handles loading Haxe standard library types from source files
//! rather than hardcoding them. This follows Haxe's actual implementation.

use log::{info, warn};
use parser::{parse_haxe_file_with_diagnostics, ErrorFormatter, HaxeFile};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Strip ANSI escape codes from a string for cleaner error output
fn strip_ansi_codes(input: &str) -> String {
    // Simple regex to remove ANSI escape codes
    // This matches ESC[ followed by any number of digits, semicolons, and ends with a letter
    let mut result = String::new();
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            // Skip the ESC[
            chars.next(); // consume '['

            // Skip until we find a letter (the final character of the escape sequence)
            while let Some(c) = chars.next() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }

    result
}

/// Standard library loader configuration
#[derive(Debug, Clone)]
pub struct StdLibConfig {
    /// Paths to search for standard library files
    pub std_paths: Vec<PathBuf>,

    /// Whether to load import.hx files automatically
    pub load_import_hx: bool,

    /// Package imports that are always available (top-level)
    pub default_imports: Vec<String>,
}

impl Default for StdLibConfig {
    fn default() -> Self {
        Self {
            std_paths: vec![
                // Common Haxe standard library locations
                PathBuf::from("/usr/lib/haxe/std"),
                PathBuf::from("/usr/local/lib/haxe/std"),
                PathBuf::from("~/.haxe/std"),
            ],
            load_import_hx: true,
            default_imports: vec![
                // Top-level types that are always imported
                "StdTypes.hx".to_string(), // Contains Int, Float, String, Bool, etc.
            ],
        }
    }
}

/// Loader for Haxe standard library
pub struct StdLibLoader {
    config: StdLibConfig,
    /// Cache of loaded files to avoid re-parsing
    loaded_files: BTreeMap<PathBuf, HaxeFile>,
    /// Preprocessor config for conditional compilation (#if wasm, etc.)
    pp_config: parser::preprocessor::PreprocessorConfig,
}

impl StdLibLoader {
    pub fn new(config: StdLibConfig) -> Self {
        Self {
            config,
            loaded_files: BTreeMap::new(),
            pp_config: parser::preprocessor::PreprocessorConfig::default(),
        }
    }

    /// Set extra preprocessor defines (e.g., "wasm" for WASM targets).
    pub fn set_preprocessor_config(&mut self, pp_config: parser::preprocessor::PreprocessorConfig) {
        self.pp_config = pp_config;
    }

    /// Load a standard library file by name
    pub fn load_std_file(&mut self, filename: &str) -> Result<&HaxeFile, String> {
        // Filter out StdTypes typedefs that don't have separate files
        // Iterator, KeyValueIterator, Iterable, KeyValueIterable are defined in StdTypes.hx
        if filename == "Iterator.hx"
            || filename == "KeyValueIterator.hx"
            || filename == "Iterable.hx"
            || filename == "KeyValueIterable.hx"
        {
            return Err(format!(
                "'{}' is a typedef in StdTypes.hx, not a separate file",
                filename
            ));
        }

        // Try to find the file in standard paths
        for std_path in &self.config.std_paths {
            let file_path = std_path.join(filename);
            if file_path.exists() {
                return self.load_file(&file_path);
            }
        }

        Err(format!(
            "Standard library file '{}' not found in any std path",
            filename
        ))
    }

    /// Load a specific file
    fn load_file(&mut self, path: &Path) -> Result<&HaxeFile, String> {
        // Check cache first
        if self.loaded_files.contains_key(path) {
            return Ok(self.loaded_files.get(path).unwrap());
        }

        // Read and parse the file
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        // Parse with preprocessor config to handle #if wasm etc.
        let mut haxe_file = parser::haxe_parser::parse_haxe_file_with_config(
            path.to_str().unwrap_or("unknown.hx"),
            &content,
            true,
            true,
            &self.pp_config,
        )
        .map_err(|e| {
            let clean_error = strip_ansi_codes(&e);
            format!("Failed to parse {}: {}", path.display(), clean_error)
        })?;

        // IMPORTANT: Preserve the source code so it can be compiled later
        // The parser doesn't preserve source by default to save memory,
        // but we need it for the compilation pipeline
        haxe_file.input = Some(content);

        // Cache the parsed file
        self.loaded_files.insert(path.to_path_buf(), haxe_file);
        Ok(self.loaded_files.get(path).unwrap())
    }

    /// Load all default imports (top-level types)
    pub fn load_default_imports(&mut self) -> Vec<HaxeFile> {
        let mut files = Vec::new();

        let default_imports = self.config.default_imports.clone();
        for import_file in &default_imports {
            match self.load_std_file(import_file) {
                Ok(file) => files.push(file.clone()),
                Err(e) => {
                    // Log warning but continue - some files might not exist
                    warn!("{}", e);
                }
            }
        }

        files
    }

    /// Load only root-level stdlib files (not subdirectories)
    /// Files in subdirectories (haxe/, sys/, etc.) should be loaded on-demand via imports
    pub fn load_root_stdlib(&mut self) -> Vec<HaxeFile> {
        let mut files = Vec::new();

        // Find the first valid stdlib path
        let stdlib_path = self.config.std_paths.iter().find(|p| p.exists()).cloned();

        if let Some(path) = stdlib_path {
            info!("Loading root stdlib from: {}", path.display());

            // Only load .hx files directly in the root directory
            let entries = match std::fs::read_dir(&path) {
                Ok(e) => e,
                Err(err) => {
                    warn!("Failed to read directory {:?}: {}", path, err);
                    return files;
                }
            };

            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                let file_path = entry.path();

                // Only load .hx files, skip directories
                if file_path.is_file()
                    && file_path.extension().and_then(|s| s.to_str()) == Some("hx")
                {
                    match self.load_file(&file_path) {
                        Ok(file) => {
                            files.push(file.clone());
                        }
                        Err(e) => {
                            let clean_error = strip_ansi_codes(&e);
                            warn!("Failed to parse {}: {}", file_path.display(), clean_error);
                        }
                    }
                }
            }

            // Load additional essential files from subdirectories that are referenced by root files
            // Preload all iterator types since they're commonly used and Array.hx imports ArrayKeyValueIterator
            // Preload all exception types since they reference each other (NotImplementedException extends PosException)
            let essential_subdirectory_files = vec![
                // All iterators from haxe/iterators/ (12 files)
                "haxe/iterators/ArrayIterator.hx",
                "haxe/iterators/ArrayKeyValueIterator.hx",
                "haxe/iterators/DynamicAccessIterator.hx",
                "haxe/iterators/DynamicAccessKeyValueIterator.hx",
                "haxe/iterators/HashMapKeyValueIterator.hx",
                "haxe/iterators/MapKeyValueIterator.hx",
                "haxe/iterators/RestIterator.hx",
                "haxe/iterators/RestKeyValueIterator.hx",
                "haxe/iterators/StringIterator.hx",
                "haxe/iterators/StringIteratorUnicode.hx",
                "haxe/iterators/StringKeyValueIterator.hx",
                "haxe/iterators/StringKeyValueIteratorUnicode.hx",
                // All exceptions from haxe/exceptions/ (3 files)
                "haxe/exceptions/PosException.hx",
                "haxe/exceptions/ArgumentException.hx",
                "haxe/exceptions/NotImplementedException.hx",
            ];

            for file_rel_path in essential_subdirectory_files {
                let file_path = path.join(file_rel_path);
                if file_path.exists() {
                    match self.load_file(&file_path) {
                        Ok(file) => {
                            files.push(file.clone());
                        }
                        Err(e) => {
                            let clean_error = strip_ansi_codes(&e);
                            warn!("Failed to parse {}: {}", file_path.display(), clean_error);
                        }
                    }
                }
            }

            info!("Loaded {} root stdlib files", files.len());
        } else {
            warn!("No valid stdlib path found");
        }

        files
    }

    /// Recursively scan and load ALL .hx files from the stdlib directory
    /// This enables automatic discovery of all stdlib types
    /// Note: Prefer load_root_stdlib() for production use
    pub fn load_all_stdlib(&mut self) -> Vec<HaxeFile> {
        let mut files = Vec::new();

        // Find the first valid stdlib path
        let stdlib_path = self.config.std_paths.iter().find(|p| p.exists()).cloned();

        if let Some(path) = stdlib_path {
            info!("Loading all stdlib from: {}", path.display());
            self.scan_directory_recursive(&path, &mut files);
            info!("Loaded {} stdlib files", files.len());
        } else {
            warn!("No valid stdlib path found");
        }

        files
    }

    /// Recursively scan a directory for .hx files
    fn scan_directory_recursive(&mut self, dir: &Path, files: &mut Vec<HaxeFile>) {
        if !dir.is_dir() {
            return;
        }

        // Skip platform-specific directories that are not Rayzor
        // These directories contain target-language-specific implementations
        let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let skip_dirs = [
            "cpp", "cs", "eval", "flash", "hl", "java", "js", "jvm", "lua", "neko", "php",
            "python", "flash8", "_std",
        ];

        if skip_dirs.contains(&dir_name) {
            return;
        }

        // Read directory entries
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(err) => {
                warn!("Failed to read directory {:?}: {}", dir, err);
                return;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();

            if path.is_dir() {
                // Recursively scan subdirectories
                self.scan_directory_recursive(&path, files);
            } else if path.extension().and_then(|s| s.to_str()) == Some("hx") {
                // Load .hx file
                match self.load_file(&path) {
                    Ok(file) => {
                        files.push(file.clone());
                    }
                    Err(e) => {
                        // Just log the warning, don't fail - some files may have parse errors
                        let clean_error = strip_ansi_codes(&e);
                        warn!("Failed to parse {}: {}", path.display(), clean_error);
                    }
                }
            }
        }
    }

    /// Find and load import.hx files in a directory
    pub fn load_import_hx(&mut self, dir: &Path) -> Vec<HaxeFile> {
        if !self.config.load_import_hx {
            return Vec::new();
        }

        let mut files = Vec::new();
        let import_path = dir.join("import.hx");

        if import_path.exists() {
            match self.load_file(&import_path) {
                Ok(file) => files.push(file.clone()),
                Err(e) => warn!("Failed to load import.hx: {}", e),
            }
        }

        files
    }
}
