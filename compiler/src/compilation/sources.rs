//! Taking source files in: parsing, pre-registration, and the stdlib.

use super::*;

impl CompilationUnit {

    /// Parse a file with the compilation unit's preprocessor defines.
    pub(crate) fn parse_file(&self, filename: &str, source: &str) -> Result<parser::HaxeFile, String> {
        let config = self.preprocessor_config();
        let file = parser::haxe_parser::parse_haxe_file_with_config(
            filename, source, true, true, &config,
        )?;
        // Every canonical parse feeds the static-signature index so call
        // sites can type not-yet-lowered statics from their declarations.
        {
            let mut index = self.static_sig_index.borrow_mut();
            index.preprocessor = config;
            index.index_file(&file);
        }
        Ok(file)
    }


    /// Load standard library files
    /// This should be called FIRST, before any user files are added
    pub fn load_stdlib(&mut self) -> Result<(), String> {
        if !self.config.load_stdlib {
            return Ok(());
        }

        // Configure stdlib loader
        let mut loader_config = StdLibConfig::default();
        loader_config.std_paths = self.config.stdlib_paths.clone();
        loader_config.default_imports = self.config.default_stdlib_imports.clone();

        let mut loader = StdLibLoader::new(loader_config);
        loader.set_preprocessor_config(self.preprocessor_config());

        // Configure namespace resolver with stdlib paths for on-demand loading
        self.namespace_resolver
            .set_stdlib_paths(self.config.stdlib_paths.clone());

        // Load pre-compiled symbols from BLADE manifest if caching is enabled
        // Skip if lazy_stdlib is enabled (for faster cold start)
        let bsym_loaded = if self.config.enable_cache && !self.config.lazy_stdlib {
            if self.load_stdlib_symbols() {
                debug!("BLADE symbols loaded, stdlib configured for cached resolution");
                true
            } else {
                debug!("No BLADE symbols available, falling back to on-demand loading");
                false
            }
        } else if self.config.lazy_stdlib {
            debug!("Lazy stdlib enabled - skipping upfront symbol registration for faster startup");
            self.register_builtin_globals();
            false
        } else {
            false
        };
        self.stdlib_manifest_loaded = bsym_loaded;

        // Load default stdlib imports (StdTypes, etc.). The BLADE manifest
        // restores their SYMBOLS, so re-running pre-registration on top would
        // duplicate them — but it carries no DECLARATIONS, and the
        // static-signature index reads declarations to type call sites. Parse
        // the same files for that index alone, so a call resolves its declared
        // parameter types identically on both paths.
        if bsym_loaded {
            for file in loader.load_default_imports() {
                let Some(source) = file.input.as_deref() else {
                    continue;
                };
                self.parse_file(&file.filename, source)
                    .map_err(|e| format!("Parse error in {}: {}", file.filename, e))?;
            }
        } else {
            let default_files = loader.load_default_imports();
            for file in default_files {
                if let Some(source) = file.input.as_deref() {
                    self.pre_register_and_enums_from_source(&file.filename, source)?;
                }
                self.stdlib_files.push(file);
            }
            debug!("Loaded {} default stdlib imports", self.stdlib_files.len());
        }

        Ok(())
    }


    /// Set source paths for user code (for on-demand import loading)
    /// These paths are checked first when resolving imports
    pub fn set_source_paths(&mut self, paths: Vec<PathBuf>) {
        self.namespace_resolver.set_source_paths(paths);
    }


    /// Pre-register type declarations from a file without full compilation
    /// This is the first pass that registers class/interface/enum names in the namespace
    /// so they can be referenced by other files during full compilation
    pub(crate) fn pre_register_file_types(&mut self, filename: &str, source: &str) -> Result<(), String> {
        use crate::tast::ast_lowering::AstLowering;

        let ast_file = self
            .parse_file(filename, source)
            .map_err(|e| format!("Parse error in {}: {}", filename, e))?;

        // Create a temporary AstLowering instance just for pre-registration
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Pre-register only - call the pre_register_file method
        lowering
            .pre_register_file(&ast_file)
            .map_err(|e| format!("Pre-registration error in {}: {:?}", filename, e))?;

        Ok(())
    }


    /// Register only enum declarations from source into the symbol table.
    ///
    /// Used when loading from BLADE cache — the cached MIR has the compiled code
    /// but the symbol table needs enum declarations registered so that user code
    /// can resolve imported enum types and their variants.
    pub(crate) fn register_enums_from_source(&mut self, filename: &str, source: &str) {
        use crate::tast::ast_lowering::AstLowering;

        let ast_file = match self.parse_file(filename, source) {
            Ok(f) => f,
            Err(_) => return,
        };
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Set package context from the parsed file
        if let Some(ref pkg) = ast_file.package {
            lowering.set_package_from_parts(&pkg.path);
        }

        // Lower enum and abstract declarations from cached files.
        // Class registration must go through the normal TAST pipeline
        // to avoid overwriting user imports.
        for decl in &ast_file.declarations {
            match decl {
                parser::TypeDeclaration::Enum(enum_decl) => {
                    let _ = lowering.lower_enum_declaration_public(enum_decl);
                }
                parser::TypeDeclaration::Abstract(_) => {
                    // Pre-register abstract declarations so import resolution
                    // creates Abstract symbols instead of Class placeholders
                    let _ = lowering.pre_register_declaration(decl);
                }
                _ => {}
            }
        }
    }


    /// Combined pre-register types + enum registration from a single parse.
    /// Eliminates the double-parse that occurred when pre_register_file_types
    /// and register_enums_from_source were called separately.
    pub(crate) fn pre_register_and_enums_from_source(
        &mut self,
        filename: &str,
        source: &str,
    ) -> Result<(), String> {
        use crate::tast::ast_lowering::AstLowering;

        let ast_file = self
            .parse_file(filename, source)
            .map_err(|e| format!("Parse error in {}: {}", filename, e))?;
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Pre-register types (classes, interfaces, enums, abstracts)
        lowering
            .pre_register_file(&ast_file)
            .map_err(|e| format!("Pre-registration error in {}: {:?}", filename, e))?;

        // Also register enums and abstracts from the same parsed AST
        if let Some(ref pkg) = ast_file.package {
            lowering.set_package_from_parts(&pkg.path);
        }
        for decl in &ast_file.declarations {
            match decl {
                parser::TypeDeclaration::Enum(enum_decl) => {
                    let _ = lowering.lower_enum_declaration_public(enum_decl);
                }
                parser::TypeDeclaration::Abstract(_) => {
                    let _ = lowering.pre_register_declaration(decl);
                }
                parser::TypeDeclaration::Typedef(typedef_decl) => {
                    let _ = lowering.lower_typedef_declaration(typedef_decl);
                }
                _ => {}
            }
        }

        Ok(())
    }


    /// Add a user source file to the compilation unit
    pub fn add_file(&mut self, source: &str, file_path: &str) -> Result<(), String> {
        let haxe_file = self
            .parse_file(file_path, source)
            .map_err(|e| format!("Parse error in {}: {}", file_path, e))?;

        self.user_files.push(haxe_file);
        Ok(())
    }


    /// Add a file from filesystem path
    /// This resolves the file's path and loads it, making it easier to work with
    /// real projects on disk
    pub fn add_file_from_path(&mut self, path: &PathBuf) -> Result<(), String> {
        use std::fs;

        let source = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path, e))?;

        let file_path_str = path
            .to_str()
            .ok_or_else(|| format!("Invalid UTF-8 in path: {:?}", path))?;

        self.add_file(&source, file_path_str)
    }


    /// Add all .hx files from a directory (recursively)
    /// This is useful for loading entire source trees
    ///
    /// # Arguments
    /// * `dir_path` - The directory to scan for .hx files
    /// * `recursive` - Whether to scan subdirectories
    pub fn add_directory(&mut self, dir_path: &PathBuf, recursive: bool) -> Result<usize, String> {
        use std::fs;

        let mut added_count = 0;

        let mut paths: Vec<PathBuf> = fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory {:?}: {}", dir_path, e))?
            .map(|entry| {
                entry
                    .map(|e| e.path())
                    .map_err(|e| format!("Failed to read directory entry: {}", e))
            })
            .collect::<Result<_, _>>()?;
        paths.sort();

        for path in paths {
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "hx" {
                        self.add_file_from_path(&path)?;
                        added_count += 1;
                    }
                }
            } else if path.is_dir() && recursive {
                added_count += self.add_directory(&path, recursive)?;
            }
        }

        Ok(added_count)
    }


    /// Add an additional source path for import resolution (e.g. from an rpkg package).
    pub fn add_source_path(&mut self, path: PathBuf) {
        self.namespace_resolver.add_source_path(path);
    }
}
