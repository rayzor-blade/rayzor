//! Compilation settings: target discovery, cache locations and the
//! presets a caller picks between.

use super::*;

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            stdlib_paths: Self::discover_stdlib_paths(),
            default_stdlib_imports: vec![
                "StdTypes.hx".to_string(), // Contains Iterator typedef
                "String.hx".to_string(),
                "Array.hx".to_string(),
                "Math.hx".to_string(), // Top-level Math functions (sqrt, sin, cos, etc.)
                "Std.hx".to_string(),  // Top-level conversion utilities
                "Type.hx".to_string(), // ValueType enum + Type reflection APIs
                // Concurrent types
                "rayzor/concurrent/Thread.hx".to_string(),
                "rayzor/concurrent/Channel.hx".to_string(),
                "rayzor/concurrent/Mutex.hx".to_string(),
                "rayzor/concurrent/Arc.hx".to_string(),
                // Array iterator classes (compiled as regular Haxe, not runtime-backed)
                "haxe/iterators/ArrayIterator.hx".to_string(),
                "haxe/iterators/ArrayKeyValueIterator.hx".to_string(),
            ],
            load_stdlib: true,
            stdlib_root_package: Some("haxe".to_string()), // Prefix stdlib with "haxe.*" namespace
            global_import_hx_files: Vec::new(),            // No global import.hx by default
            enable_cache: true, // Cache enabled - BLADE manifest now includes Math, Std, Date, etc.
            cache_dir: None,    // Auto-discover cache directory when needed
            lazy_stdlib: false, // Default to eager loading for compatibility
            pipeline_config: PipelineConfig::default(),
            hdll_search_paths: vec![PathBuf::from(".")],
            emit_safety_warnings: true,
            extra_defines: Vec::new(),
            profile_typecheck: false,
        }
    }
}

impl CompilationConfig {
    /// Discover standard library paths from environment and standard locations
    ///
    /// Search order:
    /// Discover rayzor's own stdlib (haxe-std).
    ///
    /// Resolution order:
    /// 1. RAYZOR_STD_PATH environment variable (explicit override)
    /// 2. Relative to the rayzor binary (../haxe-std, ../compiler/haxe-std)
    /// 3. Relative to cwd (compiler/haxe-std, ./haxe-std, ../haxe-std)
    ///
    /// NOTE: System Haxe installations (/usr/local/lib/haxe/std etc.) are NOT
    /// searched. Rayzor uses its own stdlib with rayzor-specific extensions.
    /// Mixing system Haxe stdlib causes subtle compilation errors.
    pub fn discover_stdlib_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // 1. Explicit override via RAYZOR_STD_PATH
        if let Ok(std_path) = std::env::var("RAYZOR_STD_PATH") {
            let path = PathBuf::from(&std_path);
            if path.exists() {
                info!("Found stdlib at RAYZOR_STD_PATH: {}", std_path);
                paths.push(path);
                return paths;
            } else {
                warn!(
                    "RAYZOR_STD_PATH set but directory doesn't exist: {}",
                    std_path
                );
            }
        }

        // 2. Walk up from the binary location looking for haxe-std/
        if let Ok(exe) = std::env::current_exe() {
            if let Some(mut dir) = exe.parent().map(|p| p.to_path_buf()) {
                for _ in 0..5 {
                    for name in &["haxe-std", "compiler/haxe-std"] {
                        let candidate = dir.join(name);
                        if candidate.is_dir() {
                            paths.push(candidate);
                        }
                    }
                    if !dir.pop() {
                        break;
                    }
                }
            }
        }

        // 3. Walk up from cwd looking for haxe-std/
        if let Ok(mut dir) = std::env::current_dir() {
            for _ in 0..5 {
                for name in &["haxe-std", "compiler/haxe-std"] {
                    let candidate = dir.join(name);
                    if candidate.is_dir() {
                        // Deduplicate by canonical path
                        let dominated = paths.iter().any(|p| {
                            matches!(
                                (p.canonicalize(), candidate.canonicalize()),
                                (Ok(a), Ok(b)) if a == b
                            )
                        });
                        if !dominated {
                            paths.push(candidate);
                        }
                    }
                }
                if !dir.pop() {
                    break;
                }
            }
        }

        if paths.is_empty() {
            warn!("No rayzor stdlib found. Set RAYZOR_STD_PATH environment variable.");
            // Fallback for development
            paths.push(PathBuf::from("compiler/haxe-std"));
            paths.push(PathBuf::from("./haxe-std"));
        }

        paths
    }

    /// Get the current target triple (e.g., "x86_64-macos", "aarch64-linux")
    pub fn get_target_triple() -> String {
        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;
        format!("{}-{}", arch, os)
    }

    /// Target discriminator for cache paths.
    ///
    /// The same `.hx` source lowers to DIFFERENT MIR depending on
    /// `extra_defines` (`#if wasm`, host-import vs native-runtime bindings,
    /// etc.). Keying the cache only by source path made native and wasm builds
    /// of the same file share one `.blade` slot — running native then wasm (or
    /// vice versa) served the wrong-target MIR until `.rayzor` was deleted by
    /// hand. Folding the sorted defines into the cache directory gives native
    /// and wasm fully separate cache trees.
    pub fn cache_discriminator(&self) -> String {
        Self::discriminator_for(&self.extra_defines)
    }

    /// Target discriminator for the standard library's cache entries.
    ///
    /// A define reaches a module only through `#if`/`#elseif`, so one the
    /// library never names cannot change what it lowers to. The library's key
    /// therefore counts only the defines it actually tests
    /// ([`STDLIB_OBSERVED_DEFINES`], generated from `haxe-std`). Counting the
    /// rest lets an unrelated flag move the key — the CLI contributes one
    /// define per loaded plugin — so the carried snapshot holds no entry under
    /// the discriminator asked for and the library is lowered from source again.
    pub fn stdlib_cache_discriminator(&self) -> String {
        Self::discriminator_for(&self.stdlib_relevant_defines())
    }

    /// The defines that can change what a standard-library module lowers to:
    /// the ones its own sources test, plus the target selector the compiler
    /// itself branches on. `wasm` names a different set of runtime bindings
    /// rather than a `#if`, so it separates artifacts even though no library
    /// source mentions it.
    pub fn stdlib_relevant_defines(&self) -> Vec<String> {
        self.extra_defines
            .iter()
            .filter(|d| d.as_str() == "wasm" || STDLIB_OBSERVED_DEFINES.contains(&d.as_str()))
            .cloned()
            .collect()
    }

    fn discriminator_for(defines: &[String]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        if defines.is_empty() {
            return "native".to_string();
        }
        let mut defines = defines.to_vec();
        defines.sort();
        let tag = if defines.iter().any(|d| d == "wasm") {
            "wasm"
        } else {
            "native"
        };
        let mut hasher = DefaultHasher::new();
        defines.hash(&mut hasher);
        format!("{}-{:08x}", tag, hasher.finish() as u32)
    }

    /// Get or create the cache directory (target-discriminated — see
    /// [`cache_discriminator`]).
    pub fn get_cache_dir(&self) -> PathBuf {
        self.cache_dir_for(&self.cache_discriminator())
    }

    /// The cache directory a standard-library module belongs in — keyed by the
    /// defines the library can observe, so an unrelated flag does not file it
    /// somewhere nothing looks up. See [`stdlib_cache_discriminator`].
    pub fn get_stdlib_cache_dir(&self) -> PathBuf {
        self.cache_dir_for(&self.stdlib_cache_discriminator())
    }

    fn cache_dir_for(&self, discriminator: &str) -> PathBuf {
        // Base is `.rayzor/blade/cache` (separate from the Rust target folder),
        // or an explicit `--cache-dir`. Either way the per-target subdir keeps
        // native and wasm artifacts from colliding.
        let base = self
            .cache_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(".rayzor/blade/cache"));
        let dir = base.join(discriminator);

        // Try to create it if it doesn't exist
        if !dir.exists() {
            let _ = std::fs::create_dir_all(&dir);
        }

        dir
    }

    /// Get the target directory for the given profile
    pub fn get_target_dir(profile: &str) -> PathBuf {
        let triple = Self::get_target_triple();
        PathBuf::from("target").join(triple).join(profile)
    }

    /// Get the build directory for intermediate artifacts
    pub fn get_build_dir(profile: &str) -> PathBuf {
        Self::get_target_dir(profile).join("build")
    }

    /// Get the cache directory for a specific profile
    pub fn get_profile_cache_dir(profile: &str) -> PathBuf {
        Self::get_target_dir(profile).join("cache")
    }

    /// Get the output directory for executables
    pub fn get_output_dir(profile: &str) -> PathBuf {
        Self::get_target_dir(profile)
    }

    /// Get the cache file path for a given source file
    pub fn get_cache_path(&self, source_path: &Path) -> PathBuf {
        let cache_dir = self.get_cache_dir();

        // Create a cache filename based on the source path
        // Convert path to a safe filename by replacing separators with underscores
        let source_str = source_path.to_string_lossy();
        let cache_name = source_str
            .replace(['/', '\\', ':'], "_")
            .replace(".hx", ".blade");

        cache_dir.join(cache_name)
    }

    /// Create a fast compilation config optimized for interpreter cold start
    ///
    /// This configuration prioritizes startup speed over type safety:
    /// - Lazy stdlib loading (symbols loaded on-demand)
    /// - Cache enabled for subsequent runs
    ///
    /// Ideal for REPL, development mode, and interpreted execution.
    pub fn fast() -> Self {
        Self {
            lazy_stdlib: true,
            ..Default::default()
        }
    }

    /// Create a strict compilation config with full type checking
    ///
    /// This is the default behavior - all symbols loaded upfront,
    /// full type analysis enabled.
    pub fn strict() -> Self {
        Self {
            lazy_stdlib: false,
            ..Default::default()
        }
    }
}
