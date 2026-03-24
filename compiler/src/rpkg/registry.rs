//! Local package registry for rpkg packages.
//!
//! Packages are stored in `~/.rayzor/packages/` with a JSON index file
//! tracking installed packages, versions, and metadata.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Registry index persisted as JSON at `~/.rayzor/packages/registry.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryIndex {
    /// Map of package name → package metadata.
    pub packages: BTreeMap<String, PackageEntry>,
}

/// Metadata for a single installed package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageEntry {
    /// Package name.
    pub name: String,
    /// Version string (semver or freeform).
    pub version: Option<String>,
    /// Relative path to the .rpkg file inside the packages directory.
    pub rpkg_file: String,
    /// Timestamp when the package was installed (seconds since epoch).
    pub installed_at: u64,
    /// Size of the .rpkg file in bytes.
    pub size_bytes: u64,
    /// Whether the package contains native code.
    pub has_native: bool,
    /// Number of Haxe source files in the package.
    pub haxe_file_count: usize,
}

/// The local package registry.
pub struct LocalRegistry {
    /// Root directory (`~/.rayzor/packages/`).
    root: PathBuf,
    /// In-memory index.
    index: RegistryIndex,
}

impl LocalRegistry {
    /// Open (or create) the local registry at the default location.
    pub fn open_default() -> Result<Self, String> {
        let home = dirs_or_home()?;
        let root = home.join(".rayzor").join("packages");
        Self::open(root)
    }

    /// Open (or create) the local registry at a specific path.
    pub fn open(root: PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&root)
            .map_err(|e| format!("Failed to create registry dir {:?}: {}", root, e))?;

        let index_path = root.join("registry.json");
        let index = if index_path.exists() {
            let data = std::fs::read_to_string(&index_path)
                .map_err(|e| format!("Failed to read registry index: {}", e))?;
            serde_json::from_str(&data)
                .map_err(|e| format!("Failed to parse registry index: {}", e))?
        } else {
            RegistryIndex::default()
        };

        Ok(Self { root, index })
    }

    /// Install an .rpkg file into the registry.
    /// Copies the file and updates the index.
    pub fn install(&mut self, rpkg_path: &Path) -> Result<&PackageEntry, String> {
        use super::load_rpkg;

        // Load and validate the rpkg
        let loaded = load_rpkg(rpkg_path)
            .map_err(|e| format!("Invalid rpkg file {:?}: {}", rpkg_path, e))?;

        let name = loaded.package_name.clone();
        let dest_filename = format!("{}.rpkg", name);
        let dest_path = self.root.join(&dest_filename);

        // Copy rpkg to registry
        std::fs::copy(rpkg_path, &dest_path)
            .map_err(|e| format!("Failed to copy to registry: {}", e))?;

        let size_bytes = std::fs::metadata(&dest_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let entry = PackageEntry {
            name: name.clone(),
            version: None,
            rpkg_file: dest_filename,
            installed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            size_bytes,
            has_native: loaded.native_lib_bytes.is_some(),
            haxe_file_count: loaded.haxe_sources.len(),
        };

        self.index.packages.insert(name.clone(), entry);
        self.save_index()?;

        Ok(self.index.packages.get(&name).unwrap())
    }

    /// Remove a package from the registry.
    pub fn remove(&mut self, name: &str) -> Result<(), String> {
        let entry = self
            .index
            .packages
            .remove(name)
            .ok_or_else(|| format!("Package '{}' is not installed", name))?;

        // Remove the rpkg file
        let rpkg_path = self.root.join(&entry.rpkg_file);
        if rpkg_path.exists() {
            let _ = std::fs::remove_file(&rpkg_path);
        }

        self.save_index()?;
        Ok(())
    }

    /// List all installed packages.
    pub fn list(&self) -> &BTreeMap<String, PackageEntry> {
        &self.index.packages
    }

    /// Get a specific package entry.
    pub fn get(&self, name: &str) -> Option<&PackageEntry> {
        self.index.packages.get(name)
    }

    /// Get the path to an installed package's .rpkg file.
    pub fn rpkg_path(&self, name: &str) -> Option<PathBuf> {
        self.index
            .packages
            .get(name)
            .map(|e| self.root.join(&e.rpkg_file))
    }

    /// Get the root directory of the registry.
    pub fn root_dir(&self) -> &Path {
        &self.root
    }

    fn save_index(&self) -> Result<(), String> {
        let index_path = self.root.join("registry.json");
        let json = serde_json::to_string_pretty(&self.index)
            .map_err(|e| format!("Failed to serialize registry index: {}", e))?;
        std::fs::write(&index_path, json)
            .map_err(|e| format!("Failed to write registry index: {}", e))?;
        Ok(())
    }
}

fn dirs_or_home() -> Result<PathBuf, String> {
    std::env::var("HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("USERPROFILE").map(PathBuf::from))
        .map_err(|_| "Could not determine home directory".to_string())
}
