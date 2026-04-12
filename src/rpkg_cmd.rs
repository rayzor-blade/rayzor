//! Rayzor package (rpkg) commands: pack / strip / install / add / remove /
//! list / inspect. Handles multi-platform dylib bundling, pure Haxe packages,
//! WASM component packages, and the user-level package registry under
//! `~/.rayzor/packages/`.

use std::path::{Path, PathBuf};

use crate::compile_helpers::compile_haxe_to_mir;
use crate::tui;
use crate::wasm_cmd::find_wasm_runtime;

#[allow(clippy::too_many_arguments)]
pub fn cmd_rpkg_pack(
    dylibs: Vec<PathBuf>,
    os_tags: Vec<String>,
    arch_tags: Vec<String>,
    wasm: Option<PathBuf>,
    js_host_args: Vec<String>,
    haxe_dir: PathBuf,
    output: PathBuf,
    name: Option<String>,
) -> Result<(), String> {
    let package_name = name.unwrap_or_else(|| {
        output
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string())
    });

    if dylibs.is_empty() && wasm.is_none() {
        println!(
            "Packing rpkg '{}' from {} (pure Haxe)",
            package_name,
            haxe_dir.display()
        );
        compiler::rpkg::pack::build_from_haxe_dir(&package_name, &haxe_dir, &output)?;
    } else if dylibs.is_empty() && wasm.is_some() {
        // WASM-only package (no native libs)
        println!(
            "Packing rpkg '{}' from {} + WASM component",
            package_name,
            haxe_dir.display()
        );
        let mut builder = compiler::rpkg::pack::RpkgBuilder::new(&package_name);
        if haxe_dir.is_dir() {
            compiler::rpkg::pack::collect_haxe_sources(&mut builder, &haxe_dir, &haxe_dir)?;
        }

        let wasm_bytes = match &wasm {
            Some(path) if path.to_string_lossy() != "__auto__" => {
                // Use pre-built WASM file
                let bytes =
                    std::fs::read(path).map_err(|e| format!("failed to read WASM: {}", e))?;
                println!(
                    "  wasm: {} ({:.1} KB)",
                    path.display(),
                    bytes.len() as f64 / 1024.0
                );
                bytes
            }
            _ => {
                // Auto-compile Haxe sources to WASM component
                println!("  compiling Haxe sources to WASM component...");
                build_wasm_component_from_haxe_dir(&haxe_dir)?
            }
        };
        builder.add_wasm_component(&package_name, &wasm_bytes);
        builder
            .write(&output)
            .map_err(|e| format!("failed to write rpkg: {}", e))?;
    } else {
        // Build platform entries: pair each dylib with its os/arch tag
        let current_os = if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "windows"
        } else {
            "unknown"
        };
        let current_arch = if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else {
            "unknown"
        };

        let mut platform_dylibs = Vec::new();
        for (i, dylib_path) in dylibs.iter().enumerate() {
            let os = os_tags.get(i).map(|s| s.as_str()).unwrap_or(current_os);
            let arch = arch_tags.get(i).map(|s| s.as_str()).unwrap_or(current_arch);
            platform_dylibs.push((dylib_path.as_path(), os, arch));
        }

        println!(
            "Packing rpkg '{}' with {} native lib(s) + {}",
            package_name,
            platform_dylibs.len(),
            haxe_dir.display()
        );
        for (path, os, arch) in &platform_dylibs {
            println!("  {}-{}: {}", os, arch, path.display());
        }

        compiler::rpkg::pack::build_from_dylibs(
            &package_name,
            &platform_dylibs,
            &haxe_dir,
            &output,
        )?;
    }

    // Collect JS host modules from CLI args + rayzor.toml [wasm] config
    let mut js_hosts: Vec<(String, String)> = Vec::new();

    // Source 1: CLI --js-host MODULE=FILE args
    for arg in &js_host_args {
        if let Some((module_name, file_path)) = arg.split_once('=') {
            let path = PathBuf::from(file_path);
            match std::fs::read_to_string(&path) {
                Ok(source) => {
                    println!(
                        "  js-host: {} ({:.1} KB)",
                        module_name,
                        source.len() as f64 / 1024.0
                    );
                    // Check for companion _bg.wasm (wasm-bindgen output)
                    let bg_wasm_path = path.with_file_name(
                        path.file_stem().unwrap().to_string_lossy().to_string() + "_bg.wasm",
                    );
                    if bg_wasm_path.exists() {
                        let bg_size = std::fs::metadata(&bg_wasm_path)
                            .map(|m| m.len())
                            .unwrap_or(0);
                        println!(
                            "  js-host-wasm: {} ({:.1} KB)",
                            bg_wasm_path.display(),
                            bg_size as f64 / 1024.0
                        );
                    }
                    js_hosts.push((module_name.to_string(), source));
                }
                Err(e) => eprintln!("  warning: JS host {} not found: {}", path.display(), e),
            }
        }
    }

    // Source 2: rayzor.toml [wasm] hosts (if no CLI args provided)
    if js_hosts.is_empty() {
        let manifest_root = haxe_dir.parent().and_then(|p| {
            let abs = if p.is_absolute() {
                p.to_path_buf()
            } else {
                std::env::current_dir().unwrap_or_default().join(p)
            };
            compiler::workspace::find_project_root(&abs)
        });
        if let Some(root) = manifest_root {
            if let Ok(project) = compiler::workspace::load_project(&root) {
                for (module_name, abs_path) in project.resolved_wasm_hosts() {
                    match std::fs::read_to_string(&abs_path) {
                        Ok(source) => {
                            println!(
                                "  js-host: {} ({:.1} KB)",
                                module_name,
                                source.len() as f64 / 1024.0
                            );
                            js_hosts.push((module_name, source));
                        }
                        Err(e) => {
                            eprintln!("  warning: JS host {} not found: {}", abs_path.display(), e)
                        }
                    }
                }
            }
        }
    }

    // Inject JS hosts into the rpkg if any were found
    if !js_hosts.is_empty() {
        let loaded = compiler::rpkg::load_rpkg(&output)
            .map_err(|e| format!("failed to reload rpkg: {}", e))?;
        let mut builder = compiler::rpkg::pack::RpkgBuilder::new(&loaded.package_name);
        for (module_path, source) in &loaded.haxe_sources {
            builder.add_haxe_source(module_path, source);
        }
        if !loaded.methods.is_empty() {
            let plugin = loaded
                .plugin_name
                .as_deref()
                .unwrap_or(&loaded.package_name);
            builder.add_method_table(plugin, &loaded.methods);
        }
        if let Some(ref wasm_bytes) = loaded.wasm_component_bytes {
            builder.add_wasm_component(&loaded.package_name, wasm_bytes);
        }
        for (module_name, js_source) in &js_hosts {
            // Check for companion _bg.wasm alongside the JS host
            // Reconstruct the path from the CLI arg to find the companion
            let bg_path = js_host_args.iter().find_map(|arg| {
                if let Some((name, file)) = arg.split_once('=') {
                    if name == module_name {
                        let js_path = PathBuf::from(file);
                        let bg = js_path.with_file_name(
                            js_path.file_stem().unwrap().to_string_lossy().to_string() + "_bg.wasm",
                        );
                        if bg.exists() {
                            return Some(bg);
                        }
                    }
                }
                None
            });
            if let Some(bg) = bg_path {
                if let Ok(wasm_bytes) = std::fs::read(&bg) {
                    builder.add_js_host_with_wasm(module_name, js_source, &wasm_bytes);
                    continue;
                }
            }
            builder.add_js_host(module_name, js_source);
        }
        // Rewrite with JS hosts included
        builder
            .write(&output)
            .map_err(|e| format!("failed to write rpkg: {}", e))?;
    }

    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "  wrote {} ({:.1} KB)",
        output.display(),
        size as f64 / 1024.0
    );

    Ok(())
}

/// Auto-compile Haxe sources in a directory to a WASM P2 Component.
///
/// Finds .hx files, compiles them through the full pipeline (parse → TAST → HIR → MIR → WASM),
/// links with the WASM runtime, and wraps as a P2 Component.
pub fn build_wasm_component_from_haxe_dir(haxe_dir: &Path) -> Result<Vec<u8>, String> {
    // Find all .hx files
    let mut hx_files: Vec<PathBuf> = Vec::new();
    collect_hx_files(haxe_dir, &mut hx_files)?;

    if hx_files.is_empty() {
        return Err(format!("no .hx files found in {}", haxe_dir.display()));
    }

    // Find the primary file (one with Main class, or first file)
    let primary = hx_files
        .iter()
        .find(|p| {
            p.file_stem()
                .map(|s| {
                    let name = s.to_string_lossy();
                    name.contains("Main") || name.contains("main")
                })
                .unwrap_or(false)
        })
        .unwrap_or(&hx_files[0])
        .clone();

    let source = std::fs::read_to_string(&primary)
        .map_err(|e| format!("failed to read {}: {}", primary.display(), e))?;

    // Compile to MIR (don't pass haxe_dir as extra source — it's already the primary file)
    let (mir_module, _diagnostics) = compile_haxe_to_mir(
        &source,
        primary.to_str().unwrap_or("unknown"),
        Vec::new(),
        &[],
        false,
    )?;

    // MIR → WASM
    let user_wasm =
        compiler::codegen::wasm_backend::WasmBackend::compile(&[&mir_module], Some("main"))?;

    // Link with runtime
    let runtime_path = find_wasm_runtime();
    let linked = if let Some(rt_path) = &runtime_path {
        let rt_bytes =
            std::fs::read(rt_path).map_err(|e| format!("failed to read runtime: {}", e))?;
        compiler::codegen::wasm_linker::WasmLinker::link(&user_wasm, &rt_bytes)?
    } else {
        user_wasm
    };

    // Wrap as P2 Component (command — has _start from Main.main)
    let component = compiler::codegen::wasm_component::wrap_as_component(
        &linked,
        compiler::codegen::wasm_component::ComponentKind::Command,
    )?;

    println!(
        "  wasm component: {:.1} KB (core: {:.1} KB)",
        component.len() as f64 / 1024.0,
        linked.len() as f64 / 1024.0
    );

    Ok(component)
}

/// Recursively collect .hx files from a directory.
fn collect_hx_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.is_dir() {
        return Ok(());
    }
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read dir {}: {}", dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("dir entry error: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            collect_hx_files(&path, files)?;
        } else if path.extension().map(|e| e == "hx").unwrap_or(false) {
            files.push(path);
        }
    }
    Ok(())
}

pub fn cmd_rpkg_strip(
    input: PathBuf,
    os: Option<String>,
    arch: Option<String>,
    output: PathBuf,
) -> Result<(), String> {
    let target_os = os.unwrap_or_else(|| {
        if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else {
            "windows"
        }
        .to_string()
    });
    let target_arch = arch.unwrap_or_else(|| {
        if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            "x86_64"
        }
        .to_string()
    });

    println!(
        "Stripping {} → {} (target: {}-{})",
        input.display(),
        output.display(),
        target_os,
        target_arch
    );

    compiler::rpkg::strip_rpkg(&input, &target_os, &target_arch, &output)
        .map_err(|e| format!("strip failed: {}", e))?;

    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "  wrote {} ({:.1} KB)",
        output.display(),
        size as f64 / 1024.0
    );

    Ok(())
}

pub fn cmd_rpkg_install(file: PathBuf) -> Result<(), String> {
    use compiler::rpkg::registry::LocalRegistry;

    let mut registry = LocalRegistry::open_default()?;
    let entry = registry.install(&file)?;

    let rows = vec![
        tui::panel::InfoRow::colored("Package", &entry.name, ratatui::style::Color::Cyan),
        tui::panel::InfoRow::new("Size", &format_bytes(entry.size_bytes)),
        tui::panel::InfoRow::new("Haxe files", &entry.haxe_file_count.to_string()),
        tui::panel::InfoRow::new("Native", if entry.has_native { "yes" } else { "no" }),
        tui::panel::InfoRow::new("Location", &registry.root_dir().display().to_string()),
    ];
    let _ = tui::panel::render_info_panel("Package Installed", &rows, None);
    Ok(())
}

pub fn cmd_rpkg_add(name: String) -> Result<(), String> {
    use compiler::rpkg::registry::LocalRegistry;

    // Verify the package is installed in the registry
    let registry = LocalRegistry::open_default()?;
    if registry.get(&name).is_none() {
        return Err(format!(
            "Package '{}' is not installed. Run `rayzor rpkg install <file.rpkg>` first.",
            name
        ));
    }

    // Find rayzor.toml in current directory
    let manifest_path = std::path::PathBuf::from("rayzor.toml");
    if !manifest_path.exists() {
        return Err("No rayzor.toml found in current directory".to_string());
    }

    let content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read rayzor.toml: {}", e))?;

    // Check if dependency already exists
    if content.contains("[dependencies]") && content.contains(&format!("{} ", name)) {
        return Err(format!(
            "Dependency '{}' already exists in rayzor.toml",
            name
        ));
    }

    // Append [dependencies] section if missing, or add to existing
    let updated = if content.contains("[dependencies]") {
        // Add to existing section
        content.replace(
            "[dependencies]",
            &format!("[dependencies]\n{} = {{ rpkg = \"{}\" }}", name, name),
        )
    } else {
        format!(
            "{}\n[dependencies]\n{} = {{ rpkg = \"{}\" }}\n",
            content.trim_end(),
            name,
            name
        )
    };

    std::fs::write(&manifest_path, updated)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let rows = vec![
        tui::panel::InfoRow::colored("Added", &name, ratatui::style::Color::Green),
        tui::panel::InfoRow::new("Source", "rpkg registry"),
    ];
    let _ = tui::panel::render_info_panel("Dependency Added", &rows, None);
    Ok(())
}

pub fn cmd_rpkg_remove(name: String) -> Result<(), String> {
    let manifest_path = std::path::PathBuf::from("rayzor.toml");
    if !manifest_path.exists() {
        return Err("No rayzor.toml found in current directory".to_string());
    }

    let content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read rayzor.toml: {}", e))?;

    // Remove the dependency line
    let lines: Vec<&str> = content.lines().collect();
    let filtered: Vec<&str> = lines
        .into_iter()
        .filter(|line| {
            let trimmed = line.trim();
            // Remove lines like: mylib = { rpkg = "mylib" } or mylib = "1.0"
            !(trimmed.starts_with(&name) && (trimmed.contains("=") && !trimmed.starts_with("[")))
        })
        .collect();

    let updated = filtered.join("\n") + "\n";
    std::fs::write(&manifest_path, updated)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let rows = vec![tui::panel::InfoRow::colored(
        "Removed",
        &name,
        ratatui::style::Color::Yellow,
    )];
    let _ = tui::panel::render_info_panel("Dependency Removed", &rows, None);
    Ok(())
}

pub fn cmd_rpkg_list() -> Result<(), String> {
    use compiler::rpkg::registry::LocalRegistry;

    let registry = LocalRegistry::open_default()?;
    let packages = registry.list();

    if packages.is_empty() {
        let rows = vec![tui::panel::InfoRow::new("Status", "No packages installed")];
        let _ = tui::panel::render_info_panel(
            "Package Registry",
            &rows,
            Some("Install with: rayzor rpkg install <file.rpkg>"),
        );
        return Ok(());
    }

    let mut rows = Vec::new();
    for (name, entry) in packages {
        let info = format!(
            "{} | {} hx files{}",
            format_bytes(entry.size_bytes),
            entry.haxe_file_count,
            if entry.has_native { " | native" } else { "" }
        );
        rows.push(tui::panel::InfoRow::colored(
            name,
            &info,
            if entry.has_native {
                ratatui::style::Color::Magenta
            } else {
                ratatui::style::Color::Cyan
            },
        ));
    }

    let _ = tui::panel::render_info_panel(
        &format!("Package Registry ({} packages)", packages.len()),
        &rows,
        Some(&registry.root_dir().display().to_string()),
    );
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn cmd_rpkg_inspect(file: PathBuf) -> Result<(), String> {
    let loaded = compiler::rpkg::load_rpkg(&file)
        .map_err(|e| format!("failed to load {}: {}", file.display(), e))?;

    println!("RPKG: {}", file.display());
    println!("  package: {}", loaded.package_name);
    println!();

    if let Some(ref name) = loaded.plugin_name {
        println!("  Method Table (plugin: {})", name);
        for m in &loaded.methods {
            let kind = if m.is_static { "static" } else { "instance" };
            println!(
                "    {} {}.{}  →  {} (params: {}, ret: {})",
                kind, m.class_name, m.method_name, m.symbol_name, m.param_count, m.return_type
            );
        }
        println!();
    }

    if !loaded.haxe_sources.is_empty() {
        println!("  Haxe Sources ({}):", loaded.haxe_sources.len());
        for path in loaded.haxe_sources.keys() {
            println!("    {}", path);
        }
        println!();
    }

    if loaded.native_lib_bytes.is_some() {
        println!(
            "  Native Library: present for current platform ({}-{})",
            if cfg!(target_os = "macos") {
                "macos"
            } else if cfg!(target_os = "linux") {
                "linux"
            } else {
                "other"
            },
            if cfg!(target_arch = "aarch64") {
                "aarch64"
            } else {
                "x86_64"
            }
        );
    } else {
        println!("  Native Library: not available for current platform");
    }

    if let Some(ref wasm_bytes) = loaded.wasm_component_bytes {
        println!(
            "  WASM Component: present ({:.1} KB) — universal fallback",
            wasm_bytes.len() as f64 / 1024.0,
        );
    }

    if !loaded.js_hosts.is_empty() {
        println!("\n  JS Hosts ({}):", loaded.js_hosts.len());
        for (module_name, source) in &loaded.js_hosts {
            println!(
                "    {} ({:.1} KB)",
                module_name,
                source.len() as f64 / 1024.0,
            );
            if let Some(wasm_bytes) = loaded.js_host_wasms.get(module_name) {
                println!(
                    "      + companion WASM ({:.1} KB)",
                    wasm_bytes.len() as f64 / 1024.0,
                );
            }
        }
    }

    Ok(())
}
