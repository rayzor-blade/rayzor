//! Project and workspace scaffolding.

use std::fs;
use std::path::Path;

/// Project template for `rayzor init`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectTemplate {
    /// Application with main() entry point (default)
    App,
    /// Library project — no main(), exposes src/
    Lib,
    /// Benchmark harness with timing
    Benchmark,
    /// Empty skeleton — just rayzor.toml
    Empty,
}

impl ProjectTemplate {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "app" => Some(Self::App),
            "lib" => Some(Self::Lib),
            "benchmark" | "bench" => Some(Self::Benchmark),
            "empty" => Some(Self::Empty),
            _ => None,
        }
    }

    pub fn all_names() -> &'static [&'static str] {
        &["app", "lib", "benchmark", "empty"]
    }
}

/// Initialize a new Rayzor project with a template.
pub fn init_project(name: &str, dir: &Path, template: ProjectTemplate) -> Result<(), String> {
    fs::create_dir_all(dir.join(".rayzor").join("cache"))
        .map_err(|e| format!("Failed to create .rayzor/cache/: {}", e))?;

    match template {
        ProjectTemplate::App => init_app_project(name, dir),
        ProjectTemplate::Lib => init_lib_project(name, dir),
        ProjectTemplate::Benchmark => init_benchmark_project(name, dir),
        ProjectTemplate::Empty => init_empty_project(name, dir),
    }?;

    // .gitignore (all templates)
    let gitignore = "build/\n.rayzor/cache/\n*.rzb\n";
    fs::write(dir.join(".gitignore"), gitignore)
        .map_err(|e| format!("Failed to write .gitignore: {}", e))?;

    Ok(())
}

fn init_app_project(name: &str, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir.join("src")).map_err(|e| format!("Failed to create src/: {}", e))?;

    let manifest = format!(
        r#"[project]
name = "{name}"
version = "0.1.0"
entry = "src/Main.hx"

[build]
class-paths = ["src"]
opt-level = 2
preset = "application"
output = "build/{name}"

[cache]
enabled = true
"#,
    );
    fs::write(dir.join("rayzor.toml"), manifest)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let main_hx = r#"class Main {
    static function main() {
        trace("Hello from Rayzor!");
    }
}
"#;
    fs::write(dir.join("src").join("Main.hx"), main_hx)
        .map_err(|e| format!("Failed to write Main.hx: {}", e))?;

    Ok(())
}

fn init_lib_project(name: &str, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir.join("src")).map_err(|e| format!("Failed to create src/: {}", e))?;

    let manifest = format!(
        r#"[project]
name = "{name}"
version = "0.1.0"

[lib]
expose = ["src"]

[build]
class-paths = ["src"]
opt-level = 2

[cache]
enabled = true
"#,
    );
    fs::write(dir.join("rayzor.toml"), manifest)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let lib_hx = format!(
        r#"class {class_name} {{
    public static function hello():String {{
        return "Hello from {name}!";
    }}
}}
"#,
        class_name = to_class_name(name),
    );
    fs::write(
        dir.join("src").join(format!("{}.hx", to_class_name(name))),
        lib_hx,
    )
    .map_err(|e| format!("Failed to write source: {}", e))?;

    Ok(())
}

fn init_benchmark_project(name: &str, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir.join("src")).map_err(|e| format!("Failed to create src/: {}", e))?;

    let manifest = format!(
        r#"[project]
name = "{name}"
version = "0.1.0"
entry = "src/Main.hx"

[build]
class-paths = ["src"]
opt-level = 3
preset = "benchmark"
output = "build/{name}"

[cache]
enabled = true
"#,
    );
    fs::write(dir.join("rayzor.toml"), manifest)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let main_hx = r#"class Main {
    static function main() {
        var iterations = 1000000;
        var start = Sys.time();

        var sum = 0.0;
        for (i in 0...iterations) {
            sum += Math.sqrt(cast(i, Float));
        }

        var elapsed = Sys.time() - start;
        trace('Result: $sum');
        trace('Time: ${elapsed}s (${iterations} iterations)');
    }
}
"#;
    fs::write(dir.join("src").join("Main.hx"), main_hx)
        .map_err(|e| format!("Failed to write Main.hx: {}", e))?;

    Ok(())
}

fn init_empty_project(name: &str, dir: &Path) -> Result<(), String> {
    let manifest = format!(
        r#"[project]
name = "{name}"
version = "0.1.0"

[build]
class-paths = ["src"]

[cache]
enabled = true
"#,
    );
    fs::write(dir.join("rayzor.toml"), manifest)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    Ok(())
}

/// Initialize a new Rayzor workspace with optional member projects.
pub fn init_workspace(name: &str, dir: &Path, members: &[String]) -> Result<(), String> {
    fs::create_dir_all(dir.join(".rayzor").join("cache"))
        .map_err(|e| format!("Failed to create .rayzor/cache/: {}", e))?;

    let members_toml = if members.is_empty() {
        "members = []".to_string()
    } else {
        let quoted: Vec<String> = members.iter().map(|m| format!("\"{}\"", m)).collect();
        format!("members = [{}]", quoted.join(", "))
    };

    let manifest = format!(
        r#"[workspace]
{members_toml}

[workspace.cache]
dir = ".rayzor/cache"
"#,
    );

    fs::write(dir.join("rayzor.toml"), manifest)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let gitignore = "build/\n.rayzor/cache/\n*.rzb\n";
    fs::write(dir.join(".gitignore"), gitignore)
        .map_err(|e| format!("Failed to write .gitignore: {}", e))?;

    // Create each member as an app project
    for member in members {
        let member_dir = dir.join(member);
        init_project(member, &member_dir, ProjectTemplate::App)?;
    }

    Ok(())
}

/// Generate a rayzor.toml from an existing HXML build file.
pub fn init_from_hxml(hxml_path: &Path, dir: &Path) -> Result<(), String> {
    let hxml_content = std::fs::read_to_string(hxml_path)
        .map_err(|e| format!("Failed to read {}: {}", hxml_path.display(), e))?;

    let config = crate::hxml::HxmlConfig::from_string(&hxml_content)?;

    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("my-project")
        .to_string();

    fs::create_dir_all(dir.join(".rayzor").join("cache"))
        .map_err(|e| format!("Failed to create .rayzor/cache/: {}", e))?;

    // Build entry from main class + class paths
    let entry = if let Some(ref main_class) = config.main_class {
        // Convert Main.Sub to src/Main/Sub.hx (check class paths)
        let relative = main_class.replace('.', "/") + ".hx";
        let mut found = None;
        for cp in &config.class_paths {
            let candidate = cp.join(&relative);
            if candidate.exists() {
                found = Some(format!("{}", candidate.display()));
                break;
            }
        }
        found.unwrap_or(relative)
    } else {
        "src/Main.hx".to_string()
    };

    let class_paths: Vec<String> = if config.class_paths.is_empty() {
        vec!["src".to_string()]
    } else {
        config
            .class_paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect()
    };
    let cp_toml: Vec<String> = class_paths.iter().map(|p| format!("\"{}\"", p)).collect();

    let defines_toml = if config.defines.is_empty() {
        String::new()
    } else {
        let entries: Vec<String> = config
            .defines
            .iter()
            .map(|(k, v)| match v {
                Some(val) => format!("{} = \"{}\"", k, val),
                None => format!("{} = true", k),
            })
            .collect();
        format!("\n[build.defines]\n{}\n", entries.join("\n"))
    };

    let manifest = format!(
        r#"[project]
name = "{name}"
version = "0.1.0"
entry = "{entry}"
# Migrated from: {hxml_file}

[build]
class-paths = [{class_paths}]
opt-level = 2
preset = "application"
{defines}
[cache]
enabled = true
"#,
        hxml_file = hxml_path.display(),
        class_paths = cp_toml.join(", "),
        defines = defines_toml,
    );

    fs::write(dir.join("rayzor.toml"), manifest)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let gitignore = "build/\n.rayzor/cache/\n*.rzb\n";
    fs::write(dir.join(".gitignore"), gitignore)
        .map_err(|e| format!("Failed to write .gitignore: {}", e))?;

    Ok(())
}

/// Detect if a directory has existing Haxe sources and auto-configure.
pub fn detect_existing_sources(dir: &Path) -> Option<(String, Vec<String>)> {
    let src_dir = dir.join("src");
    if !src_dir.is_dir() {
        return None;
    }

    let mut hx_files = Vec::new();
    if let Ok(entries) = fs::read_dir(&src_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("hx") {
                hx_files.push(path);
            }
        }
    }

    if hx_files.is_empty() {
        return None;
    }

    // Look for a Main.hx
    let entry = hx_files
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n == "Main.hx")
                .unwrap_or(false)
        })
        .map(|p| format!("src/{}", p.file_name().unwrap().to_string_lossy()))
        .unwrap_or_else(|| format!("src/{}", hx_files[0].file_name().unwrap().to_string_lossy()));

    Some((entry, vec!["src".to_string()]))
}

/// Convert a kebab-case or snake_case name to PascalCase class name.
fn to_class_name(name: &str) -> String {
    name.split(['-', '_'])
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + chars.as_str(),
            }
        })
        .collect()
}
