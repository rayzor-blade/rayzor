//! Resolve editor projects without merging incompatible workspace members.

use compiler::workspace::{Project, RayzorManifest};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Options {
    pub rayzor: bool,
    pub extra_defines: Vec<String>,
    pub native_libs: Vec<PathBuf>,
    pub packages: Vec<PathBuf>,
    pub resources: Vec<(PathBuf, String)>,
    pub configuration_files: Vec<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub cache_enabled: bool,
}

#[derive(Clone, Default)]
pub(crate) struct Settings {
    pub root: PathBuf,
    pub class_paths: Vec<PathBuf>,
    pub defines: Vec<String>,
    pub macros: Vec<String>,
    pub hxml: Option<PathBuf>,
    pub options: Options,
    pub problems: Vec<String>,
}

/// An opened nested library keeps the build's flags and gains its own source root.
pub(crate) fn include_source_library(settings: &mut Settings, source: &Path) {
    let Some(manifest) = source
        .parent()
        .into_iter()
        .flat_map(Path::ancestors)
        .map(|root| root.join("haxelib.json"))
        .find(|path| path.is_file())
    else {
        return;
    };
    let identity = manifest.canonicalize().unwrap_or_else(|_| manifest.clone());
    if settings
        .options
        .configuration_files
        .iter()
        .any(|path| path.canonicalize().unwrap_or_else(|_| path.clone()) == identity)
    {
        return;
    }
    let root = manifest.parent().unwrap();
    let metadata = crate::project::library_metadata(root, &[], &mut settings.problems);
    let mut provided = metadata.provided;
    provided.extend(
        crate::project::library_metadata(
            &settings.root,
            &settings.class_paths,
            &mut settings.problems,
        )
        .provided,
    );
    let (libraries, problems) =
        compiler::haxelib::resolve_dependencies(&metadata.libraries, root, &provided);
    settings.problems.extend(problems);
    settings.class_paths.extend(metadata.class_paths);
    settings
        .options
        .configuration_files
        .extend(metadata.input_files);
    for library in libraries {
        settings.class_paths.extend(library.class_paths);
        settings.macros.extend(library.macros);
        settings
            .options
            .configuration_files
            .extend(library.input_files);
        for (name, value) in library.defines {
            if !settings
                .defines
                .iter()
                .any(|define| define.split('=').next() == Some(&name))
            {
                settings
                    .defines
                    .push(value.map_or_else(|| name.clone(), |value| format!("{name}={value}")));
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    settings
        .class_paths
        .retain(|path| seen.insert(path.clone()));
}

/// Prefer the narrowest owning source path, including dependencies outside a root.
pub(crate) fn source_depth<'a>(
    path: &Path,
    root: &'a Path,
    class_paths: impl IntoIterator<Item = &'a PathBuf>,
) -> Option<usize> {
    std::iter::once(root)
        .chain(class_paths.into_iter().map(PathBuf::as_path))
        .filter_map(|root| {
            let root = root.canonicalize().unwrap_or_else(|_| root.into());
            path.starts_with(&root).then(|| root.components().count())
        })
        .max()
}

pub(crate) fn load(root: &Path, hxml: Option<&str>) -> Vec<Settings> {
    if hxml.is_none()
        && let Ok(RayzorManifest::Workspace(workspace)) = compiler::workspace::load_manifest(root)
    {
        let mut projects = Vec::new();
        for member in workspace.members {
            let member_root = root.join(member);
            let mut settings = load_project(&member_root, None);
            settings
                .options
                .configuration_files
                .push(root.join("rayzor.toml"));
            if settings.options.cache_dir.is_none()
                && let Some(directory) = workspace
                    .cache
                    .as_ref()
                    .and_then(|cache| cache.dir.as_ref())
            {
                settings.options.cache_dir = Some(root.join(directory));
            }
            projects.push(settings);
        }
        if !projects.is_empty() {
            return projects;
        }
    }
    vec![load_project(root, hxml)]
}

fn load_project(root: &Path, explicit_hxml: Option<&str>) -> Settings {
    let mut settings = Settings {
        root: root.into(),
        options: Options {
            cache_enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    settings
        .options
        .configuration_files
        .extend([root.join("rayzor.toml"), root.join("haxelib.json")]);
    let manifest_path = root.join("rayzor.toml");
    let project = if manifest_path.is_file() {
        settings.options.configuration_files.push(manifest_path);
        match compiler::workspace::load_project(root) {
            Ok(project) => Some(project),
            Err(message) => {
                settings.problems.push(message);
                None
            }
        }
    } else {
        None
    };
    settings.options.rayzor = project.is_some();
    settings.hxml = explicit_hxml
        .map(|path| root.join(path))
        .or_else(|| {
            project
                .as_ref()
                .and_then(|project| project.manifest.hxml.as_ref())
                .map(|path| root.join(path))
        })
        .or_else(|| {
            project
                .is_none()
                .then(|| crate::project::find_hxml(root, &mut settings.problems))
                .flatten()
        });
    let mut requested = if let Some(path) = settings.hxml.clone() {
        apply_hxml(&mut settings, &path)
    } else {
        Vec::new()
    };
    if let Some(project) = &project {
        settings.class_paths.extend(project.resolved_class_paths());
        settings.options.native_libs = project.resolved_native_libs();
        settings.options.resources.extend(project.resources());
        settings.options.cache_enabled = project.cache_enabled();
        settings.options.cache_dir = project
            .manifest
            .cache
            .as_ref()
            .and_then(|cache| cache.dir.as_ref())
            .map(|path| root.join(path));
        settings.options.extra_defines.push(
            if project
                .manifest
                .build
                .as_ref()
                .and_then(|build| build.target.as_deref())
                == Some("wasm")
            {
                "wasm"
            } else if project.preset() == "embedded" {
                "interp"
            } else {
                "jit"
            }
            .into(),
        );
        for (name, value) in project.defines() {
            settings
                .defines
                .retain(|define| define.split('=').next() != Some(name.as_str()));
            if value.as_deref().is_none_or(|value| {
                !value.trim().is_empty()
                    && value.trim() != "0"
                    && !value.trim().eq_ignore_ascii_case("false")
            }) {
                settings.options.extra_defines.push(
                    value
                        .as_ref()
                        .map_or_else(|| name.clone(), |value| format!("{name}={value}")),
                );
                settings
                    .defines
                    .push(value.map_or(name.clone(), |value| format!("{name}={value}")));
            }
        }
        resolve_packages(
            &mut settings,
            project,
            &mut std::collections::HashSet::new(),
        );
        prepare_packages(&mut settings);
        for path in settings.options.native_libs.clone() {
            match compiler::native_libs::load_manifest_native_lib_for_analysis(&path) {
                Ok((_library, plugin)) => {
                    use compiler::compiler_plugin::CompilerPlugin;
                    add_plugin_define(&mut settings.options.extra_defines, plugin.name());
                }
                Err(message) => settings
                    .problems
                    .push(format!("{}: {message}", path.display())),
            }
        }
    }
    if settings.hxml.is_none()
        && root.join("haxelib.json").is_file()
        && root.join("extraParams.hxml").is_file()
    {
        requested.extend(apply_hxml(&mut settings, &root.join("extraParams.hxml")));
    }
    let mut metadata =
        crate::project::library_metadata(root, &settings.class_paths, &mut settings.problems);
    metadata.provided.retain(|name| {
        !requested.iter().any(|lib| &lib.name == name)
            && !metadata
                .libraries
                .iter()
                .any(|lib| &lib.name == name && lib.is_local())
    });
    settings
        .options
        .configuration_files
        .extend(metadata.input_files);
    settings.class_paths.extend(metadata.class_paths);
    for dependency in metadata.libraries {
        if let Some(existing) = requested.iter_mut().find(|lib| lib.name == dependency.name) {
            if existing.is_unpinned() && dependency.is_local() {
                *existing = dependency;
            }
        } else {
            requested.push(dependency);
        }
    }
    let (libraries, problems) =
        compiler::haxelib::resolve_dependencies(&requested, root, &metadata.provided);
    settings.problems.extend(problems);
    for library in libraries {
        settings
            .options
            .configuration_files
            .extend(library.input_files);
        for class_path in &library.class_paths {
            if let Some(manifest) = class_path
                .ancestors()
                .map(|directory| directory.join("haxelib.json"))
                .find(|path| path.is_file())
            {
                settings
                    .options
                    .configuration_files
                    .push(manifest.parent().unwrap().join("extraParams.hxml"));
                settings.options.configuration_files.push(manifest);
            }
        }
        settings.class_paths.extend(library.class_paths);
        settings.macros.extend(library.macros);
        settings
            .defines
            .extend(library.defines.into_iter().map(|(name, value)| {
                value.map_or_else(|| name.clone(), |value| format!("{name}={value}"))
            }));
    }
    for (path, _) in &settings.options.resources {
        if let Err(error) = std::fs::read(path) {
            settings
                .problems
                .push(format!("Resource {}: {error}", path.display()));
        }
    }
    settings.class_paths.push(settings.root.clone());
    let mut seen = std::collections::HashSet::new();
    settings
        .class_paths
        .retain(|path| seen.insert(path.canonicalize().unwrap_or_else(|_| path.clone())));
    settings
}

fn apply_hxml(settings: &mut Settings, path: &Path) -> Vec<compiler::haxelib::Dependency> {
    settings.options.configuration_files.push(path.into());
    let build = match compiler::hxml::HxmlConfig::builds_from_file(path) {
        Ok(builds) => builds.into_iter().next().unwrap_or_default(),
        Err(message) => {
            settings.problems.push(message);
            return Vec::new();
        }
    };
    settings.problems.extend(
        build
            .warnings
            .iter()
            .map(|(file, line, message)| format!("{}:{line}: {message}", file.display())),
    );
    settings
        .options
        .configuration_files
        .extend(build.input_files.iter().cloned());
    let target_defines = build
        .target_defines()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    settings
        .class_paths
        .extend(build.class_paths.into_iter().rev());
    settings.macros.extend(build.macros);
    settings
        .defines
        .extend(build.defines.into_iter().map(|(name, value)| {
            value.map_or_else(|| name.clone(), |value| format!("{name}={value}"))
        }));
    settings
        .options
        .extra_defines
        .extend(target_defines.iter().cloned());
    settings.defines.extend(target_defines);
    settings
        .options
        .resources
        .extend(build.resources.into_iter().map(|(file, name)| {
            let name = name.unwrap_or_else(|| file.to_string_lossy().into_owned());
            (file, name)
        }));
    build
        .libraries
        .iter()
        .map(|spec| {
            compiler::haxelib::Dependency::parse(spec, path.parent().unwrap_or(&settings.root))
        })
        .collect()
}

fn resolve_packages(
    settings: &mut Settings,
    project: &Project,
    visited: &mut std::collections::HashSet<PathBuf>,
) {
    let root = project
        .root
        .canonicalize()
        .unwrap_or_else(|_| project.root.clone());
    if !visited.insert(root) {
        return;
    }
    let Some(dependencies) = project.manifest.dependencies.as_ref() else {
        return;
    };
    for (name, dependency) in dependencies {
        if let compiler::workspace::manifest::DependencySpec::Table {
            path: Some(path), ..
        } = dependency
        {
            let path = project.root.join(path);
            if path.join("rayzor.toml").is_file() {
                match compiler::workspace::load_project(&path) {
                    Ok(dependency) => {
                        settings
                            .options
                            .configuration_files
                            .push(path.join("rayzor.toml"));
                        settings
                            .class_paths
                            .extend(dependency.resolved_class_paths());
                        settings.class_paths.push(path);
                        settings
                            .options
                            .native_libs
                            .extend(dependency.resolved_native_libs());
                        resolve_packages(settings, &dependency, visited);
                    }
                    Err(message) => settings.problems.push(message),
                }
                continue;
            }
        }
        let mut single = project.manifest.clone();
        single.dependencies = Some(std::collections::BTreeMap::from([(
            name.clone(),
            dependency.clone(),
        )]));
        match compiler::workspace::resolve_dependencies(&single, &project.root) {
            Ok(paths) => settings.options.packages.extend(paths),
            Err(message) => settings.problems.push(message),
        }
    }
}

pub(crate) fn add_plugin_define(defines: &mut Vec<String>, name: &str) {
    let short = name
        .trim_start_matches("lib")
        .trim_start_matches("rayzor_")
        .trim_end_matches("_plugins");
    if !short.is_empty() && !defines.iter().any(|define| define == short) {
        defines.push(short.into());
    }
}

fn prepare_packages(settings: &mut Settings) {
    use std::hash::{Hash, Hasher};
    for path in settings.options.packages.clone() {
        let loaded = match compiler::rpkg::load_rpkg(&path) {
            Ok(loaded) => loaded,
            Err(message) => {
                settings
                    .problems
                    .push(format!("{}: {message}", path.display()));
                continue;
            }
        };
        add_plugin_define(
            &mut settings.options.extra_defines,
            loaded
                .plugin_name
                .as_deref()
                .unwrap_or(&loaded.package_name),
        );
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        loaded.haxe_sources.hash(&mut hash);
        let directory = settings
            .options
            .cache_dir
            .clone()
            .unwrap_or_else(|| settings.root.join(".rayzor/blade"))
            .join(".packages")
            .join(format!("{:016x}", hash.finish()));
        let mut valid = true;
        for (module, source) in loaded.haxe_sources {
            let relative = Path::new(&module);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                settings
                    .problems
                    .push(format!("{}: invalid source path {module}", path.display()));
                valid = false;
                continue;
            }
            let file = directory.join(relative);
            let result = (|| -> std::io::Result<()> {
                if std::fs::read_to_string(&file).ok().as_deref() == Some(&source) {
                    return Ok(());
                }
                if let Some(parent) = file.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&file, source)
            })();
            if let Err(error) = result {
                settings
                    .problems
                    .push(format!("{}: {error}", file.display()));
                valid = false;
            }
        }
        if valid && directory.is_dir() {
            settings.class_paths.push(directory);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn manifest_disables_delegated_flags_and_keeps_values_and_target() {
        let root =
            std::env::temp_dir().join(format!("rayzor-manifest-flags-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("rayzor.toml"), "[project]\nname = 'flags'\nhxml = 'build.hxml'\n[build]\ntarget = 'wasm'\n[build.defines]\nnumber = false\nversion = '2.3'\n").unwrap();
        std::fs::write(root.join("build.hxml"), "-D number\n-D version=old\n").unwrap();
        let settings = super::load(&root, None).remove(0);
        assert!(settings.problems.is_empty(), "{:?}", settings.problems);
        assert!(!settings.defines.iter().any(|define| define == "number"));
        assert!(
            settings
                .defines
                .iter()
                .any(|define| define == "version=2.3")
        );
        assert!(
            !settings
                .defines
                .iter()
                .any(|define| define == "version=old")
        );
        assert!(
            settings
                .options
                .extra_defines
                .iter()
                .any(|define| define == "wasm")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
