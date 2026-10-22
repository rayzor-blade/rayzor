//! HXML Build File Parser
//!
//! Reads `.hxml` files the way haxe does: one argument per line, a flag taking
//! the rest of its line as its value, a bare `*.hxml` line including that file,
//! `--next` starting a new build and arguments before `--each` shared by all.
//! Rayzor runs every build as `--rayzor-jit` (default) or `--rayzor-compile`;
//! a haxe target flag is recorded with the defines that target implies.
//!
//! # Example HXML File
//! ```hxml
//! -cp src
//! -main Main
//! -lib lime
//! -D analyzer-optimize
//! --rayzor-jit
//! ```

use std::path::{Path, PathBuf};

/// HXML compilation configuration
#[derive(Debug, Clone)]
pub struct HxmlConfig {
    /// Class paths (-cp)
    pub class_paths: Vec<PathBuf>,
    /// Build files read while expanding includes.
    pub input_files: Vec<PathBuf>,

    /// Main class (-main)
    pub main_class: Option<String>,

    /// Output file (optional for JIT, required for compile)
    pub output: Option<PathBuf>,

    /// Rayzor compilation mode
    pub mode: RayzorMode,

    /// Libraries to include (-lib), as `name` or `name:version`
    pub libraries: Vec<String>,

    /// Defines (-D)
    pub defines: Vec<(String, Option<String>)>,

    /// Resources to embed (-resource)
    pub resources: Vec<(PathBuf, Option<String>)>,

    /// Debug mode (-debug)
    pub debug: bool,

    /// Verbose output (-v)
    pub verbose: bool,

    /// Flags recognised but not acted on (`--dce full`, `--no-inline`, ...)
    pub compiler_flags: Vec<String>,

    /// `.hx` files named directly
    pub source_files: Vec<PathBuf>,

    /// Types named directly to be compiled (`pack.Type` lines)
    pub classes: Vec<String>,

    /// The haxe target the build names (`--js out.js` gives `js`)
    pub target: Option<String>,

    /// `--macro` calls, in order
    pub macros: Vec<String>,

    /// Lines that could not be understood: (file, line number, message)
    pub warnings: Vec<(PathBuf, usize, String)>,
}

impl Default for HxmlConfig {
    fn default() -> Self {
        Self {
            class_paths: Vec::new(),
            input_files: Vec::new(),
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
            classes: Vec::new(),
            target: None,
            macros: Vec::new(),
            warnings: Vec::new(),
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

/// The haxe targets and the defines each one sets for conditional compilation.
const TARGETS: &[(&str, &str, &[&str])] = &[
    ("--js", "js", &["js"]),
    ("--swf", "flash", &["flash", "swf"]),
    ("--neko", "neko", &["neko", "sys"]),
    ("--php", "php", &["php", "sys"]),
    ("--cpp", "cpp", &["cpp", "sys"]),
    ("--cppia", "cppia", &["cppia", "cpp", "sys"]),
    ("--cs", "cs", &["cs", "sys"]),
    ("--java", "java", &["java", "sys"]),
    ("--jvm", "jvm", &["jvm", "java", "sys"]),
    ("--python", "python", &["python", "sys"]),
    ("--lua", "lua", &["lua"]),
    ("--hl", "hl", &["hl", "sys"]),
    ("--interp", "eval", &["eval", "interp", "sys"]),
    ("--run", "eval", &["eval", "interp", "sys"]),
];

/// Flags taking a value that rayzor records without acting on.
const VALUED_FLAGS: &[&str] = &[
    "--dce",
    "-dce",
    "--cwd",
    "-C",
    "--connect",
    "--wait",
    "--server-connect",
    "--server-listen",
    "--display",
    "--remap",
    "--xml",
    "--json",
    "--times",
    "--cmd",
    "-w",
    "--debug-port",
    "--undefine",
    "--resource-path",
];

/// Flags without a value that rayzor records without acting on.
const BARE_FLAGS: &[&str] = &[
    "--no-inline",
    "--no-opt",
    "--no-output",
    "--no-traces",
    "--times",
    "--prompt",
    "--interp-mode",
    "--each",
    "--next",
    "--help",
    "--version",
];

impl HxmlConfig {
    /// Parse an HXML file. With `--next`, the first build.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        Ok(Self::builds_from_file(path)?.swap_remove(0))
    }

    /// Parse HXML content from a string. With `--next`, the first build.
    pub fn from_string(content: &str) -> Result<Self, String> {
        Ok(Self::builds_from_string(content, Path::new("."))?.swap_remove(0))
    }

    /// Every build an HXML file describes, one per `--next` section, each
    /// with the arguments given before `--each`.
    pub fn builds_from_file(path: &Path) -> Result<Vec<Self>, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read HXML file {}: {}", path.display(), e))?;
        let mut args = Vec::new();
        let mut seen = vec![path.canonicalize().unwrap_or_else(|_| path.to_path_buf())];
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        collect_args(&content, path, &dir, &mut args, &mut seen);
        let mut builds = Self::from_args(&args);
        for build in &mut builds {
            build.input_files = seen.clone();
        }
        Ok(builds)
    }

    /// As [`Self::builds_from_file`], for content whose includes resolve
    /// against `dir`.
    pub fn builds_from_string(content: &str, dir: &Path) -> Result<Vec<Self>, String> {
        let mut args = Vec::new();
        let origin = dir.join("<string>.hxml");
        let mut seen = Vec::new();
        collect_args(content, &origin, dir, &mut args, &mut seen);
        let mut builds = Self::from_args(&args);
        for build in &mut builds {
            build.input_files = seen.clone();
        }
        Ok(builds)
    }

    fn from_args(args: &[Arg]) -> Vec<Self> {
        let mut shared: Vec<&Arg> = Vec::new();
        let mut sections: Vec<Vec<&Arg>> = vec![Vec::new()];
        for arg in args {
            let section = sections.last_mut().expect("never empty");
            match arg.flag.as_str() {
                "--each" => shared.append(section),
                "--next" => sections.push(Vec::new()),
                _ => section.push(arg),
            }
        }
        if sections.len() > 1 {
            sections.retain(|s| !s.is_empty());
        }
        if sections.is_empty() {
            sections.push(Vec::new());
        }
        sections
            .into_iter()
            .map(|section| {
                let mut config = HxmlConfig::default();
                for arg in shared.iter().chain(&section) {
                    config.apply(arg);
                }
                config
            })
            .collect()
    }

    fn apply(&mut self, arg: &Arg) {
        let value = arg.value.as_deref().unwrap_or("");
        let path = || arg.dir.join(value);
        match arg.flag.as_str() {
            "-cp" | "-p" | "--class-path" => self.class_paths.push(path()),
            "-main" | "-m" | "--main" => self.main_class = Some(value.to_string()),
            "-lib" | "-L" | "--library" => self.libraries.push(value.to_string()),
            "-D" | "--define" => {
                let (name, v) = match value.split_once('=') {
                    Some((n, v)) => (n, Some(v.to_string())),
                    None => (value, None),
                };
                self.defines.push((name.to_string(), v));
            }
            "-resource" | "-r" | "--resource" => {
                let (file, name) = match value.split_once('@') {
                    Some((f, n)) => (f, Some(n.to_string())),
                    None => (value, None),
                };
                self.resources.push((arg.dir.join(file), name));
            }
            "-debug" | "--debug" => self.debug = true,
            "-v" | "--verbose" => self.verbose = true,
            "--macro" => self.macros.push(value.to_string()),
            "--rayzor-jit" => {
                self.mode = RayzorMode::Jit;
                if !value.is_empty() {
                    self.output = Some(path());
                }
            }
            "--rayzor-compile" => {
                self.mode = RayzorMode::Compile;
                if !value.is_empty() {
                    self.output = Some(path());
                }
            }
            "-output" | "--output" => self.output = Some(path()),
            "" => {
                if value.ends_with(".hx") {
                    self.source_files.push(path());
                } else {
                    self.classes.push(value.to_string());
                }
            }
            flag => {
                if let Some((_, target, _)) = TARGETS.iter().find(|(f, ..)| *f == flag) {
                    self.target = Some(target.to_string());
                    if flag == "--run" {
                        let class = value.split_whitespace().next().unwrap_or("");
                        self.main_class = Some(class.to_string());
                    }
                } else if VALUED_FLAGS.contains(&flag) || BARE_FLAGS.contains(&flag) {
                    self.compiler_flags.push(arg.text());
                } else {
                    self.warnings.push((
                        arg.file.clone(),
                        arg.line,
                        format!("unknown flag `{}`", arg.text()),
                    ));
                    self.compiler_flags.push(arg.text());
                }
            }
        }
    }

    /// Defines the build's target sets, on top of the `-D` ones.
    pub fn target_defines(&self) -> Vec<&'static str> {
        TARGETS
            .iter()
            .find(|(_, t, _)| Some(*t) == self.target.as_deref())
            .map(|(_, _, d)| d.to_vec())
            .unwrap_or_default()
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
        s.push_str(&format!("Target: {:?}\n", self.target));
        s.push_str(&format!("Output: {:?}\n", self.output));
        s.push_str(&format!("Libraries: {}\n", self.libraries.len()));
        s.push_str(&format!("Defines: {}\n", self.defines.len()));
        s.push_str(&format!("Debug: {}\n", self.debug));
        for (file, line, message) in &self.warnings {
            s.push_str(&format!("{}:{}: {}\n", file.display(), line, message));
        }
        s
    }
}

/// One argument: a flag and its value, or a bare value (`flag` empty).
#[derive(Debug, Clone)]
struct Arg {
    flag: String,
    value: Option<String>,
    /// Directory relative paths in the value resolve against.
    dir: PathBuf,
    file: PathBuf,
    line: usize,
}

impl Arg {
    fn text(&self) -> String {
        match &self.value {
            Some(v) if !self.flag.is_empty() => format!("{} {}", self.flag, v),
            Some(v) => v.clone(),
            None => self.flag.clone(),
        }
    }
}

/// Appends the arguments of `content` (read from `file`) to `out`, expanding
/// included `.hxml` files once each. Every relative path, those in included
/// files too, resolves against `dir`, as haxe resolves them against its
/// working directory.
fn collect_args(
    content: &str,
    file: &Path,
    dir: &Path,
    out: &mut Vec<Arg>,
    seen: &mut Vec<PathBuf>,
) {
    for (n, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (flag, value) = if line.starts_with('-') {
            match line.split_once(char::is_whitespace) {
                Some((f, v)) => (f.to_string(), Some(unquote(v.trim()))),
                None => (line.to_string(), None),
            }
        } else {
            (String::new(), Some(unquote(line)))
        };
        if flag.is_empty()
            && let Some(name) = value.as_deref().filter(|v| v.ends_with(".hxml"))
        {
            let included = dir.join(name);
            let key = included.canonicalize().unwrap_or_else(|_| included.clone());
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            match std::fs::read_to_string(&included) {
                Ok(text) => collect_args(&text, &included, dir, out, seen),
                Err(e) => out.push(Arg {
                    flag: "--missing-include".into(),
                    value: Some(format!("{}: {}", included.display(), e)),
                    dir: dir.to_path_buf(),
                    file: file.to_path_buf(),
                    line: n + 1,
                }),
            }
            continue;
        }
        out.push(Arg {
            flag,
            value,
            dir: dir.to_path_buf(),
            file: file.to_path_buf(),
            line: n + 1,
        });
    }
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    v.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(v)
        .to_string()
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
        assert!(config.debug);
        assert!(config.verbose);
        assert!(config.warnings.is_empty(), "{:?}", config.warnings);
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
        assert_eq!(config.output, Some(PathBuf::from("./output.bin")));
    }

    #[test]
    fn test_target_is_recorded_with_its_defines() {
        let hxml = "-main Test\n--js output.js\n--rayzor-jit";
        let config = HxmlConfig::from_string(hxml).unwrap();
        assert_eq!(config.mode, RayzorMode::Jit);
        assert_eq!(config.target.as_deref(), Some("js"));
        assert_eq!(config.target_defines(), vec!["js"]);
    }

    #[test]
    fn test_default_to_jit() {
        let hxml = "-main Test\n-cp src";
        let config = HxmlConfig::from_string(hxml).unwrap();
        // Should default to JIT mode
        assert_eq!(config.mode, RayzorMode::Jit);
    }

    #[test]
    fn long_forms_quotes_and_unknown_flags() {
        let hxml = "--class-path \"my src\"\n-p lib\n--main pack.Main\n--library heaps:2.0\n--define NAME=1\n--dce full\n--bogus 3\npack.Extra\n";
        let config = HxmlConfig::from_string(hxml).unwrap();
        assert_eq!(
            config.class_paths,
            vec![PathBuf::from("./my src"), PathBuf::from("./lib")]
        );
        assert_eq!(config.main_class.as_deref(), Some("pack.Main"));
        assert_eq!(config.libraries, vec!["heaps:2.0"]);
        assert_eq!(config.defines, vec![("NAME".into(), Some("1".into()))]);
        assert_eq!(config.classes, vec!["pack.Extra"]);
        assert_eq!(config.warnings.len(), 1);
        assert_eq!(config.warnings[0].1, 7);
    }

    #[test]
    fn next_and_each_give_one_build_per_section() {
        let hxml = "-cp src\n-main Main\n--each\n--js a.js\n--next\n--hl b.hl\n";
        let builds = HxmlConfig::builds_from_string(hxml, Path::new(".")).unwrap();
        assert_eq!(builds.len(), 2);
        for b in &builds {
            assert_eq!(b.main_class.as_deref(), Some("Main"));
            assert_eq!(b.class_paths, vec![PathBuf::from("./src")]);
        }
        assert_eq!(builds[0].target.as_deref(), Some("js"));
        assert_eq!(builds[1].target.as_deref(), Some("hl"));
    }

    #[test]
    fn includes_expand_once_against_the_base_directory() {
        let dir = std::env::temp_dir().join(format!("rayzor-hxml-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("conf")).unwrap();
        std::fs::write(dir.join("conf/common.hxml"), "-cp shared\n-D common\n").unwrap();
        std::fs::write(
            dir.join("build.hxml"),
            "conf/common.hxml\nconf/common.hxml\n-main Main\n",
        )
        .unwrap();
        let config = HxmlConfig::from_file(&dir.join("build.hxml")).unwrap();
        assert_eq!(config.class_paths, vec![dir.join("shared")]);
        assert_eq!(config.defines.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
