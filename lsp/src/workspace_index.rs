//! Source declarations across configured class paths. Entries use qualified
//! names and source ranges; compiler-local symbol IDs never leave an analysis.

use crate::documentation::SourceComments;
use parser::{ClassFieldKind, HaxeFile, Span, TypeDeclaration};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};

#[derive(Clone)]
pub(crate) struct Declaration {
    pub name: String,
    pub qualified: String,
    aliases: Vec<String>,
    pub documentation: Option<String>,
    pub signature: String,
    kind: lsp_types::SymbolKind,
    range: lsp_types::Range,
}

enum Event {
    File(PathBuf, Vec<Declaration>),
    Finished,
}

#[derive(Default)]
struct State {
    files: BTreeMap<PathBuf, Vec<Declaration>>,
    overrides: BTreeSet<PathBuf>,
    receiver: Option<Receiver<Event>>,
}

#[derive(Default)]
pub(crate) struct WorkspaceIndex {
    state: RefCell<State>,
    defines: Vec<String>,
    roots: Vec<(PathBuf, PathBuf)>,
}

impl WorkspaceIndex {
    pub fn start(paths: Vec<PathBuf>, defines: Vec<String>) -> Self {
        let mut roots: Vec<_> = paths.iter().map(|p| (canonical(p), p.clone())).collect();
        roots.sort_by_key(|(path, _)| std::cmp::Reverse(path.components().count()));
        let (sender, receiver) = mpsc::channel();
        let worker_defines = defines.clone();
        let worker = std::thread::Builder::new()
            .name("lsp-index".into())
            .stack_size(64 << 20)
            .spawn(move || {
                let mut visited = BTreeSet::new();
                for path in paths {
                    let mut pending = vec![path];
                    while let Some(path) = pending.pop() {
                        let path = canonical(&path);
                        if !visited.insert(path.clone()) {
                            continue;
                        }
                        if path.is_dir() {
                            let Ok(entries) = std::fs::read_dir(&path) else {
                                continue;
                            };
                            let mut children: Vec<_> = entries
                                .flatten()
                                .filter_map(|entry| {
                                    let kind = entry.file_type().ok()?;
                                    if kind.is_symlink() {
                                        return None;
                                    }
                                    let name = entry.file_name();
                                    let name = name.to_string_lossy();
                                    if kind.is_dir()
                                        && (name.starts_with('.')
                                            || matches!(
                                                name.as_ref(),
                                                "target"
                                                    | "node_modules"
                                                    | "bin"
                                                    | "out"
                                                    | "dist"
                                                    | "vendor"
                                            ))
                                    {
                                        return None;
                                    }
                                    Some(entry.path())
                                })
                                .collect();
                            children.sort();
                            pending.extend(children.into_iter().rev());
                        } else if path.extension().is_some_and(|ext| ext == "hx") {
                            let Ok(source) = std::fs::read_to_string(&path) else {
                                continue;
                            };
                            let parsed = std::panic::catch_unwind(|| {
                                declarations(&path, &source, &worker_defines)
                            })
                            .unwrap_or_default();
                            if sender.send(Event::File(path, parsed)).is_err() {
                                return;
                            }
                        }
                    }
                }
                let _ = sender.send(Event::Finished);
            });
        Self {
            state: RefCell::new(State {
                receiver: worker.ok().map(|_| receiver),
                ..Default::default()
            }),
            defines,
            roots,
        }
    }

    fn poll(&self, wait: bool) {
        let mut state = self.state.borrow_mut();
        let Some(receiver) = state.receiver.take() else {
            return;
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let event = if wait {
                receiver
                    .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                    .ok()
            } else {
                receiver.try_recv().ok()
            };
            match event {
                Some(Event::File(path, declarations)) => {
                    if !state.overrides.contains(&path) {
                        state.files.entry(path).or_insert(declarations);
                    }
                }
                Some(Event::Finished) => return,
                None => {
                    state.receiver = Some(receiver);
                    return;
                }
            }
        }
    }

    pub fn update(&self, path: &Path, source: Option<&str>) {
        let path = canonical(path);
        let parsed = source.map(|s| declarations(&path, s, &self.defines));
        let mut state = self.state.borrow_mut();
        state.overrides.insert(path.clone());
        if let Some(parsed) = parsed {
            state.files.insert(path, parsed);
        } else {
            state.files.remove(&path);
        }
    }

    pub fn ensure_parsed_file(&self, file: &HaxeFile) {
        let path = canonical(Path::new(&file.filename));
        self.poll(false);
        if self.state.borrow().files.contains_key(&path) {
            return;
        }
        let disk_source;
        let source = if let Some(source) = file.input.as_deref() {
            source
        } else {
            disk_source = std::fs::read_to_string(&path).ok();
            let Some(source) = disk_source.as_deref() else {
                return;
            };
            source
        };
        self.state
            .borrow_mut()
            .files
            .insert(path, collect(file, source));
    }

    pub fn ensure_disk_file(&self, path: &Path) {
        let path = canonical(path);
        self.poll(false);
        if self.state.borrow().files.contains_key(&path) {
            return;
        }
        if let Ok(source) = std::fs::read_to_string(&path) {
            self.state
                .borrow_mut()
                .files
                .insert(path.clone(), declarations(&path, &source, &self.defines));
        }
    }

    pub fn declaration(&self, qualified: &str) -> Option<Declaration> {
        self.poll(false);
        let state = self.state.borrow();
        let mut matches = state.files.values().flatten().filter(|entry| {
            entry.qualified == qualified || entry.aliases.iter().any(|alias| alias == qualified)
        });
        let first = matches.next()?;
        matches.next().is_none().then(|| first.clone())
    }

    pub fn search(&self, query: &str) -> Vec<serde_json::Value> {
        self.poll(true);
        let query = query.to_lowercase();
        self.state.borrow().files.iter().flat_map(|(path, declarations)| declarations.iter().filter(|d| d.qualified.to_lowercase().contains(&query)).map(|d| {
            let display = self.roots.iter().find_map(|(canonical,original)|path.strip_prefix(canonical).ok().map(|rest|original.join(rest))).unwrap_or_else(||path.clone());
            serde_json::json!({"name":d.name,"kind":d.kind,"containerName":d.qualified.rsplit_once('.').map(|(owner,_)|owner),"location":{"uri":crate::context::file_uri(&display.to_string_lossy()),"range":d.range}})
        })).collect()
    }
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| {
        // Deleted files retain the identity of their existing parent directory.
        match (path.parent(), path.file_name()) {
            (Some(parent), Some(name)) if parent != path => canonical(parent).join(name),
            _ => path.to_path_buf(),
        }
    })
}

fn declarations(path: &Path, source: &str, defines: &[String]) -> Vec<Declaration> {
    let mut config = parser::preprocessor::PreprocessorConfig::default();
    config.defines.insert("eval".into());
    config.defines.extend(
        defines
            .iter()
            .map(|d| d.split('=').next().unwrap().to_string()),
    );
    let text = parser::preprocessor::preprocess(source, &config);
    let filename = path.to_string_lossy();
    let Ok(file) = parser::rd::rd_parse(
        &text,
        &filename,
        path.file_name().is_some_and(|n| n == "import.hx"),
        false,
    ) else {
        return Vec::new();
    };
    collect(&file, &text)
}

fn collect(file: &HaxeFile, source: &str) -> Vec<Declaration> {
    let Some(comments) = SourceComments::new(source) else {
        return Vec::new();
    };
    let Ok(tokens) = parser::lexer::Lexer::new(source).tokenize() else {
        return Vec::new();
    };
    let package = file
        .package
        .as_ref()
        .map(|p| p.path.join("."))
        .unwrap_or_default();
    let module = Path::new(&file.filename)
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let qualify = |name: &str| {
        if package.is_empty() {
            name.to_string()
        } else {
            format!("{package}.{name}")
        }
    };
    let position = |offset: usize| {
        let before = &source[..offset];
        lsp_types::Position::new(
            before.bytes().filter(|&b| b == b'\n').count() as u32,
            before
                .rsplit('\n')
                .next()
                .unwrap_or("")
                .encode_utf16()
                .count() as u32,
        )
    };
    let mut entries = Vec::new();
    for declaration in &file.declarations {
        let (name, fields, kind) = match declaration {
            TypeDeclaration::Class(c) => {
                (&c.name, c.fields.as_slice(), lsp_types::SymbolKind::CLASS)
            }
            TypeDeclaration::Interface(c) => (
                &c.name,
                c.fields.as_slice(),
                lsp_types::SymbolKind::INTERFACE,
            ),
            TypeDeclaration::Abstract(c) => {
                (&c.name, c.fields.as_slice(), lsp_types::SymbolKind::CLASS)
            }
            TypeDeclaration::Enum(c) => (&c.name, &[][..], lsp_types::SymbolKind::ENUM),
            TypeDeclaration::Typedef(c) => (&c.name, &[][..], lsp_types::SymbolKind::STRUCT),
            _ => continue,
        };
        let owner = if name == module {
            qualify(name)
        } else {
            qualify(&format!("{module}.{name}"))
        };
        let mut add = |name: &str,
                       qualified: String,
                       alias: String,
                       span: Span,
                       kind,
                       signature_end: Option<usize>| {
            let Some(token) = tokens.iter().find(|t| {
                t.span.start >= span.start
                    && t.span.end <= span.end
                    && source.get(t.span.start..t.span.end) == Some(name)
            }) else {
                return;
            };
            let end = signature_end.unwrap_or(token.span.end).min(source.len());
            entries.push(Declaration {
                name: name.into(),
                qualified,
                aliases: vec![alias],
                kind,
                range: lsp_types::Range::new(position(token.span.start), position(token.span.end)),
                documentation: comments.at(span.start),
                signature: source
                    .get(span.start..end)
                    .unwrap_or_default()
                    .trim()
                    .trim_end_matches(';')
                    .trim()
                    .to_string(),
            });
        };
        add(
            name,
            owner.clone(),
            qualify(name),
            declaration.span(),
            kind,
            None,
        );
        for field in fields {
            let (name, kind, end) = match &field.kind {
                ClassFieldKind::Function(f) => (
                    f.name.as_str(),
                    lsp_types::SymbolKind::METHOD,
                    Some(f.body.as_ref().map_or(f.span.end, |body| body.span.start)),
                ),
                ClassFieldKind::Var { name, .. } | ClassFieldKind::Final { name, .. } => {
                    (name.as_str(), lsp_types::SymbolKind::FIELD, None)
                }
                ClassFieldKind::Property { name, .. } => {
                    (name.as_str(), lsp_types::SymbolKind::PROPERTY, None)
                }
            };
            add(
                name,
                format!("{owner}.{name}"),
                format!("{}.{name}", qualify(declaration_name(declaration))),
                field.span,
                kind,
                end,
            );
        }
    }
    entries
}

fn declaration_name(declaration: &TypeDeclaration) -> &str {
    match declaration {
        TypeDeclaration::Class(c) => &c.name,
        TypeDeclaration::Interface(c) => &c.name,
        TypeDeclaration::Abstract(c) => &c.name,
        TypeDeclaration::Enum(c) => &c.name,
        TypeDeclaration::Typedef(c) => &c.name,
        _ => "",
    }
}
