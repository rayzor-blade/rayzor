//! Source declarations across configured class paths. Entries use qualified
//! names and source ranges; compiler-local symbol IDs never leave an analysis.

use crate::documentation::SourceComments;
use compiler::tast::symbols::SymbolKind;
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
    pub type_kind: Option<SymbolKind>,
    pub(crate) kind: lsp_types::SymbolKind,
    pub(crate) range: lsp_types::Range,
}

pub(crate) struct TypeIdentity {
    pub qualified: String,
    pub kind: SymbolKind,
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

pub(crate) enum IndexActivity {
    Scanning {
        path: PathBuf,
        subject: String,
        completed: usize,
    },
    Finished {
        completed: usize,
    },
}

pub(crate) type IndexProgress = std::sync::Arc<dyn Fn(IndexActivity) + Send + Sync>;

#[derive(Clone, Default)]
pub(crate) struct WorkspaceIndex {
    state: std::rc::Rc<RefCell<State>>,
    defines: Vec<String>,
    profiles: Vec<(PathBuf, Vec<String>)>,
    roots: Vec<(PathBuf, PathBuf)>,
}

impl WorkspaceIndex {
    pub fn start(
        paths: Vec<PathBuf>,
        defines: Vec<String>,
        profiles: Vec<(PathBuf, Vec<String>)>,
        progress: Option<IndexProgress>,
    ) -> Self {
        let mut roots: Vec<_> = paths.iter().map(|p| (canonical(p), p.clone())).collect();
        roots.sort_by_key(|(path, _)| std::cmp::Reverse(path.components().count()));
        let (sender, receiver) = mpsc::sync_channel(32);
        let worker_defines = defines.clone();
        let worker_profiles = profiles.clone();
        let progress_roots = roots
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        let worker = std::thread::Builder::new()
            .name("lsp-index".into())
            .stack_size(64 << 20)
            .spawn(move || {
                let mut visited = BTreeSet::new();
                let mut completed = 0;
                let mut reported = None::<std::time::Instant>;
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
                            if let Some(progress) = &progress
                                && reported.is_none_or(|at| {
                                    at.elapsed() >= std::time::Duration::from_millis(100)
                                })
                            {
                                let relative = progress_roots
                                    .iter()
                                    .find_map(|root| path.strip_prefix(root).ok())
                                    .unwrap_or(&path);
                                let subject = relative
                                    .with_extension("")
                                    .components()
                                    .map(|part| part.as_os_str().to_string_lossy())
                                    .collect::<Vec<_>>()
                                    .join(".");
                                progress(IndexActivity::Scanning {
                                    path: path.clone(),
                                    subject,
                                    completed,
                                });
                                reported = Some(std::time::Instant::now());
                            }
                            let parsed = std::panic::catch_unwind(|| {
                                declarations(
                                    &path,
                                    &source,
                                    profile_defines(&path, &worker_defines, &worker_profiles),
                                )
                            })
                            .unwrap_or_default();
                            completed += 1;
                            if sender.send(Event::File(path, parsed)).is_err() {
                                return;
                            }
                        }
                    }
                }
                if let Some(progress) = &progress {
                    progress(IndexActivity::Finished { completed });
                }
                let _ = sender.send(Event::Finished);
            });
        Self {
            state: std::rc::Rc::new(RefCell::new(State {
                receiver: worker.ok().map(|_| receiver),
                ..Default::default()
            })),
            defines,
            profiles,
            roots,
        }
    }

    pub fn pump(&self) -> bool {
        self.poll(false);
        self.state.borrow().receiver.is_some()
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
        let parsed = source.map(|s| {
            declarations(
                &path,
                s,
                profile_defines(&path, &self.defines, &self.profiles),
            )
        });
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
        // Cached or transformed AST spans can differ from the editor source.
        let entries = declarations(
            &path,
            source,
            profile_defines(&path, &self.defines, &self.profiles),
        );
        self.state.borrow_mut().files.insert(path, entries);
    }

    pub fn ensure_disk_file(&self, path: &Path) {
        let path = canonical(path);
        self.poll(false);
        if self.state.borrow().files.contains_key(&path) {
            return;
        }
        if let Ok(source) = std::fs::read_to_string(&path) {
            self.state.borrow_mut().files.insert(
                path.clone(),
                declarations(
                    &path,
                    &source,
                    profile_defines(&path, &self.defines, &self.profiles),
                ),
            );
        }
    }

    pub fn primary_type(&self, path: &Path) -> Option<String> {
        self.poll(false);
        self.state
            .borrow()
            .files
            .get(&canonical(path))?
            .iter()
            .find(|entry| entry.type_kind.is_some())
            .map(|entry| entry.qualified.clone())
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

    pub fn definition(&self, qualified: &str) -> Option<(PathBuf, lsp_types::Range)> {
        self.poll(false);
        let state = self.state.borrow();
        let mut matches = state.files.iter().flat_map(|(path, entries)| {
            entries
                .iter()
                .filter(|entry| {
                    entry.qualified == qualified
                        || entry.aliases.iter().any(|alias| alias == qualified)
                })
                .map(move |entry| (path, entry.range))
        });
        let (path, range) = matches.next()?;
        matches.next().is_none().then(|| (path.clone(), range))
    }

    pub fn type_identities(&self) -> BTreeMap<String, TypeIdentity> {
        self.poll(false);
        let state = self.state.borrow();
        let mut names = BTreeMap::new();
        for declaration in state
            .files
            .values()
            .flatten()
            .filter(|d| d.type_kind.is_some())
        {
            for name in std::iter::once(&declaration.qualified)
                .chain(&declaration.aliases)
                .collect::<BTreeSet<_>>()
            {
                names
                    .entry(name.clone())
                    .and_modify(|entry| *entry = None)
                    .or_insert(Some(declaration));
            }
        }
        names
            .into_iter()
            .filter_map(|(name, declaration)| {
                let declaration = declaration?;
                Some((
                    name,
                    TypeIdentity {
                        qualified: declaration.qualified.clone(),
                        kind: declaration.type_kind?,
                    },
                ))
            })
            .collect()
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

fn profile_defines<'a>(
    path: &Path,
    fallback: &'a [String],
    profiles: &'a [(PathBuf, Vec<String>)],
) -> &'a [String] {
    profiles
        .iter()
        .filter(|(root, _)| path.starts_with(root))
        .max_by_key(|(root, _)| root.components().count())
        .map_or(fallback, |(_, defines)| defines)
}

pub(crate) fn declarations(path: &Path, source: &str, defines: &[String]) -> Vec<Declaration> {
    let mut config = parser::preprocessor::PreprocessorConfig::default();
    config.defines.insert("eval".into());
    config.add_defines(defines.iter());
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
        let (name, fields, kind, type_kind) = match declaration {
            TypeDeclaration::Class(c) => (
                &c.name,
                c.fields.as_slice(),
                lsp_types::SymbolKind::CLASS,
                SymbolKind::Class,
            ),
            TypeDeclaration::Interface(c) => (
                &c.name,
                c.fields.as_slice(),
                lsp_types::SymbolKind::INTERFACE,
                SymbolKind::Interface,
            ),
            TypeDeclaration::Abstract(c) => (
                &c.name,
                c.fields.as_slice(),
                lsp_types::SymbolKind::STRUCT,
                SymbolKind::Abstract,
            ),
            TypeDeclaration::Enum(c) => (
                &c.name,
                &[][..],
                lsp_types::SymbolKind::ENUM,
                SymbolKind::Enum,
            ),
            TypeDeclaration::Typedef(c) => (
                &c.name,
                &[][..],
                lsp_types::SymbolKind::STRUCT,
                SymbolKind::TypeAlias,
            ),
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
                       type_kind,
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
                type_kind,
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
            Some(type_kind),
            Some(type_signature_end(declaration, &tokens)),
        );
        if let TypeDeclaration::Typedef(alias) = declaration {
            for field in crate::structural_fields::source_fields(&alias.type_def) {
                add(
                    &field.name,
                    format!("{owner}.{}", field.name),
                    format!("{}.{}", qualify(name), field.name),
                    field.span,
                    lsp_types::SymbolKind::FIELD,
                    None,
                    Some(field.span.end),
                );
            }
        }
        for field in fields {
            let (name, kind, end) = match &field.kind {
                ClassFieldKind::Function(f) => (
                    f.name.as_str(),
                    lsp_types::SymbolKind::METHOD,
                    Some(f.body.as_ref().map_or(f.span.end, |body| body.span.start)),
                ),
                ClassFieldKind::Var { name, .. } | ClassFieldKind::Final { name, .. } => (
                    name.as_str(),
                    lsp_types::SymbolKind::FIELD,
                    Some(
                        tokens
                            .iter()
                            .find(|token| {
                                token.span.start >= field.span.start
                                    && token.span.end <= field.span.end
                                    && token.kind == parser::token::TokenKind::Assign
                            })
                            .map_or(field.span.end, |token| token.span.start),
                    ),
                ),
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
                None,
                end,
            );
        }
    }
    entries
}

fn type_signature_end(declaration: &TypeDeclaration, tokens: &[parser::token::Token]) -> usize {
    use parser::token::TokenKind;
    let span = declaration.span();
    if matches!(declaration, TypeDeclaration::Typedef(_)) {
        return span.end;
    }
    // The final balanced brace pair is the body; constraints and underlying
    // anonymous types can contain earlier pairs in the declaration header.
    let mut depth = 0;
    for token in tokens
        .iter()
        .rev()
        .filter(|t| t.span.start >= span.start && t.span.end <= span.end)
    {
        match token.kind {
            TokenKind::RBrace => depth += 1,
            TokenKind::LBrace if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    return token.span.start;
                }
            }
            _ => {}
        }
    }
    span.end
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_ast_spans_do_not_leak_bodies_into_type_headers() {
        let mut file = parser::rd::rd_parse(
            "extern class Box<T> { function new():Void; }",
            "Box.hx",
            false,
            false,
        )
        .unwrap();
        file.input = Some("extern class Box<T> {\n /** Stored value. */ public var value:T;\n public function new():Void;\n}".into());
        let index = WorkspaceIndex::default();
        index.ensure_parsed_file(&file);
        let entry = index.declaration("Box").unwrap();
        assert!(
            entry.signature.ends_with("class Box<T>"),
            "{}",
            entry.signature
        );
        assert!(!entry.signature.contains('{'), "{}", entry.signature);
        assert_eq!(
            index
                .declaration("Box.value")
                .unwrap()
                .documentation
                .as_deref(),
            Some("Stored value.")
        );
    }

    #[test]
    fn type_headers_preserve_structure_and_exclude_bodies() {
        let source = "package models;\n\
            typedef View<T> = {value:T, ?label:String};\n\
            typedef Callback = (value:Int) -> String;\n\
            interface Reader<T> extends Base<T> { function read():T; }\n\
            abstract Token({id:Int}) from {id:Int} to {id:Int} { public function id() return this.id; }\n\
            enum abstract Mode(Int) { var On = 1; }\n\
            enum Color { Red; }\n\
            class Box<T:{id:Int}> implements Reader<T> { public var value:T; }";
        let entries = declarations(Path::new("models/Types.hx"), source, &[]);
        for (name, kind, signature) in [
            (
                "View",
                SymbolKind::TypeAlias,
                "typedef View<T> = {value:T, ?label:String}",
            ),
            (
                "Callback",
                SymbolKind::TypeAlias,
                "typedef Callback = (value:Int) -> String",
            ),
            (
                "Reader",
                SymbolKind::Interface,
                "interface Reader<T> extends Base<T>",
            ),
            (
                "Token",
                SymbolKind::Abstract,
                "abstract Token({id:Int}) from {id:Int} to {id:Int}",
            ),
            ("Mode", SymbolKind::Abstract, "enum abstract Mode(Int)"),
            ("Color", SymbolKind::Enum, "enum Color"),
            (
                "Box",
                SymbolKind::Class,
                "class Box<T:{id:Int}> implements Reader<T>",
            ),
        ] {
            let entry = entries
                .iter()
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(entry.type_kind, Some(kind), "{name}");
            assert_eq!(entry.signature, signature, "{name}");
        }
    }
}
