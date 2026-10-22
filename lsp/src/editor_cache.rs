//! BLADE editor indexes are independent of compiler graph lifetimes.

use crate::binary_response::{BinaryResponse, Responses};
use crate::session_cache::{Inputs, SessionKey};
use lsp_server::{Notification, Request};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap, VecDeque, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

fn hash(value: &impl Hash) -> u64 {
    let mut hash = DefaultHasher::new();
    value.hash(&mut hash);
    hash.finish()
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Name {
    pub line: u32,
    pub column: u32,
    pub len: u32,
    pub metadata: Option<usize>,
    pub responses: Responses,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct Snapshot {
    pub complete: bool,
    pub names: Vec<Name>,
    pub records: Vec<Responses>,
    pub members: BTreeMap<(u32, u32), usize>,
    pub member_results: Vec<BinaryResponse>,
    pub document: Responses,
    pub expansions: Vec<Expansion>,
    pub diagnostics: Vec<crate::context::LspDiagnostic>,
    pub dependencies: Vec<compiler::macro_system::dependencies::Dependency>,
}

impl Snapshot {
    fn heap_bytes(&self) -> usize {
        fn nodes<K, V>(len: usize) -> usize {
            if len == 0 {
                return 0;
            }
            // Include spare B-tree slots and internal-node edges.
            ((len - 1).div_ceil(5) + 1) * (11 * std::mem::size_of::<(K, V)>() + 128)
        }
        std::mem::size_of::<Self>()
            + self.names.capacity() * std::mem::size_of::<Name>()
            + self
                .names
                .iter()
                .map(|name| name.responses.heap_bytes())
                .sum::<usize>()
            + self.records.capacity() * std::mem::size_of::<Responses>()
            + self
                .records
                .iter()
                .map(Responses::heap_bytes)
                .sum::<usize>()
            + nodes::<(u32, u32), usize>(self.members.len())
            + self.member_results.capacity() * std::mem::size_of::<BinaryResponse>()
            + self
                .member_results
                .iter()
                .map(BinaryResponse::heap_bytes)
                .sum::<usize>()
            + self.document.heap_bytes()
            + self.expansions.capacity() * std::mem::size_of::<Expansion>()
            + self
                .expansions
                .iter()
                .map(|expansion| expansion.response.heap_bytes())
                .sum::<usize>()
            + self.dependencies.capacity()
                * std::mem::size_of::<compiler::macro_system::dependencies::Dependency>()
            + self
                .dependencies
                .iter()
                .map(|dependency| dependency.path.capacity())
                .sum::<usize>()
            + self.diagnostics.capacity() * std::mem::size_of::<crate::context::LspDiagnostic>()
            + self
                .diagnostics
                .iter()
                .map(|diagnostic| {
                    diagnostic.file.capacity()
                        + diagnostic.message.capacity()
                        + diagnostic.suggestion.as_ref().map_or(0, String::capacity)
                        + diagnostic.related.capacity() * std::mem::size_of::<String>()
                        + diagnostic
                            .related
                            .iter()
                            .map(String::capacity)
                            .sum::<usize>()
                        + diagnostic.related_information.capacity()
                            * std::mem::size_of::<lsp_types::DiagnosticRelatedInformation>()
                        + diagnostic
                            .related_information
                            .iter()
                            .map(|related| {
                                related.message.capacity()
                                    + related.location.uri.as_str().len()
                                    + 128
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    pub fn answer(&self, request: &Request, source: &str) -> Option<Value> {
        let method = request.method.as_str();
        if method != "textDocument/completion"
            && let Some(value) = self.document.get(method)
        {
            if method == "textDocument/inlayHint" {
                let start = request.params["range"]["start"]["line"].as_u64()?;
                let end = request.params["range"]["end"]["line"].as_u64()?;
                return Some(Value::Array(
                    value
                        .as_array()?
                        .iter()
                        .filter(|hint| {
                            hint["position"]["line"]
                                .as_u64()
                                .is_some_and(|line| start <= line && line <= end)
                        })
                        .cloned()
                        .collect(),
                ));
            }
            return Some(value);
        }
        let line = request.params["position"]["line"].as_u64()? as u32;
        let column = request.params["position"]["character"].as_u64()? as u32;
        if method == "textDocument/signatureHelp" {
            let call = crate::analysis::enclosing_call(source, line + 1, column + 1)?;
            let name = self.names.iter().find(|name| {
                name.line + 1 == call.callee_line && name.column + 1 == call.callee_col
            })?;
            let mut response = name.responses.get(method)?;
            if response.is_object() {
                response["activeParameter"] = Value::from(call.argument);
            }
            return Some(response);
        }
        if method == "textDocument/completion" {
            return if let Some((line, column, _)) =
                crate::analysis::receiver_before(source, line + 1, column + 1)
            {
                self.members
                    .get(&(line - 1, column - 1))
                    .and_then(|&index| self.member_results.get(index))
                    .and_then(BinaryResponse::decode)
            } else {
                self.document.get(method)
            };
        }
        let name = self.names.iter().find(|name| {
            name.line == line
                && name.column <= column
                && column < name.column.saturating_add(name.len)
        });
        let key = if method == "textDocument/references" {
            format!(
                "{method}:{}",
                request.params["context"]["includeDeclaration"]
                    .as_bool()
                    .unwrap_or(false)
            )
        } else {
            method.to_string()
        };
        if let Some(value) = name.and_then(|name| {
            name.responses
                .get(&key)
                .or_else(|| self.records.get(name.metadata?)?.get(&key))
        }) {
            return Some(value);
        }
        if matches!(method, "textDocument/hover" | "rayzor/expandMacro") {
            let position = (line as u64, column as u64);
            let expansion = self
                .expansions
                .iter()
                .filter(|expansion| {
                    let range = expansion.range;
                    (range.start.line as u64, range.start.character as u64) <= position
                        && position < (range.end.line as u64, range.end.character as u64)
                })
                .min_by_key(|expansion| {
                    (
                        expansion
                            .range
                            .end
                            .line
                            .saturating_sub(expansion.range.start.line),
                        expansion.text_len,
                    )
                })
                .and_then(|expansion| expansion.response.decode());
            if expansion.is_none() && !self.complete {
                return None;
            }
            if method == "rayzor/expandMacro" {
                return Some(expansion.unwrap_or(Value::Null));
            }
            return Some(expansion.map(|expansion| serde_json::json!({"contents":{"kind":"markdown","value":format!("**Macro expansion: `{}`**\n\n```haxe\n{}\n```", expansion["name"].as_str().unwrap_or("macro"), expansion["expansion"].as_str().unwrap_or(""))},"range":expansion["range"]})).unwrap_or(Value::Null));
        }
        None
    }
}

// Payload schema is independent of the outer BLADE envelope.
const SCHEMA: &[u8; 4] = b"LSP2";

fn encode_snapshot(snapshot: &Snapshot) -> Result<Vec<u8>, postcard::Error> {
    let mut bytes = SCHEMA.to_vec();
    bytes.extend(postcard::to_allocvec(snapshot)?);
    Ok(bytes)
}

fn decode_snapshot(bytes: &[u8]) -> Option<Snapshot> {
    postcard::from_bytes(bytes.strip_prefix(SCHEMA)?).ok()
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Expansion {
    range: lsp_types::Range,
    text_len: usize,
    response: BinaryResponse,
}

impl Expansion {
    pub fn from_json(value: Value) -> Option<Self> {
        Some(Self {
            range: serde_json::from_value(value["range"].clone()).ok()?,
            text_len: value["expansion"].as_str().map_or(0, str::len),
            response: value.into(),
        })
    }
}

struct Config {
    key: SessionKey,
    root: PathBuf,
    directory: PathBuf,
    build_id: String,
}
struct Retained {
    uri: String,
    inputs: u64,
    snapshot: Arc<Snapshot>,
    bytes: usize,
}
#[derive(Default)]
struct State {
    configs: Vec<Config>,
    documents: HashMap<String, String>,
    signatures: HashMap<PathBuf, (u64, Instant)>,
    retained: VecDeque<Retained>,
}

#[derive(Clone, Default)]
pub(crate) struct EditorCache(Arc<Mutex<State>>);

impl EditorCache {
    pub fn reset_projects(&self) {
        let mut state = self.0.lock().unwrap();
        state.configs.clear();
        state.signatures.clear();
        state.retained.clear();
    }

    #[cfg(test)]
    pub fn configure(&self, mut key: SessionKey, root: &std::path::Path) {
        key.options.cache_enabled = true;
        self.configure_project(key, root);
    }

    pub fn configure_project(&self, key: SessionKey, root: &std::path::Path) {
        let mut state = self.0.lock().unwrap();
        let root = root.canonicalize().unwrap_or_else(|_| root.into());
        let directory = key
            .options
            .cache_dir
            .as_ref()
            .map(|path| path.join("editor"))
            .unwrap_or_else(|| root.join(".rayzor/blade/editor"));
        state.configs.retain(|config| config.root != root);
        state.configs.push(Config {
            key,
            root,
            directory,
            build_id: format!("{}:{}", compiler::BUILD_ID, env!("RAYZOR_LSP_BUILD_ID")),
        });
        state.signatures.clear();
    }

    pub fn track(&self, notification: &Notification) {
        let mut state = self.0.lock().unwrap();
        let params = &notification.params;
        let uri = params["textDocument"]["uri"].as_str();
        match notification.method.as_str() {
            "textDocument/didOpen" => {
                if let (Some(uri), Some(text)) = (uri, params["textDocument"]["text"].as_str()) {
                    state.documents.insert(uri.into(), text.into());
                    state.signatures.clear();
                }
            }
            "textDocument/didChange" => {
                if let (Some(uri), Some(text)) = (
                    uri,
                    params["contentChanges"]
                        .as_array()
                        .and_then(|changes| changes.last())
                        .and_then(|change| change["text"].as_str()),
                ) {
                    state.documents.insert(uri.into(), text.into());
                    state.signatures.clear();
                }
            }
            "textDocument/didClose" => {
                if let Some(uri) = uri {
                    state.documents.remove(uri);
                    state.signatures.clear();
                }
            }
            "workspace/didChangeWatchedFiles" => {
                state.signatures.clear();
            }
            "textDocument/didSave" => {
                state.signatures.clear();
                if uri.is_some_and(|uri| !crate::context::uri_to_path(uri).ends_with(".hx")) {
                    state.retained.clear();
                    state.configs.clear();
                }
            }
            "workspace/didChangeWorkspaceFolders" | "workspace/didChangeConfiguration" => {
                state.signatures.clear();
                state.retained.clear();
                state.configs.clear();
            }
            _ => {}
        }
    }

    pub fn lookup(&self, uri: &str) -> Option<Arc<Snapshot>> {
        let mut state = self.0.lock().unwrap();
        let (input_hash, path, build_id, persist) = state.identity(uri)?;
        if let Some(index) = state
            .retained
            .iter()
            .position(|entry| entry.uri == uri && entry.inputs == input_hash)
        {
            let entry = state.retained.remove(index)?;
            let result = entry.snapshot.clone();
            if !result
                .dependencies
                .iter()
                .all(|dependency| dependency.is_current())
            {
                return None;
            }
            state.retained.push_back(entry);
            return Some(result);
        }
        if !persist {
            return None;
        }
        let bytes = compiler::ir::blade::load_editor_index(path, &build_id, input_hash).ok()??;
        let snapshot = Arc::new(decode_snapshot(&bytes)?);
        if !snapshot
            .dependencies
            .iter()
            .all(|dependency| dependency.is_current())
        {
            return None;
        }
        let heap_bytes = snapshot.heap_bytes();
        state.retain(uri, input_hash, snapshot.clone(), heap_bytes);
        Some(snapshot)
    }

    pub fn answer(&self, request: &Request) -> Option<Value> {
        let uri = request.params["textDocument"]["uri"].as_str()?;
        let snapshot = self.lookup(uri)?;
        let state = self.0.lock().unwrap();
        snapshot.answer(request, state.documents.get(uri)?)
    }

    pub fn fingerprint(&self, uri: &str, documents: &HashMap<String, String>) -> Option<u64> {
        let state = self.0.lock().unwrap();
        let config = state.config_for(uri)?;
        let inputs = Inputs::capture(&config.key, documents);
        inputs.reusable.then(|| {
            hash(&(
                hash(&(compiler::BUILD_ID, &config.key, inputs)),
                uri,
                documents.get(uri),
            ))
        })
    }

    pub fn save(&self, uri: &str, input_hash: u64, snapshot: Snapshot) {
        if !snapshot.complete {
            return;
        }
        let Ok(bytes) = encode_snapshot(&snapshot) else {
            return;
        };
        if bytes.len() > 8 << 20 {
            return;
        }
        let mut state = self.0.lock().unwrap();
        let Some(config) = state.config_for(uri) else {
            return;
        };
        let path = config.directory.join(format!("{:016x}.bidx", hash(&uri)));
        let build_id = config.build_id.clone();
        let heap_bytes = snapshot.heap_bytes();
        if config.key.options.cache_enabled {
            let _ = compiler::ir::blade::save_editor_index(path, &build_id, input_hash, bytes);
        }
        state.retain(uri, input_hash, Arc::new(snapshot), heap_bytes);
    }

    pub fn publish_partial(&self, uri: &str, input_hash: u64, snapshot: Snapshot) {
        let heap_bytes = snapshot.heap_bytes();
        self.0
            .lock()
            .unwrap()
            .retain(uri, input_hash, Arc::new(snapshot), heap_bytes);
    }
}

impl State {
    fn config_for(&self, uri: &str) -> Option<&Config> {
        let path = PathBuf::from(crate::context::uri_to_path(uri));
        let path = path.canonicalize().unwrap_or(path);
        self.configs
            .iter()
            .filter_map(|config| {
                crate::build_config::source_depth(
                    &path,
                    &config.root,
                    config
                        .key
                        .class_paths
                        .iter()
                        .chain(&config.key.stdlib_paths),
                )
                .map(|depth| (depth, config))
            })
            .max_by_key(|(depth, _)| *depth)
            .map(|(_, config)| config)
    }

    fn identity(&mut self, uri: &str) -> Option<(u64, PathBuf, String, bool)> {
        let root = self.config_for(uri)?.root.clone();
        if self
            .signatures
            .get(&root)
            .is_none_or(|(_, checked)| checked.elapsed() >= Duration::from_millis(500))
        {
            let inputs = Inputs::capture(&self.config_for(uri)?.key, &self.documents);
            if !inputs.reusable {
                return None;
            }
            let signature = hash(&(compiler::BUILD_ID, &self.config_for(uri)?.key, inputs));
            self.signatures
                .insert(root.clone(), (signature, Instant::now()));
        }
        let config = self.config_for(uri)?;
        let signature = hash(&(self.signatures.get(&root)?.0, uri, self.documents.get(uri)));
        Some((
            signature,
            config.directory.join(format!("{:016x}.bidx", hash(&uri))),
            config.build_id.clone(),
            config.key.options.cache_enabled,
        ))
    }
    fn retain(&mut self, uri: &str, inputs: u64, snapshot: Arc<Snapshot>, bytes: usize) {
        self.retained.retain(|entry| entry.uri != uri);
        self.retained.push_back(Retained {
            uri: uri.into(),
            inputs,
            snapshot,
            bytes,
        });
        while self.retained.len() > 32
            || self.retained.iter().map(|entry| entry.bytes).sum::<usize>() > 64 << 20
        {
            self.retained.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn binary_snapshot_preserves_diagnostics_and_expansions() {
        let uri = "file:///workspace/Main.hx";
        let expansion = json!({"name":"Build.run","expansion":"new Div()",
            "range":{"start":{"line":2,"character":4},"end":{"line":2,"character":10}},
            "definition":{"uri":uri,"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}}}});
        let mut diagnostic =
            crate::context::LspDiagnostic::error("Main.hx", 2, 4, "example".into());
        diagnostic
            .related_information
            .push(lsp_types::DiagnosticRelatedInformation {
                location: lsp_types::Location {
                    uri: uri.parse().unwrap(),
                    range: lsp_types::Range::default(),
                },
                message: "macro declaration".into(),
            });
        let snapshot = Snapshot {
            complete: true,
            diagnostics: vec![diagnostic],
            expansions: vec![Expansion::from_json(expansion.clone()).unwrap()],
            ..Default::default()
        };
        let bytes = encode_snapshot(&snapshot).unwrap();
        assert!(bytes.starts_with(SCHEMA));
        assert!(decode_snapshot(br#"{"complete":true}"#).is_none());
        assert!(decode_snapshot(&bytes[..bytes.len() / 2]).is_none());
        let restored = decode_snapshot(&bytes).unwrap();
        assert_eq!(
            restored.diagnostics[0].related_information[0].message,
            "macro declaration"
        );
        let request = Request::new(
            1.into(),
            "rayzor/expandMacro".into(),
            json!({"textDocument":{"uri":uri},"position":{"line":2,"character":6}}),
        );
        assert_eq!(restored.answer(&request, "").unwrap(), expansion);
    }

    #[test]
    fn encoded_response_budget_evicts_to_blade() {
        let root =
            std::env::temp_dir().join(format!("rayzor-editor-memory-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        for index in 0..10 {
            std::fs::write(
                root.join(format!("Page{index}.hx")),
                format!("class Page{index} {{}}"),
            )
            .unwrap();
        }
        let cache = EditorCache::default();
        cache.configure(
            SessionKey {
                class_paths: vec![root.clone()],
                stdlib_paths: vec![],
                defines: vec![],
                macros: vec![],
                macro_budget: Duration::from_secs(5),
                options: Default::default(),
            },
            &root,
        );
        let mut first = String::new();
        for index in 0..10 {
            let path = root.join(format!("Page{index}.hx"));
            let uri = crate::context::file_uri(path.to_str().unwrap());
            let source = std::fs::read_to_string(path).unwrap();
            cache.track(&Notification::new(
                "textDocument/didOpen".into(),
                json!({"textDocument":{"uri":uri,"text":source}}),
            ));
            let fingerprint = cache
                .fingerprint(&uri, &HashMap::from([(uri.clone(), source)]))
                .unwrap();
            let mut snapshot = Snapshot {
                complete: true,
                ..Default::default()
            };
            snapshot.document.insert(
                "textDocument/semanticTokens/full",
                json!({"data":"x".repeat(7 << 20)}),
            );
            let binary_bytes = encode_snapshot(&snapshot).unwrap().len();
            assert!(snapshot.heap_bytes() >= 7 << 20);
            assert!(snapshot.heap_bytes() < binary_bytes + 4096);
            cache.save(&uri, fingerprint, snapshot);
            if index == 0 {
                first = uri;
            }
        }
        {
            let state = cache.0.lock().unwrap();
            assert!(state.retained.len() < 10);
            assert!(state.retained.iter().all(|entry| entry.uri != first));
            assert!(
                state
                    .retained
                    .iter()
                    .map(|entry| entry.bytes)
                    .sum::<usize>()
                    <= 64 << 20
            );
        }
        assert!(
            cache.lookup(&first).is_some(),
            "evicted responses must load from BLADE"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn external_dependency_and_stdlib_pages_restore_from_the_owning_project_cache() {
        let root =
            std::env::temp_dir().join(format!("rayzor-external-editor-{}", std::process::id()));
        for directory in ["app", "dependency", "std", "unrelated"] {
            std::fs::create_dir_all(root.join(directory)).unwrap();
        }
        for (file, name) in [
            ("dependency/Helper.hx", "Helper"),
            ("std/Library.hx", "Library"),
        ] {
            std::fs::write(root.join(file), format!("class {name} {{}}")).unwrap();
        }
        let key = SessionKey {
            class_paths: vec![root.join("app"), root.join("dependency")],
            stdlib_paths: vec![root.join("std")],
            defines: vec![],
            macros: vec![],
            macro_budget: Duration::from_secs(5),
            options: Default::default(),
        };
        let cache = EditorCache::default();
        cache.configure(key.clone(), &root.join("app"));
        let mut requests = Vec::new();
        for (file, name) in [
            ("dependency/Helper.hx", "Helper"),
            ("std/Library.hx", "Library"),
        ] {
            let source = format!("class {name} {{}}");
            let uri = crate::context::file_uri(root.join(file).to_str().unwrap());
            let open = Notification::new(
                "textDocument/didOpen".into(),
                json!({"textDocument":{"uri":uri,"text":source}}),
            );
            cache.track(&open);
            let inputs = cache
                .fingerprint(&uri, &HashMap::from([(uri.clone(), source)]))
                .unwrap();
            let snapshot = Snapshot {
                complete: true,
                document: Responses::from([("textDocument/hover", json!({"contents":name}))]),
                ..Default::default()
            };
            cache.save(&uri, inputs, snapshot);
            requests.push((
                open,
                Request::new(
                    1.into(),
                    "textDocument/hover".into(),
                    json!({"textDocument":{"uri":uri},"position":{"line":0,"character":7}}),
                ),
                name,
            ));
        }
        let restored = EditorCache::default();
        restored.configure(key, &root.join("app"));
        for (open, request, expected) in requests {
            restored.track(&open);
            assert_eq!(restored.answer(&request).unwrap()["contents"], expected);
        }
        let unrelated = crate::context::file_uri(root.join("unrelated/Other.hx").to_str().unwrap());
        assert!(restored.lookup(&unrelated).is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn blade_indexes_survive_restart_and_reject_changed_inputs() {
        let root = std::env::temp_dir().join(format!("rayzor-editor-blade-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let source = "class Main {}";
        std::fs::write(root.join("Main.hx"), source).unwrap();
        std::fs::write(root.join("Dependency.hx"), "class Dependency {}").unwrap();
        let uri = crate::context::file_uri(root.join("Main.hx").to_str().unwrap());
        let key = SessionKey {
            class_paths: vec![root.clone()],
            stdlib_paths: vec![],
            defines: vec![],
            macros: vec![],
            macro_budget: Duration::from_secs(5),
            options: Default::default(),
        };
        let open = Notification::new(
            "textDocument/didOpen".into(),
            json!({"textDocument":{"uri":uri,"text":source}}),
        );
        let cache = EditorCache::default();
        cache.configure(key.clone(), &root);
        cache.track(&open);
        let inputs = cache
            .fingerprint(&uri, &HashMap::from([(uri.clone(), source.into())]))
            .unwrap();
        let mut snapshot = Snapshot {
            complete: true,
            ..Default::default()
        };
        let asset = root.with_extension("asset");
        std::fs::write(&asset, "original asset").unwrap();
        let asset_time = std::fs::metadata(&asset).unwrap().modified().unwrap();
        let dependencies = compiler::macro_system::dependencies::Dependencies::default();
        dependencies.register("Main".into(), asset.clone());
        dependencies.register_read(root.join("Dependency.hx"));
        snapshot.dependencies = dependencies.snapshot();
        snapshot.names.push(Name {
            line: 0,
            column: 6,
            len: 4,
            metadata: None,
            responses: Responses::from([("textDocument/hover", json!({"contents":"class Main"}))]),
        });
        cache.save(&uri, inputs, snapshot);
        let restarted = EditorCache::default();
        restarted.configure(key.clone(), &root);
        restarted.track(&open);
        let request = Request::new(
            1.into(),
            "textDocument/hover".into(),
            json!({"textDocument":{"uri":uri},"position":{"line":0,"character":7}}),
        );
        assert_eq!(
            restarted.answer(&request).unwrap()["contents"],
            "class Main"
        );
        std::fs::write(&asset, "changed external asset").unwrap();
        assert!(restarted.answer(&request).is_none());
        let stale = EditorCache::default();
        stale.configure(key.clone(), &root);
        stale.track(&open);
        assert!(stale.answer(&request).is_none());
        std::fs::write(&asset, "original asset").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&asset)
            .unwrap()
            .set_modified(asset_time)
            .unwrap();
        assert!(restarted.answer(&request).is_some());
        restarted.track(&Notification::new(
            "textDocument/didChange".into(),
            json!({"textDocument":{"uri":uri},"contentChanges":[{"text":"class Other {}"}]}),
        ));
        assert!(restarted.answer(&request).is_none());
        std::fs::write(
            root.join("Dependency.hx"),
            "class Dependency { var changed:Int; }",
        )
        .unwrap();
        let changed = EditorCache::default();
        changed.configure(key, &root);
        changed.track(&open);
        assert!(changed.answer(&request).is_none());
        std::fs::remove_file(asset).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
