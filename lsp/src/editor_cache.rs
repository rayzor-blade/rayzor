//! BLADE editor indexes are independent of compiler graph lifetimes.

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
    pub responses: BTreeMap<String, Value>,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct Snapshot {
    pub complete: bool,
    pub names: Vec<Name>,
    pub records: Vec<BTreeMap<String, Value>>,
    pub members: BTreeMap<String, usize>,
    pub member_results: Vec<Value>,
    pub document: BTreeMap<String, Value>,
    pub expansions: Vec<Value>,
    pub diagnostics: Vec<crate::context::LspDiagnostic>,
    pub dependencies: Vec<compiler::macro_system::dependencies::Dependency>,
}

impl Snapshot {
    pub fn answer(&self, request: &Request, source: &str) -> Option<Value> {
        let method = request.method.as_str();
        if let Some(value) = self.document.get(method) {
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
            if method != "textDocument/completion" {
                return Some(value.clone());
            }
        }
        let line = request.params["position"]["line"].as_u64()? as u32;
        let column = request.params["position"]["character"].as_u64()? as u32;
        if method == "textDocument/signatureHelp" {
            let call = crate::analysis::enclosing_call(source, line + 1, column + 1)?;
            let name = self.names.iter().find(|name| {
                name.line + 1 == call.callee_line && name.column + 1 == call.callee_col
            })?;
            let mut response = name.responses.get(method)?.clone();
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
                    .get(&format!("{}:{}", line - 1, column - 1))
                    .and_then(|&index| self.member_results.get(index))
                    .cloned()
            } else {
                self.document.get(method).cloned()
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
            return Some(value.clone());
        }
        if matches!(method, "textDocument/hover" | "rayzor/expandMacro") {
            let position = (line as u64, column as u64);
            let expansion = self
                .expansions
                .iter()
                .filter(|expansion| {
                    let range = &expansion["range"];
                    let start = (
                        range["start"]["line"].as_u64().unwrap_or(u64::MAX),
                        range["start"]["character"].as_u64().unwrap_or(u64::MAX),
                    );
                    let end = (
                        range["end"]["line"].as_u64().unwrap_or(0),
                        range["end"]["character"].as_u64().unwrap_or(0),
                    );
                    start <= position && position < end
                })
                .min_by_key(|expansion| {
                    (
                        expansion["range"]["end"]["line"]
                            .as_u64()
                            .unwrap_or(0)
                            .saturating_sub(
                                expansion["range"]["start"]["line"].as_u64().unwrap_or(0),
                            ),
                        expansion["expansion"].as_str().map_or(0, str::len),
                    )
                });
            if expansion.is_none() && !self.complete {
                return None;
            }
            if method == "rayzor/expandMacro" {
                return Some(expansion.cloned().unwrap_or(Value::Null));
            }
            return Some(expansion.map(|expansion| serde_json::json!({"contents":{"kind":"markdown","value":format!("**Macro expansion: `{}`**\n\n```haxe\n{}\n```", expansion["name"].as_str().unwrap_or("macro"), expansion["expansion"].as_str().unwrap_or(""))},"range":expansion["range"]})).unwrap_or(Value::Null));
        }
        None
    }
}

struct Config {
    key: SessionKey,
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
    config: Option<Config>,
    documents: HashMap<String, String>,
    signature: Option<(u64, Instant)>,
    retained: VecDeque<Retained>,
}

#[derive(Clone, Default)]
pub(crate) struct EditorCache(Arc<Mutex<State>>);

impl EditorCache {
    pub fn configure(&self, key: SessionKey, root: &std::path::Path) {
        let mut state = self.0.lock().unwrap();
        state.config = Some(Config {
            key,
            directory: root.join(".rayzor/blade/editor"),
            build_id: format!("{}:{}", compiler::BUILD_ID, env!("RAYZOR_LSP_BUILD_ID")),
        });
        state.signature = None;
    }

    pub fn track(&self, notification: &Notification) {
        let mut state = self.0.lock().unwrap();
        let params = &notification.params;
        let uri = params["textDocument"]["uri"].as_str();
        match notification.method.as_str() {
            "textDocument/didOpen" => {
                if let (Some(uri), Some(text)) = (uri, params["textDocument"]["text"].as_str()) {
                    state.documents.insert(uri.into(), text.into());
                    state.signature = None;
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
                    state.signature = None;
                }
            }
            "textDocument/didClose" => {
                if let Some(uri) = uri {
                    state.documents.remove(uri);
                    state.signature = None;
                }
            }
            "workspace/didChangeWatchedFiles" | "workspace/didChangeConfiguration" => {
                state.signature = None;
                state.retained.clear();
            }
            _ => {}
        }
    }

    pub fn lookup(&self, uri: &str) -> Option<Arc<Snapshot>> {
        let mut state = self.0.lock().unwrap();
        let (input_hash, path, build_id) = state.identity(uri)?;
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
        let bytes = compiler::ir::blade::load_editor_index(path, &build_id, input_hash).ok()??;
        let snapshot = Arc::new(serde_json::from_slice::<Snapshot>(&bytes).ok()?);
        if !snapshot
            .dependencies
            .iter()
            .all(|dependency| dependency.is_current())
        {
            return None;
        }
        state.retain(uri, input_hash, snapshot.clone(), bytes.len());
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
        let config = state.config.as_ref()?;
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
        let Ok(bytes) = serde_json::to_vec(&snapshot) else {
            return;
        };
        if bytes.len() > 8 << 20 {
            return;
        }
        let mut state = self.0.lock().unwrap();
        let Some(config) = state.config.as_ref() else {
            return;
        };
        let path = config.directory.join(format!("{:016x}.bidx", hash(&uri)));
        let build_id = config.build_id.clone();
        let len = bytes.len();
        let _ = compiler::ir::blade::save_editor_index(path, &build_id, input_hash, bytes);
        state.retain(uri, input_hash, Arc::new(snapshot), len);
    }

    pub fn publish_partial(&self, uri: &str, input_hash: u64, snapshot: Snapshot) {
        self.0
            .lock()
            .unwrap()
            .retain(uri, input_hash, Arc::new(snapshot), 8 << 20);
    }
}

impl State {
    fn identity(&mut self, uri: &str) -> Option<(u64, PathBuf, String)> {
        let config = self.config.as_ref()?;
        if self
            .signature
            .is_none_or(|(_, checked)| checked.elapsed() >= Duration::from_millis(500))
        {
            let inputs = Inputs::capture(&config.key, &self.documents);
            if !inputs.reusable {
                return None;
            }
            self.signature = Some((
                hash(&(compiler::BUILD_ID, &config.key, inputs)),
                Instant::now(),
            ));
        }
        let signature = hash(&(self.signature?.0, uri, self.documents.get(uri)));
        Some((
            signature,
            config.directory.join(format!("{:016x}.bidx", hash(&uri))),
            config.build_id.clone(),
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
        snapshot.dependencies = dependencies.snapshot();
        snapshot.names.push(Name {
            line: 0,
            column: 6,
            len: 4,
            metadata: None,
            responses: BTreeMap::from([(
                "textDocument/hover".into(),
                json!({"contents":"class Main"}),
            )]),
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
