//! Drives a real language-server session in-process and checks its answers
//! against exact source positions.

use lsp_server::{Connection, Message, Notification, Request, RequestId};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Session {
    client: Connection,
    server: Option<std::thread::JoinHandle<Result<(), String>>>,
    root: PathBuf,
    next_id: i32,
    diagnostics: Vec<Value>,
    logs: Vec<String>,
    activity: Vec<Value>,
    temporary: bool,
    timeout: std::time::Duration,
}

impl Session {
    /// Writes `files` into a fresh workspace and initialises a server on it.
    fn start(files: &[(&str, &str)]) -> Self {
        Self::start_with(files, json!({}))
    }

    fn start_with(files: &[(&str, &str)], options: Value) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "rayzor-lsp-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        for (name, text) in files {
            let path = root.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        Self::connect(root, options, true)
    }

    fn connect(root: PathBuf, options: Value, temporary: bool) -> Self {
        Self::connect_folders(root, options, temporary, None)
    }

    fn connect_folders(
        root: PathBuf,
        options: Value,
        temporary: bool,
        folders: Option<Vec<PathBuf>>,
    ) -> Self {
        let (server_end, client) = Connection::memory();
        let server = rayzor_lsp::spawn_session(server_end).unwrap();
        let mut session = Session {
            client,
            server: Some(server),
            root,
            next_id: 0,
            diagnostics: Vec::new(),
            logs: Vec::new(),
            activity: Vec::new(),
            temporary,
            timeout: std::time::Duration::from_secs(120),
        };
        let root_uri = session.uri("");
        let mut params = json!({"processId": null, "rootUri": root_uri, "capabilities": {}, "initializationOptions": options});
        if let Some(folders) = folders {
            params["workspaceFolders"] = json!(folders.iter().map(|path| json!({"uri":format!("file://{}", path.display()),"name":path.file_name().unwrap().to_string_lossy()})).collect::<Vec<_>>());
        }
        session.request("initialize", params);
        session.notify("initialized", json!({}));
        session
    }

    fn uri(&self, name: &str) -> String {
        format!("file://{}/{}", self.root.display(), name)
            .trim_end_matches('/')
            .to_string()
    }

    fn notify(&self, method: &str, params: Value) {
        let n = Notification::new(method.to_string(), params);
        self.client.sender.send(Message::Notification(n)).unwrap();
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = RequestId::from(self.next_id);
        let r = Request::new(id.clone(), method.to_string(), params);
        self.client.sender.send(Message::Request(r)).unwrap();
        let deadline = std::time::Instant::now() + self.timeout;
        loop {
            match self
                .client
                .receiver
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                .expect("LSP response timeout")
            {
                Message::Response(resp) if resp.id == id => {
                    return resp.result.unwrap_or(Value::Null);
                }
                Message::Notification(n) if n.method == "window/logMessage" => {
                    self.logs
                        .push(n.params["message"].as_str().unwrap_or("").into());
                }
                Message::Notification(n) if n.method == "textDocument/publishDiagnostics" => {
                    self.diagnostics.push(n.params);
                }
                Message::Notification(n) if n.method == "rayzor/analysisStatus" => {
                    self.activity.push(n.params);
                }
                _ => {}
            }
        }
    }

    /// Opens `name` with its on-disk text and returns its diagnostics.
    fn open(&mut self, name: &str) -> Vec<Value> {
        let text = std::fs::read_to_string(self.root.join(name)).unwrap();
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {"uri": self.uri(name), "languageId": "haxe", "version": 1, "text": text}}),
        );
        self.wait_ready(name, 1)
    }

    /// Replaces `name`'s text in the editor (not on disk) and returns its
    /// diagnostics.
    fn change(&mut self, name: &str, text: &str) -> Vec<Value> {
        self.next_id += 1;
        self.notify(
            "textDocument/didChange",
            json!({"textDocument": {"uri": self.uri(name), "version": self.next_id}, "contentChanges": [{"text": text}]}),
        );
        self.wait_ready(name, self.next_id)
    }

    // Ordinary assertions use completed analysis; partial-index tests send
    // notifications directly and query while the worker is still running.
    fn wait_ready(&mut self, name: &str, version: i32) -> Vec<Value> {
        let mut diagnostics = Vec::new();
        loop {
            if let Message::Notification(notification) = self
                .client
                .receiver
                .recv_timeout(self.timeout)
                .expect("analysis timeout")
            {
                match notification.method.as_str() {
                    "window/logMessage" => self
                        .logs
                        .push(notification.params["message"].as_str().unwrap_or("").into()),
                    "textDocument/publishDiagnostics" => {
                        if notification.params["uri"] == self.uri(name) {
                            diagnostics = notification.params["diagnostics"]
                                .as_array()
                                .cloned()
                                .unwrap_or_default();
                        }
                        self.diagnostics.push(notification.params);
                    }
                    "rayzor/analysisStatus" => {
                        let ready = notification.params["uri"] == self.uri(name)
                            && notification.params["version"] == version
                            && notification.params["state"] == "ready";
                        self.activity.push(notification.params);
                        if ready {
                            return diagnostics;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn labels(&mut self, name: &str, line: u32, character: u32) -> Vec<String> {
        let at = self.at(name, line, character);
        let items = self.request("textDocument/completion", at);
        items
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["label"].as_str().unwrap().to_string())
            .collect()
    }

    fn at(&self, name: &str, line: u32, character: u32) -> Value {
        json!({"textDocument": {"uri": self.uri(name)}, "position": {"line": line, "character": character}})
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.next_id += 1;
        let id = RequestId::from(self.next_id);
        let _ = self.client.sender.send(Message::Request(Request::new(
            id,
            "shutdown".into(),
            Value::Null,
        )));
        // Complete shutdown before sending exit; the server batches queued messages.
        while let Ok(message) = self
            .client
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
        {
            if matches!(message, Message::Response(_)) {
                break;
            }
        }
        let _ = self
            .client
            .sender
            .send(Message::Notification(Notification::new(
                "exit".into(),
                Value::Null,
            )));
        if let Some(server) = self.server.take() {
            let _ = server.join();
        }
        if self.temporary {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

/// The (line, character) a location or range starts at.
fn start(v: &Value) -> (u64, u64) {
    let s = &v["range"]["start"];
    (
        s["line"].as_u64().unwrap(),
        s["character"].as_u64().unwrap(),
    )
}

fn hover_text(v: &Value) -> String {
    v["contents"]["value"].as_str().unwrap_or("").to_string()
}

const MAIN: &str = "class Main {
    static function add(a:Int, b:Int):Int {
        return a + b;
    }
    static function main() {
        var total = add(1, 2);
        trace(total);
    }
}
";

#[test]
fn a_burst_of_code_tabs_eventually_completes_every_page() {
    let files: Vec<_> = (0..65)
        .map(|index| {
            (
                format!("Page{index}.hx"),
                format!("class Page{index} {{ public var value:Int = {index}; }}"),
            )
        })
        .collect();
    let borrowed: Vec<_> = files
        .iter()
        .map(|(name, text)| (name.as_str(), text.as_str()))
        .collect();
    let mut s = Session::start(&borrowed);
    for (name, text) in &files {
        s.notify(
            "textDocument/didOpen",
            json!({"textDocument":{
                "uri":s.uri(name),"languageId":"haxe","version":1,"text":text
            }}),
        );
    }
    let mut ready = std::collections::HashSet::new();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while ready.len() < files.len() {
        if let Message::Notification(note) = s
            .client
            .receiver
            .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
            .expect("queued page did not finish")
        {
            if note.method == "textDocument/publishDiagnostics" {
                assert_eq!(note.params["diagnostics"], json!([]), "{:?}", note.params);
            }
            if note.method == "rayzor/analysisStatus" && note.params["state"] == "ready" {
                ready.insert(note.params["uri"].as_str().unwrap().to_string());
            }
        }
    }
    for (index, (name, _)) in files.iter().enumerate() {
        let hover = hover_text(&s.request("textDocument/hover", s.at(name, 0, 7)));
        assert!(hover.contains(&format!("class Page{index}")), "{hover}");
    }
}

#[test]
fn a_fresh_page_finishes_while_another_pages_macro_is_running() {
    let mut s = busy_page_session();
    start_busy_page(&s);
    s.notify(
        "textDocument/didOpen",
        json!({"textDocument":{
            "uri":s.uri("Fast.hx"),"languageId":"haxe","version":1,
            "text":std::fs::read_to_string(s.root.join("Fast.hx")).unwrap()
        }}),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let message = s
            .client
            .receiver
            .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
            .unwrap();
        if let Message::Notification(note) = message {
            if note.method == "textDocument/publishDiagnostics"
                && note.params["uri"] == s.uri("Fast.hx")
            {
                assert_eq!(note.params["diagnostics"], json!([]), "{:?}", note.params);
            }
            if note.method == "rayzor/analysisStatus" && note.params["state"] == "ready" {
                assert_eq!(
                    note.params["uri"],
                    s.uri("Fast.hx"),
                    "the slow page monopolized indexing"
                );
                break;
            }
        }
    }
    let source = std::fs::read_to_string(s.root.join("Fast.hx")).unwrap();
    let at = s.at("Fast.hx", 0, source.find("answer").unwrap() as u32 + 1);
    assert!(!s.request("textDocument/hover", at).is_null());
}

#[test]
fn slow_imports_report_their_module_and_yield_to_another_page() {
    let main = "/** Available while imports are being typed. */\nclass Main { var dependency:Dependency; }";
    let mut s = Session::start(&[
        ("Main.hx", main),
        ("Dependency.hx", "@:build(Gate.build()) class Dependency {}"),
        (
            "Fast.hx",
            "class Fast { public static var answer:Int = 42; }",
        ),
        ("gate.txt", "closed"),
        (".rayzor/entered", ""),
        (
            "Gate.hx",
            r#"class Gate {
 public static macro function build():Array<haxe.macro.Expr.Field> {
  var gate = haxe.macro.Context.resolvePath("gate.txt");
  sys.io.File.saveContent(haxe.macro.Context.resolvePath(".rayzor/entered"), "waiting");
  while (sys.io.File.getContent(gate) != "open") {}
  return haxe.macro.Context.getBuildFields();
 }
}"#,
        ),
    ]);
    struct Release(PathBuf);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.0, "open");
        }
    }
    let release = Release(s.root.join("gate.txt"));
    s.timeout = std::time::Duration::from_secs(10);
    s.notify("textDocument/didOpen", json!({"textDocument":{"uri":s.uri("Main.hx"),"languageId":"haxe","version":1,"text":main}}));
    let deadline = std::time::Instant::now() + s.timeout;
    let mut progress = false;
    while !progress || std::fs::read_to_string(s.root.join(".rayzor/entered")).unwrap() != "waiting"
    {
        assert!(
            std::time::Instant::now() < deadline,
            "import macro never started"
        );
        if let Ok(Message::Notification(note)) = s
            .client
            .receiver
            .recv_timeout(std::time::Duration::from_millis(5))
        {
            if note.method == "rayzor/indexProgress"
                && note.params["phase"] == "import-types"
                && note.params["subject"] == "Dependency"
            {
                assert_eq!(note.params["uri"], s.uri("Main.hx"));
                assert_eq!(note.params["version"], 1);
                progress = true;
            }
            assert!(!(note.method == "rayzor/analysisStatus" && note.params["state"] == "ready"));
        }
    }
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 1, 7)));
    assert!(hover.contains("Available while imports"), "{hover}");
    assert!(s.open("Fast.hx").is_empty(), "{:?}", s.diagnostics);
    assert!(
        !s.activity
            .iter()
            .any(|note| note["uri"] == s.uri("Main.hx") && note["state"] == "ready")
    );
    drop(release);
    assert!(s.wait_ready("Main.hx", 1).is_empty(), "{:?}", s.diagnostics);
}

#[test]
fn parameters_and_inferred_locals_answer_before_import_macros_finish() {
    let source = "class Main {\n function run(input:String) {\n  var count = 7;\n  var copy = input;\n  var items = Factory.wrap(count);\n  for (item in items) trace(item);\n  { var count = 'shadow'; trace(count); }\n  trace(count);\n  var callback = (value:Int) -> { trace(copy); return value + 1; };\n  var computed = callback(1);\n }\n}";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "Factory.hx",
            "@:build(Gate.build()) class Factory { public static function wrap<T>(value:T):Array<T> return [value]; }",
        ),
        ("gate.txt", "closed"),
        (".rayzor/entered", ""),
        (
            "Gate.hx",
            r#"class Gate {
 public static macro function build():Array<haxe.macro.Expr.Field> {
  var gate = haxe.macro.Context.resolvePath("gate.txt");
  sys.io.File.saveContent(haxe.macro.Context.resolvePath(".rayzor/entered"), "waiting");
  while (sys.io.File.getContent(gate) != "open") {}
  return haxe.macro.Context.getBuildFields();
 }
}"#,
        ),
    ]);
    struct Release(PathBuf);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.0, "open");
        }
    }
    let release = Release(s.root.join("gate.txt"));
    s.timeout = std::time::Duration::from_secs(10);
    s.notify("textDocument/didOpen", json!({"textDocument":{"uri":s.uri("Main.hx"),"languageId":"haxe","version":1,"text":source}}));
    let deadline = std::time::Instant::now() + s.timeout;
    while std::fs::read_to_string(s.root.join(".rayzor/entered")).unwrap() != "waiting" {
        assert!(
            std::time::Instant::now() < deadline,
            "full import macro did not start after preview"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    for (line, needle, expected) in [
        (1, "input", ":String"),
        (2, "count", ":Int"),
        (3, "copy", ":String"),
        (4, "items", ":Array<Int>"),
        (5, "item in", ":Int"),
        (6, "trace(count)", ":String"),
        (7, "count", ":Int"),
        (8, "callback", "-> Int"),
        (8, "value", ":Int"),
        (9, "computed", ":Int"),
    ] {
        let text = source.lines().nth(line).unwrap();
        let column = text.find(needle).unwrap() + if needle.starts_with("trace(") { 6 } else { 0 };
        let hover = hover_text(&s.request(
            "textDocument/hover",
            s.at("Main.hx", line as u32, column as u32 + 1),
        ));
        assert!(hover.contains(expected), "{needle}: {hover}");
    }
    assert!(
        !s.activity
            .iter()
            .any(|note| note["uri"] == s.uri("Main.hx") && note["state"] == "ready")
    );
    assert!(
        !s.diagnostics
            .iter()
            .any(|note| note["uri"] == s.uri("Main.hx"))
    );
    let changed = source
        .replace("input:String", "input:Bool")
        .replace("count = 7", "count = true");
    s.notify("textDocument/didChange", json!({"textDocument":{"uri":s.uri("Main.hx"),"version":2},"contentChanges":[{"text":changed}]}));
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 2, 7)));
    assert!(hover.contains("count:Bool"), "{hover}");
    drop(release);
    assert!(s.wait_ready("Main.hx", 2).is_empty(), "{:?}", s.diagnostics);
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 2, 7)));
    assert!(
        hover.contains("count:Bool") && !hover.contains("Provisional"),
        "{hover}"
    );
}

#[test]
#[ignore = "requires the Pocket Studio demo workspace"]
fn pocket_studio_task_inspector_has_early_local_types() {
    let root = std::env::var_os("RAYZOR_POCKET_STUDIO")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap())
                .join(".rayde-session/demos/pocket-studio")
        });
    let file = "src/ui/TaskInspector.hx";
    let source = std::fs::read_to_string(root.join(file)).unwrap();
    let mut s = Session::connect(root, json!({"macroBudgetMs":0}), false);
    s.timeout = std::time::Duration::from_secs(10);
    s.notify(
        "textDocument/didOpen",
        json!({"textDocument":{"uri":s.uri(file),"languageId":"haxe","version":1,"text":source}}),
    );
    for (needle, expected) in [
        ("description =", "Signal<String>"),
        ("task=Computed", "Computed<Null<model.Task>>"),
        ("Component;", "class Component"),
        ("Signal;", "abstract Signal"),
        ("Computed;", "abstract Computed"),
        ("Computed.make", "abstract Computed"),
        ("Signal.make", "abstract Signal"),
        ("BoardStore;", "class BoardStore"),
        ("current()", "current():Null<model.Task>"),
    ] {
        let offset = source.find(needle).unwrap();
        let before = &source[..offset];
        let line = before.bytes().filter(|&byte| byte == b'\n').count() as u32;
        let column = before.rsplit('\n').next().unwrap().encode_utf16().count() as u32 + 1;
        let hover = hover_text(&s.request("textDocument/hover", s.at(file, line, column)));
        assert!(hover.contains(expected), "{needle}: {hover}");
    }
    assert!(
        !s.activity
            .iter()
            .any(|note| note["uri"] == s.uri(file) && note["state"] == "ready")
    );
}

#[test]
fn standard_library_type_names_have_hovers_in_packaged_modules() {
    let source = "package demo;\nclass Main {\n var names:Array<String> = [];\n function make(value:String):Array<String> return new Array<String>();\n}";
    let mut s = Session::start(&[("demo/Main.hx", source)]);
    assert!(s.open("demo/Main.hx").is_empty(), "{:?}", s.diagnostics);
    for name in ["Array", "String"] {
        for (offset, _) in source.match_indices(name) {
            let before = &source[..offset];
            let line = before.bytes().filter(|&byte| byte == b'\n').count() as u32;
            let column = before.rsplit('\n').next().unwrap().encode_utf16().count() as u32;
            let hover =
                hover_text(&s.request("textDocument/hover", s.at("demo/Main.hx", line, column)));
            assert!(
                hover.contains(name) && hover.contains("```haxe"),
                "{name} at {line}:{column}: {hover}"
            );
            if name == "String" {
                assert!(hover.contains("The basic String class."), "{hover}");
            }
        }
    }
}

#[test]
fn referenced_api_types_and_members_hover_before_indexing_finishes() {
    let source = "package demo;\nimport api.Box;\nimport api.Model.Record;\nimport api.Model.View;\nimport api.Model.Scale;\nimport api.Model.Size;\nimport Slow;\nclass Main {\n var names:Array<String> = [];\n var value:Record;\n function render(input:View):String {\n  var box = new Box<String>('value');\n  var text = box.get();\n  box.inferred();\n  var selected:Record = box.record();\n  var scale:Scale = Scale.Small;\n  return api.Tools.describe(selected, Size.Md);\n }\n}";
    let mut s = Session::start(&[
        ("demo/Main.hx", source),
        (
            "api/Box.hx",
            "package api;\n/** Stores a typed value. */\nclass Box<T> {\n /** Creates a typed box. */ public function new(value:T) {}\n /** Reads the stored value. */ public function get():T return cast null;\n /** Inferred result. */ public function inferred() return 1;\n /** Returns a record. */ public function record():Model.Record return {name:'value'};\n}",
        ),
        (
            "api/Model.hx",
            "package api;\nclass Model {}\n/** Record details. */ typedef Record = {name:String};\n/** View contract. */ interface View {}\n/** Scale choices. */ enum abstract Scale(Int) from Int to Int { var Small = 1; }\n/** Size choices. */ enum Size { Md; Lg; }",
        ),
        (
            "api/Tools.hx",
            "package api;\nclass Tools { /** Describes a record. */ public static function describe(value:Model.Record, size:Model.Size):String return value.name; }",
        ),
        ("Slow.hx", "@:build(Gate.build()) class Slow {}"),
        ("gate.txt", "closed"),
        (".rayzor/entered", ""),
        (
            "Gate.hx",
            r#"class Gate {
 public static macro function build():Array<haxe.macro.Expr.Field> {
  var gate = haxe.macro.Context.resolvePath("gate.txt");
  sys.io.File.saveContent(haxe.macro.Context.resolvePath(".rayzor/entered"), "waiting");
  while (sys.io.File.getContent(gate) != "open") {}
  return haxe.macro.Context.getBuildFields();
 }
}"#,
        ),
    ]);
    struct Release(PathBuf);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.0, "open");
        }
    }
    let release = Release(s.root.join("gate.txt"));
    s.timeout = std::time::Duration::from_secs(10);
    s.notify("textDocument/didOpen", json!({"textDocument":{"uri":s.uri("demo/Main.hx"),"languageId":"haxe","version":1,"text":source}}));
    let deadline = std::time::Instant::now() + s.timeout;
    while std::fs::read_to_string(s.root.join(".rayzor/entered")).unwrap() != "waiting" {
        assert!(
            std::time::Instant::now() < deadline,
            "full dependency did not start"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    for (needle, delta, expected) in [
        ("api.Box", 4, "Stores a typed value."),
        ("Array<String>", 0, "class Array<T>"),
        ("String>", 0, "The basic String class."),
        ("value:Record", 6, "Record details."),
        ("input:View", 6, "View contract."),
        ("new Box", 4, "new Box("),
        ("box.get", 4, "Reads the stored value."),
        ("box.get", 4, "get():String"),
        ("box.inferred", 4, "function inferred()"),
        ("box.record", 4, "Returns a record."),
        ("scale:Scale", 6, "Scale choices."),
        ("api.Tools.describe", 4, "class Tools"),
        ("api.Tools.describe", 10, "Describes a record."),
        ("Size.Md", 0, "Size choices."),
    ] {
        let offset = source.find(needle).unwrap() + delta;
        let before = &source[..offset];
        let line = before.bytes().filter(|&byte| byte == b'\n').count() as u32;
        let column = before.rsplit('\n').next().unwrap().encode_utf16().count() as u32;
        let hover =
            hover_text(&s.request("textDocument/hover", s.at("demo/Main.hx", line, column)));
        assert!(hover.contains(expected), "{needle}: {hover}");
        assert!(
            !hover.contains("Dynamic") && !hover.contains("Unknown"),
            "{needle}: {hover}"
        );
    }
    assert!(
        !s.activity
            .iter()
            .any(|note| note["uri"] == s.uri("demo/Main.hx") && note["state"] == "ready")
    );
    drop(release);
    assert!(
        s.wait_ready("demo/Main.hx", 1).is_empty(),
        "{:?}",
        s.diagnostics
    );
}

#[test]
fn preview_expression_macros_defer_external_effects_without_catching_them() {
    let source =
        "class Main { static function main() { var value = Host.value(); trace(value); } }";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (".rayzor/count", "0"),
        (
            "Host.hx",
            r#"class Host {
 public static macro function value():haxe.macro.Expr {
  var path = haxe.macro.Context.resolvePath(".rayzor/count");
  try {
   var count = Std.parseInt(sys.io.File.getContent(path));
   sys.io.File.saveContent(path, Std.string(count + 1));
  } catch (_:Dynamic) { return macro false; }
  return macro 7;
 }
}"#,
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
    assert_eq!(
        std::fs::read_to_string(s.root.join(".rayzor/count")).unwrap(),
        "1"
    );
    let column = source.find("value =").unwrap() as u32 + 1;
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 0, column)));
    assert!(hover.contains("value:Int"), "{hover}");
    assert!(!hover.contains("Provisional"), "{hover}");
}

#[test]
fn a_waiting_request_can_be_cancelled_while_its_page_is_indexing() {
    let s = busy_page_session();
    start_busy_page(&s);
    s.client
        .sender
        .send(Message::Request(Request::new(
            999.into(),
            "rayzor/expandMacro".into(),
            s.at("Slow.hx", 0, 50),
        )))
        .unwrap();
    s.notify("$/cancelRequest", json!({"id":999}));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Message::Response(response) = s
            .client
            .receiver
            .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
            .unwrap()
            && response.id == 999.into()
        {
            assert_eq!(response.error.unwrap().code, -32800);
            break;
        }
    }
}

#[test]
fn editing_a_suspended_page_discards_the_old_generation() {
    let mut s = busy_page_session();
    start_busy_page(&s);
    let replacement = "class Slow { static function main() { var answer:Bool = true; } }";
    s.notify(
        "textDocument/didChange",
        json!({"textDocument":{"uri":s.uri("Slow.hx"),"version":2},
        "contentChanges":[{"text":replacement}]}),
    );
    assert!(s.wait_ready("Slow.hx", 2).is_empty(), "{:?}", s.diagnostics);
    assert!(
        !s.activity.iter().any(|note| note["uri"] == s.uri("Slow.hx")
            && note["version"] == 1
            && note["state"] == "ready")
    );
    assert!(
        !s.diagnostics
            .iter()
            .any(|note| note["uri"] == s.uri("Slow.hx") && note["version"] == 1)
    );
    let at = s.at("Slow.hx", 0, replacement.find("answer").unwrap() as u32 + 1);
    let hover = hover_text(&s.request("textDocument/hover", at));
    assert!(hover.contains("Bool"), "{hover}");
}

fn busy_page_session() -> Session {
    let s = Session::start_with(
        &[
            ("build.hxml", "-cp .\n"),
            (
                "Busy.hx",
                "import haxe.macro.Expr; class Busy { public static macro function make():Expr { sys.io.File.saveContent('MARKER', 'started'); while (true) {} return macro 0; } }",
            ),
            (
                "Quick.hx",
                "class Quick { public static macro function value() return macro 42; }",
            ),
            (
                "Slow.hx",
                "class Slow { static function main() { var answer = Busy.make(); } }",
            ),
            (
                "Fast.hx",
                "class Fast { static function main() { var answer:Int = Quick.value(); } }",
            ),
        ],
        json!({}),
    );
    let path = s.root.join("Busy.hx");
    let source = std::fs::read_to_string(&path).unwrap().replace(
        "'MARKER'",
        &format!("{:?}", s.root.join("started").to_string_lossy()),
    );
    std::fs::write(path, source).unwrap();
    s
}

fn start_busy_page(s: &Session) {
    s.notify(
        "textDocument/didOpen",
        json!({"textDocument":{
            "uri":s.uri("Slow.hx"),"languageId":"haxe","version":1,
            "text":std::fs::read_to_string(s.root.join("Slow.hx")).unwrap()
        }}),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !s.root.join("started").exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "slow macro never started"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn default_and_zero_macro_budgets_allow_unbounded_evaluation() {
    let source = "class Main { static function main() { var value = Builder.make(); } }";
    for options in [json!({}), json!({"macroBudgetMs":0})] {
        let mut s = Session::start_with(
            &[
                (
                    "Builder.hx",
                    "class Builder { public static macro function make() { var total = 0; for (i in 0...4096) total += i; return macro $v{total}; } }",
                ),
                ("Main.hx", source),
            ],
            options,
        );
        let diagnostics = s.open("Main.hx");
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let at = s.at("Main.hx", 0, source.find("Builder.make").unwrap() as u32);
        let expansion = s.request("rayzor/expandMacro", at);
        assert!(
            expansion["expansion"]
                .as_str()
                .is_some_and(|text| text == "8386560"),
            "{expansion}"
        );
    }
}

#[test]
fn hover_answers_at_declarations_and_uses() {
    let mut s = Session::start(&[("Main.hx", MAIN)]);
    assert!(s.open("Main.hx").is_empty());

    let at = s.at("Main.hx", 1, 21);
    let decl = s.request("textDocument/hover", at);
    assert!(
        hover_text(&decl).contains("function add(a:Int, b:Int):Int"),
        "{decl}"
    );
    let at = s.at("Main.hx", 5, 21);
    let call = s.request("textDocument/hover", at);
    assert!(hover_text(&call).contains("function add"), "{call}");
    let at = s.at("Main.hx", 6, 15);
    let local = s.request("textDocument/hover", at);
    assert!(hover_text(&local).contains("var total:Int"), "{local}");
    let at = s.at("Main.hx", 2, 15);
    let param = s.request("textDocument/hover", at);
    assert!(hover_text(&param).contains("a:Int"), "{param}");
    let at = s.at("Main.hx", 3, 1);
    assert!(s.request("textDocument/hover", at).is_null());
}

#[test]
fn hover_includes_declaration_comments_and_refreshes_after_edits() {
    let source = r#"/** Main **entry point**. */
@:keep
class Main {
    /** Stored count. */
    static var count:Int = 1;
    /**
     * Adds a value.
     *
     * `value` is added to the stored count.
     * @param value The amount to add.
     * @return The resulting count.
     */
    @:keep
    static inline function add(value:Int):Int {
        return count + value;
    }
    static function main() {
        trace(add(DocTools.amount));
        trace(DocTools.twice(count));
    }
}
/** Named integer type. */
typedef Count = Int;
/** Unit wrapper. */
abstract Unit(Int) {}
/** Available choices. */
enum Choice {
    /** Select the first choice. */
    First;
}
/** Service contract. */
interface Service {
    // Perform the work.
    public function run():Void;
}
"#;
    let imported = r#"/** Shared utilities. */
class DocTools {
    /** Default amount. */
    public static var amount:Int = 2;
    /**
     * Doubles the input.
     * @param {Int} value - Input amount.
     * @returns {Int} Twice the amount.
     **/
    public static function twice(value:Int):Int return value * 2;
}
"#;
    let mut s = Session::start(&[("Main.hx", source), ("DocTools.hx", imported)]);
    assert!(s.open("Main.hx").is_empty());
    let hover_at = |s: &mut Session, text: &str, needle: &str| {
        let offset = text.find(needle).unwrap();
        let before = &text[..offset];
        let line = before.bytes().filter(|&b| b == b'\n').count() as u32;
        let col = before.rsplit('\n').next().unwrap().encode_utf16().count() as u32;
        let at = s.at("Main.hx", line, col);
        hover_text(&s.request("textDocument/hover", at))
    };
    for (needle, expected) in [
        ("Main {", "Main **entry point**."),
        ("count:Int", "Stored count."),
        ("count +", "Stored count."),
        (
            "add(value",
            "Adds a value.\n\n`value` is added to the stored count.",
        ),
        ("add(DocTools", "Adds a value."),
        ("add(value", "**Parameter** `value`: The amount to add."),
        ("amount)", "Default amount."),
        ("DocTools.amount", "Shared utilities."),
        ("twice(count", "Doubles the input."),
        ("twice(count", "**Returns**\n\n`Int` Twice the amount."),
        ("Count =", "Named integer type."),
        ("Unit(Int)", "Unit wrapper."),
        ("Choice {", "Available choices."),
        ("First;", "Select the first choice."),
        ("Service {", "Service contract."),
        ("run()", "Perform the work."),
    ] {
        let hover = hover_at(&mut s, source, needle);
        assert!(hover.contains(expected), "{needle}: {hover}");
        assert!(!hover.contains("/**"), "{hover}");
    }
    let changed = source.replace("Stored count.", "Current total.");
    s.change("Main.hx", &changed);
    let hover = hover_at(&mut s, &changed, "count +");
    assert!(hover.contains("Current total."), "{hover}");
    assert!(!hover.contains("Stored count."), "{hover}");
}

#[test]
fn workspace_index_finds_unopened_types_and_refreshes_files() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp src\n"),
        (
            "src/tools/Unopened.hx",
            "#if rayzor\npackage tools; /** Available without opening this file. */ class Unopened {}\n#end",
        ),
        ("target/Ignored.hx", "class Ignored {}"),
    ]);
    let symbols = s.request("workspace/symbol", json!({"query":"Unopened"}));
    let symbols = symbols.as_array().unwrap();
    assert_eq!(symbols.len(), 1, "{symbols:?}");
    assert_eq!(
        symbols[0]["location"]["uri"],
        s.uri("src/tools/Unopened.hx")
    );
    assert!(
        s.request("workspace/symbol", json!({"query":"Ignored"}))
            .as_array()
            .unwrap()
            .is_empty()
    );
    std::fs::write(
        s.root.join("src/tools/Unopened.hx"),
        "package tools; class Replacement {}",
    )
    .unwrap();
    s.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":s.uri("src/tools/Unopened.hx"),"type":2}]}),
    );
    assert!(
        s.request("workspace/symbol", json!({"query":"Unopened"}))
            .as_array()
            .unwrap()
            .iter()
            .all(|entry| entry["name"] != "Unopened")
    );
    assert_eq!(
        s.request("workspace/symbol", json!({"query":"Replacement"}))[0]["name"],
        "Replacement"
    );
    std::fs::remove_file(s.root.join("src/tools/Unopened.hx")).unwrap();
    s.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":s.uri("src/tools/Unopened.hx"),"type":3}]}),
    );
    assert!(
        s.request("workspace/symbol", json!({"query":"Replacement"}))
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn imported_type_docs_and_named_signatures_survive_across_modules() {
    let box_source = "package models;\n/** A reusable box. */\nclass Box<T> {\n public var value:T;\n public function new(value:T) { this.value = value; }\n}";
    let api_source = "package models;\n/** Box operations. */\nclass Api {\n /** Wrap a box in an array. */\n public static function wrap(box:Box<String>):Array<Box<String>> return [box];\n}";
    let main = "import models.Box;\nimport models.Api;\nclass Main {\n static function main() {\n  var box = new Box(\"hi\");\n  var result = Api.wrap(box);\n }\n}";
    let mut s = Session::start(&[
        ("Main.hx", main),
        ("models/Box.hx", box_source),
        ("models/Api.hx", api_source),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    for (line, column, expected) in [
        (0, 15, "A reusable box."),
        (1, 15, "Box operations."),
        (4, 17, "A reusable box."),
        (5, 15, "Box operations."),
    ] {
        let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", line, column)));
        assert!(hover.contains(expected), "{line}:{column}: {hover}");
    }
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 5, 20)));
    assert!(hover.contains("Wrap a box in an array."), "{hover}");
    assert!(hover.contains("models.Box<String>"), "{hover}");
    assert!(
        !hover.contains("SymbolId") && !hover.contains("TypeId"),
        "{hover}"
    );
    let signature = s.request("textDocument/signatureHelp", s.at("Main.hx", 5, 25));
    assert!(
        signature.to_string().contains("models.Box<String>"),
        "{signature}"
    );
    assert!(!signature.to_string().contains("SymbolId"), "{signature}");
    std::fs::write(
        s.root.join("models/Box.hx"),
        box_source.replace("A reusable box.", "Updated box documentation."),
    )
    .unwrap();
    s.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":s.uri("models/Box.hx"),"type":2}]}),
    );
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 4, 17)));
    assert!(
        hover.contains("Updated box documentation.") && !hover.contains("A reusable box."),
        "{hover}"
    );
}

#[test]
fn dialog_switch_subject_keeps_source_field_identity() {
    let source = "typedef Props = { /** Is the panel open? */ ?open:Bool, ?id:String };\nclass Base<T> { public var props:T; public function new(props:T) { this.props = props; } }\nclass Main extends Base<Props> {\n function render():Bool {\n  return switch props.open { case null: false; case value: value; }\n }\n}";
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let line = source.lines().nth(4).unwrap();
    let hover = hover_text(&s.request(
        "textDocument/hover",
        s.at("Main.hx", 4, line.find("open").unwrap() as u32),
    ));
    assert!(hover.contains("open:") && hover.contains("Bool"), "{hover}");
    assert!(hover.contains("Is the panel open?"), "{hover}");
    assert!(!hover.contains("__switch"), "{hover}");
    let labels = s.labels("Main.hx", 4, line.find("props.").unwrap() as u32 + 6);
    assert!(
        labels.contains(&"open".into()) && labels.contains(&"id".into()),
        "{labels:?}"
    );
}

#[test]
fn dialog_forwarded_signal_members_keep_the_payload_type() {
    let source = "interface ISignal<T> { public function get():T; public function set(value:T):T; }\n@:forward(get, set) abstract Signal<T>(ISignal<T>) from ISignal<T> {}\nclass Main { static function read(signal:Signal<Bool>):Bool {\n signal.set(true);\n return signal.get();\n} }";
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let members = s.request("textDocument/completion", s.at("Main.hx", 4, 15));
    let get = members
        .as_array()
        .unwrap()
        .iter()
        .find(|member| member["label"] == "get")
        .unwrap();
    assert_eq!(get["detail"], "() -> Bool", "{get}");
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 4, 15)));
    assert!(hover.contains("get():Bool"), "{hover}");
    let definition = s.request("textDocument/definition", s.at("Main.hx", 4, 15));
    assert_eq!(start(&definition), (0, 39), "{definition}");
    let signature = s.request("textDocument/signatureHelp", s.at("Main.hx", 3, 12));
    assert!(
        signature.to_string().contains("Bool") && !signature.to_string().contains("Unknown"),
        "{signature}"
    );
}

#[test]
fn dialog_enum_pattern_bindings_have_hover_and_definition() {
    let source = "enum State<T> { Const(value:T); Bound(value:T); }\nclass Main { static function read(state:State<Bool>):Bool {\n return switch state {\n  case Const(value): value;\n  case Bound(other): other;\n };\n} }";
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 3, 13)));
    assert!(hover.contains("value:Bool"), "{hover}");
    let definition = s.request("textDocument/definition", s.at("Main.hx", 3, 22));
    assert_eq!(start(&definition), (3, 13), "{definition}");
}

#[test]
fn dialog_abstract_enum_patterns_bind_the_underlying_payload() {
    let source = "enum State<T> { Const(value:T); Bound(value:T); }\n@:forward abstract Reactive<T>(State<T>) from State<T> to State<T> {}\nclass Main { static function read(state:Reactive<Bool>):Bool {\n return switch state {\n  case Const(value): value;\n  case Bound(other): other;\n };\n} }";
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 4, 13)));
    assert!(hover.contains("value:Bool"), "{hover}");
    let diagnostics = s.change(
        "Main.hx",
        &source.replace("case Const(value): value", "case Const(value): 42"),
    );
    assert!(
        !diagnostics.is_empty(),
        "the Bool return must still reject Int"
    );
}

#[test]
fn dialog_optional_map_literals_coerce_their_entries() {
    let source = "abstract Reactive<T>(T) { @:from public static function of<T>(value:T):Reactive<T> return cast value; }\nclass Main {\n static function part(?data:Map<String,Reactive<Bool>>):Void {}\n static function main() { part([\"open\" => true]); }\n}";
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let diagnostics = s.change("Main.hx", &source.replace("=> true", "=> 42"));
    assert!(
        !diagnostics.is_empty(),
        "a map entry must retain its declared payload type"
    );
}

#[test]
fn dialog_aliases_do_not_replace_a_same_named_imported_class() {
    let source = "import api.Registry;\nclass Main { static function main() {\n var own = new Alias<Bool>(); own.add(true);\n var imported = new Registry<Int>(); imported.add(42);\n} }\ntypedef Alias<T> = local.Registry<T>;";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "api/Registry.hx",
            "package api; class Registry<T> { public function new() {} public function add(value:T):Void {} }",
        ),
        (
            "local/Registry.hx",
            "package local; typedef Registry<T> = api.Registry<T>;",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 2, 34)));
    assert!(hover.contains("Bool"), "{hover}");
    let diagnostics = s.change(
        "Main.hx",
        &source.replace("imported.add(42)", "imported.add(true)"),
    );
    assert!(
        !diagnostics.is_empty(),
        "the imported Int registry must reject Bool"
    );
}

#[test]
fn dialog_macro_helpers_keep_the_callers_filename_and_search_bounds() {
    let source = "class Main { static function main() { var path:String = Helpers.caller(); } }";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "Helpers.hx",
            "import haxe.macro.Context; class Helpers { macro public static function caller() { var file = Context.getPosInfos(Context.currentPos()).file; var path = sys.FileSystem.fullPath(file); var slash = path.lastIndexOf(\"/\"); if (path.indexOf(\"Main.hx\", slash) < 0) Context.error(\"wrong caller\", Context.currentPos()); return macro $v{path}; } }",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let preview = s.request(
        "rayzor/expandMacro",
        s.at("Main.hx", 0, source.find("Helpers.caller").unwrap() as u32),
    );
    assert!(preview.to_string().contains("Main.hx"), "{preview}");
}

#[test]
fn dialog_constructor_hints_use_utf16_positions() {
    let source = "class Main { static function main() { /* 🙂é */ var value = new Box(\"x\"); value.get(); } }\nclass Box { public function new(label:String) {} public function get():Bool return true; }";
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let at = |needle: &str| {
        source[..source.find(needle).unwrap()]
            .encode_utf16()
            .count() as u32
    };
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 0, at("Box("))));
    assert!(hover.contains("new Box(label:String)"), "{hover}");
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 0, at("get();"))));
    assert!(hover.contains("get():Bool"), "{hover}");
    assert!(s.labels("Main.hx", 0, at("get();")).contains(&"get".into()));
}

#[test]
fn dialog_generic_members_and_inherited_constructor_are_instantiated() {
    let main = "import api.Dialog;\nclass Main { static function main() {\n var registry = new Alias<Dialog>();\n registry.add(new Dialog({open:true}));\n var dialog = new Dialog({open:true});\n} }\ntypedef Alias<T> = Registry<T>;\nclass Registry<T> { public function new() {} public function add(value:T):Void {} }";
    let mut s = Session::start(&[
        ("Main.hx", main),
        (
            "api/Dialog.hx",
            "package api;\ntypedef Props = { ?open:Bool };\nclass Base<T> { public var props:T; public function new(props:T, ?children:Array<String>) { this.props = props; } }\nclass Dialog extends Base<Props> {}",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 3, 10)));
    assert!(
        hover.contains("api.Dialog") && !hover.contains("add(T)"),
        "{hover}"
    );
    let items = s.request("textDocument/completion", s.at("Main.hx", 3, 10));
    let add = items
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["label"] == "add")
        .unwrap();
    assert!(
        add["detail"].as_str().unwrap().contains("api.Dialog"),
        "{add}"
    );
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 4, 18)));
    assert!(
        hover.contains("new Dialog(") && hover.contains("api.Props"),
        "{hover}"
    );
    let help = s.request("textDocument/signatureHelp", s.at("Main.hx", 4, 25));
    assert!(help.to_string().contains("api.Props"), "{help}");
    s.notify(
        "textDocument/didClose",
        json!({"textDocument":{"uri":s.uri("Main.hx")}}),
    );
    s.activity.clear();
    assert!(s.open("Main.hx").is_empty());
    assert!(
        !s.activity
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:#?}",
        s.activity
    );
    let restored = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 4, 18)));
    assert_eq!(restored, hover);
    let diagnostics = s.change(
        "Main.hx",
        &main.replace(
            "registry.add(new Dialog({open:true}))",
            "registry.add(\"wrong\")",
        ),
    );
    assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
    assert!(
        diagnostics[0]["message"]
            .as_str()
            .unwrap()
            .contains("Dialog"),
        "{diagnostics:#?}"
    );
}

#[test]
fn dialog_module_qualified_types_and_enum_definitions_have_source_locations() {
    let source = "class Main {\n var callback:api.Dialog.CancelEvent->Void;\n function placement():api.TopLayer.Placement return Centered;\n}";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "api/Dialog.hx",
            "package api;\nclass Dialog {}\n/** Cancellation details. */\ntypedef CancelEvent = { prevented:Bool };",
        ),
        (
            "api/TopLayer.hx",
            "package api;\nclass TopLayer {}\nenum Placement { Centered; Edge; }",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 1, 26)));
    assert!(
        hover.contains("CancelEvent") && hover.contains("Cancellation details."),
        "{hover}"
    );
    let definition = s.request("textDocument/definition", s.at("Main.hx", 1, 26));
    assert!(
        definition["uri"]
            .as_str()
            .unwrap_or("")
            .ends_with("api/Dialog.hx"),
        "{definition}"
    );
    let definition = s.request("textDocument/definition", s.at("Main.hx", 2, 52));
    assert!(
        definition["uri"]
            .as_str()
            .unwrap_or("")
            .ends_with("api/TopLayer.hx"),
        "{definition}"
    );
}

#[test]
fn cached_and_partial_hovers_work_while_a_build_macro_is_waiting() {
    let mut s = Session::start(&[
        (
            "First.hx",
            "class First { public static var value:Int = 1; }",
        ),
        (
            "Slow.hx",
            "/** Available before its build completes. */\n@:build(Gate.build())\nclass Slow {}",
        ),
        ("gate.txt", "closed"),
        (".rayzor/entered", ""),
        (
            "Gate.hx",
            r#"import haxe.macro.Context; import haxe.macro.Expr;
class Gate { public static macro function build():Array<Field> {
 var path = Context.resolvePath("gate.txt");
 sys.io.File.saveContent(Context.resolvePath(".rayzor/entered"), "waiting");
 while (sys.io.File.getContent(path) != "open") {}
 return Context.getBuildFields();
} }"#,
        ),
    ]);
    struct Release(std::path::PathBuf);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::write(&self.0, "open");
        }
    }
    let release = Release(s.root.join("gate.txt"));
    assert!(s.open("First.hx").is_empty());
    // A worker round trip drains the ready notification after its index is saved.
    s.request("workspace/symbol", json!({"query":"First"}));
    s.notify("textDocument/didOpen", json!({"textDocument":{"uri":s.uri("Slow.hx"),"version":1,"languageId":"haxe","text":std::fs::read_to_string(s.root.join("Slow.hx")).unwrap()}}));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::fs::read_to_string(s.root.join(".rayzor/entered")).unwrap_or_default() != "waiting" {
        assert!(
            std::time::Instant::now() < deadline,
            "build macro never started"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let hover = hover_text(&s.request("textDocument/hover", s.at("First.hx", 0, 34)));
    assert!(hover.contains("value:Int"), "{hover}");
    let partial = hover_text(&s.request("textDocument/hover", s.at("Slow.hx", 2, 7)));
    assert!(
        partial.contains("class Slow") && partial.contains("Available before"),
        "{partial}"
    );
    assert!(
        !s.activity
            .iter()
            .any(|activity| activity["uri"] == s.uri("Slow.hx") && activity["state"] == "ready"),
        "macro completed before cached queries were answered"
    );
    drop(release);
    s.request("workspace/symbol", json!({"query":"Slow"}));
}

#[test]
fn blade_editor_data_is_reused_after_a_server_restart() {
    let source = "class Main { /** Retained field. */ public static var value:Int = 7; }";
    let mut first = Session::start(&[("Main.hx", source)]);
    assert!(first.open("Main.hx").is_empty());
    first.request("workspace/symbol", json!({"query":"Main"}));
    let root = first.root.clone();
    first.temporary = false;
    drop(first);
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    assert!(
        std::fs::read_dir(root.join(".rayzor/blade/editor"))
            .unwrap()
            .any(|entry| entry
                .unwrap()
                .path()
                .extension()
                .is_some_and(|extension| extension == "bidx"))
    );
    let mut second = Session::connect(root, json!({}), false);
    assert!(second.open("Main.hx").is_empty());
    let column = source.find("value").unwrap() as u32;
    let hover = hover_text(&second.request("textDocument/hover", second.at("Main.hx", 0, column)));
    assert!(
        hover.contains("value:Int") && hover.contains("Retained field."),
        "{hover}"
    );
    second.request("workspace/symbol", json!({"query":"Main"}));
    assert!(
        !second
            .activity
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:#?}",
        second.activity
    );
    let changed = source.replace("value:Int = 7", "value:String = \"new\"");
    assert!(second.change("Main.hx", &changed).is_empty());
    let hover = hover_text(&second.request("textDocument/hover", second.at("Main.hx", 0, column)));
    assert!(
        hover.contains("value:String") && !hover.contains("value:Int"),
        "{hover}"
    );
}

#[test]
fn a_new_dependency_tab_reuses_macro_types_and_persists_its_editor_index() {
    let shared = "@:build(Build.fields())\nclass Shared { public static function value():Dynamic return generated; }";
    let mut first = Session::start(&[
        (
            "Main.hx",
            "class Main { static function main() trace(Shared.value()); }",
        ),
        ("Shared.hx", shared),
        ("Unrelated.hx", "class Unrelated {}"),
        ("notes.txt", "original notes"),
        (".rayzor/runs", "0"),
        (
            "Build.hx",
            r#"class Build {
            public static macro function fields():Array<haxe.macro.Expr.Field> {
                var path = haxe.macro.Context.resolvePath(".rayzor/runs");
                var runs = Std.parseInt(sys.io.File.getContent(path));
                sys.io.File.saveContent(path, Std.string(runs + 1));
                var fields = haxe.macro.Context.getBuildFields();
                fields.push({name:"generated", access:[APublic, AStatic], kind:FVar(macro:Int, macro 7), pos:haxe.macro.Context.currentPos()});
                return fields;
            }
        }"#,
        ),
    ]);
    assert!(first.open("Main.hx").is_empty(), "{:?}", first.diagnostics);
    assert_eq!(
        std::fs::read_to_string(first.root.join(".rayzor/runs")).unwrap(),
        "1"
    );
    std::fs::write(
        first.root.join("Unrelated.hx"),
        "class Unrelated { var unused:String; }",
    )
    .unwrap();
    std::fs::write(first.root.join("notes.txt"), "updated notes").unwrap();
    first.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":first.uri("Unrelated.hx"),"type":2}]}),
    );
    assert!(
        first.open("Shared.hx").is_empty(),
        "{:?}",
        first.diagnostics
    );
    let column = shared.lines().nth(1).unwrap().find("generated").unwrap() as u32;
    let hover = hover_text(&first.request("textDocument/hover", first.at("Shared.hx", 1, column)));
    assert!(hover.contains("generated:Int"), "{hover}");
    assert_eq!(
        std::fs::read_to_string(first.root.join(".rayzor/runs")).unwrap(),
        "1"
    );
    first.notify(
        "textDocument/didClose",
        json!({"textDocument":{"uri":first.uri("Shared.hx")}}),
    );
    first.activity.clear();
    std::fs::write(first.root.join("notes.txt"), "changed again").unwrap();
    assert!(first.open("Shared.hx").is_empty());
    assert!(
        !first
            .activity
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:?}",
        first.activity
    );
    let root = first.root.clone();
    first.temporary = false;
    drop(first);
    let mut second = Session::connect(root, json!({}), true);
    assert!(second.open("Shared.hx").is_empty());
    let hover =
        hover_text(&second.request("textDocument/hover", second.at("Shared.hx", 1, column)));
    assert!(hover.contains("generated:Int"), "{hover}");
    assert!(
        !second
            .activity
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:?}",
        second.activity
    );
    assert_eq!(
        std::fs::read_to_string(second.root.join(".rayzor/runs")).unwrap(),
        "1"
    );
    std::fs::write(
        second.root.join("Build.hx"),
        r#"class Build {
        public static macro function fields():Array<haxe.macro.Expr.Field> {
            var fields = haxe.macro.Context.getBuildFields();
            fields.push({name:"generated", access:[APublic,AStatic], kind:FVar(macro:String,macro "updated"), pos:haxe.macro.Context.currentPos()});
            return fields;
        }
    }"#,
    )
    .unwrap();
    second.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":second.uri("Build.hx"),"type":2}]}),
    );
    // The root source remains unchanged; the macro helper is the changed input.
    assert!(second.change("Shared.hx", shared).is_empty());
    let hover =
        hover_text(&second.request("textDocument/hover", second.at("Shared.hx", 1, column)));
    assert!(hover.contains("generated:String"), "{hover}");
}

#[test]
fn macro_filesystem_reads_invalidate_persisted_types() {
    let source =
        "@:build(Build.fields())\nclass Main { static function test():Dynamic return generated; }";
    let mut session = Session::start(&[
        ("Main.hx", source),
        (".rayzor/anchor", "anchor"),
        (".rayzor/assets/value.txt", "Int"),
        (
            "Build.hx",
            r#"class Build {
    public static macro function fields():Array<haxe.macro.Expr.Field> {
        var anchor = haxe.macro.Context.resolvePath(".rayzor/anchor");
        var root = anchor.substr(0, anchor.length - "anchor".length);
        var value = sys.io.File.getContent(root + "assets/value.txt");
        var type = value == "Int" ? macro:Int : macro:String;
        if (sys.FileSystem.exists(root + "enabled")) type = macro:Bool;
        if (sys.FileSystem.readDirectory(root + "assets").length > 1) type = macro:Float;
        var fields = haxe.macro.Context.getBuildFields();
        fields.push({name:"generated", access:[APublic,AStatic], kind:FVar(type), pos:haxe.macro.Context.currentPos()});
        return fields;
    }
}"#,
        ),
    ]);
    let root = session.root.clone();
    let column = source.lines().nth(1).unwrap().find("generated").unwrap() as u32;
    for (stage, expected) in ["Int", "String", "Bool", "Float"].into_iter().enumerate() {
        if stage != 0 {
            session.temporary = false;
            drop(session);
            match stage {
                1 => std::fs::write(root.join(".rayzor/assets/value.txt"), "String").unwrap(),
                2 => std::fs::write(root.join(".rayzor/enabled"), "enabled").unwrap(),
                3 => std::fs::write(root.join(".rayzor/assets/extra.txt"), "extra").unwrap(),
                _ => unreachable!(),
            }
            session = Session::connect(root.clone(), json!({}), true);
        }
        assert!(
            session.open("Main.hx").is_empty(),
            "{:?}",
            session.diagnostics
        );
        let hover =
            hover_text(&session.request("textDocument/hover", session.at("Main.hx", 1, column)));
        assert!(hover.contains(&format!("generated:{expected}")), "{hover}");
        assert!(
            session
                .activity
                .iter()
                .any(|activity| activity["state"] == "checking")
        );
    }
}

#[test]
fn a_request_sent_with_didopen_after_restart_is_answered_from_the_index() {
    let source = "class Main { /** Retained field. */ public static var value:Int = 7; }";
    let mut first = Session::start(&[("Main.hx", source)]);
    assert!(first.open("Main.hx").is_empty());
    let root = first.root.clone();
    first.temporary = false;
    drop(first);
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let mut second = Session::connect(root, json!({}), false);
    let uri = second.uri("Main.hx");
    second.notify(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": uri, "languageId": "haxe", "version": 1, "text": source}}),
    );
    let column = source.find("value").unwrap() as u32;
    let hover = hover_text(&second.request("textDocument/hover", second.at("Main.hx", 0, column)));
    assert!(hover.contains("Retained field."), "{hover}");
    assert!(
        !second
            .activity
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:#?}",
        second.activity
    );
}

#[test]
fn typedef_fields_have_types_and_docs_at_declarations_and_imported_reads() {
    let types = "package models;\ntypedef State = {\n /** Horizontal offset. */\n panX:Float,\n /** Display label. */\n ?label:String,\n /** Nested position. */\n position:{x:Int, y:Int}\n};\ntypedef Alias = State;";
    let main = "import models.Types.State;\nimport models.Types.Alias;\nclass Main {\n static function main() {\n  var state:State = {panX:1.0, position:{x:1, y:2}};\n  var alias:Alias = state;\n  state.panX;\n  alias.panX;\n  state.position.x;\n }\n}";
    let mut s = Session::start(&[("Main.hx", main), ("models/Types.hx", types)]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    for line in [6, 7] {
        let text = main.lines().nth(line).unwrap();
        let col = text.find("panX").unwrap();
        let hover = hover_text(&s.request(
            "textDocument/hover",
            s.at("Main.hx", line as u32, col as u32 + 1),
        ));
        assert!(
            hover.contains("panX:Float") && hover.contains("Horizontal offset."),
            "{hover}"
        );
        let items = s.request(
            "textDocument/completion",
            s.at("Main.hx", line as u32, col as u32),
        );
        let field = items
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["label"] == "panX")
            .unwrap();
        assert_eq!(field["detail"], "Float");
        assert!(
            field["documentation"]["value"]
                .as_str()
                .unwrap_or("")
                .contains("Horizontal offset."),
            "{field}"
        );
    }
    s.open("models/Types.hx");
    for (line, name, ty, doc) in [
        (3, "panX", "Float", "Horizontal offset."),
        (5, "label", "String", "Display label."),
        (7, "x", "Int", ""),
    ] {
        let col = types.lines().nth(line).unwrap().find(name).unwrap();
        let hover = hover_text(&s.request(
            "textDocument/hover",
            s.at("models/Types.hx", line as u32, col as u32),
        ));
        assert!(
            hover.contains(&format!("{name}:{ty}")) && hover.contains(doc),
            "{hover}"
        );
    }
}

#[test]
fn hover_preserves_type_declaration_kinds_and_structural_types() {
    let types = "package models;\n/** View state. */\ntypedef State = {panX:Float, ?label:String};\n/** A count. */\ntypedef Count = Int;\n/** A collection. */\ntypedef Items<T> = Array<T>;\n/** A readable value. */\ninterface Readable { public function read():String; }\n/** An identifier. */\nabstract Token(Int) from Int to Int {}\n/** A mode. */\nenum abstract Mode(Int) { var Enabled = 1; }\n/** A color. */\nenum Color { Red; }\n/** A box. */\nclass Box<T> { public var value:T; }";
    let main = "import models.Types.State;\nimport models.Types.Count;\nimport models.Types.Items;\nimport models.Types.Readable;\nimport models.Types.Token;\nimport models.Types.Mode;\nimport models.Types.Color;\nimport models.Types.Box;\nclass Main {\n static function main() {\n  var state:State = {panX:1.0};\n  var shape:{panX:Float, ?label:String} = {panX:2.0};\n  var inferred = {panX:3.0, label:\"view\"};\n  shape.panX;\n  inferred.panX;\n  state.panX;\n }\n}";
    let mut s = Session::start(&[("Main.hx", main), ("models/Types.hx", types)]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    s.open("models/Types.hx");
    for (name, expected, doc) in [
        (
            "State",
            "typedef State = {panX:Float, ?label:String}",
            "View state.",
        ),
        ("Count", "typedef Count = Int", "A count."),
        ("Items", "typedef Items<T> = Array<T>", "A collection."),
        ("Readable", "interface Readable", "A readable value."),
        (
            "Token",
            "abstract Token(Int) from Int to Int",
            "An identifier.",
        ),
        ("Mode", "enum abstract Mode(Int)", "A mode."),
        ("Color", "enum Color", "A color."),
        ("Box", "class Box<T>", "A box."),
    ] {
        for (file, text) in [("Main.hx", main), ("models/Types.hx", types)] {
            let (line, contents) = text
                .lines()
                .enumerate()
                .find(|(_, l)| l.contains(name))
                .unwrap();
            let col = contents.find(name).unwrap();
            let hover = hover_text(&s.request(
                "textDocument/hover",
                s.at(file, line as u32, col as u32 + 1),
            ));
            assert!(
                hover.contains(expected) && hover.contains(doc),
                "{file} {name}: {hover}"
            );
            assert!(
                !hover.contains("SymbolId") && !hover.contains("TypeId"),
                "{hover}"
            );
        }
    }
    for (line, name, expected) in [
        (11, "shape", "?label:String"),
        (12, "inferred", "panX:Float"),
    ] {
        let col = main.lines().nth(line).unwrap().find(name).unwrap();
        let hover = hover_text(&s.request(
            "textDocument/hover",
            s.at("Main.hx", line as u32, col as u32 + 1),
        ));
        assert!(hover.contains(expected) && hover.contains('{'), "{hover}");
    }
    let symbols = s.request(
        "textDocument/documentSymbol",
        json!({"textDocument":{"uri":s.uri("models/Types.hx")}}),
    );
    for (line, name) in [(13, "shape"), (14, "inferred"), (15, "state")] {
        let items = s.request(
            "textDocument/completion",
            s.at("Main.hx", line, name.len() as u32 + 3),
        );
        assert!(
            items
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["label"] == "panX" && item["detail"] == "Float"),
            "{name}: {items}"
        );
    }
    let completions = s.request("textDocument/completion", s.at("Main.hx", 10, 2));
    for (name, symbol_kind, completion_kind) in [
        (
            "State",
            lsp_types::SymbolKind::STRUCT,
            lsp_types::CompletionItemKind::STRUCT,
        ),
        (
            "Count",
            lsp_types::SymbolKind::STRUCT,
            lsp_types::CompletionItemKind::STRUCT,
        ),
        (
            "Items",
            lsp_types::SymbolKind::STRUCT,
            lsp_types::CompletionItemKind::STRUCT,
        ),
        (
            "Readable",
            lsp_types::SymbolKind::INTERFACE,
            lsp_types::CompletionItemKind::INTERFACE,
        ),
        (
            "Token",
            lsp_types::SymbolKind::STRUCT,
            lsp_types::CompletionItemKind::STRUCT,
        ),
        (
            "Mode",
            lsp_types::SymbolKind::STRUCT,
            lsp_types::CompletionItemKind::STRUCT,
        ),
        (
            "Color",
            lsp_types::SymbolKind::ENUM,
            lsp_types::CompletionItemKind::ENUM,
        ),
        (
            "Box",
            lsp_types::SymbolKind::CLASS,
            lsp_types::CompletionItemKind::CLASS,
        ),
    ] {
        let symbol = symbols
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == name)
            .unwrap_or_else(|| panic!("missing {name} in {symbols}"));
        assert_eq!(symbol["kind"], json!(symbol_kind), "{symbol}");
        let completion = completions
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["label"] == name)
            .unwrap_or_else(|| panic!("missing {name} in {completions}"));
        assert_eq!(completion["kind"], json!(completion_kind), "{completion}");
    }
}

#[test]
fn fully_qualified_standard_library_calls_are_typed() {
    let source = "class Main {\n static function main() {\n  var extension = haxe.io.Path.extension(\"view.png\");\n  var lower = extension.toLowerCase();\n }\n}";
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 2, 34)));
    assert!(
        hover.contains("function extension") && hover.contains(":String"),
        "{hover}"
    );
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 3, 7)));
    assert!(hover.contains("var lower:String"), "{hover}");
    let diagnostics = s.change(
        "Main.hx",
        &source
            .replace("extension(", "withoutExtension(")
            .replace("\"view.png\"", "123"),
    );
    assert!(
        diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|m| m.contains("String") && m.contains("Int"))),
        "{diagnostics:#?}"
    );
}

#[test]
fn a_namesake_enum_constructor_does_not_replace_a_standard_library_class() {
    let source = "enum Shape { Path(d:String); }\nclass Main {\n static function main() {\n  var extension = haxe.io.Path.extension(\"view.png\");\n  var shape = Shape.Path(\"M0 0\");\n }\n}";
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 3, 34)));
    assert!(
        hover.contains("function extension") && hover.contains(":String"),
        "{hover}"
    );
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 4, 7)));
    assert!(hover.contains("var shape:Shape"), "{hover}");
}

#[test]
fn hashlink_library_selects_the_target_and_other_missing_targets_warn() {
    let source = "class Main { static function main() {\n#if hl\nvar selected:Int = 1;\n#elseif js\nvar selected:Bool = true;\n#else\nvar selected:String = 'check';\n#end\n} }";
    for (hxml, line, expected, missing) in [
        ("-cp .\n", 6, "String", true),
        ("-cp .\n-L hashlink\n", 2, "Int", false),
        ("-cp .\n-L hashlink:1.0.0\n--no-output\n", 2, "Int", false),
        ("-cp .\n-L hashlink\n--js app.js\n", 4, "Bool", false),
        ("-cp .\n--no-output\n", 6, "String", true),
    ] {
        let mut session = Session::start(&[
            ("build.hxml", hxml),
            ("Main.hx", source),
            (
                "haxelib.json",
                r#"{"name":"hashlink","version":"1.0.0","classPath":"."}"#,
            ),
        ]);
        assert!(
            session.open("Main.hx").is_empty(),
            "{:?}",
            session.diagnostics
        );
        let position = session.at("Main.hx", line, 5);
        let hover = hover_text(&session.request("textDocument/hover", position));
        assert!(
            hover.contains(&format!("selected:{expected}")),
            "{hxml}: {hover}"
        );
        let warning = session
            .diagnostics
            .iter()
            .filter(|note| note["uri"] == session.uri("build.hxml"))
            .flat_map(|note| note["diagnostics"].as_array().unwrap())
            .any(|diagnostic| {
                diagnostic["severity"] == 2
                    && diagnostic["message"]
                        .as_str()
                        .is_some_and(|message| message.contains("No Haxe target"))
            });
        assert_eq!(warning, missing, "{hxml}: {:?}", session.diagnostics);
    }
}

#[test]
fn missing_target_warning_is_cleared_when_configuration_adds_a_target() {
    let mut session = Session::start(&[
        ("build.hxml", "-cp .\n"),
        ("Main.hx", "class Main { static function main() {} }"),
    ]);
    session.open("Main.hx");
    assert!(
        session
            .diagnostics
            .iter()
            .any(|note| note["uri"] == session.uri("build.hxml")
                && note["diagnostics"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|d| d["severity"] == 2)),
        "{:?}",
        session.diagnostics
    );
    std::fs::write(session.root.join("build.hxml"), "-cp .\n--js app.js\n").unwrap();
    session.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":session.uri("build.hxml"),"type":2}]}),
    );
    session.request("workspace/symbol", json!({"query":"Main"}));
    let last = session
        .diagnostics
        .iter()
        .rev()
        .find(|note| note["uri"] == session.uri("build.hxml"))
        .unwrap();
    assert_eq!(last["diagnostics"], json!([]), "{last}");
}

#[test]
fn analysis_status_and_diagnostics_track_document_versions() {
    let source = "class Main { static function main() {} }";
    let mut s = Session::start(&[("Main.hx", source)]);
    for (version, text) in [
        (10, source),
        (
            11,
            "class Main { static function main() { var bad:Int = \"bad\"; } }",
        ),
    ] {
        if version == 10 {
            s.notify("textDocument/didOpen", json!({"textDocument":{"uri":s.uri("Main.hx"),"version":version,"languageId":"haxe","text":text}}));
        } else {
            s.notify("textDocument/didChange", json!({"textDocument":{"uri":s.uri("Main.hx"),"version":version},"contentChanges":[{"text":text}]}));
        }
        s.request("textDocument/hover", s.at("Main.hx", 0, 7));
        s.request("workspace/symbol", json!({"query":"Main"}));
        // A partial hover can answer before the document's diagnostics finish.
        if !s
            .activity
            .iter()
            .any(|event| event["version"] == version && event["state"] == "ready")
        {
            s.wait_ready("Main.hx", version);
        }
        let activity: Vec<_> = s
            .activity
            .iter()
            .filter(|event| event["version"] == version)
            .map(|event| event["state"].as_str().unwrap())
            .collect();
        assert_eq!(
            activity,
            ["queued", "checking", "ready"],
            "{:?}",
            s.activity
        );
        assert!(
            s.diagnostics
                .iter()
                .any(|diagnostic| diagnostic["uri"] == s.uri("Main.hx")
                    && diagnostic["version"] == version),
            "{:?}",
            s.diagnostics
        );
    }
}

#[test]
fn imported_subclasses_are_assignable_to_their_base_classes() {
    let mut s = Session::start(&[
        ("Base.hx", "class Base { public function new() {} }"),
        ("Middle.hx", "class Middle extends Base {}"),
        ("Child.hx", "class Child extends Middle {}"),
        (
            "Main.hx",
            "class Main { static function fill(value:Base):Void {} static function main() { var child = new Child(); fill(child); var items:Array<Base> = [child]; } }",
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn enum_abstract_values_do_not_replace_imported_class_identities() {
    let source = "import keys.NamedKey; import widgets.ContextMenu; class Main { static function accept(value:widgets.Base):Void {} static function main() { var key = NamedKey.ContextMenu; var offset = widgets.Scroll.of(2); var menu = Builder.make(<context-menu/>); accept(menu); } }";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "keys/NamedKey.hx",
            "package keys; enum abstract NamedKey(Int) { var ContextMenu = 42; var Scroll = 1; }",
        ),
        (
            "widgets/Base.hx",
            "package widgets; class Base { public function new() {} }",
        ),
        (
            "widgets/ContextMenu.hx",
            "package widgets; class ContextMenu extends Base {}",
        ),
        (
            "widgets/Scroll.hx",
            "package widgets; class Scroll { public static function of(value:Int):Int return value; }",
        ),
        (
            "Builder.hx",
            r#"
import haxe.macro.Expr;
import haxe.macro.Context;
class Builder {
    public static macro function make(template:Expr):Expr {
        switch Context.getType("ContextMenu") {
            case TInst(c, _): if (c.get().pack.join(".") != "widgets") throw "wrong class";
            case _: throw "expected a class";
        }
        var p = Context.getPosInfos(template.pos);
        var tag = Context.makePosition({file:p.file, min:p.min + 13, max:p.min + 14});
        return macro @:pos(tag) new widgets.ContextMenu();
    }
}

"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let at = s.at("Main.hx", 0, source.find("context-menu").unwrap() as u32);
    let hover = hover_text(&s.request("textDocument/hover", at));
    assert!(hover.contains("class ContextMenu extends Base"), "{hover}");
}

#[test]
fn generic_switch_branches_use_the_inferred_common_base() {
    for (result, explicit, valid) in [
        ("Element", false, true),
        ("Element", true, true),
        ("For<Int>", true, false),
    ] {
        let [ints, bools, strings] = if explicit {
            ["For<Int>", "For<Bool>", "For<String>"]
        } else {
            ["For", "For", "For"]
        };
        let source = format!(
            "class Main {{ static function pane(which:String):{result} return switch which {{case \"one\": new {ints}(() -> [1], n -> new Element()); case \"two\": new {bools}(() -> [true], n -> new Element()); case _:new {strings}(() -> [\"x\"], n -> new Element());}}; static function main() {{}} }}"
        );
        let mut s = Session::start(&[
            ("Main.hx", &source),
            ("Element.hx", "class Element { public function new() {} }"),
            (
                "For.hx",
                "class For<T> extends Element { public function new(each:Void->Array<T>, item:T->Element) {super();} }",
            ),
        ]);
        let diagnostics = s.open("Main.hx");
        assert_eq!(diagnostics.is_empty(), valid, "{result}: {diagnostics:#?}");
    }
}

#[test]
fn deferred_macro_switch_branches_use_the_inferred_common_base() {
    let source = r#"import widgets.For;
class Main {
 static function pane(which:String):widgets.Element return switch which {
  case "activity": Builder.each([[1, 2]]);
  case "elements": Builder.each([["a"]]);
  case "debug": Builder.each([true]);
  case _: Builder.each([[{text:"hello"}]]);
 };
 static function main() {}
}"#;
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "Builder.hx",
            "import haxe.macro.Context; import haxe.macro.Expr; class Builder { public static macro function each(values:Expr):Expr { Context.typeof(values); return macro new widgets.For(() -> $values, value -> new widgets.Element()); } }",
        ),
        (
            "widgets/Element.hx",
            "package widgets; class Element { public function new() {} }",
        ),
        (
            "widgets/For.hx",
            "package widgets; class For<T> extends Element { public function new(each:Void->Array<T>, item:T->Element) {super();} }",
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let offset = source.find("Builder.each([true])").unwrap();
    let before = &source[..offset];
    let at = s.at(
        "Main.hx",
        before.bytes().filter(|&byte| byte == b'\n').count() as u32,
        before.rsplit('\n').next().unwrap().len() as u32,
    );
    let expanded = s.request("rayzor/expandMacro", at);
    assert!(
        expanded["expansion"]
            .as_str()
            .is_some_and(|text| text.contains("new widgets.For")),
        "{expanded}"
    );
}

#[test]
fn generic_property_keys_bind_abstract_conversion_value_types() {
    for (project_file, configuration) in [
        ("build.hxml", "--hl out.hl\n"),
        (
            "rayzor.toml",
            "[project]\nname=\"generic-properties\"\n[cache]\nenabled=false\n",
        ),
    ] {
        for (value, valid) in [
            ("116", true),
            ("(116:Single)", true),
            ("widgets.Computed.make((116:Single))", true),
            ("new widgets.State<Single>((116:Single))", true),
            ("widgets.Computed.make(\"wrong\")", false),
            ("widgets.Styles.Display.Flex", false),
            ("\"wrong\"", false),
        ] {
            let source = format!(
                "import widgets.Prop; import widgets.Node; import widgets.Flex; import widgets.Other; class Main {{ static function main() {{ var node = new Node(); node.set(Prop.Width, {value}); node.set(Prop.Display, widgets.Styles.Display.Flex); node.set(Prop.Display, Flex); }} }}"
            );
            let mut s = Session::start(&[
                ("Main.hx", &source),
                (project_file, configuration),
                (
                    "widgets/Styles.hx",
                    "package widgets; class Styles {} enum abstract Display(Int) to Int { var Flex = 0; }",
                ),
                (
                    "widgets/Prop.hx",
                    "package widgets; abstract Prop<T>(Int) { public static inline var Width:Prop<Single> = cast 1; public static inline var Display:Prop<Styles.Display> = cast 2; }",
                ),
                (
                    "widgets/IntoReactive.hx",
                    "package widgets; enum ReactiveValue<T> { Const(value:T); } abstract IntoReactive<T>(ReactiveValue<T>) from ReactiveValue<T> { @:from public static function fromConst<T>(value:T):IntoReactive<T> return Const(value); @:from public static function fromInt(value:Int):IntoReactive<Single> return Const((value:Single)); @:from public static function fromComputed<T>(value:Computed<T>):IntoReactive<T> return Const(cast value); @:from public static function fromState<T>(value:State<T>):IntoReactive<T> return Const(value.value); }",
                ),
                (
                    "widgets/Computed.hx",
                    "package widgets; abstract Computed<T>(T) { public static function make<T>(value:T):Computed<T> return cast value; }",
                ),
                (
                    "widgets/State.hx",
                    "package widgets; class State<T> { public var value:T; public function new(value:T) this.value = value; }",
                ),
                (
                    "widgets/Flex.hx",
                    "package widgets; class Flex { public function new() {} }",
                ),
                (
                    "widgets/Other.hx",
                    "package widgets; private enum abstract Display(Int) { var Flex = 7; } class Other {}",
                ),
                (
                    "widgets/Node.hx",
                    "package widgets; class Node { public function new() {} public function set<T>(prop:Prop<T>, value:IntoReactive<T>):Void {} }",
                ),
            ]);
            let diagnostics = s.open("Main.hx");
            assert_eq!(diagnostics.is_empty(), valid, "{value}: {diagnostics:#?}");
        }
    }
}

#[test]
fn throwing_switch_branches_do_not_supply_void_return_values() {
    for (branches, valid) in [
        (
            "case File(bytes): bytes; case Dir: throw 'directory'; case null: throw 'missing';",
            true,
        ),
        (
            "case Dir: throw 'directory'; case null: throw 'missing'; case File(bytes): bytes;",
            true,
        ),
        (
            "case File(bytes): bytes; case Dir: { trace('directory'); throw 'directory'; } case null: throw 'missing';",
            true,
        ),
        ("case File(bytes): bytes; default: throw 'missing';", true),
        (
            "case File(bytes): node != null ? bytes : throw 'missing'; case Dir: throw 'directory'; case null: throw 'missing';",
            true,
        ),
        (
            "case File(bytes): bytes; case Dir: { if (node != null) throw 'directory'; else throw 'missing'; } case null: throw 'missing';",
            true,
        ),
        (
            "case File(bytes): bytes; case Dir: nothing(); case null: throw 'missing';",
            false,
        ),
        (
            "case File(bytes): bytes; case Dir: 0; case null: throw 'missing';",
            false,
        ),
        (
            "case File(bytes): bytes; case Dir: return 0; case null: throw 'missing';",
            false,
        ),
    ] {
        let source = format!(
            "import haxe.io.Bytes; enum Node {{ File(bytes:Bytes); Dir; }} class Main {{ static function nothing():Void {{}} static function read(node:Null<Node>):Bytes return switch node {{ {branches} }} static function main() {{ read(File(Bytes.alloc(0))); }} }}"
        );
        let mut s = Session::start(&[("Main.hx", &source)]);
        let diagnostics = s.open("Main.hx");
        assert_eq!(
            diagnostics.is_empty(),
            valid,
            "{branches}: {diagnostics:#?}"
        );
    }
}

#[test]
#[ignore = "requires the sibling RayDE workspace"]
fn rayde_memory_workspace_keeps_bytes_returning_switches() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../RayDE/src/rayde/workspace");
    let source = std::fs::read_to_string(root.join("MemoryWorkspace.hx")).unwrap();
    let interface = std::fs::read_to_string(root.join("Workspace.hx")).unwrap();
    let mut s = Session::start(&[
        ("build.hxml", "-cp .\n--hl out.hl\n"),
        ("rayde/workspace/MemoryWorkspace.hx", &source),
        ("rayde/workspace/Workspace.hx", &interface),
    ]);
    let diagnostics = s.open("rayde/workspace/MemoryWorkspace.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn sibling_components_in_arrays_use_their_common_element_type() {
    for (kind, valid) in [("Element", true), ("Dynamic", true), ("Title", false)] {
        let source = format!(
            "class Main {{ static function main() {{ var children:Array<{kind}> = [new Title(), new Description()]; }} }} class Element {{ public function new() {{}} }} class Component<T> extends Element {{}} class Title extends Component<{{title:String}}> {{}} class Description extends Component<{{description:String}}> {{}}"
        );
        let mut s = Session::start(&[("Main.hx", &source)]);
        let diagnostics = s.open("Main.hx");
        assert_eq!(diagnostics.is_empty(), valid, "{kind}: {diagnostics:#?}");
    }
}

#[test]
fn identical_macro_asset_rewrites_do_not_restart_analysis_for_hover() {
    let source =
        "class Main { static function main() { var value = Assets.read(); trace(value); } }";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "Assets.hx",
            "class Assets { public static macro function read() { var path = haxe.macro.Context.resolvePath('asset.bin'); var bytes = sys.io.File.getBytes(path); return macro $v{bytes.toString()}; } }",
        ),
        ("asset.bin", "same"),
    ]);
    assert!(s.open("Main.hx").is_empty());
    let before = s.activity.len();
    let path = s.root.join("asset.bin");
    let next =
        std::fs::metadata(&path).unwrap().modified().unwrap() + std::time::Duration::from_secs(1);
    std::fs::write(&path, "same").unwrap();
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(next)
        .unwrap();
    let at = s.at("Main.hx", 0, source.find("value").unwrap() as u32);
    let hover = hover_text(&s.request("textDocument/hover", at));
    assert!(hover.contains("String"), "{hover}");
    assert!(
        !s.activity[before..]
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:?}",
        &s.activity[before..]
    );
}

#[test]
fn method_callbacks_use_structural_function_compatibility() {
    let source = r#"typedef Callback = Int->Void;
class Main {
 public function new() {}
 function listen(callback:Callback):Main { callback(1); return this; }
 static function main() {
  var owner = new Main();
  owner.listen((value:Int) -> {}).listen((value:Int) -> value + 1);
 }
}"#;
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn method_callbacks_reject_incompatible_parameter_types() {
    let source = r#"class Main {
 public function new() {}
 function listen(callback:Int->Void):Void callback(1);
 static function main() new Main().listen((value:String) -> {});
}"#;
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    assert!(
        !diagnostics.is_empty(),
        "incompatible callback was accepted"
    );
}

#[test]
fn discarded_conditionals_allow_different_branch_results() {
    let source = r#"class Main {
 static function interaction():Main return new Main();
 static function fill():Void {}
 public function new() {}
 static function main() {
  var scrolling = true;
  if (scrolling) { interaction(); } else fill();
  if (scrolling) { if (scrolling) interaction(); else fill(); } else fill();
  while (scrolling) { if (scrolling) interaction(); else fill(); break; }
  fill();
 }
}"#;
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn discarded_conditionals_still_check_conditions_and_branch_contents() {
    let source = r#"class Main {
 static function takesInt(value:Int):Void {}
 static function main() {
  if (123) takesInt("wrong"); else takesInt(0);
 }
}"#;
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    let messages = format!("{diagnostics:#?}");
    assert!(messages.contains("Bool"), "{messages}");
    assert!(
        messages.contains("String") && messages.contains("Int"),
        "{messages}"
    );
}

#[test]
fn conditional_values_still_require_compatible_branches() {
    let source = r#"class Main {
 static function main() {
  var condition = true;
  var value:Int = if (condition) 1 else "wrong";
 }
}"#;
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    let messages = format!("{diagnostics:#?}");
    assert!(
        messages.contains("String") && messages.contains("Int"),
        "{messages}"
    );
}

#[test]
fn inherited_generic_props_and_ordinary_closures_are_valid() {
    let props = "package models; typedef Props = {file:Void->String, ?label:String};";
    let component = "package models; class Component<P> { public var props:P; public function new(props:P) this.props = props; }";
    let source = "import models.Component;\nimport models.Props;\nclass Main extends Component<Props> {\n function render():String {\n  var serial = 0;\n  var cancel:Void->Void = () -> {};\n  Owner.onCleanup(() -> { serial++; cancel(); });\n  cancel = () -> {};\n  var label = props.label;\n  return props.file();\n }\n static function main() {}\n}";
    let owner =
        "class Owner { public static function onCleanup(callback:Void->Void):Void callback(); }";
    let mut s = Session::start(&[
        ("Main.hx", source),
        ("Owner.hx", owner),
        ("models/Component.hx", component),
        ("models/Props.hx", props),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 9, 17)));
    assert!(
        hover.contains("file") && !hover.contains("TypeAlias") && !hover.contains("SymbolId"),
        "{hover}"
    );
}

#[test]
fn build_traversal_does_not_label_ordinary_closures_as_expansions() {
    let macros = r#"import haxe.macro.Context;
import haxe.macro.Expr;
import haxe.macro.ExprTools;
class Builder {
 static function walk(e:Expr):Expr {
  return switch e.expr {
   case EConst(CInt("7")): macro 8;
   case _: ExprTools.map(e, walk);
  }
 }
 public static function build():Array<Field> {
  var fields = Context.getBuildFields();
  for (field in fields) switch field.kind {
   case FFun(fn): fn.expr = walk(fn.expr);
   case _:
  }
  return fields;
 }
}"#;
    let source = "@:build(Builder.build())\nclass Main {\n static function main() {\n  Owner.onCleanup(() -> { cleanup(); });\n  var marker = 7;\n }\n static function cleanup() {}\n}";
    let mut s = Session::start(&[
        ("Main.hx", source),
        ("Builder.hx", macros),
        (
            "Owner.hx",
            "class Owner { public static function onCleanup(callback:Void->Void):Void callback(); }",
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    for column in [9, 18, 25] {
        let preview = s.request("rayzor/expandMacro", s.at("Main.hx", 3, column));
        assert!(preview.is_null(), "{preview}");
        let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 3, column)));
        assert!(!hover.contains("Macro expansion"), "{hover}");
    }
    let preview = s.request("rayzor/expandMacro", s.at("Main.hx", 4, 15));
    assert!(
        preview["expansion"]
            .as_str()
            .is_some_and(|text| text.contains('8')),
        "{preview}"
    );
}

#[test]
fn standard_library_completion_includes_documentation() {
    let source = "class Main {\n    static function main() {\n        var text = \"hello\";\n        text.charAt(0);\n    }\n}";
    let mut s = Session::start(&[("Main.hx", source)]);
    s.open("Main.hx");
    let at = s.at("Main.hx", 3, 13);
    let items = s.request("textDocument/completion", at);
    let item = items
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["label"] == "charAt")
        .unwrap();
    assert!(
        item["documentation"]["value"]
            .as_str()
            .is_some_and(|d| d.contains("character")),
        "{item}"
    );
}

#[test]
fn definition_references_and_highlights_find_every_name() {
    let mut s = Session::start(&[("Main.hx", MAIN)]);
    s.open("Main.hx");

    let at = s.at("Main.hx", 5, 21);
    assert_eq!(start(&s.request("textDocument/definition", at)), (1, 20));
    let at = s.at("Main.hx", 6, 15);
    assert_eq!(start(&s.request("textDocument/definition", at)), (5, 12));

    let mut at = s.at("Main.hx", 6, 15);
    at["context"] = json!({"includeDeclaration": true});
    let refs = s.request("textDocument/references", at);
    let mut found: Vec<_> = refs.as_array().unwrap().iter().map(start).collect();
    found.sort();
    assert_eq!(found, vec![(5, 12), (6, 14)]);

    let at = s.at("Main.hx", 5, 13);
    let highlights = s.request("textDocument/documentHighlight", at);
    assert_eq!(highlights.as_array().map(Vec::len), Some(2), "{highlights}");
}

#[test]
fn navigation_crosses_files_in_the_workspace() {
    let util = "class Util {
    public static function twice(x:Int):Int {
        return x * 2;
    }
}
";
    let main = "class Main {
    static function main() {
        var n = Util.twice(3);
        trace(n);
    }
}
";
    let mut s = Session::start(&[("Util.hx", util), ("Main.hx", main)]);
    assert!(s.open("Main.hx").is_empty());

    let at = s.at("Main.hx", 2, 22);
    let def = s.request("textDocument/definition", at);
    assert!(def["uri"].as_str().unwrap().ends_with("/Util.hx"), "{def}");
    assert_eq!(start(&def), (1, 27));

    let mut at = s.at("Main.hx", 2, 22);
    at["context"] = json!({"includeDeclaration": true});
    let refs = s.request("textDocument/references", at);
    let files: Vec<_> = refs
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r["uri"]
                .as_str()
                .unwrap()
                .rsplit('/')
                .next()
                .unwrap()
                .to_string()
        })
        .collect();
    assert!(
        files.contains(&"Util.hx".into()) && files.contains(&"Main.hx".into()),
        "{refs}"
    );
}

#[test]
fn outline_tokens_and_completion_see_the_file() {
    let mut s = Session::start(&[("Main.hx", MAIN)]);
    s.open("Main.hx");
    let uri = s.uri("Main.hx");

    let symbols = s.request(
        "textDocument/documentSymbol",
        json!({"textDocument": {"uri": uri}}),
    );
    let names: Vec<_> = symbols
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    for name in ["Main", "add", "main"] {
        assert!(names.contains(&name), "{names:?}");
    }

    let tokens = s.request(
        "textDocument/semanticTokens/full",
        json!({"textDocument": {"uri": uri}}),
    );
    let data = tokens["data"].as_array().unwrap();
    // Five numbers per token: every declaration and use of add, a, b, total.
    assert!(data.len() / 5 >= 9, "{} tokens", data.len() / 5);

    let at = s.at("Main.hx", 6, 8);
    let items = s.request("textDocument/completion", at);
    let labels: Vec<_> = items
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    for name in ["total", "add"] {
        assert!(labels.contains(&name), "missing {name}");
    }
}

#[test]
fn diagnostics_point_at_the_error() {
    let src = "class Main {
    static function main() {
        undefinedThing(1);
    }
}
";
    let mut s = Session::start(&[("Main.hx", src)]);
    let diags = s.open("Main.hx");
    assert_eq!(diags.len(), 1, "{diags:?}");
    assert!(
        diags[0]["message"]
            .as_str()
            .unwrap()
            .contains("undefinedThing")
    );
    assert_eq!(start(&diags[0]), (2, 8));
}

#[test]
fn an_hxml_project_gets_its_class_paths_libraries_and_target() {
    let main = "class Main {
    static function main() {
        #if js
        var v = Lib.answer();
        #else
        var v = onlyWhenNotJs();
        #end
        trace(v);
    }
}
";
    let lib = "class Lib {
    public static function answer():Int {
        return 42;
    }
}
";
    let mut s = Session::start(&[
        (
            "build.hxml",
            "-cp src\n-lib mylib\n-main Main\n--js out.js\n",
        ),
        ("src/Main.hx", main),
        (".haxelib/mylib/.current", "1.0.0"),
        (
            ".haxelib/mylib/1,0,0/haxelib.json",
            r#"{"name":"mylib","classPath":"src"}"#,
        ),
        (".haxelib/mylib/1,0,0/src/Lib.hx", lib),
    ]);
    let diags = s.open("src/Main.hx");
    assert!(diags.is_empty(), "{diags:?}");

    let at = s.at("src/Main.hx", 3, 22);
    let def = s.request("textDocument/definition", at);
    let uri = def["uri"].as_str().unwrap_or("");
    assert!(uri.ends_with("/.haxelib/mylib/1,0,0/src/Lib.hx"), "{def}");
}

#[test]
fn an_hxml_hashlink_short_flag_applies_to_the_standard_library() {
    let source = "class Main {\n static function accept(value:Single):Void {}\n static function main() { accept(0.5); }\n}";
    let mut session = Session::start(&[("build.hxml", "-cp .\n-hl out.hl\n"), ("Main.hx", source)]);
    assert!(
        session.open("Main.hx").is_empty(),
        "{:?}",
        session.diagnostics
    );
    assert!(
        session
            .diagnostics
            .iter()
            .all(|update| update["diagnostics"].as_array().unwrap().is_empty()),
        "{:?}",
        session.diagnostics
    );
    let errors = session.change(
        "Main.hx",
        &source.replace("accept(0.5)", "accept(\"wrong\")"),
    );
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]["message"].as_str().unwrap().contains("Single"),
        "{errors:?}"
    );
}

const POINT: &str = "class Point {
    public var x:Int;
    public static var origin:Point = null;
    public function new(x:Int) {
        this.x = x;
    }
    public function moved(by:Int, scale:Float):Point {
        return new Point(x + by);
    }
    public static function make():Point {
        return new Point(0);
    }
}
class Main {
    static function main() {
        var p = Point.make();
        var s = \"hi\";
        trace(p.x + s.length);
    }
}
";

#[test]
fn completion_after_a_dot_lists_the_receivers_members_while_editing() {
    let mut s = Session::start(&[("Main.hx", POINT)]);
    assert!(s.open("Main.hx").is_empty());

    let editing = POINT.replace("trace(p.x + s.length);", "trace(p.mo");
    s.change("Main.hx", &editing);
    let line = 17;
    let col = "        trace(p.mo".len() as u32;
    let members = s.labels("Main.hx", line, col);
    assert!(
        members.contains(&"moved".into()) && members.contains(&"x".into()),
        "{members:?}"
    );
    assert!(
        !members.contains(&"make".into()) && !members.contains(&"class".into()),
        "{members:?}"
    );

    let editing = POINT.replace("trace(p.x + s.length);", "Point.");
    s.change("Main.hx", &editing);
    let statics = s.labels("Main.hx", line, "        Point.".len() as u32);
    assert!(
        statics.contains(&"make".into()) && statics.contains(&"origin".into()),
        "{statics:?}"
    );
    assert!(!statics.contains(&"moved".into()), "{statics:?}");

    let editing = POINT.replace("trace(p.x + s.length);", "s.");
    s.change("Main.hx", &editing);
    let string = s.labels("Main.hx", line, "        s.".len() as u32);
    assert!(
        string.contains(&"charAt".into()) && string.contains(&"length".into()),
        "{string:?}"
    );
}

#[test]
fn local_function_parameters_complete_names_and_imported_typedef_members() {
    let main = r#"import Actions.CenterAction;
class Main {
    static function main() {
        function row(a:CenterAction, recordingNow:Bool):String {
            return a.id;
        }
        row({id:"record", title:"Record"}, false);
    }
}
"#;
    let mut s = Session::start(&[
        ("Main.hx", main),
        (
            "Actions.hx",
            "class Actions {}\ntypedef CenterAction = {id:String, title:String};",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty());
    let names = s.labels("Main.hx", 4, 12);
    assert!(
        names.contains(&"a".into()) && names.contains(&"recordingNow".into()),
        "{names:?}"
    );
    let editing = main.replace("return a.id;", "a.");
    s.change("Main.hx", &editing);
    let members = s.labels("Main.hx", 4, "            a.".len() as u32);
    assert!(
        members.contains(&"id".into()) && members.contains(&"title".into()),
        "{members:?}"
    );
}

#[test]
fn signature_help_follows_the_argument_being_typed() {
    let mut s = Session::start(&[("Main.hx", POINT)]);
    s.open("Main.hx");
    let editing = POINT.replace("trace(p.x + s.length);", "p.moved(1, ");
    s.change("Main.hx", &editing);
    let at = s.at("Main.hx", 17, "        p.moved(1, ".len() as u32);
    let help = s.request("textDocument/signatureHelp", at);
    assert!(
        help["signatures"][0]["label"]
            .as_str()
            .unwrap_or("")
            .starts_with("moved("),
        "{help}"
    );
    assert_eq!(help["activeParameter"], 1, "{help}");
}

/// A build macro that never finishes, one that throws and one that recurses
/// without end each produce a diagnostic, and the server keeps answering.
#[test]
fn failing_macros_are_reported_and_the_server_survives() {
    let use_macro = |m: &str| {
        format!(
            "@:build({m}.build())\nclass Use{m} {{\n    static function main() {{\n        var x = 1;\n        trace(x);\n    }}\n}}\n"
        )
    };
    let field = "Array<haxe.macro.Expr.Field>";
    let macros = [
        (
            "MLoops",
            format!("macro static function build():{field} {{ while (true) {{}} return null; }}"),
        ),
        (
            "MCatches",
            format!(
                "macro static function build():{field} {{ while (true) try {{ while (true) {{}} }} catch (e:Dynamic) {{}} return null; }}"
            ),
        ),
        (
            "MThrows",
            format!("macro static function build():{field} {{ throw \"boom\"; }}"),
        ),
        (
            "MRecurses",
            format!(
                "macro static function build():{field} {{ return deep(0); }}\n    static function deep(n:Int):{field} {{ return deep(n + 1); }}"
            ),
        ),
    ];
    let mut files = Vec::new();
    for (name, body) in &macros {
        files.push((
            format!("{name}.hx"),
            format!("class {name} {{\n    {body}\n}}\n"),
        ));
        files.push((format!("Use{name}.hx"), use_macro(name)));
    }
    files.push(("Main.hx".into(), MAIN.into()));
    let borrowed: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let mut s = Session::start_with(&borrowed, json!({"macroBudgetMs": 300}));

    for (name, _) in &macros {
        let started = std::time::Instant::now();
        let previous = s.diagnostics.len();
        let diags = s.open(&format!("Use{name}.hx"));
        assert!(
            !diags.is_empty()
                || s.diagnostics[previous..]
                    .iter()
                    .any(|d| d["diagnostics"].as_array().is_some_and(|d| !d.is_empty())),
            "{name} gave no diagnostic"
        );
        assert!(
            started.elapsed().as_secs() < 10,
            "{name} took {:?}",
            started.elapsed()
        );
        // The class stays as written, so the rest of the file still answers.
        let at = s.at(&format!("Use{name}.hx"), 3, 12);
        let hover = s.request("textDocument/hover", at);
        assert!(hover_text(&hover).contains("var x:Int"), "{name}: {hover}");
    }
    s.open("Main.hx");
    let at = s.at("Main.hx", 1, 21);
    let hover = s.request("textDocument/hover", at);
    assert!(
        hover_text(&hover).contains("function add"),
        "server stopped answering: {hover}"
    );
}

#[test]
fn initialization_macros_transform_inline_markup_and_add_completions() {
    check_initialization_macro_editor_services(MacroProject::Hxml);
}

#[test]
fn discovered_haxelib_dependency_macros_transform_inline_markup() {
    check_initialization_macro_editor_services(MacroProject::InstalledLibrary);
}

#[test]
fn local_path_dependency_macros_transform_inline_markup() {
    check_initialization_macro_editor_services(MacroProject::LocalLibrary);
}

enum MacroProject {
    Hxml,
    InstalledLibrary,
    LocalLibrary,
}

fn check_initialization_macro_editor_services(project: MacroProject) {
    let main = "class Main {\n    #if eval\n    static function view():Int { var first = \"🦀\", second = 2; return <counter/>; }\n    static function main() { var count = view(); }\n    static var count(default, never):Int = 3;\n    static final LIMIT:Int = 10;\n    #end\n}\n";
    let macros = r#"#if macro
import haxe.macro.Context;
import haxe.macro.Expr;
#end
class Setup {
    public static function enable() {
        haxe.macro.Compiler.addGlobalMetadata("Main", "@:build(Setup.build())");
    }
    public static function build():Array<Field> {
        var fields = Context.getBuildFields();
        var fieldType = Array;
        if (!Std.isOfType(fields, fieldType)) Context.error("expected build fields", Context.currentPos());
        function lower(e:Expr):Expr {
            return switch e.expr {
                case EMeta({name: ":markup"}, _): macro 42;
                case _: haxe.macro.ExprTools.map(e, lower);
            };
        }
        for (field in fields) switch field.kind {
            case FFun(fn) if (fn.expr != null): fn.expr = lower(fn.expr);
            case _:
        }
        fields.push({name:"generated", access:[APublic, AStatic], kind:FVar(macro:Int, macro 7), pos:Context.currentPos()});
        return fields;
    }
}"#;
    let files = match project {
        MacroProject::InstalledLibrary => vec![
            ("haxelib.json", r#"{"dependencies":{"markup":"1.0.0"}}"#),
            (
                ".haxelib/markup/1,0,0/haxelib.json",
                r#"{"classPath":"src"}"#,
            ),
            (
                ".haxelib/markup/1,0,0/extraParams.hxml",
                "--macro Setup.enable()\n",
            ),
            (".haxelib/markup/1,0,0/src/Setup.hx", macros),
            ("Main.hx", main),
        ],
        MacroProject::LocalLibrary => vec![
            (
                "haxelib.json",
                r#"{"dependencies":{"markup":{"path":"./packages/markup"}}}"#,
            ),
            ("packages/markup/haxelib.json", r#"{"classPath":"src"}"#),
            (
                "packages/markup/extraParams.hxml",
                "--macro Setup.enable()\n",
            ),
            ("packages/markup/src/Setup.hx", macros),
            ("Main.hx", main),
        ],
        MacroProject::Hxml => vec![
            ("build.hxml", "--macro Setup.enable()\n"),
            ("Setup.hx", macros),
            ("Main.hx", main),
        ],
    };
    let mut s = Session::start(&files);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let view_line = main
        .lines()
        .position(|line| line.contains("function view"))
        .unwrap() as u32;
    let main_line = main
        .lines()
        .position(|line| line.contains("function main"))
        .unwrap() as u32;
    let labels = s.labels("Main.hx", main_line, 44);
    assert!(labels.iter().any(|l| l == "generated"), "{labels:?}");
    let hover = s.request("textDocument/hover", s.at("Main.hx", main_line, 41));
    assert!(!hover.is_null(), "missing hover after markup expansion");
    let line = main.lines().nth(view_line as usize).unwrap();
    let column = line[..line.find("<counter/>").unwrap()]
        .encode_utf16()
        .count() as u32;
    let preview = s.request("rayzor/expandMacro", s.at("Main.hx", view_line, column + 1));
    assert_eq!(preview["expansion"], "42", "{preview}");
    assert_eq!(
        preview["range"],
        json!({"start":{"line":view_line,"character":column},"end":{"line":view_line,"character":column + 10}})
    );
    assert!(
        s.request("rayzor/expandMacro", s.at("Main.hx", view_line, 30))
            .is_null()
    );
    assert!(
        s.request(
            "rayzor/expandMacro",
            s.at("Main.hx", view_line, column + 10)
        )
        .is_null()
    );
    let generated = s.request("rayzor/expandMacro", s.at("Main.hx", 0, 2));
    let generated = generated["expansion"].as_str().unwrap();
    assert!(generated.contains("generated"), "{generated}");
    assert!(
        !generated.contains("class Main")
            && !generated.contains("count(")
            && !generated.contains("view("),
        "{generated}"
    );
    let hints = s.request("textDocument/inlayHint", json!({"textDocument":{"uri":s.uri("Main.hx")},"range":{"start":{"line":view_line,"character":0},"end":{"line":view_line,"character":1000}}}));
    assert_eq!(hints.as_array().unwrap().len(), 1, "{hints}");
    assert_eq!(
        hints[0]["position"],
        json!({"line":view_line,"character":column+10})
    );
    assert!(
        preview["definition"]["uri"]
            .as_str()
            .unwrap()
            .ends_with("Setup.hx"),
        "{preview}"
    );
    assert_eq!(
        preview["definition"]["range"]["start"]["line"],
        macros
            .lines()
            .position(|line| line.contains("function build"))
            .unwrap() as u32,
        "{preview}"
    );
}

/// A markup macro answering display requests as Haxe macros do: after `<` a
/// field access on a type listing the tags, among attributes a structure
/// literal checked against the props.
#[test]
fn markup_macros_answer_display_requests_for_tags_and_attributes() {
    let macros = r#"#if macro
import haxe.macro.Compiler;
import haxe.macro.Context;
import haxe.macro.Expr;
#end
class Setup {
    public static function enable() {
        Compiler.addGlobalMetadata("Main", "@:build(Setup.build())");
    }
    public static function build():Array<Field> {
        var fields = Context.getBuildFields();
        function lower(e:Expr):Expr {
            return switch e.expr {
                case EMeta({name: ":markup"}, {expr: EConst(CString(source, _))}) if (Context.containsDisplayPosition(e.pos)):
                    var focus = displayed(e.pos, source);
                    macro {
                        $focus;
                        0;
                    };
                case EMeta({name: ":markup"}, _): macro 0;
                case _: haxe.macro.ExprTools.map(e, lower);
            };
        }
        for (field in fields) switch field.kind {
            case FFun(fn) if (fn.expr != null): fn.expr = lower(fn.expr);
            case _:
        }
        return fields;
    }
    static function displayed(pos:Position, source:String):Expr {
        // Typing a type defers the build until the typer runs, as Ashui's does.
        Context.getType("Props");
        var at = Compiler.getDisplayPos();
        var inside = source.substring(1, at.pos - Context.getPosInfos(pos).min);
        var here = Context.makePosition({file: at.file, min: at.pos, max: at.pos});
        if (~/^[a-z]*$/.match(inside))
            return {expr: EDisplay(macro @:pos(here) (null : ui.Tags), DKDot), pos: here};
        var literal:Expr = {expr: EDisplay({expr: EObjectDecl([]), pos: here}, DKStructure), pos: here};
        return {expr: ECheckType(literal, macro :Props), pos: here};
    }
    public static function tags():Array<Field> {
        return [for (name in ["card", "row"]) {name: name, access: [APublic], kind: FVar(macro :Bool), pos: Context.currentPos()}];
    }
}"#;
    let main = "typedef Props = {title:String, ?wide:Bool};\nclass Main {\n    static function view():Int return <card title=\"a\" ></card>;\n    static function tag():Int return <ca></ca>;\n    static function main() { view(); tag(); var x = 1 < 2; }\n}\n";
    let mut s = Session::start(&[
        ("build.hxml", "--macro Setup.enable()\n"),
        ("Setup.hx", macros),
        (
            "ui/Tags.hx",
            "package ui;\n\n/** Fields named after the tags. **/\n@:build(Setup.tags())\nclass Tags {}\n",
        ),
        ("Main.hx", main),
    ]);
    s.open("Main.hx");
    let at = |needle: &str| {
        let line = main.lines().position(|l| l.contains(needle)).unwrap();
        let column = main.lines().nth(line).unwrap().find(needle).unwrap() + needle.len();
        (line as u32, column as u32)
    };
    let mut complete = |(line, column): (u32, u32), trigger: Option<&str>| {
        let mut params = s.at("Main.hx", line, column);
        if let Some(trigger) = trigger {
            params["context"] = json!({"triggerKind": 2, "triggerCharacter": trigger});
        }
        let items = s.request("textDocument/completion", params);
        let mut labels: Vec<String> = items
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["label"].as_str().unwrap().to_string())
            .collect();
        labels.sort();
        labels
    };
    assert_eq!(complete(at("title=\"a\" "), Some(" ")), ["title", "wide"]);
    assert_eq!(complete(at("<ca"), None), ["card", "row"]);
    assert!(complete(at("1 <"), Some("<")).is_empty());
}

/// A file a macro read changes after analysis, as a library's shared cache
/// can: requests are answered from the last analysis while a new one runs.
#[test]
fn changed_macro_inputs_are_revalidated_without_holding_requests() {
    let macros = r#"import haxe.macro.Context;
class Setup {
    public static function enable() {
        haxe.macro.Compiler.addGlobalMetadata("Main", "@:build(Setup.build())");
    }
    public static function build() {
        var fields = Context.getBuildFields();
        var text = sys.io.File.getContent(Context.resolvePath("data.txt"));
        fields.push({name: "generated", access: [APublic, AStatic], kind: FVar(macro :Int, macro $v{text.length}), pos: Context.currentPos()});
        return fields;
    }
}"#;
    let main = "class Main {\n    static function main() { var count = Main.generated; }\n}\n";
    let mut s = Session::start(&[
        ("build.hxml", "--macro Setup.enable()\n"),
        ("Setup.hx", macros),
        ("data.txt", "first"),
        ("Main.hx", main),
    ]);
    assert!(s.open("Main.hx").is_empty());
    let at = s.at("Main.hx", 1, 34);
    assert!(hover_text(&s.request("textDocument/hover", at.clone())).contains("count"));
    std::fs::write(s.root.join("data.txt"), "changed").unwrap();
    s.activity.clear();
    assert!(hover_text(&s.request("textDocument/hover", at)).contains("count"));
    assert!(
        !s.activity
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:#?}",
        s.activity
    );
    s.wait_ready("Main.hx", 1);
    assert!(
        s.activity
            .iter()
            .any(|activity| activity["state"] == "checking"),
        "{:#?}",
        s.activity
    );
}

#[test]
fn global_build_without_changes_has_no_expansion_preview() {
    let macros = r#"import haxe.macro.Context;
class Setup {
    public static function enable() {
        haxe.macro.Compiler.addGlobalMetadata("KeyMap", "@:build(Setup.build())");
    }
    public static function build() return Context.getBuildFields();
}"#;
    let main = "class KeyMap {\n public static function encode(key:String):String return key;\n static var count(default, never):Int = 3;\n}";
    let mut s = Session::start(&[
        ("build.hxml", "--macro Setup.enable()\n"),
        ("Setup.hx", macros),
        ("KeyMap.hx", main),
    ]);
    assert!(s.open("KeyMap.hx").is_empty(), "{:#?}", s.diagnostics);
    for (line, column) in [(0, 2), (0, 7), (1, 26), (2, 13)] {
        assert!(
            s.request("rayzor/expandMacro", s.at("KeyMap.hx", line, column))
                .is_null()
        );
    }
    let hover = s.request("textDocument/hover", s.at("KeyMap.hx", 1, 26));
    assert!(!hover_text(&hover).contains("Macro expansion"), "{hover}");
    let hints = s.request("textDocument/inlayHint", json!({"textDocument":{"uri":s.uri("KeyMap.hx")},"range":{"start":{"line":0,"character":0},"end":{"line":4,"character":0}}}));
    assert!(
        hints.is_null() || hints.as_array().is_some_and(Vec::is_empty),
        "{hints}"
    );
}

#[test]
fn nested_macro_helpers_generate_fields_and_expansions() {
    let macros = r#"import haxe.macro.Context;
import haxe.macro.Expr;
class DepthMacros {
    #if macro
    public static function enable() {
        if (!MapDepth.check()) throw "initialization maps";
        haxe.macro.Compiler.addGlobalMetadata("Main", "@:build(DepthMacros.build())");
        Context.onGenerate(function(_) { if (!MapDepth.check()) throw "generation maps"; });
    }
    public static function build():Array<Field> {
        if (!MapDepth.check()) throw "build maps";
        var fields = Context.getBuildFields();
        fields.push({name:"generated", access:[APublic, AStatic], kind:FVar(macro:Bool, macro true), pos:Context.currentPos()});
        return fields;
    }
    #end
    public static macro function value():Expr {
        if (!MapDepth.check()) throw "expression maps";
        return macro true;
    }
}"#;
    let main = "class Main {\n static function main() { var value = DepthMacros.value(); }\n}";
    let mut s = Session::start(&[
        ("build.hxml", "--macro DepthMacros.enable()\n"),
        (
            "MapDepth.hx",
            include_str!("../../compiler/tests/haxe/macro_map_depth/MapDepth.hx"),
        ),
        ("DepthMacros.hx", macros),
        ("Main.hx", main),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:#?}", s.diagnostics);
    assert!(
        s.labels("Main.hx", 1, 36)
            .iter()
            .any(|label| label == "generated")
    );
    let column = main
        .lines()
        .nth(1)
        .unwrap()
        .find("DepthMacros.value")
        .unwrap() as u32;
    let preview = s.request("rayzor/expandMacro", s.at("Main.hx", 1, column + 12));
    assert_eq!(preview["expansion"], "true", "{preview}");
}

#[test]
fn macro_resources_use_project_class_paths_in_every_phase() {
    let main = "class Main {\n    static function main() { var text = Assets.text(); }\n}\n";
    let macros = r#"import haxe.macro.Context;
import haxe.macro.Expr;
import sys.FileSystem;
import sys.io.File;
class Assets {
    #if macro
    public static function enable() {
        haxe.macro.Compiler.define("asset_file", "value.txt");
        if (read() != "asset value") throw "initialization resource";
        haxe.macro.Compiler.addGlobalMetadata("Main", "@:build(Assets.build())");
        Context.onGenerate(function(_) {
            var file = Context.resolvePath(Context.definedValue("asset_root") + "/" + Context.definedValue("asset_file"));
            if (File.getContent(file) != "asset value") throw "generation resource";
        });
    }
    static function read():String {
        #if !asset_root
        throw "missing asset_root condition";
        #end
        if (!Context.defined("asset_root")) throw "missing macro defines";
        if (Context.getClassPath().length == 0) throw "no class paths";
        var directory = Context.resolvePath(Context.definedValue("asset_root"));
        if (!FileSystem.isDirectory(directory)) throw "not a directory";
        var entries = FileSystem.readDirectory(directory);
        if (entries.length != 1 || entries[0] != "value.txt") throw "directory entries";
        var file = Context.resolvePath(Context.definedValue("asset_root") + "/" + Context.definedValue("asset_file"));
        Context.registerModuleDependency(Context.getLocalModule(), file);
        if (FileSystem.isDirectory(file)) throw "not a file";
        if (!FileSystem.exists(FileSystem.fullPath(file))) throw "full path";
        if (Context.resolvePath(FileSystem.fullPath(file)) != FileSystem.fullPath(file)) throw "absolute resource";
        var missing = FileSystem.absolutePath(directory + "/missing.txt");
        if (FileSystem.absolutePath("") != FileSystem.absolutePath(".")) throw "empty absolute path";
        if (FileSystem.absolutePath("unused/../macro-missing.txt") != FileSystem.absolutePath("macro-missing.txt")) throw "relative normalization";
        var unchanged = FileSystem.fullPath(directory) + "/unused/../missing.txt";
        if (FileSystem.absolutePath(unchanged) != unchanged) throw "absolute path changed";
        if (FileSystem.exists(missing)) throw "missing exists";
        if (FileSystem.isDirectory(missing)) throw "missing directory exists";
        var failed = false;
        try Context.resolvePath("icons/missing.txt") catch (_:Dynamic) failed = true;
        if (!failed) throw "missing resource did not throw";
        failed = false;
        try FileSystem.readDirectory(missing) catch (_:Dynamic) failed = true;
        if (!failed) throw "missing directory did not throw";
        return File.getContent(file);
    }
    public static function build():Array<Field> {
        var fields = Context.getBuildFields();
        fields.push({name:"asset", access:[APublic, AStatic], kind:FVar(macro:String, Context.makeExpr(read(), Context.currentPos())), pos:Context.currentPos()});
        return fields;
    }
    #end
    public static macro function text():Expr {
        return Context.makeExpr(read(), Context.currentPos());
    }
}"#;
    let mut s = Session::start(&[
        (
            "build.hxml",
            "-cp src\n-cp decoy\n-cp resources\n-D asset_root=icons\n--macro Assets.enable()\n",
        ),
        ("decoy/icons/value.txt", "wrong asset"),
        ("resources/icons/value.txt", "asset value"),
        ("src/Assets.hx", macros),
        ("src/Main.hx", main),
    ]);
    let diagnostics = s.open("src/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let labels = s.labels("src/Main.hx", 1, 40);
    assert!(labels.iter().any(|label| label == "asset"), "{labels:?}");
    let preview = s.request("rayzor/expandMacro", s.at("src/Main.hx", 1, 43));
    assert!(
        preview["expansion"]
            .as_str()
            .unwrap_or("")
            .contains("asset value"),
        "{preview}"
    );
}

#[test]
#[ignore = "requires the sibling RayDE workspace and its libraries"]
fn rayde_image_preview_preserves_types_and_stdlib_calls() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../RayDE")
        .canonicalize()
        .unwrap();
    let mut s = Session::connect(root, json!({"macroBudgetMs":5000}), false);
    let name = "src/rayde/ui/ImagePreview.hx";
    let diagnostics = s.open(name);
    assert!(
        !diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|m| m.contains("Cannot find type 'haxe.io'"))),
        "{diagnostics:#?}"
    );
    let source = std::fs::read_to_string(s.root.join(name)).unwrap();
    for (needle, expected) in [
        ("ImageViewState", "typedef ImageViewState = {panX:Float"),
        ("Path.extension", "function extension"),
    ] {
        let (line, text) = source
            .lines()
            .enumerate()
            .find(|(_, text)| text.contains(needle))
            .unwrap();
        let column = text.find(needle).unwrap() + if needle == "Path.extension" { 6 } else { 1 };
        let hover =
            hover_text(&s.request("textDocument/hover", s.at(name, line as u32, column as u32)));
        assert!(hover.contains(expected), "{needle}: {hover}");
    }
}

#[test]
#[ignore = "requires the sibling RayDE workspace and its libraries"]
fn rayde_macro_workspace_keeps_editor_services() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../RayDE")
        .canonicalize()
        .unwrap();
    let mut s = Session::connect(root, json!({"macroBudgetMs": 5000}), false);
    let name = "src/rayde/ui/Welcome.hx";
    let diagnostics = s.open(name);
    eprintln!(
        "RayDE diagnostics ({}): {:#?}",
        diagnostics.len(),
        &diagnostics[..diagnostics.len().min(20)]
    );
    assert!(!diagnostics.iter().any(|d| {
        d["message"]
            .as_str()
            .is_some_and(|m| m.contains("expected expression, found '<'"))
    }));
    let text = std::fs::read_to_string(s.root.join(name)).unwrap();
    let line = text
        .lines()
        .position(|l| l.contains("function darkScheme()"))
        .unwrap() as u32;
    let column = text
        .lines()
        .nth(line as usize)
        .unwrap()
        .find("darkScheme")
        .unwrap() as u32;
    let hover = s.request("textDocument/hover", s.at(name, line, column + 1));
    eprintln!(
        "RayDE hover: {hover}; completions: {:?}",
        s.labels(name, line, column)
    );
    assert!(
        hover_text(&hover).contains("function darkScheme"),
        "missing typed hover in macro-enabled file: {hover}"
    );
    let labels = s.labels(name, line, column);
    assert!(
        labels.iter().any(|l| l == "darkScheme"),
        "missing local completions"
    );
    let symbols = s.request(
        "textDocument/documentSymbol",
        json!({"textDocument":{"uri":s.uri(name)}}),
    );
    assert!(!symbols.is_null(), "missing document symbols");
}

#[test]
fn macro_previews_hints_and_definition_errors_have_source_positions() {
    let macros = "import haxe.macro.Expr;\nclass Macros {\n public static macro function value():Expr return macro 21 + 21;\n public static macro function broken():Expr { throw \"macro failed\"; }\n}";
    let main = "class Main {\n static function main() { var result = Macros.value(); }\n}";
    let mut s = Session::start(&[("Macros.hx", macros), ("Main.hx", main)]);
    assert!(s.open("Main.hx").is_empty());
    let column = main.lines().nth(1).unwrap().find("Macros.value").unwrap() as u32;
    let preview = s.request("rayzor/expandMacro", s.at("Main.hx", 1, column + 8));
    assert_eq!(preview["expansion"], "21 + 21", "{preview}");
    assert!(
        preview["definition"]["uri"]
            .as_str()
            .unwrap()
            .ends_with("Macros.hx")
    );
    let hints = s.request("textDocument/inlayHint", json!({"textDocument":{"uri":s.uri("Main.hx")},"range":{"start":{"line":0,"character":0},"end":{"line":3,"character":0}}}));
    assert_eq!(hints.as_array().unwrap().len(), 1, "{hints}");
    assert_eq!(hints[0]["position"]["line"], 1);
    let hover = s.request("textDocument/hover", s.at("Main.hx", 1, column + 8));
    assert!(hover_text(&hover).contains("21 + 21"), "{hover}");
    let edited = main
        .replace("Macros.value()", "ordinaryValue")
        .replace("class Main", "class MainNew");
    let edited = edited.strip_suffix('}').unwrap();
    assert!(!s.change("Main.hx", edited).is_empty());
    assert!(
        s.request("textDocument/hover", s.at("Main.hx", 0, 7))
            .is_null()
    );
    assert!(
        s.request("rayzor/expandMacro", s.at("Main.hx", 1, column + 8))
            .is_null()
    );
    let stale_hints = s.request("textDocument/inlayHint", json!({"textDocument":{"uri":s.uri("Main.hx")},"range":{"start":{"line":0,"character":0},"end":{"line":4,"character":0}}}));
    assert!(stale_hints.is_null(), "{stale_hints}");
    assert!(s.change("Main.hx", main).is_empty());
    let before = s.diagnostics.len();
    s.change(
        "Main.hx",
        &main.replace("Macros.value()", "Macros.broken()"),
    );
    let error = s.diagnostics[before..]
        .iter()
        .find(|d| d["uri"] == s.uri("Macros.hx"))
        .expect("macro declaration diagnostic");
    assert_eq!(
        error["diagnostics"][0]["range"]["start"]["line"], 3,
        "{error}"
    );
    assert_eq!(
        error["diagnostics"][0]["relatedInformation"][0]["location"]["uri"],
        s.uri("Main.hx")
    );
    assert_eq!(error["diagnostics"][0]["severity"], 1);
    assert!(
        error["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("macro failed")
    );
}

#[test]
fn missing_libraries_and_failed_documents_keep_the_session_alive() {
    let main = "class Main {\n /** The useful field. */ public static var available:Int = 42;\n static function main() {}\n}";
    let mut s = Session::start(&[
        (
            "build.hxml",
            "-lib rayzor_lsp_test_missing_a04356\n-lib rayzor_lsp_test_missing_b04356\n",
        ),
        (
            "Broken.hx",
            "class Broken { static function f() return Unknown.field; }",
        ),
        ("Main.hx", main),
    ]);
    assert!(!s.open("Broken.hx").is_empty());
    assert!(
        s.logs
            .iter()
            .any(|message| message.contains("rayzor_lsp_test_missing_a04356")),
        "{:?}",
        s.logs
    );
    assert!(
        s.logs
            .iter()
            .any(|message| message.contains("rayzor_lsp_test_missing_b04356")),
        "{:?}",
        s.logs
    );
    assert!(s.open("Main.hx").is_empty());
    assert!(
        s.labels("Main.hx", 2, 15)
            .iter()
            .any(|label| label == "available")
    );
    let column = main.lines().nth(1).unwrap().find("available").unwrap() as u32;
    let hover = s.request("textDocument/hover", s.at("Main.hx", 1, column + 1));
    assert!(hover_text(&hover).contains("The useful field"), "{hover}");
    s.change("Main.hx", &main.replace("available", "updated"));
    assert!(
        s.labels("Main.hx", 2, 15)
            .iter()
            .any(|label| label == "updated")
    );
}

#[test]
fn generic_receivers_nullable_strings_and_optional_local_calls_are_valid() {
    let main = r#"class Main {
    static function identity<T>(value:T):T return value;
    static function main() {
        var seen = new Map<String, Bool>();
        function add(path:Null<String>, ?name:String):Void {
            if (path == null || path == "" || "" == path || !sys.FileSystem.exists(path)) return;
            var label = name != null ? name : StringTools.replace(haxe.io.Path.withoutDirectory(path), ".exe", "");
            if (seen.exists(label)) return;
            seen.set(label, true);
        }
        add(Sys.getEnv("SHELL"));
        add("/bin/sh", "sh");
        var value = identity(true);
        var child = new Child();
        child.store(value);
        var callback:Dynamic = function() return 1;
        callback();
        var bytes = haxe.io.Bytes.ofString("text");
    }
}
class Parent<T> {
    public function new() {}
    public function store(value:T):Void {}
}
class Child extends Parent<Bool> {}
"#;
    let mut s = Session::start(&[("Main.hx", main)]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn chained_calls_convert_generic_abstract_arguments() {
    let main = r#"import controls.Identity;
class Main {
    static function main() {
        Identity.of().setClasses(["hover-docs"]);
    }
}"#;
    let mut s = Session::start(&[
        ("Main.hx", main),
        (
            "controls/Identity.hx",
            r#"package controls;
class Identity {
    public function new() {}
    public static function of():Identity return new Identity();
    public function setClasses(value:Reactive<Array<String>>):Void {}
}
abstract Reactive<T>(T) {
    @:from public static function constant<T>(value:T):Reactive<T> return cast value;
}"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn fully_qualified_project_types_work_without_an_import() {
    let main = r#"class Main {
    static function main() {
        var scroll = widgets.input.Scroll.of(1);
        scroll.move(2);
    }
}"#;
    let mut s = Session::start(&[
        ("Main.hx", main),
        (
            "widgets/input/Scroll.hx",
            r#"package widgets.input;
class Scroll {
    public function new() {}
    public static function of(id:Int):Scroll return new Scroll();
    public function move(value:Int):Void {}
}"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn build_macros_load_helpers_and_their_macro_only_dependencies() {
    let mut s = Session::start(&[
        (
            "Main.hx",
            "@:build(Build.fields())\nclass Main { static function main() { var value = generated; } }",
        ),
        (
            "Build.hx",
            r#"class Build {
    #if macro
    public static function fields():Array<haxe.macro.Expr.Field> {
        var fields = haxe.macro.Context.getBuildFields();
        var name = Helpers.name();
        fields.push({name:name, access:[APublic, AStatic], kind:FVar(macro:String, macro "generated"), pos:haxe.macro.Context.currentPos()});
        return fields;
    }
    #end
}"#,
        ),
        (
            "Helpers.hx",
            r#"class Helpers {
    #if macro
    public static function name():String {
        var quoted = macro 'test$id';
        if (!quoted.expr.match(EConst(CString("test$id", SingleQuotes)))) throw "quote identity";
        if (String.fromCharCode(233) != "é" || "é".charCodeAt(0) != 233
            || StringTools.unsafeCodeAt("é", 0) != 233 || StringTools.fastCodeAt("x", 2) != -1) throw "character codes";
        if (Math.fround(-1.5) != -1 || Math.sqrt(9) != 3 || Math.cos(0) != 1 || !Math.isFinite(Math.PI)) throw "math primitives";
        if (StringTools.htmlEscape("<&", true) != "&lt;&amp;") throw "stdlib fallback";
        var iterator = [7, 9].iterator();
        if (!iterator.hasNext() || iterator.next() != 7 || iterator.next() != 9 || iterator.hasNext()) throw "array iterator";
        var counter = new Counter();
        for (item in counter) { if (item != 0) throw "iterator start"; break; }
        if (counter.next() != 1) throw "iterator mutation";
        if (State.After != 4) throw "abstract constants";
        var document = Xml.parse('<asset name="generated"><child/></asset>');
        var root = document.firstElement();
        if (root.nodeName != "asset" || root.elements().next().nodeName != "child") throw "XML traversal";
        var result = Payload(root.get("name"));
        return switch result { case Ready: "wrong branch"; case Payload(name): name; };
    }
    #end
}
private enum abstract State(Int) { var Before = 3; var After; }
private enum Result { Ready; Payload(name:String); }
private class Counter {
    var current = 0;
    public function new() {}
    public function hasNext():Bool return true;
    public function next():Int return current++;
}
"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{:#?}", s.diagnostics);
    assert!(s.labels("Main.hx", 1, 30).contains(&"generated".into()));
}

#[test]
fn string_buffers_work_in_build_macros_with_either_standard_library() {
    for project in ["rayzor.toml", "build.hxml"] {
        let mut s = Session::start(&[
            (
                project,
                if project == "rayzor.toml" {
                    "[project]\nname = 'macro-string-buffer'\n"
                } else {
                    "-cp .\n-main Main\n"
                },
            ),
            (
                "Main.hx",
                "@:build(Build.fields())\nclass Main { static function main() { var value = generated; } }",
            ),
            (
                "Build.hx",
                r#"import haxe.macro.Expr;
class Build {
    #if macro
    public static function fields():Array<Field> {
        var buffer = new StringBuf();
        buffer.add("é");
        buffer.addChar(88);
        buffer.addSub("abcd", 1, 2);
        buffer.add(null);
        if (buffer.toString() != "éXbcnull" || buffer.length != 8) throw "string buffer content";
        buffer.addSub("abcd", 2);
        if (buffer.toString() != "éXbcnullcd" || buffer.length != 10) throw "string buffer substring";
        var fields = haxe.macro.Context.getBuildFields();
        fields.push({name:"generated", access:[APublic, AStatic], kind:FVar(macro:String, macro "ok"), pos:haxe.macro.Context.currentPos()});
        return fields;
    }
    #end
}"#,
            ),
        ]);
        let diagnostics = s.open("Main.hx");
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        assert!(s.labels("Main.hx", 1, 30).contains(&"generated".into()));
    }
}

#[test]
fn instantiated_receivers_still_reject_wrong_values_and_required_arguments() {
    let main = r#"class Main {
    static function main() {
        var seen = new Map<String, Bool>();
        seen.set("bad", 123);
        function add(path:Null<String>, ?name:String):Void {}
        add();
        var number = 123;
        number();
    }
}"#;
    let mut s = Session::start(&[("Main.hx", main)]);
    let diagnostics = s.open("Main.hx");
    for line in [3, 5, 7] {
        assert!(
            diagnostics
                .iter()
                .any(|d| d["range"]["start"]["line"] == line),
            "missing error on line {line}: {diagnostics:#?}"
        );
    }
    assert!(
        diagnostics
            .iter()
            .filter_map(|d| d["message"].as_str())
            .all(|message| !message.contains("expected `T`")),
        "{diagnostics:#?}"
    );
}

#[test]
fn initialization_failures_belong_to_configuration_and_survive_cache_reuse() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp .\n--macro Setup.install()\n"),
        (
            "Setup.hx",
            "class Setup { public static function install() { throw \"setup failed\"; } }",
        ),
        ("Main.hx", "class Main { public static var value:Int = 1; }"),
        (
            "Other.hx",
            "class Other { public static var value:Int = 2; }",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty());
    let config = s.uri("build.hxml");
    assert!(
        s.diagnostics.iter().any(|update| update["uri"] == config
            && update["diagnostics"]
                .as_array()
                .is_some_and(|diags| diags.iter().any(|diag| diag["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("setup failed"))
                    && diag["range"]["start"]["line"] == 1))),
        "{:?}",
        s.diagnostics
    );
    assert!(s.open("Other.hx").is_empty());
    let root = s.root.clone();
    s.temporary = false;
    drop(s);
    let mut restarted = Session::connect(root, json!({}), true);
    assert!(restarted.open("Main.hx").is_empty());
    assert!(
        restarted
            .activity
            .iter()
            .all(|event| event["state"] != "checking")
    );
    assert!(
        restarted
            .diagnostics
            .iter()
            .any(|update| update["uri"] == config
                && update["diagnostics"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|diagnostic| {
                        diagnostic["message"]
                            .as_str()
                            .is_some_and(|message| message.contains("setup failed"))
                    })),
        "{:?}",
        restarted.diagnostics
    );
    let uri = restarted.uri("Main.hx");
    restarted.notify("textDocument/didClose", json!({"textDocument":{"uri":uri}}));
    restarted.request("workspace/symbol", json!({"query":"Main"}));
    let remaining = restarted
        .diagnostics
        .iter()
        .rfind(|update| update["uri"] == config)
        .unwrap()["diagnostics"]
        .as_array()
        .unwrap();
    assert_eq!(remaining.len(), 1, "{remaining:?}");
    assert_eq!(remaining[0]["severity"], 2, "{remaining:?}");
    assert!(
        remaining[0]["message"]
            .as_str()
            .unwrap()
            .contains("No Haxe target"),
        "{remaining:?}"
    );
}

#[test]
fn enum_constructor_and_static_method_values_do_not_reenter_lookup() {
    let main = r#"import haxe.io.Bytes;
import haxe.io.BytesInput;
class Main {
    static function main() {
        var input = new BytesInput(Bytes.ofString("value"));
        var readByte = input.readByte;
        var alias = Target;
        var Target = alias;
        var read = Target.value;
        var enumType = Choice;
        var choice = enumType.Item;
    }
}
class Target { public static function value():Int return 42; }
enum Choice { Item; }"#;
    let mut s = Session::start(&[("Main.hx", main)]);
    let diagnostics = s.open("Main.hx");
    assert!(
        !diagnostics.iter().any(|item| item["message"]
            .as_str()
            .unwrap_or("")
            .contains("internal compiler error")),
        "{diagnostics:#?}"
    );
    let symbols = s.request(
        "textDocument/documentSymbol",
        json!({"textDocument":{"uri":s.uri("Main.hx")}}),
    );
    assert!(!symbols.as_array().unwrap().is_empty());
    s.change(
        "Main.hx",
        "class Main { public static var alive:Int = 42; }",
    );
    assert!(
        s.labels("Main.hx", 0, 30)
            .iter()
            .any(|label| label == "alive")
    );
}

#[test]
fn module_enum_constructors_precede_foreign_ambient_type_names() {
    let source = "package app;\nimport foreign.Tools;\nprivate enum Event { Message(text:String); }\nclass Main { static function main() {\n var constructor = Message;\n var event = constructor(\"ok\");\n} }";
    let mut session = Session::start(&[
        ("app/Main.hx", source),
        (
            "foreign/Tools.hx",
            "package foreign; enum Message { Info; } class Tools {}",
        ),
    ]);
    let diagnostics = session.open("app/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let hover = hover_text(&session.request("textDocument/hover", session.at("app/Main.hx", 4, 6)));
    assert!(
        hover.contains("String") && hover.contains("Event"),
        "{hover}"
    );
    assert!(!hover.contains("foreign.Message"), "{hover}");
}

#[test]
fn rayzor_manifest_uses_native_stdlib_defines_and_declared_sources() {
    let source = "import rayzor.concurrent.Channel;\nclass Main {\n#if enabled\npublic static function selected():Int return 7;\n#else\npublic static function selected():String return 'wrong';\n#end\n#if disabled\npublic var fail:MissingDisabledType;\n#end\n#if jit\npublic function channel(value:Channel<Int>):Channel<Int> return value;\n#end\npublic var value:Int = selected();\n}";
    let mut session = Session::start(&[
        (
            "rayzor.toml",
            "[project]\nname = 'native'\ntype = 'library'\n[build]\nclass-paths = ['src']\n[build.defines]\nenabled = true\ndisabled = false\n",
        ),
        ("build.hxml", "-cp wrong\n-D disabled\n"),
        ("src/Main.hx", source),
    ]);
    let diagnostics = session.open("src/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("src/Main.hx", 3, 28)));
    assert!(hover.contains("selected():Int"), "{hover}");
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("src/Main.hx", 11, 32)));
    assert!(hover.contains("Channel"), "{hover}");
}

#[test]
fn rayzor_manifest_source_dependencies_supply_types_and_documentation() {
    let mut session = Session::start(&[
        (
            "rayzor.toml",
            "[project]\nname = 'consumer'\n[build]\nclass-paths = ['src']\n[dependencies]\nhelper = {path = 'deps/helper'}\n",
        ),
        (
            "src/Main.hx",
            "class Main { public var value:Helper; public function new(value:Helper) this.value = value; }",
        ),
        (
            "deps/helper/rayzor.toml",
            "[project]\nname = 'helper'\ntype = 'library'\n[build]\nclass-paths = ['src']\n",
        ),
        (
            "deps/helper/src/Helper.hx",
            "/** A local native-project dependency. */ class Helper { public function new() {} }",
        ),
    ]);
    let diagnostics = session.open("src/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let source = std::fs::read_to_string(session.root.join("src/Main.hx")).unwrap();
    let column = source.find("Helper").unwrap() as u32;
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("src/Main.hx", 0, column)));
    assert!(
        hover.contains("class Helper") && hover.contains("local native-project dependency"),
        "{hover}"
    );
}

#[test]
fn workspace_members_keep_defines_and_persisted_hovers_separate() {
    let source = "class Main {\n#if number\npublic var value:Int = 7;\n#else\npublic var value:String = 'text';\n#end\n}";
    let mut session = Session::start(&[
        (
            "rayzor.toml",
            "[workspace]\nmembers = ['one', 'two']\n[workspace.cache]\ndir = '.indexes'\n",
        ),
        (
            "one/rayzor.toml",
            "[project]\nname = 'one'\n[build]\nclass-paths = ['src']\n[build.defines]\nnumber = true\n",
        ),
        (
            "two/rayzor.toml",
            "[project]\nname = 'two'\n[build]\nclass-paths = ['src']\n",
        ),
        ("one/src/Main.hx", source),
        ("two/src/Main.hx", source),
    ]);
    for (file, line, expected) in [
        ("one/src/Main.hx", 2, "value:Int"),
        ("two/src/Main.hx", 4, "value:String"),
        ("one/src/Main.hx", 2, "value:Int"),
    ] {
        assert!(session.open(file).is_empty(), "{:#?}", session.diagnostics);
        let hover = hover_text(&session.request("textDocument/hover", session.at(file, line, 12)));
        assert!(hover.contains(expected), "{file}: {hover}");
    }
    let root = session.root.clone();
    session.temporary = false;
    drop(session);
    let mut restarted = Session::connect(root, json!({}), true);
    for (file, line, expected) in [
        ("one/src/Main.hx", 2, "value:Int"),
        ("two/src/Main.hx", 4, "value:String"),
    ] {
        assert!(restarted.open(file).is_empty());
        let hover =
            hover_text(&restarted.request("textDocument/hover", restarted.at(file, line, 12)));
        assert!(hover.contains(expected), "{file}: {hover}");
    }
    assert!(
        !restarted
            .activity
            .iter()
            .any(|event| event["state"] == "checking"),
        "{:#?}",
        restarted.activity
    );
}

#[test]
fn changing_a_manifest_rechecks_open_documents_with_new_defines() {
    let manifest = "[project]\nname = 'reload'\n[build]\nclass-paths = ['src']\n[build.defines]\nnumber = true\n";
    let source = "class Main {\n#if number\npublic var value:Int = 7;\n#else\npublic var value:String = 'text';\n#end\n}";
    let mut session = Session::start(&[("rayzor.toml", manifest), ("src/Main.hx", source)]);
    assert!(session.open("src/Main.hx").is_empty());
    std::fs::write(
        session.root.join("rayzor.toml"),
        manifest.replace("number = true", "number = false"),
    )
    .unwrap();
    session.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":session.uri("rayzor.toml"),"type":2}]}),
    );
    assert!(session.wait_ready("src/Main.hx", 1).is_empty());
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("src/Main.hx", 4, 12)));
    assert!(hover.contains("value:String"), "{hover}");
}

#[test]
fn rayzor_manifest_delegates_to_hxml_and_reloads_included_defines() {
    let source = "class Main {\n#if number\npublic var value:Int = 7;\n#else\npublic var value:String = 'text';\n#end\n}";
    let mut session = Session::start(&[
        (
            "rayzor.toml",
            "[project]\nname = 'delegated'\nhxml = 'config/build.hxml'\n",
        ),
        ("config/build.hxml", "-cp ../src\nflags.hxml\n"),
        ("config/flags.hxml", "-D number\n"),
        ("src/Main.hx", source),
    ]);
    assert!(
        session.open("src/Main.hx").is_empty(),
        "{:#?}",
        session.diagnostics
    );
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("src/Main.hx", 2, 12)));
    assert!(hover.contains("value:Int"), "{hover}");
    std::fs::write(session.root.join("config/flags.hxml"), "# removed\n").unwrap();
    session.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":session.uri("config/flags.hxml"),"type":2}]}),
    );
    assert!(session.wait_ready("src/Main.hx", 1).is_empty());
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("src/Main.hx", 4, 12)));
    assert!(hover.contains("value:String"), "{hover}");
}

#[test]
fn disabled_manifest_cache_keeps_hovers_without_writing_an_index() {
    let mut session = Session::start(&[
        (
            "rayzor.toml",
            "[project]\nname = 'uncached'\n[cache]\nenabled = false\ndir = '.custom-cache'\n",
        ),
        ("Main.hx", "class Main { public var value:Int = 7; }"),
    ]);
    assert!(session.open("Main.hx").is_empty());
    let hover = hover_text(&session.request("textDocument/hover", session.at("Main.hx", 0, 25)));
    assert!(hover.contains("value:Int"), "{hover}");
    assert!(!session.root.join(".custom-cache/editor").exists());
}

#[test]
fn missing_manifest_dependencies_are_reported_without_losing_hints() {
    let mut session = Session::start(&[
        (
            "rayzor.toml",
            "[project]\nname = 'incomplete'\n[build]\nnative-libs = ['missing-plugin.dylib']\n[dependencies]\nmissing = {path = 'missing.rpkg'}\n",
        ),
        ("Main.hx", "class Main { public var value:Int = 7; }"),
    ]);
    assert!(session.open("Main.hx").is_empty());
    let hover = hover_text(&session.request("textDocument/hover", session.at("Main.hx", 0, 25)));
    assert!(hover.contains("value:Int"), "{hover}");
    let config = session.uri("rayzor.toml");
    let errors = session
        .diagnostics
        .iter()
        .filter(|update| update["uri"] == config)
        .flat_map(|update| update["diagnostics"].as_array().unwrap())
        .map(|error| error["message"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(
        errors.iter().any(|error| error.contains("missing-plugin")),
        "{errors:?}"
    );
    assert!(
        errors.iter().any(|error| error.contains("missing.rpkg")),
        "{errors:?}"
    );
}

#[test]
fn packaged_sources_are_indexed_without_loading_native_code() {
    let mut setup = Session::start(&[
        (
            "rayzor.toml",
            "[project]\nname = 'packaged'\n[build]\nclass-paths = ['src']\n[dependencies]\nfixture = {path = 'fixture.rpkg'}\n",
        ),
        ("src/Main.hx", "class Main { public var value:Provided; }"),
    ]);
    let mut package = compiler::rpkg::pack::RpkgBuilder::new("fixture");
    package.add_haxe_source(
        "Provided.hx",
        "/** Packaged declaration documentation. */ class Provided { public var value:Int; }",
    );
    package.add_method_table("rayzor_fixture_plugins", &[]);
    // Metadata and source analysis must not attempt dlopen on this payload.
    package.add_native_lib(
        b"not executable",
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    package.write(&setup.root.join("fixture.rpkg")).unwrap();
    let root = setup.root.clone();
    setup.temporary = false;
    drop(setup);
    let mut session = Session::connect(root, json!({}), true);
    assert!(
        session.open("src/Main.hx").is_empty(),
        "{:?}",
        session.diagnostics
    );
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("src/Main.hx", 0, 32)));
    assert!(
        hover.contains("class Provided") && hover.contains("Packaged declaration documentation"),
        "{hover}"
    );
    assert!(
        !session.logs.iter().any(|log| log.contains("dlopen")),
        "{:?}",
        session.logs
    );
}

#[test]
#[cfg(unix)]
fn native_manifest_plugins_do_not_run_runtime_initialization() {
    let mut setup = Session::start(&[("Main.hx", "class Main { public var value:Int = 7; }")]);
    let marker = setup.root.join("runtime-initialized");
    let source = setup.root.join("plugin.c");
    std::fs::write(
        &source,
        format!(
            r#"
#include <stddef.h>
#include <stdio.h>
unsigned int rayzor_plugin_abi_version(void) {{ return {}; }}
void *plugin_init(size_t *count) {{
    FILE *file = fopen({}, "w");
    if (file) fclose(file);
    *count = 0;
    return NULL;
}}
"#,
            compiler::plugin::ABI_VERSION,
            serde_json::to_string(marker.to_str().unwrap()).unwrap()
        ),
    )
    .unwrap();
    let library = setup.root.join(if cfg!(target_os = "macos") {
        "libfixture.dylib"
    } else {
        "libfixture.so"
    });
    let output = std::process::Command::new("cc")
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .arg("-fPIC")
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::write(
        setup.root.join("rayzor.toml"),
        format!(
            "[project]\nname = 'native-plugin'\n[build]\nnative-libs = [{}]\n",
            serde_json::to_string(library.file_name().unwrap().to_str().unwrap()).unwrap()
        ),
    )
    .unwrap();
    let root = setup.root.clone();
    setup.temporary = false;
    drop(setup);
    let mut session = Session::connect(root, json!({}), true);
    assert!(session.open("Main.hx").is_empty());
    let hover = hover_text(&session.request("textDocument/hover", session.at("Main.hx", 0, 25)));
    assert!(hover.contains("value:Int"), "{hover}");
    assert!(!marker.exists(), "editor analysis ran plugin_init");
    assert!(session.logs.is_empty(), "{:?}", session.logs);
}

#[test]
fn workspace_folders_load_and_replace_separate_project_settings() {
    let source = "class Main {\n#if number\npublic var value:Int = 7;\n#else\npublic var value:String = 'text';\n#end\n}";
    let mut setup = Session::start(&[
        (
            "one/rayzor.toml",
            "[project]\nname = 'one'\n[build.defines]\nnumber = true\n",
        ),
        ("two/rayzor.toml", "[project]\nname = 'two'\n"),
        ("one/Main.hx", source),
        ("one/RemovedOnly.hx", "class RemovedOnly {}"),
        ("two/Main.hx", source),
    ]);
    let root = setup.root.clone();
    setup.temporary = false;
    drop(setup);
    let folders = vec![root.join("one")];
    let mut session = Session::connect_folders(root, json!({}), true, Some(folders));
    assert!(session.open("one/Main.hx").is_empty());
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("one/Main.hx", 2, 12)));
    assert!(hover.contains("value:Int"), "{hover}");
    session.notify(
        "workspace/didChangeWorkspaceFolders",
        json!({"event":{"added":[{"uri":session.uri("two"),"name":"two"}],"removed":[]}}),
    );
    assert!(session.wait_ready("one/Main.hx", 1).is_empty());
    assert!(session.open("two/Main.hx").is_empty());
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("two/Main.hx", 4, 12)));
    assert!(hover.contains("value:String"), "{hover}");
    session.notify(
        "workspace/didChangeWorkspaceFolders",
        json!({"event":{"added":[],"removed":[{"uri":session.uri("one"),"name":"one"}]}}),
    );
    assert!(session.wait_ready("one/Main.hx", 1).is_empty());
    let hover =
        hover_text(&session.request("textDocument/hover", session.at("one/Main.hx", 2, 12)));
    assert!(
        hover.contains("value:Int"),
        "removed folder inherited another project's flags: {hover}"
    );
    // A request is a barrier for the notification; workspace search must stop
    // advertising declarations from the removed folder.
    let symbols = session.request("workspace/symbol", json!({"query":"RemovedOnly"}));
    assert!(symbols.as_array().unwrap().is_empty(), "{symbols}");
}

#[test]
fn saving_a_manifest_rechecks_haxe_documents_without_treating_toml_as_haxe() {
    let manifest = "[project]\nname = 'saved'\n[build.defines]\nnumber = true\n";
    let source = "class Main {\n#if number\npublic var value:Int = 7;\n#else\npublic var value:String = 'text';\n#end\n}";
    let mut session = Session::start(&[("rayzor.toml", manifest), ("Main.hx", source)]);
    assert!(session.open("Main.hx").is_empty());
    let updated = manifest.replace("true", "false");
    std::fs::write(session.root.join("rayzor.toml"), &updated).unwrap();
    session.notify(
        "textDocument/didSave",
        json!({"textDocument":{"uri":session.uri("rayzor.toml")},"text":updated}),
    );
    assert!(session.wait_ready("Main.hx", 1).is_empty());
    let hover = hover_text(&session.request("textDocument/hover", session.at("Main.hx", 4, 12)));
    assert!(hover.contains("value:String"), "{hover}");
    assert!(
        !session
            .activity
            .iter()
            .any(|event| event["uri"] == session.uri("rayzor.toml"))
    );
}

#[test]
fn nue_manifest_keeps_shared_tensor_and_map_identities() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../nue")
        .canonicalize()
        .unwrap();
    let mut session = Session::connect(root, json!({}), false);
    for file in ["nue/Module.hx", "nue/model/NamedTensorMap.hx"] {
        let diagnostics = session.open(file);
        assert!(diagnostics.is_empty(), "{file}: {diagnostics:#?}");
        let source = std::fs::read_to_string(session.root.join(file)).unwrap();
        let offset = source
            .rfind(if file.ends_with("Module.hx") {
                "function forward(x:Tensor)"
            } else {
                "var tensorByName:StringMap<Tensor>"
            })
            .unwrap();
        let offset = offset + source[offset..].find("Tensor").unwrap();
        let before = &source[..offset];
        let line = before.bytes().filter(|byte| *byte == b'\n').count() as u32;
        let column = before.rsplit('\n').next().unwrap().len() as u32;
        let hover =
            hover_text(&session.request("textDocument/hover", session.at(file, line, column)));
        assert!(
            hover.contains("Tensor") && !hover.contains("Unknown") && !hover.contains("SymbolId"),
            "{file}: {hover}"
        );
    }
}

#[test]
fn project_types_keep_imported_identities_when_short_names_collide() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp .\n-main Main\n"),
        (
            "Main.hx",
            r#"import payload.Bytes;
import native.Driver;
class Main {
    static var driver:Driver;
    public static function create():Bytes {
        var out = Bytes.alloc(8);
        return out;
    }
}"#,
        ),
        (
            "payload/Bytes.hx",
            "package payload; class Bytes { public function new() {} public static function alloc(size:Int):Bytes return new Bytes(); }",
        ),
        (
            "native/Bytes.hx",
            "package native; abstract Bytes(Int) from Int to Int {}",
        ),
        (
            "native/Driver.hx",
            "package native; class Driver { public var data:Bytes; public function new() {} }",
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 4, 29)));
    assert!(hover.contains("create():payload.Bytes"), "{hover}");
}

#[test]
fn project_types_allow_declared_abstract_conversions_through_aliases() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp .\n-main Main\n"),
        (
            "types/Types.hx",
            r#"package types;
typedef Number = Count;
abstract Count(Int) from Int to Int {}
abstract Label(String) {
    @:from public static function ofString(value:String):Label return cast value;
    @:to public function asString():String return this;
}
class Types {}"#,
        ),
        (
            "Main.hx",
            r#"import types.Types.Number;
import types.Types.Label;
class Main {
    static function number():Number return 7;
    static function raw(value:Number):Int return value;
    static function label():Label return "text";
    static function text(value:Label):String return value;
    static function main() { var value:Number = 3; var text:Label = "value"; }
}"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn project_defines_keep_values_in_syntax_indexing_imports_and_macros() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp .\nflags.hxml\n--macro Setup.enable()\n"),
        (
            "flags.hxml",
            "-D window-lib=0.3.1\n-D theme=dark\n-D level=4\n",
        ),
        (
            "Setup.hx",
            r#"class Setup {
#if macro
    public static function enable() {
        if (!haxe.macro.Context.defined("window_lib")) throw "missing normalized define";
        if (haxe.macro.Context.definedValue("window_lib") != "0.3.1") throw "missing version value";
#if (theme != "dark" || level < 4)
        throw "wrong macro branch";
#end
    }
#end
}"#,
        ),
        (
            "Helper.hx",
            r#"#if (window_lib == "0.3.1" && theme == "dark" && level >= 4)
class Helper { public static function value():Int return 7; }
#else
class Helper { public static function value():String return "wrong"; }
#end"#,
        ),
        (
            "Main.hx",
            r#"class Main {
#if (theme == "dark" && level >= 4)
    public var value:Int = Helper.value();
#else
    this is deliberately not valid syntax
#end
}"#,
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 2, 15)));
    assert!(hover.contains("value:Int"), "{hover}");
    std::fs::write(
        s.root.join("flags.hxml"),
        "-D window-lib=0.3.1\n-D theme=light\n-D level=2\n",
    )
    .unwrap();
    // A configuration change must invalidate the old branch's cached index.
    s.notify(
        "workspace/didChangeWatchedFiles",
        json!({"changes":[{"uri":s.uri("flags.hxml"),"type":2}]}),
    );
    assert!(
        !s.wait_ready("Main.hx", 1).is_empty(),
        "stale diagnostics after changing -D values"
    );
}

#[test]
fn project_types_reject_undeclared_and_wrong_generic_abstract_conversions() {
    let mut s = Session::start(&[
        (
            "Types.hx",
            "abstract Cell<T>(T) from T to T {} abstract Locked(Int) {} class Types {}",
        ),
        (
            "Main.hx",
            r#"import Types.Cell;
import Types.Locked;
class Main {
    static function good():Cell<String> return "text";
    static function wrong():Cell<String> return 1;
    static function locked():Locked return 1;
    static function reinterpreted():Locked return cast 1;
}"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert_eq!(diagnostics.len(), 2, "{diagnostics:#?}");
    assert!(
        diagnostics
            .iter()
            .all(|d| d["message"].as_str().unwrap().contains("Type mismatch")),
        "{diagnostics:#?}"
    );
}

#[test]
fn project_types_accept_generic_abstract_instances_and_nullable_values() {
    let mut s = Session::start(&[
        (
            "Container.hx",
            "abstract Container<T>(Array<T>) { public function new(length:Int) this = []; }",
        ),
        (
            "Main.hx",
            r#"class Main {
    static function take(value:Container<Int>):Int return 1;
    static function count(value:Null<Int>):Int return value != null ? value * 4 : 0;
    static function main() { var value = new Container<Int>(3); take(value); }
}"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn project_types_index_abstracts_through_their_declared_key_types() {
    let mut s = Session::start(&[
        (
            "Lookup.hx",
            r#"abstract Lookup<K>(Array<Int>) {
    public function new() this = [];
    @:arrayAccess public function read(key:K):Int return 7;
}"#,
        ),
        (
            "Main.hx",
            r#"class Main {
    static function main() { var lookup = new Lookup<String>(); var result = lookup["key"]; }
}"#,
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
    let diagnostics = s.change("Main.hx", "class Main { static function main() { var lookup = new Lookup<String>(); var result = lookup[true]; } }");
    assert!(!diagnostics.is_empty(), "invalid abstract key was accepted");
    assert!(
        diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|m| m.contains("String") && m.contains("Bool"))),
        "{diagnostics:#?}"
    );
}

#[test]
fn package_siblings_resolve_in_static_calls_and_field_chains() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp src\n"),
        (
            "src/widgets/Size.hx",
            "package widgets; enum abstract Size(String) to String { var Md = 'md'; }",
        ),
        (
            "src/widgets/State.hx",
            "package widgets; enum State { Open; Closed; }",
        ),
        (
            "src/widgets/Library.hx",
            "package widgets; class Library { public static function part():Int return 7; }",
        ),
        (
            "src/widgets/Menu.hx",
            "package widgets; class Menu { public static var registry:{near:()->Int} = {near:() -> 3}; }",
        ),
        (
            "src/widgets/Panel.hx",
            "package widgets; class Panel { public function new() {} public function run():Int { var optional:Null<Size> = null; var size:String = (optional == null ? Size.Md : optional : String); var state:State = State.Open; return Library.part() + Menu.registry.near() + widgets.Library.part(); } }",
        ),
    ]);
    let diagnostics = s.open("src/widgets/Panel.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn unrelated_conditional_branches_still_report_a_type_mismatch() {
    let mut s = Session::start(&[(
        "Main.hx",
        "class Main { public static function main() { var value = true ? 'text' : false; } }",
    )]);
    let diagnostics = s.open("Main.hx");
    assert!(
        diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|message| message.contains("Type mismatch")
                && message.contains("String")
                && message.contains("Bool"))),
        "{diagnostics:#?}"
    );
}

#[test]
fn expression_macro_hovers_expose_the_generated_value_type() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp src\n"),
        (
            "src/widgets/Sheet.hx",
            "package widgets; class Sheet { public function new() {} public var count:Int = 7; }",
        ),
        (
            "src/widgets/Builder.hx",
            "package widgets;\n#if macro\nimport haxe.macro.Expr;\n#end\nclass Builder { public static macro function make():ExprOf<Sheet> return macro new widgets.Sheet(); }",
        ),
        (
            "src/Main.hx",
            "import widgets.Builder;\nclass Main { public static function main() { var sheet = Builder.make(); var count:Int = sheet.count; } }",
        ),
    ]);
    let diagnostics = s.open("src/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let text = std::fs::read_to_string(s.root.join("src/Main.hx")).unwrap();
    let column = text.lines().nth(1).unwrap().find("make(").unwrap() + 1;
    let hover = hover_text(&s.request("textDocument/hover", s.at("src/Main.hx", 1, column as u32)));
    assert!(
        hover.contains("Sheet") && !hover.contains("ExprOf"),
        "{hover}"
    );
}

#[test]
fn static_callback_parameters_and_captured_locals_shadow_instance_fields() {
    let source = r#"class Main<T, R> {
 final work:Int = 0;
 final ready:Int = 0;
 final value:String = 'field';
 public function new() {}
 static function run<T, R>(value:T, work:(T, Void->Bool)->R, ready:Null<Void->Void>):R {
  var invoke = () -> perform(value, work, ready);
  return invoke();
 }
 static function perform<T, R>(value:T, work:(T, Void->Bool)->R, ready:Null<Void->Void>):R {
  var result = work(value, () -> true);
  if (ready != null) ready();
  return result;
 }
 static function locals():Int {
  var work = (value:Int, check:Void->Bool) -> value;
  var ready:Null<Void->Void> = () -> {};
  var value = 4;
  var invoke = () -> { if (ready != null) ready(); return work(value, () -> true); };
  var wrap = (work:Void->Int) -> () -> work();
  return wrap(invoke)();
 }
 static function main() {
  var result:Int = run(42, (value, check) -> value, () -> {});
  var local:Int = locals();
 }
}"#;
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
    let line = source.lines().nth(6).unwrap();
    let at = s.at("Main.hx", 6, line.find("work,").unwrap() as u32 + 1);
    let hover = hover_text(&s.request("textDocument/hover", at.clone()));
    assert!(
        hover.contains("work:") && hover.contains("Bool") && hover.contains("R"),
        "{hover}"
    );
    let definition = s.request("textDocument/definition", at);
    assert_eq!(definition["range"]["start"]["line"], 5, "{definition}");
    assert_eq!(
        definition["range"]["start"]["character"],
        source.lines().nth(5).unwrap().find("work:").unwrap(),
        "{definition}"
    );
}

#[test]
fn nullable_callback_aliases_keep_their_argument_and_return_types() {
    let source = "typedef Callback<T> = Null<T->Int>;\nclass Main { static function invoke(callback:Callback<String>):Int { if (callback != null) { var result = callback('ready'); return result; } return 0; } static function main() {} }";
    let mut s = Session::start(&[("Main.hx", source)]);
    assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
    let at = s.at(
        "Main.hx",
        1,
        source.lines().nth(1).unwrap().find("result =").unwrap() as u32 + 1,
    );
    let hover = hover_text(&s.request("textDocument/hover", at));
    assert!(hover.contains("result:Int"), "{hover}");
    for invalid in ["callback(42)", "callback()", "callback('ready', 'extra')"] {
        let diagnostics = s.change("Main.hx", &source.replace("callback('ready')", invalid));
        assert!(
            diagnostics.iter().any(|d| d["message"]
                .as_str()
                .is_some_and(|message| message.contains("Type mismatch"))),
            "{invalid}: {diagnostics:?}"
        );
    }
}

#[test]
fn real_instance_fields_still_fail_in_static_closures() {
    let source = "class Main { var value:Int = 1; var work:Void->Int = () -> 1; static function direct():Int return value; static function capture():Void->Int return () -> work(); static function qualified():Int return Main.value; static function main() {} }";
    let mut s = Session::start(&[("Main.hx", source)]);
    let diagnostics = s.open("Main.hx");
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| {
            d["message"].as_str().is_some_and(|message| {
                message.contains("Instance member") && message.contains("static context")
            })
        })
        .collect();
    assert_eq!(errors.len(), 3, "{diagnostics:?}");
}

#[test]
fn runtime_callback_names_shadow_macro_aliases_inside_closures() {
    let mut s = Session::start(&[
        (
            "Generator.hx",
            "class Generator { public static macro function choose() return macro 'global'; }",
        ),
        (
            "Main.hx",
            "class Main { static function capture(choose:Void->Int):Void->Int return () -> choose(); public static function main() { var evaluate = (choose:Void->Int) -> choose(); var number:Int = evaluate(() -> 42); { var choose = () -> 43; var callback = () -> choose(); var other:Int = callback(); } var global:String = Generator.choose(); } }",
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
}

#[test]
fn nested_expression_macros_preserve_the_generated_type_and_preview() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp src\n"),
        (
            "src/widgets/Sheet.hx",
            "package widgets; class Sheet { public function new() {} public var count:Int = 7; }",
        ),
        (
            "src/widgets/Builder.hx",
            "package widgets;\n#if macro\nimport haxe.macro.Expr;\n#end\nclass Builder { public static macro function outer(template:Expr):Expr return macro @:generated widgets.Builder.inner($template); public static macro function inner(template:Expr):Expr return macro new widgets.Sheet(); }",
        ),
        (
            "src/Main.hx",
            "class Main { public static function main() { var sheet = widgets.Builder.outer('body'); var count:Int = sheet.count; } }",
        ),
    ]);
    let diagnostics = s.open("src/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let text = std::fs::read_to_string(s.root.join("src/Main.hx")).unwrap();
    let column = text.find("sheet =").unwrap() as u32 + 1;
    let hover = hover_text(&s.request("textDocument/hover", s.at("src/Main.hx", 0, column)));
    assert!(
        hover.contains("Sheet") && !hover.contains("Dynamic"),
        "{hover}"
    );
    let column = text.find("outer(").unwrap() as u32 + 1;
    let preview = s.request("rayzor/expandMacro", s.at("src/Main.hx", 0, column));
    let expansion = preview["expansion"].as_str().unwrap_or_default();
    assert!(
        expansion.contains("new widgets.Sheet") && !expansion.contains("Builder.inner"),
        "{preview}"
    );
}

#[test]
fn unavailable_expression_macros_keep_their_declared_runtime_type() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp src\n"),
        (
            "src/widgets/Sheet.hx",
            "package widgets; class Sheet { public function new() {} public var count:Int = 7; }",
        ),
        (
            "src/widgets/Builder.hx",
            "package widgets;\n#if macro\nimport haxe.macro.Expr;\n#end\nclass Builder { public static macro function make(template:Expr):ExprOf<Sheet> { throw 'unavailable dependency'; } }",
        ),
        (
            "src/Main.hx",
            "import widgets.Builder; import widgets.Sheet;\nclass Main { public static function main() { var sheet:Sheet = Builder.make('body'); var other:Sheet = Builder.make('other'); var count:Int = sheet.count; } }",
        ),
    ]);
    let diagnostics = s.open("src/Main.hx");
    assert!(
        diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|m| m.contains("unavailable dependency"))),
        "{diagnostics:#?}"
    );
    assert!(
        !diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|m| m.contains("Type mismatch") || m.contains("Cannot find"))),
        "{diagnostics:#?}"
    );
    let text = std::fs::read_to_string(s.root.join("src/Main.hx")).unwrap();
    let column = text.lines().nth(1).unwrap().find("make(").unwrap() + 1;
    let hover = hover_text(&s.request("textDocument/hover", s.at("src/Main.hx", 1, column as u32)));
    assert!(
        hover.contains("function make") && hover.contains("Sheet") && !hover.contains("ExprOf"),
        "{hover}"
    );
}

#[test]
fn native_manifest_recovered_macros_keep_their_runtime_signature() {
    let mut s = Session::start(&[
        (
            "rayzor.toml",
            "[project]\nname = 'macro-signatures'\n[cache]\nenabled = false\n",
        ),
        (
            "Sheet.hx",
            "class Sheet { public function new() {} public var count:Int = 7; }",
        ),
        (
            "Builder.hx",
            "#if macro\nimport haxe.macro.Expr;\n#end\nclass Builder { public static macro function make(template:Expr):ExprOf<Sheet> { throw 'unavailable dependency'; } }",
        ),
        (
            "Main.hx",
            "class Main { public static function main() { var sheet:Sheet = Builder.make('body'); var other:Sheet = Builder.make('other'); var count:Int = sheet.count; } }",
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(
        diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|message| message.contains("unavailable dependency"))),
        "{diagnostics:#?}"
    );
    assert!(
        !diagnostics
            .iter()
            .any(|d| d["message"].as_str().is_some_and(|message| message
                .contains("Type mismatch")
                || message.contains("Cannot find"))),
        "{diagnostics:#?}"
    );
    let source = std::fs::read_to_string(s.root.join("Main.hx")).unwrap();
    let at = source.find("make(").unwrap() as u32 + 1;
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 0, at)));
    assert!(
        hover.contains("Sheet") && !hover.contains("Expr"),
        "{hover}"
    );
}

#[test]
fn recovered_typed_macros_still_check_the_payload_type() {
    for configuration in [
        ("build.hxml", "-cp .\n"),
        (
            "rayzor.toml",
            "[project]\nname = 'typed-macro-signatures'\n[cache]\nenabled = false\n",
        ),
    ] {
        let mut s = Session::start(&[
            configuration,
            (
                "Builder.hx",
                "#if macro\nimport haxe.macro.Expr;\n#end\nclass Builder { public static macro function make(value:ExprOf<Int>):ExprOf<Int> { throw 'unavailable dependency'; } }",
            ),
            (
                "Main.hx",
                "class Main { public static function main() { var count:Int = Builder.make('wrong'); } }",
            ),
        ]);
        let diagnostics = s.open("Main.hx");
        assert!(
            diagnostics.iter().any(|d| d["message"]
                .as_str()
                .is_some_and(|message| message == "Type mismatch: expected `Int`, found `String`")),
            "{diagnostics:#?}"
        );
        assert!(
            !diagnostics
                .iter()
                .any(|d| d["message"].as_str().is_some_and(|message| message
                    .contains("Type mismatch")
                    && message.contains("Expr"))),
            "{diagnostics:#?}"
        );
    }
}

#[test]
fn nested_library_sources_keep_the_build_flags_and_resolve_siblings() {
    let mut s = Session::start(&[
        ("build.hxml", "-cp src\n-D feature=enabled\n"),
        (
            "lib/haxelib.json",
            r#"{"name":"widgets","classPath":"haxe"}"#,
        ),
        (
            "lib/haxe/widgets/Library.hx",
            "package widgets; class Library { public static function part():Int return 7; }",
        ),
        (
            "lib/haxe/widgets/Size.hx",
            "package widgets; enum abstract Size(String) to String { var Md = 'md'; }",
        ),
        (
            "lib/haxe/widgets/Panel.hx",
            "package widgets;\n#if (feature == 'enabled')\nclass Panel { public function run():Int { var size:String = Size.Md; return Library.part(); } }\n#else\ninvalid syntax\n#end",
        ),
    ]);
    let diagnostics = s.open("lib/haxe/widgets/Panel.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let source = std::fs::read_to_string(s.root.join("lib/haxe/widgets/Panel.hx")).unwrap();
    let column = source.lines().nth(2).unwrap().find("Library.part").unwrap() as u32 + 1;
    let hover = hover_text(&s.request(
        "textDocument/hover",
        s.at("lib/haxe/widgets/Panel.hx", 2, column),
    ));
    assert!(hover.contains("Library"), "{hover}");
}

#[test]
fn quoted_arrow_parameters_reify_names_and_types() {
    let mut s = Session::start(&[
        (
            "Builder.hx",
            "#if macro\nimport haxe.macro.Expr;\n#end\nclass Builder { public static macro function make():Expr { var name = 'number'; var hint = macro : Int; return macro ($name:$hint) -> $i{name} + 1; } }",
        ),
        (
            "Main.hx",
            "class Main { public static function main() { var increment = Builder.make(); var result:Int = increment(3); } }",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
}

#[test]
fn lambda_find_accepts_arrays_as_iterables() {
    let source = "class Element { public function new() {} }\nclass Summary extends Element { public function new() { super(); } }\ntypedef Stream<T> = { function iterator():Iterator<T>; }\nclass Main {\n    static function main() {\n        var children:Array<Dynamic> = [new Summary()];\n        var summary:Null<Element> = Lambda.find(children, c -> Std.isOfType(c, Summary));\n        var numbers:Array<Int> = [1, 2, 3];\n        var found:Null<Int> = Lambda.find(numbers, n -> n > 1);\n        var stream:Stream<Int> = numbers;\n        var iterable:Iterable<Int> = numbers;\n    }\n}";
    for configuration in [
        ("build.hxml", "-cp .\n"),
        (
            "rayzor.toml",
            "[project]\nname = 'iterable-test'\n[cache]\nenabled = false\n",
        ),
    ] {
        let mut s = Session::start(&[configuration, ("Main.hx", source)]);
        let diagnostics = s.open("Main.hx");
        assert!(
            diagnostics.is_empty(),
            "{}: {diagnostics:#?}",
            configuration.0
        );
    }
}

#[test]
fn iterable_arguments_still_check_structure_and_element_types() {
    let mut s = Session::start(&[
        (
            "custom/Iterable.hx",
            "package custom; class Iterable<T> { public function new() {} }",
        ),
        (
            "Main.hx",
            "typedef Stream<T> = { function iterator():Iterator<T>; }\nclass Main {\n    static function ints(values:Iterable<Int>) {}\n    static function main() {\n        ints(['wrong']);\n        var invalid:Stream<Int> = ['wrong'];\n        var notAnIterator:Iterator<Int> = [1, 2];\n        var wrongMethod:Iterable<Int> = {iterator:() -> 'wrong'};\n        var nominal:custom.Iterable<Int> = [1, 2];\n    }\n}",
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    for line in 4..=8 {
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic["range"]["start"]["line"] == line
                    && diagnostic["message"]
                        .as_str()
                        .is_some_and(|message| message.contains("Type mismatch"))
            }),
            "line {line}: {diagnostics:#?}"
        );
    }
}

#[test]
fn compact_generics_in_imported_modules_preserve_nullable_receiver_fields() {
    let source = "import JamSession.Position;\nclass Main { final file:String; final session:Void->Null<JamSession>; public function new(file:String, session:Void->Null<JamSession>) { this.file=file; this.session=session; } function current():Null<JamSession> { var current=session(); return current != null && current.file == file ? current : null; } static function main() {} }";
    let mut s = Session::start(&[
        ("Main.hx", source),
        (
            "JamSession.hx",
            "typedef Position = { id:String }; class JamSession { public final file:String; public function new(file:String) this.file=file; function refresh():Void { var marks:Array<String>=[], positions:Array<Position>=[], nested:Array<Array<Int>>=[[]], deep:Array<Array<Array<Int>>>=[[[]]]; } }",
        ),
    ]);
    assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
    let line = source.lines().nth(1).unwrap();
    let column = line.find("current=session").unwrap() as u32 + 1;
    let hover = hover_text(&s.request("textDocument/hover", s.at("Main.hx", 1, column)));
    assert!(hover.contains("current:Null<JamSession>"), "{hover}");
    let column = line.find("current.file").unwrap() as u32 + "current.".len() as u32 + 1;
    let position = s.at("Main.hx", 1, column);
    let hover = hover_text(&s.request("textDocument/hover", position.clone()));
    assert!(hover.contains("file:String"), "{hover}");
    let definition = s.request("textDocument/definition", position);
    assert!(
        definition.to_string().contains("JamSession.hx"),
        "{definition}"
    );
    let diagnostics = s.change(
        "Main.hx",
        &source.replace("? current : null", "? current.file : null"),
    );
    assert!(
        diagnostics.iter().any(|diagnostic| diagnostic["message"]
            .as_str()
            .is_some_and(|message| message.contains("Type mismatch"))),
        "returning the String field as a session must still fail: {diagnostics:?}"
    );
}

#[test]
fn markup_tags_resolve_to_generated_classes_and_survive_cache_reload() {
    let source = "@:build(Wrapper.build()) class Main { static function main() { var value = Builder.identity(7); var panel = <arbitrary-name><clickable/></arbitrary-name>; } }";
    let builder = r#"
#if macro
import haxe.macro.Context;
import haxe.macro.Expr;
#end
class Builder {
    public static macro function identity(e:Expr):Expr {
        var type = Context.typeof(e);
        return e;
    }
    public static macro function make(template:Expr):Expr {
        var type = try Context.getType("widgets.Panel") catch (_:Dynamic) null;
        var cls = switch type { case TInst(c, _): c.get(); case _: throw "missing generated class"; }
        if (cls.superClass == null || cls.superClass.params.length != 1) throw "missing generic superclass";
        switch Context.follow(cls.superClass.params[0]) {
            case TAnonymous(a): if (a.get().fields[0].name != "label") throw "missing props";
            case _: throw "expected structural props";
        }
        var action = switch Context.getType("widgets.Controls.Action") { case TInst(c, _): c.get(); case _: throw "missing action"; }
        if (action.pack.join(".") != "widgets" || action.module != "widgets.Controls") throw "incorrect module path";
        var p = Context.getPosInfos(template.pos);
        var outer = Context.makePosition({file:p.file, min:p.min + 1, max:p.min + 15});
        var inner = Context.makePosition({file:p.file, min:p.min + 17, max:p.min + 26});
        var child:Expr = {expr:ENew({pack:["widgets"], name:"Controls", sub:"Action"}, []), pos:inner};
        return macro @:pos(outer) new widgets.Panel([$child]);
    }
}


"#;
    for config in [
        ("build.hxml", "-cp .\n--hl out.hl\n"),
        ("rayzor.toml", "[project]\nname = 'generated-tag-types'\n"),
    ] {
        let mut s = Session::start(&[
            config,
            ("Main.hx", source),
            ("Builder.hx", builder),
            (
                "Wrapper.hx",
                r#"
import haxe.macro.Expr;
import haxe.macro.Context;
using haxe.macro.ExprTools;
class Wrapper {
    static function lower(e:Expr):Expr return switch e.expr {
        case EMeta({name:":markup"}, _):
            var p = Context.getPosInfos(e.pos);
            var call = Context.makePosition({file:p.file, min:p.min, max:p.min + 1});
            macro @:pos(call) Builder.make($e);
        case _: e.map(lower);
    }
    public static macro function build():Array<Field> {
        var fields = Context.getBuildFields();
        for (field in fields) switch field.kind {
            case FFun(fn): if (fn.expr != null) fn.expr = lower(fn.expr);
            case _:
        }
        return fields;
    }
}
"#,
            ),
            (
                "widgets/Panel.hx",
                "package widgets; import widgets.Controls.Action; /** A composed panel. */ class Panel extends Base<{label:String}> { public function new(children:Array<Action>) {super();} }",
            ),
            (
                "widgets/Base.hx",
                "package widgets; class Base<T> { public function new() {} }",
            ),
            (
                "widgets/Controls.hx",
                "package widgets; class Controls {} /** A composed action. */ class Action { public function new() {} }",
            ),
        ]);
        for cached in [false, true] {
            assert!(s.open("Main.hx").is_empty(), "{:?}", s.diagnostics);
            if cached {
                assert!(
                    s.activity.iter().all(|event| event["state"] != "checking"),
                    "{:?}",
                    s.activity
                );
            }
            for (tag, class, file, docs) in [
                (
                    "arbitrary-name",
                    "Panel",
                    "widgets/Panel.hx",
                    "A composed panel",
                ),
                (
                    "clickable",
                    "Action",
                    "widgets/Controls.hx",
                    "A composed action",
                ),
            ] {
                let at = s.at("Main.hx", 0, source.find(tag).unwrap() as u32 + 1);
                let hover = hover_text(&s.request("textDocument/hover", at.clone()));
                assert!(
                    hover.contains(class) && hover.contains(docs),
                    "{tag}: {hover}"
                );
                assert!(
                    !hover.contains("Macro expansion") && !hover.contains("Builder.make"),
                    "{hover}"
                );
                let definition = s.request("textDocument/definition", at);
                assert!(definition.to_string().contains(file), "{definition}");
            }
            for offset in [0, 1, 8] {
                let at = s.at(
                    "Main.hx",
                    0,
                    source.find("<arbitrary-name").unwrap() as u32 + offset,
                );
                let expansion = s.request("rayzor/expandMacro", at);
                let text = expansion["expansion"].as_str().unwrap_or_default();
                assert!(
                    text.contains("new widgets.Panel")
                        && text.contains("new widgets.Controls.Action"),
                    "{expansion}"
                );
            }
            if !cached {
                let root = s.root.clone();
                s.temporary = false;
                drop(s);
                s = Session::connect(root, json!({}), true);
            }
        }
    }
}

#[test]
fn a_failed_method_keeps_generated_tag_references_in_other_methods() {
    let main = "class Main { static function broken():Int return missing; static function main() { var element = Builder.make(<custom-tag/>); var size = element.width; } }";
    let mut s = Session::start(&[
        ("Main.hx", main),
        (
            "Widget.hx",
            "class Widget { public var width:Int = 42; public function new() {} }",
        ),
        (
            "Builder.hx",
            r#"
import haxe.macro.Expr;
import haxe.macro.Context;
class Builder {
    public static macro function make(e:Expr):Expr {
        var info = Context.getPosInfos(e.pos);
        var tag = Context.makePosition({file:info.file, min:info.min + 1, max:info.min + 11});
        return macro @:pos(tag) new Widget();
    }
}
"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(
        diagnostics.iter().any(|diagnostic| diagnostic["message"]
            .as_str()
            .is_some_and(|message| message.contains("missing"))),
        "{diagnostics:?}"
    );
    let at = s.at("Main.hx", 0, main.find("custom-tag").unwrap() as u32 + 1);
    let hover = hover_text(&s.request("textDocument/hover", at.clone()));
    assert!(hover.contains("class Widget"), "{hover}");
    assert!(
        s.request("textDocument/definition", at)
            .to_string()
            .contains("Widget.hx")
    );
    let width = s.at("Main.hx", 0, main.find("width").unwrap() as u32 + 1);
    let hover = hover_text(&s.request("textDocument/hover", width));
    assert!(hover.contains("width:Int"), "{hover}");
}

#[test]
fn reflected_enum_fields_keep_declared_types_positions_and_typed_initializers() {
    let main = "import theme.Palette; class Main { static function main() { var colors = Builder.colors(); } }";
    let palette = "package theme;\n#if macro\nenum abstract Palette(String) { var Wrong = 'macro'; }\n#else\nenum abstract Palette(String) {\n#if palette_alternate\n var Zed = 'zed'; var Alpha = 'alpha';\n#else\n var Wrong = 'without-define';\n#end\n public static final ALL:Array<Palette> = [Zed, Alpha]; }\n#end\n";
    let mut s = Session::start(&[
        ("build.hxml", "-cp .\n-D palette_alternate\n--hl out.hl\n"),
        ("Main.hx", main),
        ("theme/Palette.hx", palette),
        (
            "Builder.hx",
            r#"
import haxe.macro.Context;
import haxe.macro.Expr;
using haxe.macro.TypeTools;
class Builder {
    public static macro function colors():Expr {
        var fields = switch Context.getType("theme.Palette") {
            case TAbstract(a, _):
                if (a.get().impl == null) throw "missing implementation: " + a.get().pack.join(".") + ":" + a.get().name + ":" + a.get().module;
                var self = a.get().pack.concat([a.get().name]).join(".");
                [for (field in a.get().impl.get().statics.get())
                    if (field.meta.has(":enum") && field.type.toString() == self) field];
            case _: throw "expected an abstract";
        }
        fields.sort((x, y) -> Context.getPosInfos(x.pos).min - Context.getPosInfos(y.pos).min);
        var names = [for (field in fields) switch field.expr() {
            case {expr:TCast({expr:TConst(TString(value))}, _)} | {expr:TConst(TString(value))}: value;
            case _: throw "expected a typed string initializer";
        }];
        if (names.join(",") != "zed,alpha") throw "wrong enum field reflection: " + names.join(",");
        return macro $v{names.join(",")};
    }
}
"#,
        ),
    ]);
    let diagnostics = s.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    s.open("theme/Palette.hx");
    s.change(
        "theme/Palette.hx",
        &palette.replace(
            "var Zed = 'zed'; var Alpha = 'alpha';",
            "var Alpha = 'alpha'; var Zed = 'zed';",
        ),
    );
    let diagnostics = s.change("Main.hx", main);
    assert!(
        diagnostics.iter().any(|diagnostic| diagnostic["message"]
            .as_str()
            .is_some_and(|message| message.contains("wrong enum field reflection: alpha,zed"))),
        "{diagnostics:#?}"
    );
}

#[test]
fn macro_expansion_uses_stdlib_signatures_and_host_commands() {
    let main = "class Main { static function main() { var panel = Builder.make(); panel.title; } }";
    let builder = r#"
import haxe.macro.Context;
import haxe.macro.Expr;
class Builder {
    public static macro function make():Expr {
        var sys = switch Context.getType("Sys") {
            case TInst(c, _): c.get();
            case _: throw "missing Sys class";
        }
        var command = null;
        for (field in sys.statics.get()) if (field.name == "command") command = field;
        if (command == null) throw "missing command declaration";
        switch command.type {
            case TFun(arguments, result):
                if (arguments.length != 2 || arguments[0].name != "cmd" || !arguments[1].opt)
                    throw "incorrect command parameters";
                if (!Context.unify(result, Context.typeof(macro 1))) throw "incorrect command result";
            case _: throw "command is not a function";
        }
        var status = Sys.systemName() == "Windows"
            ? Sys.command("cmd", ["/C", "exit 7"])
            : Sys.command("/bin/sh", ["-c", "printf macro-host-output; exit 7"]);
        if (status != 7) throw "incorrect host exit status";
        return macro new Widget();
    }
}
"#;
    for configuration in [
        ("build.hxml", "-cp .\n--hl out.hl\n"),
        ("rayzor.toml", "[project]\nname = 'macro-stdlib-snapshot'\n"),
    ] {
        let mut session = Session::start(&[
            configuration,
            ("Main.hx", main),
            ("Builder.hx", builder),
            (
                "Widget.hx",
                "class Widget { public var title:String; public function new() { title = 'panel'; } }",
            ),
        ]);
        let diagnostics = session.open("Main.hx");
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let at = session.at("Main.hx", 0, main.rfind("panel").unwrap() as u32 + 1);
        let hover = hover_text(&session.request("textDocument/hover", at));
        assert!(hover.contains("Widget"), "{hover}");
    }
}

#[test]
fn forwarded_generic_signal_reads_preserve_comprehension_and_closure_types() {
    let main = r#"import reactive.Signal;
import model.Task;
class Main {
    final tasks:Signal<Array<Task>>;
    final query:Signal<String>;
    public function new(tasks:Signal<Array<Task>>, query:Signal<String>) { this.tasks = tasks; this.query = query; }
    function move(id:Int, lane:String):Void {
        var text = query.get().toLowerCase();
        var filtered = tasks.get().filter(task -> task.title.indexOf(text) >= 0);
        tasks.set([for (task in tasks.get()) task.id != id ? task : {id:task.id, title:task.title, lane:lane}]);
        query.set("next");
    }
}"#;
    let mut session = Session::start(&[
        ("build.hxml", "-cp .\n--hl out.hl\n"),
        ("Main.hx", main),
        (
            "model/Task.hx",
            "package model; typedef Task = { final id:Int; final title:String; final lane:String; };",
        ),
        (
            "reactive/ISignal.hx",
            "package reactive; interface ISignal<T> { function get():T; function set(value:T):Void; }",
        ),
        (
            "reactive/Signal.hx",
            "package reactive; @:forward(get, set) abstract Signal<T>(ISignal<T>) from ISignal<T> to ISignal<T> {}",
        ),
    ]);
    let diagnostics = session.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    for (needle, expected) in [
        ("text =", "text:String"),
        ("task ->", "task:model.Task"),
        ("task.id", "task:model.Task"),
    ] {
        let line = main.lines().position(|line| line.contains(needle)).unwrap();
        let source_line = main.lines().nth(line).unwrap();
        let at = session.at(
            "Main.hx",
            line as u32,
            source_line.find(needle).unwrap() as u32 + 1,
        );
        let hover = hover_text(&session.request("textDocument/hover", at));
        assert!(hover.contains(expected), "expected {expected}, got {hover}");
    }
    session.notify(
        "textDocument/didClose",
        json!({"textDocument":{"uri":session.uri("Main.hx")}}),
    );
    let diagnostics = session.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let line = main
        .lines()
        .position(|line| line.contains("task.id"))
        .unwrap();
    let column = main.lines().nth(line).unwrap().find("task.id").unwrap();
    let at = session.at("Main.hx", line as u32, column as u32 + 1);
    assert!(hover_text(&session.request("textDocument/hover", at)).contains("task:model.Task"));
    let diagnostics = session.change(
        "Main.hx",
        &main.replace("query.set(\"next\")", "query.set(42)"),
    );
    assert!(
        diagnostics.iter().any(|d| d["message"]
            .as_str()
            .is_some_and(|message| message.contains("String") && message.contains("Int"))),
        "{diagnostics:#?}"
    );
    let diagnostics = session.change("Main.hx", &main.replace("title:task.title", "title:42"));
    assert!(
        !diagnostics.is_empty(),
        "an incompatible task field must remain an error"
    );
}

#[test]
fn macro_type_queries_load_unopened_sibling_components() {
    let main = "package ui; import macros.Builder; class Main { public function new() {} public function render() { var panel = Builder.make(); return panel; } }";
    let mut session = Session::start(&[
        ("build.hxml", "-cp .\n--hl out.hl\n"),
        ("ui/Main.hx", main),
        (
            "ui/One.hx",
            "package ui; class One { public function new() {} }",
        ),
        (
            "ui/Two.hx",
            "package ui; class Two { public function new() {} }",
        ),
        (
            "ui/Three.hx",
            "package ui; class Three { public function new() {} }",
        ),
        (
            "ui/Four.hx",
            "package ui; class Four { public function new() {} }",
        ),
        (
            "macros/Builder.hx",
            r#"
package macros;
import haxe.macro.Context;
import haxe.macro.Expr;
using StringTools;
class Builder {
 static var attempts = 0;
 public static macro function make():Expr {
  if (++attempts != 1) throw "suspended macro leaked its statics";
  if (StringTools.trim(" one ") != "one" || " two ".trim() != "two") throw "trim binding";
  if ([" one ", " two "].map(StringTools.trim).join(",") != "one,two") throw "trim callback";
  for (name in ["One", "Two", "Three", "Four"]) {
   var found = try Context.getType(name) catch (_:Dynamic) Context.getType("MissingFallback");
   switch found { case TInst(c, _): if (c.get().pack.join(".") != "ui") throw "wrong package"; case _: throw "not a class"; }
  }
  var missing = try { Context.getType("DefinitelyMissing"); false; } catch (_:Dynamic) true;
  if (!missing) throw "unknown type must throw";
  return macro new ui.Four();
 }
}
"#,
        ),
    ]);
    let diagnostics = session.open("ui/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let at = session.at("ui/Main.hx", 0, main.rfind("panel").unwrap() as u32 + 1);
    let hover = hover_text(&session.request("textDocument/hover", at));
    assert!(hover.contains("ui.Four"), "{hover}");
}

#[test]
#[ignore = "requires RAYZOR_TEST_COMPONENT_PROJECT pointing to an installed Ashui application"]
fn configured_ashui_component_expands_and_has_element_hover() {
    let root =
        PathBuf::from(std::env::var_os("RAYZOR_TEST_COMPONENT_PROJECT").expect("project path"));
    let file =
        std::env::var("RAYZOR_TEST_COMPONENT_FILE").unwrap_or_else(|_| "src/ui/BoardApp.hx".into());
    let source = std::fs::read_to_string(root.join(&file)).unwrap();
    let mut session = Session::connect(root, json!({"macroBudgetMs":0}), false);
    session.timeout = std::time::Duration::from_secs(900);
    let diagnostics = session.open(&file);
    eprintln!("diagnostics for {file}: {diagnostics:#?}");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let offset = source.find("<div").expect("div element") + 1;
    let before = &source[..offset];
    let at = session.at(
        &file,
        before.bytes().filter(|b| *b == b'\n').count() as u32,
        before.rsplit('\n').next().unwrap().encode_utf16().count() as u32,
    );
    let hover = hover_text(&session.request("textDocument/hover", at));
    assert!(hover.contains("class Div"), "{hover}");
}

#[test]
fn macros_can_query_and_call_a_type_defined_during_expansion() {
    let main = "package pages; import Builder; class Main { static function main() { var result = Builder.make(); trace(result); } }";
    let mut session = Session::start(&[
        ("pages/Main.hx", main),
        (
            "Builder.hx",
            r#"
import haxe.macro.Context;
import haxe.macro.Expr;
class Builder {
 public static macro function make():Expr {
  Context.defineType(macro class Generated {
   public static function value(x:Int):Int return x;
   public static function constrained<T:{a:Int}>(v:T):T return v;
  });
  switch Context.getType("Generated") {
   case TInst(c, _): if (c.get().name != "Generated") throw "wrong generated class";
   case _: throw "generated type was not published";
  }
  var invalid = try { Context.typeExpr(macro Generated.value("bad")); false; } catch (_:Dynamic) true;
  if (!invalid) throw "generated parameter types were lost";
  var constrained = try { Context.typeExpr(macro Generated.constrained({b:""})); false; } catch (_:Dynamic) true;
  if (!constrained) throw "generated generic constraints were lost";
  return macro Generated.value(1);
 }
}
"#,
        ),
    ]);
    let diagnostics = session.open("pages/Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let at = session.at("pages/Main.hx", 0, main.find("result").unwrap() as u32 + 1);
    let hover = hover_text(&session.request("textDocument/hover", at));
    assert!(hover.contains("result:Int"), "{hover}");
}

#[test]
fn generic_build_queries_publish_types_once_and_reject_later_duplicates() {
    for duplicate in [false, true] {
        let main = r#"
import haxe.macro.Expr;
class Main {
 static function main() {
  var first = new Box<Int>();
  SECOND_USE
 }
}
@:genericBuild(Builder.make()) class Box<T> {}
@:genericBuild(Builder.make()) class OtherBox<T> {}
class Builder {
 public static macro function make():ComplexType {
  haxe.macro.Context.defineType(macro class Generated { public function new() {} });
  haxe.macro.Context.getType("Generated");
  return macro : Generated;
 }
}
"#
        .replace(
            "SECOND_USE",
            if duplicate {
                "var second = new OtherBox<String>();"
            } else {
                ""
            },
        );
        let mut session = Session::start(&[("Main.hx", &main)]);
        let diagnostics = session.open("Main.hx");
        assert_eq!(
            diagnostics.is_empty(),
            !duplicate,
            "{duplicate}: {diagnostics:#?}"
        );
        if duplicate {
            assert!(
                diagnostics.iter().any(|diagnostic| diagnostic["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("already defined")),
                "{diagnostics:#?}"
            );
        } else {
            let hover =
                hover_text(&session.request("textDocument/hover", session.at("Main.hx", 4, 7)));
            assert!(hover.contains("Generated"), "{hover}");
        }
    }
}

#[test]
fn a_bad_child_does_not_narrow_other_children_to_the_first_component() {
    let source = r#"
class Main {
 static function main() {
  var children = ([new Title(), new Description(), 42] : Array<Element>);
 }
}
class Element { public function new() {} }
class Component<T> extends Element {}
class Title extends Component<{title:String}> {}
class Description extends Component<{description:String}> {}
"#;
    let mut session = Session::start(&[("Main.hx", source)]);
    let diagnostics = session.open("Main.hx");
    let errors: Vec<_> = diagnostics.iter().filter(|d| d["severity"] == 1).collect();
    assert_eq!(errors.len(), 1, "{diagnostics:#?}");
    let message = errors[0]["message"].as_str().unwrap();
    assert!(
        message.contains("expected `Element`") && message.contains("found `Int`"),
        "{message}"
    );
}

#[test]
fn deferred_expansions_load_newly_constructed_runtime_classes() {
    let main = "import Builder; import widgets.Factory; class Main { static function main() { var item = Factory.create(); var direct = Builder.make(); var count = item.value() + direct.value(); } }";
    let mut session = Session::start(&[
        ("Main.hx", main),
        (
            "Builder.hx",
            "import haxe.macro.Context; import haxe.macro.Expr; class Builder { public static macro function make():Expr { Context.typeof(macro 0); return macro new widgets.Item(); } }",
        ),
        (
            "widgets/Factory.hx",
            "package widgets; import Builder; class Factory { public static function create() return Builder.make(); }",
        ),
        (
            "widgets/Item.hx",
            "package widgets; import widgets.Base; import widgets.Values; class Item extends Base { public function new() { super(); } public function value() return Values.read(); }",
        ),
        (
            "widgets/Base.hx",
            "package widgets; class Base { public function new() {} }",
        ),
        (
            "widgets/Values.hx",
            "package widgets; class Values { public static function read() return 7; }",
        ),
    ]);
    let diagnostics = session.open("Main.hx");
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    let hover = hover_text(&session.request(
        "textDocument/hover",
        session.at("Main.hx", 0, main.find("count").unwrap() as u32 + 1),
    ));
    assert!(hover.contains("count:Int"), "{hover}");
}

#[test]
fn unicode_macro_positions_follow_the_current_document() {
    let main = "import widgets.Widget; class Main { static function main() { var label = 'λ'; var view = Builder.make(<box>…<widget/></box>); } }";
    let mut session = Session::start(&[
        ("Main.hx", main),
        (
            "widgets/Widget.hx",
            "package widgets; class Widget { public function new() {} }",
        ),
        (
            "Builder.hx",
            "import haxe.macro.Context; import haxe.macro.Expr; class Builder { public static macro function make(template:Expr):Expr { var p = Context.getPosInfos(template.pos); var tag = Context.makePosition({file:p.file, min:p.min + 7, max:p.min + 13}); return {expr: ENew({pack:['widgets'], name:'Widget'}, []), pos:tag}; } }",
        ),
    ]);
    for (version, source) in [
        main.to_string(),
        main.replace('λ', "😀β01234567890123456789"),
    ]
    .iter()
    .enumerate()
    {
        let diagnostics = if version == 0 {
            session.open("Main.hx")
        } else {
            session.change("Main.hx", source)
        };
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let offset = source.find("<widget").unwrap() + 1;
        let at = session.at("Main.hx", 0, source[..offset].encode_utf16().count() as u32);
        let hover = hover_text(&session.request("textDocument/hover", at));
        assert!(hover.contains("class Widget"), "{hover}");
    }
}
