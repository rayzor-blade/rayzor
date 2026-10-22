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
        };
        let root_uri = session.uri("");
        session.request(
            "initialize",
            json!({"processId": null, "rootUri": root_uri, "capabilities": {}, "initializationOptions": options}),
        );
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
        loop {
            match self
                .client
                .receiver
                .recv_timeout(std::time::Duration::from_secs(120))
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
                .recv_timeout(std::time::Duration::from_secs(120))
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
fn hover_answers_at_declarations_and_uses() {
    let mut s = Session::start(&[("Main.hx", MAIN)]);
    assert!(s.open("Main.hx").is_empty());

    let at = s.at("Main.hx", 1, 21);
    let decl = s.request("textDocument/hover", at);
    assert!(
        hover_text(&decl).contains("function add(Int, Int):Int"),
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
        let activity: Vec<_> = s
            .activity
            .iter()
            .filter(|event| event["version"] == version)
            .map(|event| event["state"].as_str().unwrap())
            .collect();
        assert_eq!(activity, ["checking", "ready"]);
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
