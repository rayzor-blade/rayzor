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
        loop {
            if let Message::Notification(n) = self.client.receiver.recv().unwrap() {
                if n.method == "window/logMessage" {
                    self.logs
                        .push(n.params["message"].as_str().unwrap_or("").into());
                    continue;
                }
                if n.method != "textDocument/publishDiagnostics" {
                    continue;
                }
                if n.params["uri"] == self.uri(name) {
                    return n.params["diagnostics"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default();
                }
                self.diagnostics.push(n.params);
            }
        }
    }

    /// Replaces `name`'s text in the editor (not on disk) and returns its
    /// diagnostics.
    fn change(&mut self, name: &str, text: &str) -> Vec<Value> {
        self.next_id += 1;
        self.notify(
            "textDocument/didChange",
            json!({"textDocument": {"uri": self.uri(name), "version": self.next_id}, "contentChanges": [{"text": text}]}),
        );
        loop {
            if let Message::Notification(n) = self.client.receiver.recv().unwrap()
                && n.method == "textDocument/publishDiagnostics"
            {
                if n.params["uri"] == self.uri(name) {
                    return n.params["diagnostics"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default();
                }
                self.diagnostics.push(n.params);
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
