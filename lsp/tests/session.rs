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
}

impl Session {
    /// Writes `files` into a fresh workspace and initialises a server on it.
    fn start(files: &[(&str, &str)]) -> Self {
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
        let (server_end, client) = Connection::memory();
        let server = std::thread::spawn(move || rayzor_lsp::serve(&server_end));
        let mut session = Session {
            client,
            server: Some(server),
            root,
            next_id: 0,
            diagnostics: Vec::new(),
        };
        let root_uri = session.uri("");
        session.request(
            "initialize",
            json!({"processId": null, "rootUri": root_uri, "capabilities": {}}),
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
            match self.client.receiver.recv().unwrap() {
                Message::Response(resp) if resp.id == id => {
                    return resp.result.unwrap_or(Value::Null);
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
            if let Message::Notification(n) = self.client.receiver.recv().unwrap()
                && n.method == "textDocument/publishDiagnostics"
            {
                return n.params["diagnostics"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
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
                return n.params["diagnostics"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
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
        let _ = std::fs::remove_dir_all(&self.root);
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
