use lsp_server::{Connection, Message, Notification, Request};
use serde_json::json;
use std::time::{Duration, Instant};

#[test]
fn reports_workspace_modules_and_qualified_document_types() {
    let root = std::env::temp_dir().join(format!("rayzor-index-progress-{}", std::process::id()));
    std::fs::create_dir_all(root.join("src/models")).unwrap();
    let root = root.canonicalize().unwrap();
    std::fs::write(root.join("build.hxml"), "-cp src\n-main models.Thing\n").unwrap();
    let source = "package models; class Thing { public static function main() {} }";
    std::fs::write(root.join("src/models/Thing.hx"), source).unwrap();
    for index in 0..64 {
        std::fs::write(
            root.join(format!("src/models/Extra{index}.hx")),
            format!("package models; class Extra{index} {{}}"),
        )
        .unwrap();
    }
    let uri = format!("file://{}", root.display());
    let document = format!("{uri}/src/models/Thing.hx");
    let (server, client) = Connection::memory();
    let session = rayzor_lsp::spawn_session(server).unwrap();
    client
        .sender
        .send(Message::Request(Request::new(
            1.into(),
            "initialize".into(),
            json!({
                "rootUri":uri, "capabilities":{"experimental":{"rayzorIndexProgress":true}}
            }),
        )))
        .unwrap();
    assert!(matches!(
        client
            .receiver
            .recv_timeout(Duration::from_secs(10))
            .unwrap(),
        Message::Response(_)
    ));
    client
        .sender
        .send(Message::Notification(Notification::new(
            "initialized".into(),
            json!({}),
        )))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut scanning = false;
    loop {
        let message = client
            .receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
        if let Message::Notification(note) = message
            && note.method == "rayzor/indexProgress"
            && note.params["scope"] == "workspace"
        {
            if note.params["state"] == "ready" {
                assert_eq!(note.params["completed"], 65);
                break;
            }
            assert!(
                note.params["subject"]
                    .as_str()
                    .unwrap()
                    .starts_with("models.")
            );
            assert!(
                note.params["uri"]
                    .as_str()
                    .unwrap()
                    .starts_with(&format!("{uri}/src/models/"))
            );
            assert_eq!(note.params["phase"], "scanning");
            scanning = true;
        }
    }
    assert!(scanning, "No workspace module progress was reported");
    client
        .sender
        .send(Message::Notification(Notification::new(
            "textDocument/didOpen".into(),
            json!({
                "textDocument":{"uri":document,"languageId":"haxe","version":1,"text":source}
            }),
        )))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut named_type = false;
    loop {
        let message = client
            .receiver
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
        if let Message::Notification(note) = message {
            if note.method == "rayzor/indexProgress" && note.params["scope"].is_null() {
                assert_eq!(note.params["subject"], "models.Thing");
                assert_eq!(note.params["version"], 1);
                named_type = true;
            }
            if note.method == "rayzor/analysisStatus" && note.params["state"] == "ready" {
                break;
            }
        }
    }
    assert!(named_type, "No qualified document type was reported");
    client
        .sender
        .send(Message::Request(Request::new(
            2.into(),
            "shutdown".into(),
            json!(null),
        )))
        .unwrap();
    loop {
        if let Message::Response(response) = client
            .receiver
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            && response.id == 2.into()
        {
            break;
        }
    }
    client
        .sender
        .send(Message::Notification(Notification::new(
            "exit".into(),
            json!(null),
        )))
        .unwrap();
    session.join().unwrap().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
